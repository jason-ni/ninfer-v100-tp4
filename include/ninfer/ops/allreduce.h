#pragma once

// ninfer::ops - N-device collectives (tensor parallelism over tp in {1,2,4}).
//
// These are the cross-device primitives every tensor-parallel forward pass needs: one summing
// reduction for row-parallel projections, one row gather for the vocabulary-split output head, and
// the one-way rank-0 gather/broadcast used by the rank-0-owned selector paths. All operate on an
// ExecutionContext holding `tp` DeviceContexts and use each device's own compute stream; no stream
// parameter is accepted, because "the stream a device executes on" is already a property of its
// DeviceContext and a second, disagreeing stream story would break the ordering guarantees below.
//
// The implementation is ONE rank-generic algorithm parameterized by the runtime `tp`, not a
// separate two-rank and four-rank path. At tp == 1 every collective is a validation-only no-op; at
// tp == 2 the same grouped NCCL calls are simply a two-rank exchange.
//
// TRANSPORT. Cross-device traffic runs over NCCL, one communicator per rank, which on this
// V100-SXM2 mesh uses the NVLink P2P fabric directly (measured ~48 GB/s per direction on every
// ordered pair). Every collective is issued as a grouped per-rank call: ncclGroupStart(), one
// nccl call per rank on that rank's own DeviceContext::stream, ncclGroupEnd(). Grouping is
// mandatory here because a single host thread drives all communicators; without it the first
// collective deadlocks waiting for a peer call the same thread has not issued yet.
//
// The calls are stream-ordered GPU work with no host round trip, host-memory spin flag, or
// explicit host staging, so they are CUDA-Graph capturable. Ordering between consecutive
// collectives comes from NCCL's own stream semantics; there are no cross-stream cudaEvents, which
// is what lets each rank be captured into its own independent graph.
//
// The all-reduce is a plain in-place ncclAllReduce(ncclSum) on every rank and is NOT guaranteed
// bitwise-identical across ranks (NCCL may associate the reduction differently per rank). That is
// consistent with the Op oracle (an FP64 sum rounded once to BF16) and with the reference
// tensor-parallel implementations; rank-to-rank lockstep is preserved because every token decision
// is taken on rank 0 and published to the peers with broadcast_rank0.
//
// POST-CONDITION. On return each rank's stream is ordered after the collective; an unbounded
// sequence of calls reusing the same buffers and communicators needs no host synchronization
// between calls, which is exactly what a captured CUDA graph replays. For the 27B model that is
// 128 all-reduces per decode token (64 layers x 2 row-parallel projections: the mixer output and
// the MLP down projection) plus one allgather_rows per logit column.
//
// CALLER OBLIGATION -- the legacy default stream. These ops read their inputs on
// DeviceContext::stream, which core creates with cudaStreamNonBlocking and which therefore does
// NOT implicitly synchronize with a device's legacy default stream. Work issued with the plain
// cudaMemcpy/cudaMemset/<<<...>>> forms (no stream argument) lands on that default stream and is
// unordered against these collectives. A caller that stages inputs that way must retire them
// first (a device or stream synchronize, or an event); producing the inputs on
// DeviceContext::stream needs no extra step. This trap is invisible in single-device code, where
// the default stream orders everything.
//
// After a call returns the work is enqueued but not complete; the caller synchronizes each
// device's own stream, or lets subsequent stream-ordered work depend on it, as usual.

#include "core/device.h" // DeviceContext, ExecutionContext, kMaximumDevices
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t
#include <nccl.h>         // ncclComm_t

#include <array>
#include <cstddef>

namespace ninfer::ops {

// Owns one NCCL communicator per rank for an ExecutionContext with `tp` in {1,2,4}. Construction
// is a one-time startup cost (ncclCommInitAll over the device ids); the collectives below only
// issue stream work. At tp == 1 no communicator is created and every collective is a
// validation-only no-op, so single-device inference never links a live communicator.
//
// The communicators use NVLink P2P when the topology allows it and are the single cross-device
// ordering mechanism: consecutive collectives are ordered by NCCL's stream semantics alone, with
// no cross-stream cudaEvents. Instances are not thread-safe: one instance belongs to the single
// inference thread. A moved-from instance holds no communicators and must not be passed to a
// collective (the ops reject it).
class CollectiveComms {
public:
    explicit CollectiveComms(const ExecutionContext& ec);
    ~CollectiveComms();

    CollectiveComms(const CollectiveComms&)            = delete;
    CollectiveComms& operator=(const CollectiveComms&) = delete;
    CollectiveComms(CollectiveComms&& other) noexcept;
    CollectiveComms& operator=(CollectiveComms&& other) noexcept;

    [[nodiscard]] ncclComm_t comm(int rank) const noexcept {
        return comms_[static_cast<std::size_t>(rank)];
    }

    // Number of ranks this set was created for.
    [[nodiscard]] int count() const noexcept { return tp_; }

    // False for a moved-from instance or one whose communicators failed to initialize. A tp == 1
    // set is live and its collectives are no-ops.
    [[nodiscard]] bool live() const noexcept {
        if (tp_ < 1) { return false; }
        if (tp_ == 1) { return true; }
        for (int rank = 0; rank < tp_; ++rank) {
            if (comms_[static_cast<std::size_t>(rank)] == nullptr) { return false; }
        }
        return true;
    }

private:
    int tp_ = 0;
    std::array<ncclComm_t, kMaximumDevices> comms_{};
};

/**
 * N-device summing all-reduce, written to every rank's buffer:
 *
 *   ideal[i] = sum_r buffer_rank_r[i]   for every i,
 *
 * written back to every `buffer[r]`, which afterwards hold the bitwise-identical sum.
 *
 * `buffer[r]` is a contiguous BF16 tensor resident on `ec.dev[r]`; every rank carries the same
 * shape. `staging[r]` is scratch of the same dtype and shape, also resident on `ec.dev[r]`, and
 * must not overlap `buffer[r]`. The implementation uses only rank 0's `staging[0]` as the star
 * accumulator; every rank must still supply a valid staging slot because the caller owns the
 * per-rank allocation uniformly.
 *
 * The oracle evaluates `ideal` in FP64 from the represented BF16 inputs; the observable BF16
 * output is that value rounded once to BF16 storage. NCCL's in-place ncclAllReduce(ncclSum)
 * evaluates that sum with an internal association that need not match the oracle bit-for-bit, so
 * the output is checked within the BF16 rounding bound rather than by byte equality. The Op holds
 * no persistent state and allocates nothing.
 *
 * Requires `ec.tp >= 1`, `ec.tp <= kMaximumDevices`, a live `comms` created for the same `tp`, and
 * (`ec.tp >= 2`) distinct devices. At `ec.tp == 1` it validates and returns. Consecutive calls
 * sharing the same arguments need no host synchronization between them. `staging` is validated for
 * residency and shape but is otherwise unused: ncclAllReduce reduces in place. It is retained only
 * to keep the row-parallel projection call sites unchanged and will be dropped with them.
 */
void allreduce_sum(const std::array<Tensor, kMaximumDevices>& buffer,
                   const std::array<Tensor, kMaximumDevices>& staging,
                   const ExecutionContext& ec, const CollectiveComms& comms);

/**
 * N-device row gather, exact (no arithmetic). The gathered axis is `ne[1]`, so each rank
 * contributes one contiguous block of a `[C, R]` tensor (`ne[0] == C` is the row length,
 * `ne[1] == R` the row count, `ne[2] == ne[3] == 1`). Rank r owns a contiguous row range; the
 * ranges concatenate, in rank order, to the destination row count:
 *
 *   destination[r][c, row] = part[s][c, row - offset(s)]   where offset(s) = sum_{u<s} part[u].ne[1]
 *
 * for every destination rank r and every source rank s, so each device ends up holding the
 * identical full image. `part[r]` is the row range owned by `ec.dev[r]`: rank 0 owns the leading
 * rows and higher ranks the following ranges, and the per-rank counts must sum to the destination
 * row count. `destination[r]` and `part[r]` are contiguous, share one dtype, agree on `ne[0]`,
 * live on `ec.dev[r]`, and must not overlap. A caller whose split axis is not `ne[1]` (for example
 * `[V, B]` logits split by vocabulary with B > 1) arranges the layout so that it is; this Op
 * relocates contiguous storage and performs no transpose.
 *
 * `destination[r]` is filled by grouped point-to-point NCCL send/receive (so uneven per-rank
 * blocks are allowed; the current registered models split the output head evenly). Rank s's
 * `part[s]` lands at row offset(s) of every destination, so every rank ends up holding the
 * identical full image. The transformation is a pure relocation of storage, so it is verified by
 * exact byte comparison. The Op holds no persistent state and allocates nothing.
 *
 * Requires `ec.tp >= 1`, `ec.tp <= kMaximumDevices`, a live `comms` created for the same `tp`.
 * At `ec.tp == 1` it validates and returns. Consecutive calls sharing the same arguments need no
 * host synchronization between them.
 */
void allgather_rows(const std::array<Tensor, kMaximumDevices>& destination,
                    const std::array<Tensor, kMaximumDevices>& part, const ExecutionContext& ec,
                    const CollectiveComms& comms);

/**
 * Exact one-way vocabulary gather used by DFlash's rank-0 selector. `destination` is the full
 * `[C,T]` image on rank 0; `part[r]` is rank r's contiguous width block. The first/contiguous
 * dimension is concatenated for every column. Ranks other than 0 are deliberately not written
 * because they never consume DFlash proposal logits.
 * Each peer sends its contiguous block to rank 0, which places it with the same strided addressing
 * the previous UVA path used, so the same work buffers may be reused by the next graph round.
 */
void gather_columns_rank0(const Tensor& destination,
                          const std::array<Tensor, kMaximumDevices>& part,
                          const ExecutionContext& ec, const CollectiveComms& comms);

// Exact one-way relocation from rank 0 to every peer. `destination[r]` (r >= 1) has the same
// contiguous dtype and shape as `source`, and lives on `ec.dev[r]`; `destination[0]` is ignored.
// Uses ncclBroadcast(root 0) and orders rank 0's next overwrite after every peer has consumed the
// source, including under CUDA Graph capture.
void broadcast_rank0(const Tensor& source,
                     const std::array<Tensor, kMaximumDevices>& destination,
                     const ExecutionContext& ec, const CollectiveComms& comms);

} // namespace ninfer::ops
