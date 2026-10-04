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
// tp == 2 the star reduce below degenerates to the historical two-party exchange.
//
// TRANSPORT. Payloads move with cudaMemcpyAsync(cudaMemcpyDeviceToDevice) over unified virtual
// addresses, which every 64-bit Linux CUDA context has: a device pointer already names its
// device, so this one entry point expresses a cross-device transfer as well as a local one. When
// the driver grants peer access and startup validates it, the copy is a direct device-to-device
// PCIe transfer. On this V100 topology the translated IOMMU domain prevents direct P2P; CUDA's
// UVA D2D implementation then selects its driver-managed staging path. Startup validates the
// exact UVA D2D route in either case. The collectives never allocate or copy through an explicit
// host buffer, and callers need no peer-access branch; enable_peer_access() only reports whether
// direct peer access was qualified.
//
// The equivalent cudaMemcpyPeerAsync entry point is deliberately NOT used: it is rejected inside a
// stream capture region (cudaErrorStreamCaptureUnsupported), which would make the whole
// tensor-parallel decode program uncapturable. See src/ops/common/allreduce.cu's pull_peer().
//
// ORDERING. Every transfer is a PULL: a rank reads a source into storage that same rank alone
// owns, on its own stream. Nothing a rank owns is ever written by a peer's stream, so the classic
// push hazard -- the peer overwriting a staging buffer that this rank has not finished reading yet
// -- cannot exist. The summing reduce is a star through rank 0 so that every rank ends up with the
// BITWISE-IDENTICAL result: ranks 1..N-1 publish their partial and rank 0 folds all of them into
// its own buffer in one fixed order, then broadcasts that exact buffer back. An all-to-all
// "each rank sums independently" reduce is deliberately not used: with N > 2 the ranks would sum
// the same operands in different orders and the last-bit result could differ across ranks, which
// would desynchronize the replicated residual and every per-rank GDN/KV state derived from it.
//
//   allreduce_sum(rank 0):   record(inputs_ready[0]); for s in 1..N-1 { wait(inputs_ready[s]);
//                            partial[s] -> scratch; scratch += buffer[0] }
//                            record(pull_done[0])
//   allreduce_sum(rank r>0): record(inputs_ready[r]); wait(pull_done[0]);
//                            buffer[0] -> buffer[r]; record(pull_done[r])
//   rank 0 then waits(pull_done[r]) for every r>0.
//
// The waits carry the whole correctness argument: wait(inputs_ready[s]) makes the inbound copy read
// a completed peer partial; wait(pull_done[r]) (on rank 0) is the write-after-read barrier that
// lets peer r overwrite its partial on the next call, and wait(pull_done[0]) (on a peer) both
// orders the peer after rank 0's read of its partial and publishes the summed result.
//
// POST-CONDITION. On return each rank's stream is ordered after every peer's read of that rank's
// inputs. Therefore an unbounded sequence of calls sharing the same buffers, staging, and
// PeerEvents needs NO host synchronization between calls -- which is exactly what a captured
// CUDA graph replays, and what a decode token's collectives do. For the 27B model that is
// 128 all-reduces (64 layers x 2 row-parallel projections: the mixer output and the MLP down
// projection) plus one allgather_rows per logit column.
//
// Everything in a call is stream-ordered CUDA work (memcpy, event record, event wait, kernel
// launch): there is no host round trip, host-memory spin flag, or explicit host staging allocation
// in the hot path. That is what makes the sequence capturable. With all device streams enrolled in
// ONE capture -- each peer stream joined to the origin's by an event fork -- the record/wait pairs
// above become graph EDGES rather than nodes, and the whole decode program instantiates as a
// single cross-device cudaGraphExec. Event objects are caller-owned and created once, because
// cudaEventCreate is not capturable.
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

#include <cuda_runtime.h> // cudaEvent_t

#include <array>
#include <cstddef>

namespace ninfer::ops {

// Qualifies the actual cross-device copy route at startup. On Linux, first resolve every CUDA
// device's PCI bus IDs to /sys/bus/pci/devices/<BDF>/iommu_group/type. DMA and DMA-FQ prevent
// direct P2P, so the function leaves peer access disabled and validates CUDA's UVA D2D fallback
// instead. Otherwise, when every ordered device pair advertises peer access, enable them and check
// distinct 16 KiB patterns with the collectives' UVA D2D API on their destination compute streams.
// Returns true only when direct peer access is enabled and every ordered pair's copy is exact. If
// direct P2P is unavailable or fails qualification, it is disabled and the same UVA D2D route is
// qualified with driver-managed staging; the function returns false after emitting one diagnostic.
// A failed UVA check or CUDA API error throws; inference must not continue.
//
// Call once during setup, before graph capture or concurrent inference. This function allocates
// temporary probe buffers, synchronizes each compute stream, and releases its buffers while
// restoring the caller's current device. Peer-access changes are context-level operations;
// none of this setup is graph-capturable or belongs in a hot path. A tp == 1 context returns false.
bool enable_peer_access(const ExecutionContext& ec);

// The reusable cross-device ordering events: two per device, created on that device with timing
// disabled.
//
//   inputs_ready(r) - recorded on rank r's stream once the inputs rank r contributes are complete;
//                     the peers wait on it before reading them.
//   pull_done(r)    - recorded on rank r's stream once rank r's inbound copy has finished reading
//                     the peer's source; the peer waits on it before overwriting that source.
//
// Re-recording an event a previous call already recorded is well defined: cudaStreamWaitEvent
// captures the event's state at the time the wait is issued, and the collectives always issue a
// call's waits before the next call's records, so one instance serves an unbounded number of
// sequential calls. Instances are not thread-safe: one instance belongs to one stream set. A
// moved-from instance holds no events and must not be passed to a collective (the ops reject it).
class PeerEvents {
public:
    explicit PeerEvents(const ExecutionContext& ec);
    ~PeerEvents();

    PeerEvents(const PeerEvents&)            = delete;
    PeerEvents& operator=(const PeerEvents&) = delete;
    PeerEvents(PeerEvents&& other) noexcept;
    PeerEvents& operator=(PeerEvents&& other) noexcept;

    [[nodiscard]] cudaEvent_t inputs_ready(int rank) const noexcept {
        return inputs_ready_[static_cast<std::size_t>(rank)];
    }

    [[nodiscard]] cudaEvent_t pull_done(int rank) const noexcept {
        return pull_done_[static_cast<std::size_t>(rank)];
    }

    // Number of ranks this set was created for.
    [[nodiscard]] int count() const noexcept { return tp_; }

    // False for a moved-from instance.
    [[nodiscard]] bool live() const noexcept {
        if (tp_ < 1) { return false; }
        for (int rank = 0; rank < tp_; ++rank) {
            if (inputs_ready_[static_cast<std::size_t>(rank)] == nullptr ||
                pull_done_[static_cast<std::size_t>(rank)] == nullptr) {
                return false;
            }
        }
        return true;
    }

private:
    int tp_ = 0;
    std::array<cudaEvent_t, kMaximumDevices> inputs_ready_{};
    std::array<cudaEvent_t, kMaximumDevices> pull_done_{};
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
 * output is that value rounded once to BF16 storage, and rank 0's local combine is the qualified
 * residual_add computation body (FP32 accumulation of the two BF16 operands, one
 * round-to-nearest-even on store) applied once per peer partial. The other ranks receive rank 0's
 * exact bytes. The Op holds no persistent state and allocates nothing.
 *
 * Requires `ec.tp >= 1`, `ec.tp <= kMaximumDevices`, a live `events` created for the same `tp`, and
 * (`ec.tp >= 2`) distinct devices. At `ec.tp == 1` it validates and returns. Consecutive calls
 * sharing the same arguments need no host synchronization between them.
 */
void allreduce_sum(const std::array<Tensor, kMaximumDevices>& buffer,
                   const std::array<Tensor, kMaximumDevices>& staging,
                   const ExecutionContext& ec, const PeerEvents& events);

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
 * `destination[r]` is written only by rank r's stream: rank r copies its own block locally and
 * pulls every peer's block. The transformation is a pure relocation of storage, so it is verified
 * by exact byte comparison. The Op holds no persistent state and allocates nothing.
 *
 * Requires `ec.tp >= 1`, `ec.tp <= kMaximumDevices`, and a live `events` created for the same `tp`.
 * At `ec.tp == 1` it validates and returns. Consecutive calls sharing the same arguments need no
 * host synchronization between them.
 */
void allgather_rows(const std::array<Tensor, kMaximumDevices>& destination,
                    const std::array<Tensor, kMaximumDevices>& part, const ExecutionContext& ec,
                    const PeerEvents& events);

/**
 * Exact one-way vocabulary gather used by DFlash's rank-0 selector. `destination` is the full
 * `[C,T]` image on rank 0; `part[r]` is rank r's contiguous width block. The first/contiguous
 * dimension is concatenated for every column. Ranks other than 0 are deliberately not written
 * because they never consume DFlash proposal logits.
 * The source lifetime edge still orders every peer after rank 0 has finished importing its shard,
 * so the same work buffers may be reused by the next graph round.
 */
void gather_columns_rank0(const Tensor& destination,
                          const std::array<Tensor, kMaximumDevices>& part,
                          const ExecutionContext& ec, const PeerEvents& events);

// Exact one-way relocation from rank 0 to every peer. `destination[r]` (r >= 1) has the same
// contiguous dtype and shape as `source`, and lives on `ec.dev[r]`; `destination[0]` is ignored.
// Uses the qualified UVA D2D transport and orders rank 0's next overwrite after every peer has
// consumed the source, including under CUDA Graph capture.
void broadcast_rank0(const Tensor& source,
                     const std::array<Tensor, kMaximumDevices>& destination,
                     const ExecutionContext& ec, const PeerEvents& events);

} // namespace ninfer::ops
