#pragma once

// ninfer::ops - N-rank small-message collectives over a device-visible mailbox.
//
// The 128 row-parallel all-reduces of one decode round and the vocabulary read of the output
// head are 10-128 KB BF16 messages. NCCL's small-message protocol (a flag per 8 bytes and a
// per-call kernel) costs ~26 us per message on this box regardless of the NVLink fabric; a
// purpose-built mailbox kernel moves the same slice with one flag per CTA, device-side spin, and
// a rank-ordered fold, at a fraction of that. This header owns the host-side plan and the
// fast-path entry points; the kernels and the startup probe live in tp_mailbox.cu.
//
// Two transports, probed and self-checked once at startup:
//   Host   - one mapped pinned host block shared by every rank; no GPU peer access required.
//   Device - one inbox per rank in its own device memory; every rank pushes its slice straight
//            into each peer's inbox over the P2P fabric (CUDA peer access).
// Both fold the per-rank contributions in rank order, so every rank ends with bit-identical
// bytes; both keep per-block step counters in device memory and double-buffer through parity
// slots, so repeated eager calls and CUDA-Graph replays are correct with no host synchronization.
// NCCL stays the fallback for messages over the mailbox capacity, non-BF16 traffic, and any rank
// count or shape the fast path does not cover.

#include "core/device.h"
#include "core/tensor.h"

#include <nccl.h>

#include <array>
#include <cstdint>
#include <memory>

namespace ninfer::ops::detail {

enum class TpMailboxTransport : std::int32_t { Nccl = 0, Host = 1, Device = 2 };

class TpMailboxPlan {
public:
    // Null when tp < 2, when NINFER_TP_MAILBOX=0, or when nothing passed the self-check. The
    // probe times NCCL and each surviving mailbox candidate on a decode-sized message and keeps
    // the fastest. `comms` is the caller's communicator set (the NCCL baseline).
    static std::unique_ptr<TpMailboxPlan> create(
        const ExecutionContext& ec, const std::array<ncclComm_t, kMaximumDevices>& comms);
    ~TpMailboxPlan();
    TpMailboxPlan(const TpMailboxPlan&)            = delete;
    TpMailboxPlan& operator=(const TpMailboxPlan&) = delete;

    [[nodiscard]] int tp() const noexcept;
    [[nodiscard]] TpMailboxTransport transport() const noexcept;

    // BF16 all-reduce fast path: every rank's buffer is overwritten with the rank-ordered
    // BF16-rounded sum of all ranks, bit-identical on every rank. False = the caller must use
    // NCCL.
    [[nodiscard]] bool allreduce_bf16(const std::array<Tensor, kMaximumDevices>& buffer,
                                      const ExecutionContext& ec);
    // BF16 row-gather fast path: every rank's destination is filled with all ranks' parts at
    // their prefix element offsets, bit-identical on every rank. False = the caller must use
    // NCCL.
    [[nodiscard]] bool allgather_rows_bf16(const std::array<Tensor, kMaximumDevices>& destination,
                                           const std::array<Tensor, kMaximumDevices>& part,
                                           const ExecutionContext& ec);

    struct Impl;

private:
    explicit TpMailboxPlan(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::ops::detail
