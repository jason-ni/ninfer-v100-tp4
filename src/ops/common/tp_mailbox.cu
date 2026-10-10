#include "ops/common/tp_mailbox.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops::detail {
namespace {

// Slice geometry. kMailboxSlice * kMailboxMaxBlocks is the largest all-reduce the fast path
// takes (64K bf16 = 128 KB); the gather covers 1M bf16 (2 MB) per rank, which is the TP2
// vocabulary shard at the widest verify window.
constexpr int kMailboxSlice       = 1024; // bf16 elements per CTA
constexpr int kMailboxMaxBlocks   = 64;
constexpr int kMailboxMaxElements = kMailboxSlice * kMailboxMaxBlocks;
constexpr int kGatherChunk        = 8192; // elements per CTA
constexpr int kGatherMaxBlocks    = 128;
constexpr int kGatherMaxElements  = kGatherChunk * kGatherMaxBlocks;
constexpr std::size_t kRanks      = kMaximumDevices;

// A peer that never answers must not leave the kernel spinning forever (the context would never
// exit and the supervisor's restart would stall): give up after ~60 s at the 1.4 GHz clock and
// fault the context.
constexpr long long kSpinTimeoutCycles = 83'000'000'000LL;

// [source rank][block]; each flag sits 128 B from its neighbour so CTAs on different blocks do
// not share a cache line. Slots alternate by step parity, which is safe because a rank cannot
// reach step s+2 before the peer finished step s (each rank's stream serializes its own kernels,
// and step s+1's flag is only raised after the peer published its step s+1 slice).
struct alignas(128) HostMailbox {
    std::uint64_t pad0[16];
    std::uint64_t flag[kRanks][kMailboxMaxBlocks][16];
    __nv_bfloat16 data[kRanks][2][kMailboxMaxElements];
    std::uint64_t gflag[kRanks][kGatherMaxBlocks][16];
    __nv_bfloat16 gdata[kRanks][2][kGatherMaxElements];
};

// One rank's device inbox: peers push into data[their rank], so a rank only ever reads its own
// memory. Same slice/step/parity protocol as the host block.
struct alignas(128) Inbox {
    std::uint64_t flag[kRanks][kMailboxMaxBlocks][16];
    __nv_bfloat16 data[kRanks][2][kMailboxMaxElements];
    std::uint64_t gflag[kRanks][kGatherMaxBlocks][16];
    __nv_bfloat16 gdata[kRanks][2][kGatherMaxElements];
};

// Row-gather ranges, uniform across ranks: rank r's contiguous part (count[r] elements) lands at
// destination element offset[r] on every rank. Offsets are element counts and must be multiples
// of 8 so the vector path stays 16-byte aligned.
struct GatherRanges {
    std::int64_t offset[kRanks];
    std::int64_t count[kRanks];
    std::int32_t ranks;
};

__device__ __forceinline__ void fold_rank_order_bf16x8(uint4& acc, const uint4& other) {
    auto* a        = reinterpret_cast<__nv_bfloat162*>(&acc);
    const auto* b  = reinterpret_cast<const __nv_bfloat162*>(&other);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 fa = __bfloat1622float2(a[j]);
        const float2 fb = __bfloat1622float2(b[j]);
        a[j]            = __float22bfloat162_rn(make_float2(fa.x + fb.x, fa.y + fb.y));
    }
}

__device__ __forceinline__ void fold_rank_order_bf16(__nv_bfloat16& acc, __nv_bfloat16 other) {
    acc = __float2bfloat16(__bfloat162float(acc) + __bfloat162float(other));
}

__device__ __forceinline__ void spin_wait(const volatile std::uint64_t* flag, std::uint64_t step,
                                          long long start) {
    while (*flag < step) {
        if (clock64() - start > kSpinTimeoutCycles) { __trap(); }
    }
}

// ---- all-reduce ------------------------------------------------------------------------------

// Host transport: every rank publishes its slice into the shared pinned block and folds every
// rank's slice (own included) back in rank order.
__global__ void host_mailbox_allreduce_kernel(__nv_bfloat16* x, int n, int rank, int ranks,
                                              HostMailbox* mb, std::uint64_t* steps) {
    const int blk            = static_cast<int>(blockIdx.x);
    const std::uint64_t step = steps[blk] + 1;
    const int parity         = static_cast<int>(step & 1);
    const int base           = blk * kMailboxSlice;
    const int end            = ::min(n, base + kMailboxSlice);
    const int tid            = static_cast<int>(threadIdx.x);
    const int rem            = (end - base) & 7;
    const int vec_end        = end - rem;

    for (int e = base + tid * 8; e < vec_end; e += blockDim.x * 8) {
        store_vec(&mb->data[rank][parity][e], load_vec<uint4>(&x[e]));
    }
    if (tid < rem) {
        const int e = vec_end + tid;
        store_vec(&mb->data[rank][parity][e], load_vec<__nv_bfloat16>(&x[e]));
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        *reinterpret_cast<volatile std::uint64_t*>(&mb->flag[rank][blk][0]) = step;
        const long long start = clock64();
        for (int src = 0; src < ranks; ++src) {
            if (src == rank) { continue; }
            spin_wait(reinterpret_cast<const volatile std::uint64_t*>(&mb->flag[src][blk][0]), step,
                      start);
        }
    }
    __syncthreads();
    __threadfence_system();

    for (int e = base + tid * 8; e < vec_end; e += blockDim.x * 8) {
        uint4 acc = __ldcv(reinterpret_cast<const uint4*>(&mb->data[0][parity][e]));
        for (int src = 1; src < ranks; ++src) {
            fold_rank_order_bf16x8(
                acc, __ldcv(reinterpret_cast<const uint4*>(&mb->data[src][parity][e])));
        }
        store_vec(&x[e], acc);
    }
    if (tid < rem) {
        const int e = vec_end + tid;
        __nv_bfloat16 acc = mb->data[0][parity][e];
        for (int src = 1; src < ranks; ++src) {
            fold_rank_order_bf16(acc, mb->data[src][parity][e]);
        }
        x[e] = acc;
    }
    if (tid == 0) { steps[blk] = step; }
}

// Device transport: every rank pushes its slice straight into each peer's inbox and then reads
// every rank's slice from its own inbox (own value from the source buffer).
__global__ void nvlink_allreduce_kernel(__nv_bfloat16* x, int n, int rank, int ranks, Inbox* mine,
                                        Inbox* const* peers, std::uint64_t* steps) {
    const int blk            = static_cast<int>(blockIdx.x);
    const std::uint64_t step = steps[blk] + 1;
    const int parity         = static_cast<int>(step & 1);
    const int base           = blk * kMailboxSlice;
    const int end            = ::min(n, base + kMailboxSlice);
    const int tid            = static_cast<int>(threadIdx.x);
    const int rem            = (end - base) & 7;
    const int vec_end        = end - rem;

    for (int e = base + tid * 8; e < vec_end; e += blockDim.x * 8) {
        const uint4 v = load_vec<uint4>(&x[e]);
        for (int src = 0; src < ranks; ++src) {
            if (src == rank) { continue; }
            store_vec(&peers[src]->data[rank][parity][e], v);
        }
    }
    if (tid < rem) {
        const int e = vec_end + tid;
        const __nv_bfloat16 v = load_vec<__nv_bfloat16>(&x[e]);
        for (int src = 0; src < ranks; ++src) {
            if (src == rank) { continue; }
            peers[src]->data[rank][parity][e] = v;
        }
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        for (int src = 0; src < ranks; ++src) {
            if (src == rank) { continue; }
            *reinterpret_cast<volatile std::uint64_t*>(&peers[src]->flag[rank][blk][0]) = step;
        }
        const long long start = clock64();
        for (int src = 0; src < ranks; ++src) {
            if (src == rank) { continue; }
            spin_wait(reinterpret_cast<const volatile std::uint64_t*>(&mine->flag[src][blk][0]),
                      step, start);
        }
    }
    __syncthreads();
    __threadfence_system();

    // Fold strictly in source order 0, 1, ..., ranks-1 on every rank: this rank's own value is
    // substituted for its source slot, so all ranks evaluate the identical association and end
    // with bit-identical bytes. (Folding each rank's own value first is commutative for two ranks
    // but not associative for more.)
    for (int e = base + tid * 8; e < vec_end; e += blockDim.x * 8) {
        const uint4 own = load_vec<uint4>(&x[e]);
        uint4 acc       = rank == 0
                              ? own
                              : __ldcv(reinterpret_cast<const uint4*>(&mine->data[0][parity][e]));
        for (int src = 1; src < ranks; ++src) {
            fold_rank_order_bf16x8(
                acc, src == rank ? own
                                 : __ldcv(reinterpret_cast<const uint4*>(&mine->data[src][parity][e])));
        }
        store_vec(&x[e], acc);
    }
    if (tid < rem) {
        const int e = vec_end + tid;
        const __nv_bfloat16 own = load_vec<__nv_bfloat16>(&x[e]);
        __nv_bfloat16 acc = rank == 0 ? own : mine->data[0][parity][e];
        for (int src = 1; src < ranks; ++src) {
            fold_rank_order_bf16(acc,
                                 src == rank ? own : mine->data[src][parity][e]);
        }
        x[e] = acc;
    }
    if (tid == 0) { steps[blk] = step; }
}

// ---- row gather ------------------------------------------------------------------------------

// Relocate each rank's contiguous part to its destination offset. Both transports write the
// local block first, publish it to every peer, then copy every peer's block.
__global__ void host_gather_rows_kernel(const __nv_bfloat16* local, __nv_bfloat16* out,
                                        GatherRanges ranges, int rank, HostMailbox* mb,
                                        std::uint64_t* steps) {
    const int blk            = static_cast<int>(blockIdx.x);
    const std::uint64_t step = steps[blk] + 1;
    const int parity         = static_cast<int>(step & 1);
    const int base           = blk * kGatherChunk;
    const int tid            = static_cast<int>(threadIdx.x);
    const int ranks          = static_cast<int>(ranges.ranks);
    const int own_count      = static_cast<int>(ranges.count[rank]);
    const int own_end        = ::min(own_count, base + kGatherChunk);
    const int own_rem        = own_end > base ? (own_end - base) & 7 : 0;
    const int own_vec_end    = own_end - own_rem;

    for (int e = base + tid * 8; e < own_vec_end; e += blockDim.x * 8) {
        const uint4 v = load_vec<uint4>(&local[e]);
        store_vec(&out[ranges.offset[rank] + e], v);
        for (int src = 0; src < ranks; ++src) {
            if (src == rank) { continue; }
            store_vec(&mb->gdata[rank][parity][e], v);
        }
    }
    if (tid < own_rem) {
        const int e = own_vec_end + tid;
        const __nv_bfloat16 v = local[e];
        out[ranges.offset[rank] + e] = v;
        for (int src = 0; src < ranks; ++src) {
            if (src == rank) { continue; }
            mb->gdata[rank][parity][e] = v;
        }
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        *reinterpret_cast<volatile std::uint64_t*>(&mb->gflag[rank][blk][0]) = step;
        const long long start = clock64();
        for (int src = 0; src < ranks; ++src) {
            if (src == rank) { continue; }
            spin_wait(reinterpret_cast<const volatile std::uint64_t*>(&mb->gflag[src][blk][0]),
                      step, start);
        }
    }
    __syncthreads();
    __threadfence_system();

    for (int src = 0; src < ranks; ++src) {
        if (src == rank) { continue; }
        const int count   = static_cast<int>(ranges.count[src]);
        const int end     = ::min(count, base + kGatherChunk);
        if (end <= base) { continue; }
        const int rem     = (end - base) & 7;
        const int vec_end = end - rem;
        for (int e = base + tid * 8; e < vec_end; e += blockDim.x * 8) {
            const uint4 v = __ldcv(reinterpret_cast<const uint4*>(&mb->gdata[src][parity][e]));
            store_vec(&out[ranges.offset[src] + e], v);
        }
        if (tid < rem) {
            const int e = vec_end + tid;
            out[ranges.offset[src] + e] = mb->gdata[src][parity][e];
        }
    }
    if (tid == 0) { steps[blk] = step; }
}

__global__ void nvlink_gather_rows_kernel(const __nv_bfloat16* local, __nv_bfloat16* out,
                                          GatherRanges ranges, int rank, Inbox* mine,
                                          Inbox* const* peers, std::uint64_t* steps) {
    const int blk            = static_cast<int>(blockIdx.x);
    const std::uint64_t step = steps[blk] + 1;
    const int parity         = static_cast<int>(step & 1);
    const int base           = blk * kGatherChunk;
    const int tid            = static_cast<int>(threadIdx.x);
    const int ranks          = static_cast<int>(ranges.ranks);
    const int own_count      = static_cast<int>(ranges.count[rank]);
    const int own_end        = ::min(own_count, base + kGatherChunk);
    const int own_rem        = own_end > base ? (own_end - base) & 7 : 0;
    const int own_vec_end    = own_end - own_rem;

    for (int e = base + tid * 8; e < own_vec_end; e += blockDim.x * 8) {
        const uint4 v = load_vec<uint4>(&local[e]);
        store_vec(&out[ranges.offset[rank] + e], v);
        for (int src = 0; src < ranks; ++src) {
            if (src == rank) { continue; }
            store_vec(&peers[src]->gdata[rank][parity][e], v);
        }
    }
    if (tid < own_rem) {
        const int e = own_vec_end + tid;
        const __nv_bfloat16 v = local[e];
        out[ranges.offset[rank] + e] = v;
        for (int src = 0; src < ranks; ++src) {
            if (src == rank) { continue; }
            peers[src]->gdata[rank][parity][e] = v;
        }
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        for (int src = 0; src < ranks; ++src) {
            if (src == rank) { continue; }
            *reinterpret_cast<volatile std::uint64_t*>(&peers[src]->gflag[rank][blk][0]) = step;
        }
        const long long start = clock64();
        for (int src = 0; src < ranks; ++src) {
            if (src == rank) { continue; }
            spin_wait(reinterpret_cast<const volatile std::uint64_t*>(&mine->gflag[src][blk][0]),
                      step, start);
        }
    }
    __syncthreads();
    __threadfence_system();

    for (int src = 0; src < ranks; ++src) {
        if (src == rank) { continue; }
        const int count   = static_cast<int>(ranges.count[src]);
        const int end     = ::min(count, base + kGatherChunk);
        if (end <= base) { continue; }
        const int rem     = (end - base) & 7;
        const int vec_end = end - rem;
        for (int e = base + tid * 8; e < vec_end; e += blockDim.x * 8) {
            const uint4 v = __ldcv(reinterpret_cast<const uint4*>(&mine->gdata[src][parity][e]));
            store_vec(&out[ranges.offset[src] + e], v);
        }
        if (tid < rem) {
            const int e = vec_end + tid;
            out[ranges.offset[src] + e] = mine->gdata[src][parity][e];
        }
    }
    if (tid == 0) { steps[blk] = step; }
}

// ---- host-side plan --------------------------------------------------------------------------

class DeviceGuard {
public:
    DeviceGuard() { CUDA_CHECK(cudaGetDevice(&previous_)); }
    ~DeviceGuard() { (void)cudaSetDevice(previous_); }
    DeviceGuard(const DeviceGuard&)            = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;
    static void set(int device) { CUDA_CHECK(cudaSetDevice(device)); }

private:
    int previous_ = 0;
};

[[noreturn]] void nccl_fail(ncclResult_t result, const char* what) {
    throw std::runtime_error(std::string("TP mailbox ") + what + ": " +
                             ncclGetErrorString(result));
}

void nccl_check(ncclResult_t result, const char* what) {
    if (result != ncclSuccess) { nccl_fail(result, what); }
}

} // namespace

struct TpMailboxPlan::Impl {
    int tp                       = 0;
    TpMailboxTransport transport = TpMailboxTransport::Nccl;
    HostMailbox* host            = nullptr;
    std::array<HostMailbox*, kRanks> host_dev{};
    std::array<int, kRanks> device{-1, -1, -1, -1};
    std::array<Inbox*, kRanks> inbox{};
    std::array<std::uint64_t*, kRanks> steps{};
    std::array<std::uint64_t*, kRanks> gsteps{};
    std::array<Inbox**, kRanks> peer_ptrs{};

    ~Impl() { release(); }

    void release() {
        if (host != nullptr) {
            (void)cudaFreeHost(host);
            host = nullptr;
        }
        // Device allocations belong to different devices; each free runs under its owner.
        int previous = -1;
        (void)cudaGetDevice(&previous);
        for (std::size_t r = 0; r < kRanks; ++r) {
            if (device[r] < 0) { continue; }
            (void)cudaSetDevice(device[r]);
            if (inbox[r] != nullptr) {
                (void)cudaFree(inbox[r]);
                inbox[r] = nullptr;
            }
            if (steps[r] != nullptr) {
                (void)cudaFree(steps[r]);
                steps[r] = nullptr;
            }
            if (gsteps[r] != nullptr) {
                (void)cudaFree(gsteps[r]);
                gsteps[r] = nullptr;
            }
            if (peer_ptrs[r] != nullptr) {
                (void)cudaFree(peer_ptrs[r]);
                peer_ptrs[r] = nullptr;
            }
        }
        if (previous >= 0) { (void)cudaSetDevice(previous); }
    }
};

namespace {

void launch_allreduce(const TpMailboxPlan::Impl& impl, TpMailboxTransport transport,
                      const ExecutionContext& ec, const std::array<__nv_bfloat16*, kRanks>& buffer,
                      int n) {
    const int blocks = (n + kMailboxSlice - 1) / kMailboxSlice;
    for (int r = 0; r < impl.tp; ++r) {
        DeviceGuard::set(ec.dev[r]->device);
        if (transport == TpMailboxTransport::Device) {
            nvlink_allreduce_kernel<<<blocks, 64, 0, ec.dev[r]->stream>>>(
                buffer[r], n, r, impl.tp, impl.inbox[r], impl.peer_ptrs[r], impl.steps[r]);
        } else {
            host_mailbox_allreduce_kernel<<<blocks, 64, 0, ec.dev[r]->stream>>>(
                buffer[r], n, r, impl.tp, impl.host_dev[r], impl.steps[r]);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

void launch_gather(const TpMailboxPlan::Impl& impl, TpMailboxTransport transport,
                   const ExecutionContext& ec,
                   const std::array<const __nv_bfloat16*, kRanks>& local,
                   const std::array<__nv_bfloat16*, kRanks>& out, const GatherRanges& ranges) {
    std::int64_t max_count = 0;
    for (int r = 0; r < impl.tp; ++r) { max_count = std::max(max_count, ranges.count[r]); }
    const int blocks = static_cast<int>((max_count + kGatherChunk - 1) / kGatherChunk);
    for (int r = 0; r < impl.tp; ++r) {
        DeviceGuard::set(ec.dev[r]->device);
        if (transport == TpMailboxTransport::Device) {
            nvlink_gather_rows_kernel<<<blocks, 256, 0, ec.dev[r]->stream>>>(
                local[r], out[r], ranges, r, impl.inbox[r], impl.peer_ptrs[r], impl.gsteps[r]);
        } else {
            host_gather_rows_kernel<<<blocks, 256, 0, ec.dev[r]->stream>>>(
                local[r], out[r], ranges, r, impl.host_dev[r], impl.gsteps[r]);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

void synchronize_all(const ExecutionContext& ec) {
    for (int r = 0; r < ec.tp; ++r) {
        DeviceGuard::set(ec.dev[r]->device);
        CUDA_CHECK(cudaStreamSynchronize(ec.dev[r]->stream));
    }
}

const char* transport_name(TpMailboxTransport transport) {
    switch (transport) {
    case TpMailboxTransport::Host: return "host mailbox";
    case TpMailboxTransport::Device: return "nvlink mailbox";
    case TpMailboxTransport::Nccl: return "nccl";
    }
    return "unknown";
}

// Exactness self-check: rank r contributes r + 1 everywhere, so every element must come back as
// the exact integer sum tp(tp+1)/2 on every rank. The odd size also exercises the scalar tail.
bool self_check_allreduce(const TpMailboxPlan::Impl& impl, const ExecutionContext& ec,
                          TpMailboxTransport transport,
                          const std::array<__nv_bfloat16*, kRanks>& scratch) {
    const int sizes[] = {kMailboxMaxElements, kMailboxMaxElements - 3};
    const float expected = static_cast<float>(impl.tp * (impl.tp + 1) / 2);
    for (const int n : sizes) {
        for (int r = 0; r < impl.tp; ++r) {
            std::vector<__nv_bfloat16> host(static_cast<std::size_t>(n),
                                            __float2bfloat16(static_cast<float>(r + 1)));
            DeviceGuard::set(ec.dev[r]->device);
            CUDA_CHECK(cudaMemcpyAsync(scratch[r], host.data(),
                                       static_cast<std::size_t>(n) * sizeof(__nv_bfloat16),
                                       cudaMemcpyHostToDevice, ec.dev[r]->stream));
        }
        launch_allreduce(impl, transport, ec, scratch, n);
        for (int r = 0; r < impl.tp; ++r) {
            std::vector<__nv_bfloat16> out(static_cast<std::size_t>(n));
            DeviceGuard::set(ec.dev[r]->device);
            CUDA_CHECK(cudaMemcpyAsync(out.data(), scratch[r],
                                       static_cast<std::size_t>(n) * sizeof(__nv_bfloat16),
                                       cudaMemcpyDeviceToHost, ec.dev[r]->stream));
            CUDA_CHECK(cudaStreamSynchronize(ec.dev[r]->stream));
            for (int i = 0; i < n; ++i) {
                if (__bfloat162float(out[i]) != expected) {
                    std::fprintf(stderr,
                                 "[ninfer] TP mailbox self-check failed (%s): rank %d element "
                                 "%d/%d got %f, expected %f\n",
                                 transport_name(transport), r, i, n, __bfloat162float(out[i]),
                                 expected);
                    return false;
                }
            }
        }
    }
    return true;
}

// Times `launch` (which must enqueue every rank) through per-rank events; the slowest rank is the
// message's latency. Warms first so both NCCL's lazy connection setup and the mailbox step
// counters are out of the measurement.
template <class Launch>
double time_candidate(const ExecutionContext& ec, Launch&& launch, int iterations) {
    std::array<cudaEvent_t, kRanks> begin{};
    std::array<cudaEvent_t, kRanks> end{};
    for (int r = 0; r < ec.tp; ++r) {
        DeviceGuard::set(ec.dev[r]->device);
        CUDA_CHECK(cudaEventCreate(&begin[r]));
        CUDA_CHECK(cudaEventCreate(&end[r]));
    }
    for (int i = 0; i < 50; ++i) { launch(); }
    synchronize_all(ec);
    for (int r = 0; r < ec.tp; ++r) {
        DeviceGuard::set(ec.dev[r]->device);
        CUDA_CHECK(cudaEventRecord(begin[r], ec.dev[r]->stream));
    }
    for (int i = 0; i < iterations; ++i) { launch(); }
    float max_ms = 0.0f;
    for (int r = 0; r < ec.tp; ++r) {
        DeviceGuard::set(ec.dev[r]->device);
        CUDA_CHECK(cudaEventRecord(end[r], ec.dev[r]->stream));
        CUDA_CHECK(cudaEventSynchronize(end[r]));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, begin[r], end[r]));
        max_ms = ms > max_ms ? ms : max_ms;
    }
    for (int r = 0; r < ec.tp; ++r) {
        DeviceGuard::set(ec.dev[r]->device);
        CUDA_CHECK(cudaEventDestroy(begin[r]));
        CUDA_CHECK(cudaEventDestroy(end[r]));
    }
    return static_cast<double>(max_ms) * 1000.0 / iterations;
}

void launch_nccl_allreduce(const ExecutionContext& ec,
                           const std::array<ncclComm_t, kMaximumDevices>& comms,
                           const std::array<__nv_bfloat16*, kRanks>& buffer, int n) {
    nccl_check(ncclGroupStart(), "ncclGroupStart");
    for (int r = 0; r < ec.tp; ++r) {
        DeviceGuard::set(ec.dev[r]->device);
        nccl_check(ncclAllReduce(buffer[r], buffer[r], static_cast<std::size_t>(n), ncclBfloat16,
                                 ncclSum, comms[static_cast<std::size_t>(r)], ec.dev[r]->stream),
                   "ncclAllReduce");
    }
    nccl_check(ncclGroupEnd(), "ncclGroupEnd");
}

std::unique_ptr<TpMailboxPlan::Impl> init_plan(const ExecutionContext& ec,
                                               const std::array<ncclComm_t, kMaximumDevices>& comms) {
    DeviceGuard guard;
    auto impl    = std::make_unique<TpMailboxPlan::Impl>();
    impl->tp     = ec.tp;

    // The pinned host block is one allocation shared by every rank; cudaHostGetDevicePointer
    // gives each device its alias (one UVA address on 64-bit systems).
    const std::size_t host_bytes = sizeof(HostMailbox);
    if (cudaHostAlloc(reinterpret_cast<void**>(&impl->host), host_bytes,
                      cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        (void)cudaGetLastError();
        impl->host = nullptr;
        return impl;
    }
    std::memset(impl->host, 0, host_bytes);
    for (int r = 0; r < impl->tp; ++r) {
        DeviceGuard::set(ec.dev[r]->device);
        void* alias = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&alias, impl->host, 0));
        impl->host_dev[r] = static_cast<HostMailbox*>(alias);
    }

    // Per-rank inbox, step counters, and the peer-pointer table the device kernels index. The
    // tables are published only after every inbox exists, or a table would hold a null peer.
    for (int r = 0; r < impl->tp; ++r) {
        impl->device[r] = ec.dev[r]->device;
        DeviceGuard::set(ec.dev[r]->device);
        CUDA_CHECK(cudaMalloc(&impl->inbox[r], sizeof(Inbox)));
        CUDA_CHECK(cudaMemset(impl->inbox[r], 0, sizeof(Inbox)));
        CUDA_CHECK(cudaMalloc(&impl->steps[r], sizeof(std::uint64_t) * kMailboxMaxBlocks));
        CUDA_CHECK(cudaMemset(impl->steps[r], 0, sizeof(std::uint64_t) * kMailboxMaxBlocks));
        CUDA_CHECK(cudaMalloc(&impl->gsteps[r], sizeof(std::uint64_t) * kGatherMaxBlocks));
        CUDA_CHECK(cudaMemset(impl->gsteps[r], 0, sizeof(std::uint64_t) * kGatherMaxBlocks));
    }
    std::array<Inbox*, kRanks> peer_host{};
    for (int src = 0; src < impl->tp; ++src) { peer_host[src] = impl->inbox[src]; }
    for (int r = 0; r < impl->tp; ++r) {
        DeviceGuard::set(ec.dev[r]->device);
        CUDA_CHECK(cudaMalloc(&impl->peer_ptrs[r], sizeof(Inbox*) * kRanks));
        CUDA_CHECK(cudaMemcpy(impl->peer_ptrs[r], peer_host.data(), sizeof(Inbox*) * kRanks,
                              cudaMemcpyHostToDevice));
    }

    // Device transport requires every ordered pair to be able to dereference peer memory.
    bool device_available = true;
    for (int a = 0; a < impl->tp && device_available; ++a) {
        DeviceGuard::set(ec.dev[a]->device);
        for (int b = 0; b < impl->tp; ++b) {
            if (a == b) { continue; }
            const cudaError_t err = cudaDeviceEnablePeerAccess(b, 0);
            if (err == cudaErrorPeerAccessAlreadyEnabled) {
                // A previous plan in this process already enabled the pair; the sticky error must
                // be consumed or the next CUDA_CHECK trips on it.
                (void)cudaGetLastError();
                continue;
            }
            if (err != cudaSuccess) {
                (void)cudaGetLastError();
                device_available = false;
                break;
            }
        }
    }

    std::array<__nv_bfloat16*, kRanks> scratch{};
    for (int r = 0; r < impl->tp; ++r) {
        DeviceGuard::set(ec.dev[r]->device);
        CUDA_CHECK(cudaMalloc(&scratch[r],
                              sizeof(__nv_bfloat16) * static_cast<std::size_t>(kMailboxMaxElements)));
    }

    const bool host_ok =
        impl->host != nullptr && self_check_allreduce(*impl, ec, TpMailboxTransport::Host, scratch);
    const bool device_ok = device_available &&
                           self_check_allreduce(*impl, ec, TpMailboxTransport::Device, scratch);

    // Race the survivors against NCCL on a decode-sized message (T=4 x 5120 = 40 KB) and keep the
    // fastest.
    constexpr int kProbeElements = 20480;
    constexpr int kIterations    = 200;
    const double nccl_us = time_candidate(
        ec, [&] { launch_nccl_allreduce(ec, comms, scratch, kProbeElements); }, kIterations);
    const double host_us =
        host_ok ? time_candidate(
                      ec,
                      [&] { launch_allreduce(*impl, TpMailboxTransport::Host, ec, scratch,
                                             kProbeElements); },
                      kIterations)
                : 1.0e9;
    const double device_us =
        device_ok ? time_candidate(
                        ec,
                        [&] { launch_allreduce(*impl, TpMailboxTransport::Device, ec, scratch,
                                               kProbeElements); },
                        kIterations)
                  : 1.0e9;

    impl->transport = TpMailboxTransport::Nccl;
    if (device_ok && device_us < (host_ok ? host_us : nccl_us) && device_us < nccl_us) {
        impl->transport = TpMailboxTransport::Device;
    } else if (host_ok && host_us < nccl_us) {
        impl->transport = TpMailboxTransport::Host;
    }
    std::fprintf(stderr,
                 "[ninfer] TP%d mailbox probe (40 KB all-reduce): nccl %.1f us, host %.1f us, "
                 "nvlink %.1f us -> %s\n",
                 impl->tp, nccl_us, host_us, device_us, transport_name(impl->transport));

    for (int r = 0; r < impl->tp; ++r) {
        DeviceGuard::set(ec.dev[r]->device);
        CUDA_CHECK(cudaFree(scratch[r]));
    }
    if (impl->transport == TpMailboxTransport::Nccl) { impl->release(); }
    return impl;
}

} // namespace

std::unique_ptr<TpMailboxPlan> TpMailboxPlan::create(
    const ExecutionContext& ec, const std::array<ncclComm_t, kMaximumDevices>& comms) {
    if (ec.tp < 2) { return nullptr; }
    const char* env = std::getenv("NINFER_TP_MAILBOX");
    if (env != nullptr && env[0] == '0') { return nullptr; }
    try {
        auto impl = init_plan(ec, comms);
        if (impl->transport == TpMailboxTransport::Nccl) { return nullptr; }
        return std::unique_ptr<TpMailboxPlan>(new TpMailboxPlan(std::move(impl)));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[ninfer] TP mailbox disabled: %s\n", error.what());
        return nullptr;
    }
}

TpMailboxPlan::TpMailboxPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
TpMailboxPlan::~TpMailboxPlan() = default;

int TpMailboxPlan::tp() const noexcept { return impl_->tp; }

TpMailboxTransport TpMailboxPlan::transport() const noexcept { return impl_->transport; }

bool TpMailboxPlan::allreduce_bf16(const std::array<Tensor, kMaximumDevices>& buffer,
                                   const ExecutionContext& ec) {
    if (impl_ == nullptr || impl_->transport == TpMailboxTransport::Nccl) { return false; }
    if (ec.tp != impl_->tp) { return false; }
    const std::int64_t elements = buffer[0].numel();
    if (elements <= 0 || elements > kMailboxMaxElements) { return false; }
    std::array<__nv_bfloat16*, kRanks> pointers{};
    for (int r = 0; r < impl_->tp; ++r) {
        if (buffer[r].data == nullptr || buffer[r].dtype != DType::BF16 ||
            !buffer[r].is_contiguous() || buffer[r].numel() != elements) {
            return false;
        }
        if (reinterpret_cast<std::uintptr_t>(buffer[r].data) % 16 != 0) { return false; }
        pointers[r] = static_cast<__nv_bfloat16*>(buffer[r].data);
    }
    DeviceGuard guard;
    launch_allreduce(*impl_, impl_->transport, ec, pointers, static_cast<int>(elements));
    return true;
}

bool TpMailboxPlan::allgather_rows_bf16(const std::array<Tensor, kMaximumDevices>& destination,
                                        const std::array<Tensor, kMaximumDevices>& part,
                                        const ExecutionContext& ec) {
    if (impl_ == nullptr || impl_->transport == TpMailboxTransport::Nccl) { return false; }
    if (ec.tp != impl_->tp) { return false; }
    GatherRanges ranges{};
    ranges.ranks        = impl_->tp;
    std::int64_t offset = 0;
    std::int64_t total  = -1;
    std::array<__nv_bfloat16*, kRanks> out{};
    std::array<const __nv_bfloat16*, kRanks> local{};
    for (int r = 0; r < impl_->tp; ++r) {
        const std::int64_t count = part[r].numel();
        if (count <= 0 || count > kGatherMaxElements) { return false; }
        if (offset % 8 != 0) { return false; }
        if (part[r].data == nullptr || destination[r].data == nullptr ||
            part[r].dtype != DType::BF16 || destination[r].dtype != DType::BF16 ||
            !part[r].is_contiguous() || !destination[r].is_contiguous()) {
            return false;
        }
        if (reinterpret_cast<std::uintptr_t>(part[r].data) % 16 != 0 ||
            reinterpret_cast<std::uintptr_t>(destination[r].data) % 16 != 0) {
            return false;
        }
        if (total < 0) { total = destination[r].numel(); }
        if (destination[r].numel() != total || total < offset + count) { return false; }
        ranges.offset[r] = offset;
        ranges.count[r]  = count;
        out[r]           = static_cast<__nv_bfloat16*>(destination[r].data);
        local[r]         = static_cast<const __nv_bfloat16*>(part[r].data);
        offset += count;
    }
    if (total != offset) { return false; }
    DeviceGuard guard;
    launch_gather(*impl_, impl_->transport, ec, local, out, ranges);
    return true;
}

} // namespace ninfer::ops::detail
