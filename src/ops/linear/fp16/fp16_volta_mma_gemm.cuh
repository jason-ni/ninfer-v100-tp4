#pragma once

// Fused BF16 x FP16 GEMM on Volta tensor cores (mma.sync.m8n8k4), sm_70 only.
//
// The vision-tower projections are stored as native FP16 RowMajor matrices rather than the Q5
// code/high/scale planes, so this is the q5_volta_mma_gemm.cuh tile with the codec removed: the B
// operand loads straight from FP16 storage and there is no split-K (the contraction is short enough
// that a single split covers every registered vision shape). The A operand is still the model's BF16
// hidden state, staged to FP16 in shared memory exactly as the Q5 route does.
//
// K need not be a multiple of the 32-wide tile: the final tile loads its out-of-range uint4s as
// zeros, which both avoids an out-of-row read and zero-pads the shared tail so the mma reads zeros
// past the valid contraction (fc2 has k=4304).

#include "ops/common/volta_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

struct Fp16VoltaMmaSchedule {
    static constexpr int kWarps      = 4;  // warps per CTA; each owns 8 output rows
    static constexpr int kKStep      = 32; // K elements staged per iteration (double-buffered)
    static constexpr int kTTile      = 32; // rows of the mma A operand
    static constexpr int kXPad       = 8;  // shared row padding (see Q5VoltaMmaSchedule)
    static constexpr int kRowsPerCta = kWarps * 8;
    static constexpr int kThreads    = kWarps * 32;
};

__global__ __launch_bounds__(Fp16VoltaMmaSchedule::kThreads, 8) void fp16_volta_mma_gemm_kernel(
    const __half* __restrict__ w, const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out, int out_ld, int n, int k, int t) {
    using S = Fp16VoltaMmaSchedule;
    constexpr int kVecs = S::kKStep / 8;  // uint4 (8 half) per row per tile

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int n0   = (static_cast<int>(blockIdx.x) * S::kWarps + warp) * 8;
    const int t0   = static_cast<int>(blockIdx.z) * S::kTTile;
    const int tcnt = min(S::kTTile, t - t0);

    __shared__ __align__(16) __half x_sh[2][S::kTTile][S::kKStep + S::kXPad];
    __shared__ __align__(16) __half w_sh[2][S::kWarps][8][S::kKStep + S::kXPad];

    // Split the global load from the shared store with a register carrier, so the mma loop runs
    // inside global latency (the Q5 route's long_scoreboard win, carried over). A uint4 whose 8
    // elements run past the end of K is loaded as zeros: that both avoids an out-of-row read and
    // zero-pads the shared tail of the final partial tile.
    struct Carry {
        uint4 xraw;
        bool xactive;
        uint4 wraw;
        bool wactive;
    };
    Carry carry;

    auto prefetch = [&](int kbase) {
        const int nvalid = min(S::kKStep, k - kbase);
        const int idx    = static_cast<int>(threadIdx.x);
        carry.xactive    = idx < tcnt * kVecs;
        if (carry.xactive) {
            const int row = idx / kVecs;
            const int v   = idx % kVecs;
            if (v * 8 + 8 <= nvalid) {
                carry.xraw = *reinterpret_cast<const uint4*>(
                    x + static_cast<std::int64_t>(t0 + row) * k + kbase + v * 8);
            } else {
                carry.xraw = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        const int row     = n0 + (lane >> 2);
        carry.wactive     = row < n;
        if (carry.wactive) {
            if ((lane & 3) * 8 + 8 <= nvalid) {
                carry.wraw = *reinterpret_cast<const uint4*>(
                    w + static_cast<std::int64_t>(row) * k + kbase + (lane & 3) * 8);
            } else {
                carry.wraw = make_uint4(0u, 0u, 0u, 0u);
            }
        }
    };

    auto commit = [&](const Carry& r, int buf) {
        if (r.xactive) {
            const int idx   = static_cast<int>(threadIdx.x);
            const auto* src = reinterpret_cast<const __nv_bfloat16*>(&r.xraw);
            __half tmp[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) { tmp[j] = __float2half(__bfloat162float(src[j])); }
            *reinterpret_cast<uint4*>(&x_sh[buf][idx / kVecs][(idx % kVecs) * 8]) =
                *reinterpret_cast<const uint4*>(tmp);
        }
        if (r.wactive) {
            *reinterpret_cast<uint4*>(&w_sh[buf][warp][lane >> 2][(lane & 3) * 8]) =
                *reinterpret_cast<const uint4*>(&r.wraw);
        }
    };

    float d[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    prefetch(0);
    commit(carry, 0);
    int buf = 0;
    // One barrier per iteration is enough: iteration i reads buf and writes buf^1, and the only
    // conflict (iteration i-1's reads of the buffer i is about to write) is separated by the
    // barrier at the top, because that whole body precedes it.
    for (int kbase = 0; kbase < k; kbase += S::kKStep) {
        __syncthreads();
        const int nxt       = kbase + S::kKStep;
        const bool has_next = nxt < k;
        if (has_next) { prefetch(nxt); }

#pragma unroll
        for (int kk = 0; kk < S::kKStep; kk += 8) {
            half2 a[4], b[4];
            volta_load_qp(a, reinterpret_cast<const half2*>(&x_sh[buf][0][kk]),
                          (S::kKStep + S::kXPad) / 2);
            volta_load_k(b, reinterpret_cast<const half2*>(&w_sh[buf][warp][0][kk]),
                         (S::kKStep + S::kXPad) / 2);
            volta_mma_qk(d, a, b);
        }

        if (has_next) { commit(carry, buf ^ 1); }
        buf ^= 1;
    }

#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int row_t = volta_d_get_i(l);
        const int col_n = n0 + volta_d_get_j(l);
        if (row_t < tcnt && col_n < n) {
            const std::int64_t o = static_cast<std::int64_t>(t0 + row_t) * out_ld + col_n;
            out[o] = __float2bfloat16(d[l]);
        }
    }
}

#endif // sm_70

} // namespace ninfer::ops::detail
