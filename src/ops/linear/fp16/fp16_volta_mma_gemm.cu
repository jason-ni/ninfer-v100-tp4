#include "core/arena.h"
#include "core/device.h"
#include "ops/linear/fp16/fp16_launch.h"
#include "ops/linear/fp16/fp16_volta_mma_gemm.cuh"

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {
using S = Fp16VoltaMmaSchedule;
} // namespace

void launch_fp16_volta_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n      = out.ne[0];
    const std::int32_t k      = x.ne[0];
    const std::int32_t t      = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));

    const dim3 grid(static_cast<unsigned>((n + S::kRowsPerCta - 1) / S::kRowsPerCta),
                    1u,
                    static_cast<unsigned>((t + S::kTTile - 1) / S::kTTile));
    const auto* wdata    = static_cast<const __half*>(w.qdata);
    const auto* xd       = static_cast<const __nv_bfloat16*>(x.data);
    auto* out_data       = static_cast<__nv_bfloat16*>(out.data);

    fp16_volta_mma_gemm_kernel<<<grid, S::kThreads, 0, stream>>>(wdata, xd, out_data, out_ld, n, k, t);
    CUDA_CHECK(cudaGetLastError());
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
