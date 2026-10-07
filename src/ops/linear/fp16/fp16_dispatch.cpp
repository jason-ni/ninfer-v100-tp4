#include "ops/linear/fp16/fp16_dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail {

Fp16Launch select_fp16_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
        break;
    case LinearPolicy::AllowA4:
        throw std::invalid_argument("fp16 linear: unsupported policy");
    }
    // Native FP16 RowMajor weights. The tensor-core tile loads B in uint4 (8-half) units, so the
    // contraction must be a multiple of 8; N and T are bounds-checked in the kernel. A K that is not
    // a multiple of the 32-wide tile (fc2, k=4304) zero-pads its shared tail, so any k%8==0 works.
    if (n <= 0 || k <= 0 || t <= 0 || k % 8 != 0) {
        throw std::invalid_argument("fp16 linear: unsupported shape or T");
    }
#ifdef NINFER_VOLTA_BUILD
    return launch_fp16_volta_mma;
#else
    throw std::invalid_argument("fp16 linear: requires the sm_70 build");
#endif
}

void fp16_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                   cudaStream_t stream) {
    const Fp16Launch launch = select_fp16_launch(w.n, w.k, x.ne[1], policy);
    launch(x, w, out, stream);
}

} // namespace ninfer::ops::detail
