#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

using Fp16Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

#ifdef NINFER_VOLTA_BUILD
// The single sm_70 tensor-core route for native FP16 RowMajor weights. No split-K: the contraction
// is short enough that one split covers every registered vision projection, so no workspace exists.
void launch_fp16_volta_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
#endif

} // namespace ninfer::ops::detail
