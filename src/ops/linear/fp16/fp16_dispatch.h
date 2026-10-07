#pragma once

#include "ninfer/ops/linear.h"
#include "ops/linear/fp16/fp16_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

Fp16Launch select_fp16_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy);

void fp16_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                   cudaStream_t stream);

} // namespace ninfer::ops::detail
