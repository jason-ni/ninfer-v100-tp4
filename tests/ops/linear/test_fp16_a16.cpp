#include "ninfer/ops/linear.h"

#include "ops/direct_fp16_weight.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::direct_fp16_weight;

constexpr ReductionCriterion kA16Tolerance{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};

std::vector<std::uint16_t> make_activation_bits(std::int32_t hidden, std::int32_t tokens) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(hidden) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t column = 0; column < hidden; ++column) {
            const int centered = ((column * 29 + token * 71 + 17) & 0xff) - 128;
            result[static_cast<std::size_t>(token) * hidden + column] =
                f32_to_bf16(static_cast<float>(centered) * (1.0F / 512.0F));
        }
    }
    return result;
}

std::vector<float> materialize(std::span<const std::uint16_t> bits) {
    std::vector<float> result(bits.size());
    for (std::size_t index = 0; index < bits.size(); ++index) {
        result[index] = bf16_to_f32(bits[index]);
    }
    return result;
}

std::vector<std::int32_t> sampled_rows(std::int32_t rows) {
    std::vector<std::int32_t> result{0, 1, rows / 4, rows / 2, (3 * rows) / 4, rows - 2, rows - 1};
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<std::int32_t> sampled_tokens(std::int32_t tokens) {
    if (tokens <= 32) {
        std::vector<std::int32_t> result(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) {
            result[static_cast<std::size_t>(token)] = token;
        }
        return result;
    }
    std::vector<std::int32_t> result{0, 1, tokens / 2, tokens - 2, tokens - 1};
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

int run_fp16_linear_case(DeviceWeight& weight, std::int32_t tokens) {
    const std::int32_t rows                          = weight.host.n;
    const std::int32_t hidden                        = weight.host.k;
    const std::vector<std::uint16_t> activation_bits = make_activation_bits(hidden, tokens);
    const std::vector<float> activation              = materialize(activation_bits);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedDeviceBuffer guarded_output(
        static_cast<std::size_t>(rows) * tokens * sizeof(std::uint16_t));
    guarded_output.fill(0xff);

    Tensor x(device_activation.p, DType::BF16, {hidden, tokens});
    Tensor output(guarded_output.data(), DType::BF16, {rows, tokens});
    DeviceArena workspace(256);
    ops::linear(x, weight.view(), output, ops::LinearPolicy::A16Only, workspace, nullptr);
    cuda_synchronize();

    const std::string suffix = " T=" + std::to_string(tokens);
    int failures             = guarded_output.verify_guards("FP16_A16 Linear output" + suffix);
    const std::vector<std::uint16_t> output_bits =
        from_device<std::uint16_t>(guarded_output.data(), static_cast<std::size_t>(rows) * tokens);
    for (std::size_t index = 0; index < output_bits.size(); ++index) {
        const std::uint16_t bits = output_bits[index];
        if (!std::isfinite(bf16_to_f32(bits))) {
            std::cerr << "FP16_A16 Linear output" << suffix << " element " << index
                      << " is not finite\n";
            ++failures;
            break;
        }
    }

    const std::vector<std::int32_t> sampled       = sampled_rows(rows);
    const std::vector<std::int32_t> token_samples = sampled_tokens(tokens);
    std::vector<double> actual;
    std::vector<double> expected;
    actual.reserve(sampled.size() * token_samples.size());
    expected.reserve(actual.capacity());
    for (const std::int32_t row : sampled) {
        for (const std::int32_t token : token_samples) {
            actual.push_back(
                bf16_to_f32(output_bits[static_cast<std::size_t>(token) * rows + row]));
            expected.push_back(dot_fp64(
                weight.host, row,
                std::span<const float>(
                    activation.data() + static_cast<std::size_t>(token) * hidden, hidden)));
        }
    }
    failures += verify_reduction("FP16_A16 Linear [" + std::to_string(rows) + "," +
                                     std::to_string(hidden) + "]" + suffix,
                                 actual, expected, kA16Tolerance);
    const std::vector<std::uint16_t> activation_after =
        from_device<std::uint16_t>(device_activation, activation_bits.size());
    if (activation_after != activation_bits) {
        std::cerr << "FP16_A16 Linear" << suffix << " modified its activation\n";
        ++failures;
    }
    failures += weight.verify_preserved("FP16_A16 Linear weight" + suffix);
    return failures;
}

int run_fp16_linear() {
    int failures = 0;
#ifdef NINFER_VOLTA_BUILD
    // The registered 27B vision-tower projections, [N,K] row-major FP16. mlp/fc2 (k=4304) is not a
    // multiple of the 32-wide tile and exercises the K-tail zero-pad; the rest are whole tiles.
    const std::vector<std::pair<std::int32_t, std::int32_t>> shapes = {
        {1152, 1536},  // vision/patch_embedding
        {3456, 1152},  // attention/qkv
        {1152, 1152},  // attention/output
        {4304, 1152},  // mlp/fc1
        {1152, 4304},  // mlp/fc2 (k % 32 != 0)
        {4608, 4608},  // merger/fc1
        {5120, 4608},  // merger/fc2
    };
    for (const auto& shape : shapes) {
        DeviceWeight weight(make_patterned(shape.first, shape.second, 501U));
        for (const std::int32_t tokens : {1, 2, 4, 8, 32, 128, 1024}) {
            failures += run_fp16_linear_case(weight, tokens);
        }
    }
    // One tensor-parallel shard of each form: column-parallel halves N (merger/fc1 -> 2304),
    // row-parallel halves K (merger/fc2 -> k=2304). The launcher reads extents from the tensors, so
    // the divided grid follows automatically.
    {
        DeviceWeight column(make_patterned(2304, 4608, 509U));
        DeviceWeight rowpar(make_patterned(5120, 2304, 511U));
        for (const std::int32_t tokens : {1, 8, 33}) {
            failures += run_fp16_linear_case(column, tokens);
            failures += run_fp16_linear_case(rowpar, tokens);
        }
    }
#endif
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = run_fp16_linear();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " FP16_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FP16_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
