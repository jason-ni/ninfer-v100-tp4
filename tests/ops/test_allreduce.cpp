// N-device qualification suite for the cross-device collectives declared in
// include/ninfer/ops/allreduce.h.
//
// The suite runs the same cases at tp == 1, tp == 2, and (when four devices are visible) tp == 4.
// The tp == 4 cases are the direct check that the unified N-way reduce composes four shards
// correctly; the tp == 2 cases are the regression gate for the route the bespoke two-party pull
// used to serve. tp == 1 is the documented validate-and-return no-op.
//
// With fewer visible devices than a width the suite reports the repository's skip code (77) for the
// widths it cannot run; it never fails merely because the hardware is smaller.
//
// Oracles:
//   allreduce_sum - independent FP64 elementwise sum of ALL represented BF16 partials. The Op's
//     observable output is BF16 and its star implementation rounds to BF16 after each of the tp-1
//     local additions, so the deviation from the exact sum is bounded by 0.5*2^-8*((tp-1)*sum|p| +
//     |sum|) rather than one ulp; the criterion uses exactly that accumulated-rounding bound and
//     separately requires every rank to hold bitwise-identical bytes.
//   allgather_rows/gather_columns_rank0/broadcast_rank0 - exact: the Ops only relocate bytes, so
//     every destination byte is compared bit-for-bit against an independently constructed image.
#include "ninfer/ops/allreduce.h"
#include "ops/op_tester.h"

#include "core/device.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// A BF16 mantissa has 7 explicit bits, so ulp/|x| <= 2^-7 and one round-to-nearest introduces at
// most 2^-8 relative error. Each of the tp-1 local additions contributes this bound to the
// accumulated deviation from the exact sum.
constexpr double kBf16RoundingRelative = 3.90625e-3; // 2^-8

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

// Independent oracle: the complete formula in FP64 from the represented BF16 inputs, over the
// runtime tp.
std::vector<double> allreduce_sum_oracle(const std::vector<std::vector<float>>& partials) {
    const std::size_t count = partials.front().size();
    std::vector<double> expected(count, 0.0);
    for (const std::vector<float>& partial : partials) {
        for (std::size_t i = 0; i < count; ++i) {
            expected[i] += static_cast<double>(partial[i]);
        }
    }
    return expected;
}

void set_device(const ExecutionContext& ec, int rank) {
    cuda_check(cudaSetDevice(ec.dev[rank]->device), "cudaSetDevice");
}

void synchronize_all(const ExecutionContext& ec) {
    for (int rank = 0; rank < ec.tp; ++rank) {
        set_device(ec, rank);
        cuda_check(cudaStreamSynchronize(ec.dev[rank]->stream), "cudaStreamSynchronize");
    }
}

// Inputs are staged with cudaMemcpy/cudaMemset, which the runtime issues on each device's LEGACY
// default stream. DeviceContext::stream is created with cudaStreamNonBlocking and therefore does
// NOT implicitly synchronize with that default stream, so the staging writes must be retired
// before the collective's transfers read them. This is the caller obligation the Op contract
// documents; omitting the wait makes the suite intermittently read pre-staging bytes.
void retire_staging(const ExecutionContext& ec) {
    for (int rank = 0; rank < ec.tp; ++rank) {
        set_device(ec, rank);
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
}

using Guards = std::array<std::unique_ptr<GuardedDeviceBuffer>, kMaximumDevices>;

// `ne0` is the contiguous dimension and `ne1` the outer one, so a 1-D buffer passes ne1 == 1 and
// the real row-parallel residual passes {5120, 48}. Every rank contributes an independently seeded
// BF16 partial; the Op's observable result is their sum on every rank.
int run_allreduce_case(const char* label, std::int32_t ne0, std::int32_t ne1, std::uint32_t seed,
                       const ExecutionContext& ec, const ops::CollectiveComms& events) {
    const int tp            = ec.tp;
    const std::size_t count = static_cast<std::size_t>(ne0) * static_cast<std::size_t>(ne1);

    std::vector<std::vector<float>> partials(tp);
    for (int rank = 0; rank < tp; ++rank) {
        partials[rank].resize(count);
        fill_uniform(partials[rank], seed + static_cast<std::uint32_t>(rank), -8.0f, 8.0f);
        round_to_bf16(partials[rank]);
    }
    const auto expected = allreduce_sum_oracle(partials);

    Guards buffers, staging;
    std::array<Tensor, kMaximumDevices> buffer{};
    std::array<Tensor, kMaximumDevices> staging_t{};
    for (int rank = 0; rank < tp; ++rank) {
        set_device(ec, rank);
        buffers[rank] = std::make_unique<GuardedDeviceBuffer>(count * sizeof(std::uint16_t));
        staging[rank] = std::make_unique<GuardedDeviceBuffer>(count * sizeof(std::uint16_t));
        const auto bits = encode_bf16(partials[rank]);
        buffers[rank]->copy_from_host(bits.data(), count * sizeof(std::uint16_t));
        staging[rank]->fill(0);
        buffer[rank]    = Tensor(buffers[rank]->data(), DType::BF16, {ne0, ne1});
        staging_t[rank] = Tensor(staging[rank]->data(), DType::BF16, {ne0, ne1});
    }

    retire_staging(ec);
    ops::allreduce_sum(buffer, staging_t, ec, events);
    synchronize_all(ec);

    std::vector<std::vector<double>> observed(tp);
    int failures = 0;
    for (int rank = 0; rank < tp; ++rank) {
        set_device(ec, rank);
        observed[rank] = from_device_bf16(buffers[rank]->data(), count);
        failures += buffers[rank]->verify_guards("allreduce buffer device guard");
        failures += staging[rank]->verify_guards("allreduce staging device guard");
    }

    // Distributed contract: the star reduce must leave bitwise-identical bytes on every rank.
    for (int rank = 1; rank < tp; ++rank) {
        if (observed[rank] != observed[0]) {
            std::cout << "FAIL " << label << ": rank " << rank << " is not bitwise-equal to rank 0\n";
            ++failures;
        }
    }

    // Accumulated-rounding bound against the exact FP64 sum. Each of the tp-1 local additions
    // rounds its running sum to BF16, so the error is bounded by half a BF16 ulp of the running
    // magnitude per step.
    std::vector<double> abs_sum(count, 0.0);
    for (const std::vector<float>& partial : partials) {
        for (std::size_t i = 0; i < count; ++i) {
            abs_sum[i] += std::abs(static_cast<double>(partial[i]));
        }
    }
    double worst_ratio   = 0.0;
    std::size_t worst_i  = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const double tolerance = kBf16RoundingRelative *
                                 ((tp - 1) * abs_sum[i] + std::abs(expected[i]));
        const double error = std::abs(static_cast<double>(observed[0][i]) - expected[i]);
        const double ratio = error / (tolerance + 1.0e-30);
        if (ratio > worst_ratio) {
            worst_ratio = ratio;
            worst_i     = i;
        }
    }
    if (worst_ratio > 1.0) {
        std::cout << "FAIL " << label << ": accumulated-rounding bound exceeded at index " << worst_i
                  << " (ratio " << worst_ratio << ", actual " << observed[0][worst_i]
                  << ", reference " << expected[worst_i] << ")\n";
        ++failures;
    }
    return failures;
}

// `row_length` is ne[0] (the contiguous row) and the gathered axis is ne[1]. Rank r owns
// `rows_per_rank[r]` leading rows and receives the full concatenation.
int run_allgather_case(const char* label, const std::vector<std::int32_t>& rows_per_rank,
                       std::int32_t row_length, std::uint32_t seed, const ExecutionContext& ec,
                       const ops::CollectiveComms& events) {
    const int tp        = ec.tp;
    std::int32_t rows   = 0;
    for (std::int32_t r : rows_per_rank) rows += r;

    std::vector<std::size_t> part_counts(tp);
    for (int rank = 0; rank < tp; ++rank) {
        part_counts[rank] =
            static_cast<std::size_t>(rows_per_rank[rank]) * static_cast<std::size_t>(row_length);
    }

    // Exact oracle: the gathered image is the concatenation of the owned row ranges, in rank order.
    std::vector<std::uint16_t> expected;
    expected.reserve(static_cast<std::size_t>(rows) * static_cast<std::size_t>(row_length));
    for (int rank = 0; rank < tp; ++rank) {
        std::vector<float> source(part_counts[rank]);
        fill_uniform(source, seed + static_cast<std::uint32_t>(rank), -8.0f, 8.0f);
        const auto bits = encode_bf16(source);
        expected.insert(expected.end(), bits.begin(), bits.end());
    }

    Guards parts, destinations;
    std::array<Tensor, kMaximumDevices> part{};
    std::array<Tensor, kMaximumDevices> destination{};
    std::vector<std::vector<std::uint16_t>> bits_per_rank(tp);
    for (int rank = 0; rank < tp; ++rank) {
        set_device(ec, rank);
        parts[rank] = std::make_unique<GuardedDeviceBuffer>(part_counts[rank] * sizeof(std::uint16_t));
        destinations[rank] = std::make_unique<GuardedDeviceBuffer>(
            static_cast<std::size_t>(rows) * static_cast<std::size_t>(row_length) *
            sizeof(std::uint16_t));
        std::vector<float> source(part_counts[rank]);
        fill_uniform(source, seed + static_cast<std::uint32_t>(rank), -8.0f, 8.0f);
        bits_per_rank[rank] = encode_bf16(source);
        parts[rank]->copy_from_host(bits_per_rank[rank].data(), part_counts[rank] * sizeof(std::uint16_t));
        destinations[rank]->fill(0);
        part[rank] =
            Tensor(parts[rank]->data(), DType::BF16, {row_length, rows_per_rank[rank]});
        destination[rank] = Tensor(destinations[rank]->data(), DType::BF16, {row_length, rows});
    }

    retire_staging(ec);
    ops::allgather_rows(destination, part, ec, events);
    synchronize_all(ec);

    int failures = 0;
    for (int rank = 0; rank < tp; ++rank) {
        set_device(ec, rank);
        failures += verify_exact((std::string(label) + " device " + std::to_string(rank)).c_str(),
                                 from_device<std::uint16_t>(
                                     destinations[rank]->data(),
                                     static_cast<std::size_t>(rows) * row_length),
                                 expected);
        failures += destinations[rank]->verify_guards("allgather destination device guard");
        failures += parts[rank]->verify_guards("allgather part device guard");
    }
    return failures;
}

// `widths_per_rank[r]` is rank r's contiguous width block along ne[0]; the rank-0 destination is
// the full [C, T] image, and only rank 0 is written.
int run_gather_columns_rank0_case(const char* label, const std::vector<std::int32_t>& widths_per_rank,
                                  std::int32_t columns, std::uint32_t seed,
                                  const ExecutionContext& ec, const ops::CollectiveComms& events) {
    const int tp      = ec.tp;
    std::int32_t full = 0;
    for (std::int32_t w : widths_per_rank) full += w;

    std::vector<std::vector<float>> parts_float(tp);
    for (int rank = 0; rank < tp; ++rank) {
        parts_float[rank].resize(static_cast<std::size_t>(widths_per_rank[rank]) * columns);
        fill_uniform(parts_float[rank], seed + static_cast<std::uint32_t>(rank), -8.0f, 8.0f);
    }

    // Exact oracle: ne[0] is the contiguous dimension, so destination[c*full + row] and
    // part[c*w + i] — the owned width blocks are concatenated within each of the `columns` rows.
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(full) * columns);
    std::int32_t row_base = 0;
    for (int rank = 0; rank < tp; ++rank) {
        const std::int32_t w = widths_per_rank[rank];
        for (std::int32_t c = 0; c < columns; ++c) {
            for (std::int32_t i = 0; i < w; ++i) {
                expected[static_cast<std::size_t>(c) * full + (row_base + i)] =
                    f32_to_bf16(parts_float[rank][static_cast<std::size_t>(c) * w + i]);
            }
        }
        row_base += w;
    }

    Guards parts, destinations;
    std::array<Tensor, kMaximumDevices> part{};
    for (int rank = 0; rank < tp; ++rank) {
        set_device(ec, rank);
        parts[rank] = std::make_unique<GuardedDeviceBuffer>(
            static_cast<std::size_t>(widths_per_rank[rank]) * columns * sizeof(std::uint16_t));
        const auto bits = encode_bf16(parts_float[rank]);
        parts[rank]->copy_from_host(bits.data(), bits.size() * sizeof(std::uint16_t));
        part[rank] = Tensor(parts[rank]->data(), DType::BF16, {widths_per_rank[rank], columns});
    }
    set_device(ec, 0);
    destinations[0] = std::make_unique<GuardedDeviceBuffer>(
        static_cast<std::size_t>(full) * columns * sizeof(std::uint16_t));
    destinations[0]->fill(0);
    const Tensor destination(destinations[0]->data(), DType::BF16, {full, columns});

    retire_staging(ec);
    ops::gather_columns_rank0(destination, part, ec, events);
    synchronize_all(ec);

    int failures = 0;
    set_device(ec, 0);
    failures += verify_exact(label,
                             from_device<std::uint16_t>(destinations[0]->data(),
                                                        static_cast<std::size_t>(full) * columns),
                             expected);
    failures += destinations[0]->verify_guards("gather_columns destination device guard");
    for (int rank = 0; rank < tp; ++rank) {
        failures += parts[rank]->verify_guards("gather_columns part device guard");
    }
    return failures;
}

// `source` lives on rank 0; every peer destination must become a bit-exact copy.
int run_broadcast_case(const char* label, std::int32_t ne0, std::int32_t ne1, std::uint32_t seed,
                       const ExecutionContext& ec, const ops::CollectiveComms& events) {
    const int tp            = ec.tp;
    const std::size_t count = static_cast<std::size_t>(ne0) * static_cast<std::size_t>(ne1);

    std::vector<float> source(count);
    fill_uniform(source, seed, -8.0f, 8.0f);
    const auto bits = encode_bf16(source);

    Guards destinations;
    std::array<Tensor, kMaximumDevices> destination{};
    set_device(ec, 0);
    auto source_buffer = std::make_unique<GuardedDeviceBuffer>(count * sizeof(std::uint16_t));
    source_buffer->copy_from_host(bits.data(), count * sizeof(std::uint16_t));
    const Tensor source_t(source_buffer->data(), DType::BF16, {ne0, ne1});
    for (int rank = 0; rank < tp; ++rank) {
        set_device(ec, rank);
        destinations[rank] = std::make_unique<GuardedDeviceBuffer>(count * sizeof(std::uint16_t));
        destinations[rank]->fill(0);
        destination[rank] = Tensor(destinations[rank]->data(), DType::BF16, {ne0, ne1});
    }

    retire_staging(ec);
    ops::broadcast_rank0(source_t, destination, ec, events);
    synchronize_all(ec);

    int failures = 0;
    for (int rank = 1; rank < tp; ++rank) {
        set_device(ec, rank);
        failures += verify_exact((std::string(label) + " device " + std::to_string(rank)).c_str(),
                                 from_device<std::uint16_t>(destinations[rank]->data(), count), bits);
        failures += destinations[rank]->verify_guards("broadcast destination device guard");
    }
    return failures;
}

// Back-to-back collectives with no host synchronization in between, the property the Op contract's
// post-condition guarantees. Each round multiplies every element by `tp`; the start value is chosen
// so the value stays an exact power of two through all rounds, then is all-gathered into an
// [n, tp] image. `log2(tp)` must divide `kStartExponent`.
int run_chained_case(const ExecutionContext& ec, const ops::CollectiveComms& events) {
    constexpr int kStartExponent = -60;
    const int tp                 = ec.tp;
    const int log2_tp            = static_cast<int>(std::lround(std::log2(static_cast<double>(tp))));
    const int rounds             = kStartExponent / -log2_tp;
    const std::int32_t n         = 4096;

    Guards buffers;
    std::array<Tensor, kMaximumDevices> buffer{};
    for (int rank = 0; rank < tp; ++rank) {
        set_device(ec, rank);
        buffers[rank] = std::make_unique<GuardedDeviceBuffer>(
            static_cast<std::size_t>(n) * sizeof(std::uint16_t));
        set_device(ec, rank);
        std::vector<float> init(n, std::ldexp(1.0f, kStartExponent));
        const auto bits = encode_bf16(init);
        buffers[rank]->copy_from_host(bits.data(), bits.size() * sizeof(std::uint16_t));
        buffer[rank] = Tensor(buffers[rank]->data(), DType::BF16, {n, 1});
    }

    Guards staging;
    std::array<Tensor, kMaximumDevices> staging_t{};
    for (int rank = 0; rank < tp; ++rank) {
        set_device(ec, rank);
        staging[rank] = std::make_unique<GuardedDeviceBuffer>(
            static_cast<std::size_t>(n) * sizeof(std::uint16_t));
        staging[rank]->fill(0);
        staging_t[rank] = Tensor(staging[rank]->data(), DType::BF16, {n, 1});
    }

    Guards parts, gathered;
    std::array<Tensor, kMaximumDevices> part{};
    std::array<Tensor, kMaximumDevices> gathered_t{};
    for (int rank = 0; rank < tp; ++rank) {
        set_device(ec, rank);
        parts[rank] = std::make_unique<GuardedDeviceBuffer>(
            static_cast<std::size_t>(n) * sizeof(std::uint16_t));
        gathered[rank] = std::make_unique<GuardedDeviceBuffer>(
            static_cast<std::size_t>(n) * tp * sizeof(std::uint16_t));
        gathered[rank]->fill(0);
        part[rank]       = Tensor(parts[rank]->data(), DType::BF16, {n, 1});
        gathered_t[rank] = Tensor(gathered[rank]->data(), DType::BF16, {n, tp});
    }

    retire_staging(ec);
    for (int round = 0; round < rounds; ++round) {
        ops::allreduce_sum(buffer, staging_t, ec, events);
        for (int rank = 0; rank < tp; ++rank) {
            cuda_check(cudaMemcpyAsync(parts[rank]->data(), buffer[rank].data,
                                       static_cast<std::size_t>(n) * sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToDevice, ec.dev[rank]->stream),
                       "chained part copy");
        }
        ops::allgather_rows(gathered_t, part, ec, events);
    }
    synchronize_all(ec);

    int failures             = 0;
    const float expected_val = std::ldexp(1.0f, kStartExponent + rounds * log2_tp);
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(n) * tp,
                                        f32_to_bf16(expected_val));
    for (int rank = 0; rank < tp; ++rank) {
        set_device(ec, rank);
        failures += verify_exact((std::string("chained gathered device ") + std::to_string(rank)).c_str(),
                                 from_device<std::uint16_t>(
                                     gathered[rank]->data(), static_cast<std::size_t>(n) * tp),
                                 expected);
        failures += gathered[rank]->verify_guards("chained gathered device guard");
    }
    return failures;
}

// Coordinator-free throughput probe: the collectives must sustain back-to-back calls without a host
// synchronization in the loop. Reported only; the suite does not fail on a slow machine.
int run_microbenchmark(const ExecutionContext& ec, const ops::CollectiveComms& events) {
    constexpr std::int32_t n = 5120;
    Guards buffers;
    std::array<Tensor, kMaximumDevices> buffer{};
    for (int rank = 0; rank < ec.tp; ++rank) {
        set_device(ec, rank);
        buffers[rank] =
            std::make_unique<GuardedDeviceBuffer>(static_cast<std::size_t>(n) * sizeof(std::uint16_t));
        buffers[rank]->fill(0);
        buffer[rank] = Tensor(buffers[rank]->data(), DType::BF16, {n, 1});
    }
    Guards staging;
    std::array<Tensor, kMaximumDevices> staging_t{};
    for (int rank = 0; rank < ec.tp; ++rank) {
        set_device(ec, rank);
        staging[rank] =
            std::make_unique<GuardedDeviceBuffer>(static_cast<std::size_t>(n) * sizeof(std::uint16_t));
        staging[rank]->fill(0);
        staging_t[rank] = Tensor(staging[rank]->data(), DType::BF16, {n, 1});
    }

    constexpr int kIterations = 1000;
    const auto start          = std::chrono::steady_clock::now();
    for (int i = 0; i < kIterations; ++i) {
        ops::allreduce_sum(buffer, staging_t, ec, events);
        for (int rank = 0; rank < ec.tp; ++rank) {
            cuda_check(cudaStreamSynchronize(ec.dev[rank]->stream), "microbench stream sync");
        }
    }
    const double mean_us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() /
        kIterations;
    std::cout << "allreduce_sum microbenchmark tp=" << ec.tp << ": mean " << mean_us
              << " us/call\n";
    return 0;
}

int run_suite(int tp) {
    std::vector<int> device_ids;
    for (int rank = 0; rank < tp; ++rank) device_ids.push_back(rank);
    const ExecutionContext ec(device_ids);
    const ops::CollectiveComms events(ec);

    int failures = 0;
    std::cout << "== allreduce suite tp=" << tp << " ==\n";
    // Real decode shape first: 5120 is the hidden dimension all-reduced 128 times per token.
    failures += run_allreduce_case("allreduce_sum [5120]", 5120, 1, 101u, ec, events);
    // The real row-parallel residual: a full 48-token prefill chunk, 2-D.
    failures += run_allreduce_case("allreduce_sum [5120,48]", 5120, 48, 102u, ec, events);
    failures += run_allreduce_case("allreduce_sum [4097]", 4097, 1, 103u, ec, events);
    failures += run_allreduce_case("allreduce_sum [1]", 1, 1, 104u, ec, events);
    failures += run_allreduce_case("allreduce_sum [3]", 3, 1, 105u, ec, events);
    failures += run_allreduce_case("allreduce_sum [17,3]", 17, 3, 106u, ec, events);

    // The remaining collectives are documented validate-and-return no-ops at tp == 1, so their
    // relocation cases only have meaning with at least two ranks. allreduce_sum above still runs at
    // tp == 1: the identity sum is observable in the pre-loaded buffer.
    if (tp >= 2) {
        std::vector<std::int32_t> rows_512_512(tp, 1024 / tp);
        failures +=
            run_allgather_case("allgather_rows [5120,1024]", rows_512_512, 5120, 201u, ec, events);

        // Gathered logits: 248320 vocabulary rows for one token, split by vocabulary block.
        std::vector<std::int32_t> vocab(tp, 248320 / tp);
        failures += run_allgather_case("allgather_rows [1,248320]", vocab, 1, 202u, ec, events);

        // Uneven rows: rank 0 owns the remainder so the owned row counts are not all equal.
        std::vector<std::int32_t> uneven(tp, 1);
        uneven[0] = 2;
        failures += run_allgather_case("allgather_rows [5120,3] uneven", uneven, 5120, 203u, ec, events);

        std::vector<std::int32_t> widths(tp, 124160 / tp);
        widths[0] = 124160 - (tp - 1) * (124160 / tp);
        failures += run_gather_columns_rank0_case("gather_columns_rank0 [248320,3]", widths, 3, 205u,
                                                  ec, events);

        std::vector<std::int32_t> uneven_widths(tp, 2);
        uneven_widths[0] = 3;
        failures += run_gather_columns_rank0_case("gather_columns_rank0 [11,7] uneven", uneven_widths,
                                                  7, 206u, ec, events);

        failures += run_broadcast_case("broadcast_rank0 [5120]", 5120, 1, 301u, ec, events);
        failures += run_broadcast_case("broadcast_rank0 [4097]", 4097, 1, 302u, ec, events);
        failures += run_broadcast_case("broadcast_rank0 [1]", 1, 1, 303u, ec, events);

        failures += run_chained_case(ec, events);
    }
    failures += run_microbenchmark(ec, events);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int count = 0;
    cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count < 2) {
        std::cout << "SKIP: tensor-parallel collectives require at least two devices\n";
        return 77;
    }

    try {
        int failures = 0;
        failures += run_suite(1);
        failures += run_suite(2);
        if (count >= 4) {
            failures += run_suite(4);
        } else {
            std::cout << "note: only " << count << " devices visible; skipping tp=4 suite\n";
        }
        std::cout << (failures ? "FAIL" : "OK") << " allreduce\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "allreduce test failed: " << error.what() << '\n';
        return 1;
    }
}
