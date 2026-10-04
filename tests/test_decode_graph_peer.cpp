// Isolated qualification for capturing the N-way NCCL allreduce inside per-rank CUDA Graphs.
//
// Each rank captures its OWN graph in one simultaneous-capture pass, then replays all ranks'
// graphs concurrently. Every replay is compared against the eager NCCL allreduce on the same
// inputs rank-by-rank, which is the exact graph/eager parity property the decode program depends
// on. The tp == 2 form exercises two ranks; tp == 4 exercises four.
#include "ninfer/ops/allreduce.h"
#include "ops/op_tester.h"

#include "core/decode_graph.h"
#include "core/device.h"

#include <array>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kWidth = 5120;

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

void set_device(const ExecutionContext& ec, int rank) {
    cuda_check(cudaSetDevice(ec.dev[rank]->device), "cudaSetDevice");
}

void synchronize_all(const ExecutionContext& ec) {
    for (int rank = 0; rank < ec.tp; ++rank) {
        set_device(ec, rank);
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
}

// Fills every rank's partial with an independently seeded BF16 row and returns the represented
// (already BF16-rounded) values.
std::vector<std::vector<float>> preload(const ExecutionContext& ec, std::uint32_t seed,
                                        const std::array<std::unique_ptr<GuardedDeviceBuffer>,
                                                         kMaximumDevices>& value_buf,
                                        const std::array<std::unique_ptr<GuardedDeviceBuffer>,
                                                         kMaximumDevices>& staging_buf) {
    const int tp = ec.tp;
    std::vector<std::vector<float>> partials(tp);
    for (int rank = 0; rank < tp; ++rank) {
        partials[rank].resize(kWidth);
        fill_uniform(partials[rank], seed + static_cast<std::uint32_t>(rank), -2.0f, 2.0f);
        round_to_bf16(partials[rank]);
    }
    for (int rank = 0; rank < tp; ++rank) {
        set_device(ec, rank);
        const auto bits = encode_bf16(partials[rank]);
        value_buf[rank]->copy_from_host(bits.data(), bits.size() * sizeof(std::uint16_t));
        staging_buf[rank]->fill(0);
    }
    synchronize_all(ec);
    return partials;
}

std::vector<std::uint16_t> read_bits(const ExecutionContext& ec, int rank,
                                     const std::array<std::unique_ptr<GuardedDeviceBuffer>,
                                                      kMaximumDevices>& value_buf) {
    set_device(ec, rank);
    return from_device<std::uint16_t>(value_buf[rank]->data(), kWidth);
}

int run_parity(const std::vector<int>& device_ids) {
    const int tp = static_cast<int>(device_ids.size());
    ExecutionContext ec(device_ids);
    ops::CollectiveComms eager_comms(ec);
    ops::CollectiveComms graph_comms(ec);

    using Guard = std::unique_ptr<GuardedDeviceBuffer>;
    std::array<Guard, kMaximumDevices> value_buf, staging_buf;
    std::array<Tensor, kMaximumDevices> value, staging;
    std::array<int, kMaximumDevices> devices{};
    std::array<cudaStream_t, kMaximumDevices> streams{};
    for (int rank = 0; rank < tp; ++rank) {
        set_device(ec, rank);
        value_buf[rank]   = std::make_unique<GuardedDeviceBuffer>(kWidth * sizeof(std::uint16_t));
        staging_buf[rank] = std::make_unique<GuardedDeviceBuffer>(kWidth * sizeof(std::uint16_t));
        value[rank]   = Tensor(value_buf[rank]->data(), DType::BF16, {kWidth, 1});
        staging[rank] = Tensor(staging_buf[rank]->data(), DType::BF16, {kWidth, 1});
        devices[rank] = ec.dev[rank]->device;
        streams[rank] = ec.dev[rank]->stream;
    }
    const auto device_span = std::span<const int>(devices.data(), static_cast<std::size_t>(tp));
    const auto stream_span =
        std::span<const cudaStream_t>(streams.data(), static_cast<std::size_t>(tp));

    DecodeGraphDefinition definition;
    definition.capture(device_span, stream_span,
                       [&] { ops::allreduce_sum(value, staging, ec, graph_comms); });
    if (!definition.ready()) {
        throw std::runtime_error("per-rank capture did not produce a graph");
    }
    std::cout << "tp=" << tp << " captured graphs: ranks=" << definition.rank_count()
              << " nodes=" << definition.node_count() << '\n';
    if (definition.node_count() < static_cast<std::size_t>(tp)) {
        std::cout << "FAIL tp=" << tp << ": graphs have fewer nodes than ranks\n";
        return 1;
    }

    DecodeGraphExecutable executable;
    executable.instantiate(definition, device_span, stream_span);
    executable.upload();
    synchronize_all(ec);

    int failures = 0;
    for (std::uint32_t round = 0; round < 12; ++round) {
        const std::uint32_t seed = 700u + round * 17u;

        // Eager reference, per rank.
        preload(ec, seed, value_buf, staging_buf);
        ops::allreduce_sum(value, staging, ec, eager_comms);
        synchronize_all(ec);
        std::vector<std::vector<std::uint16_t>> eager(tp);
        for (int rank = 0; rank < tp; ++rank) { eager[rank] = read_bits(ec, rank, value_buf); }

        // Re-preload the same inputs and replay the captured graphs.
        preload(ec, seed, value_buf, staging_buf);
        executable.launch();
        synchronize_all(ec);

        for (int rank = 0; rank < tp; ++rank) {
            if (read_bits(ec, rank, value_buf) != eager[rank]) {
                std::cout << "FAIL tp=" << tp << " round " << round << ": rank " << rank
                          << " graph replay differs from eager allreduce\n";
                ++failures;
            }
        }
    }
    for (int rank = 0; rank < tp; ++rank) {
        failures += value_buf[rank]->verify_guards("graph replay value guard");
        failures += staging_buf[rank]->verify_guards("graph replay staging guard");
    }
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
        std::cout << "SKIP: per-rank graph test requires at least two devices\n";
        return 77;
    }

    try {
        int failures = 0;
        failures += run_parity({0, 1});
        if (count >= 4) {
            failures += run_parity({0, 1, 2, 3});
        } else {
            std::cout << "note: only " << count << " devices visible; skipping tp=4\n";
        }
        std::cout << (failures ? "FAIL" : "OK") << " decode-graph-peer\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "decode graph peer test failed: " << error.what() << '\n';
        return 1;
    }
}
