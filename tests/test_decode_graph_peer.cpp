// Isolated qualification for the N-device CUDA-graph bridge and for capturing the N-way
// collectives inside one multi-device graph (decode-graph design §3.6).
//
// The test captures a single graph on the origin stream that forks every peer stream in, runs the
// N-way allreduce inside the capture, joins the peers back, and then replays the captured graph
// many times over re-preloaded inputs. Each replay is compared bit-for-bit against the eager
// N-way allreduce on the same inputs, which is the exact graph/eager parity property the decode
// program depends on. The tp == 2 form exercises the one-peer convenience topology; the tp == 4
// form exercises origin + three peers.
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
                                        const std::array<Tensor, kMaximumDevices>& value,
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
    (void)value;
    return partials;
}

std::vector<std::uint16_t> read_bits(const ExecutionContext& ec, int rank,
                                     const std::array<std::unique_ptr<GuardedDeviceBuffer>,
                                                      kMaximumDevices>& value_buf) {
    set_device(ec, rank);
    return from_device<std::uint16_t>(value_buf[rank]->data(), kWidth);
}

int run_parity(const std::vector<int>& device_ids) {
    const int tp      = static_cast<int>(device_ids.size());
    ExecutionContext ec(device_ids);
    if (!ops::enable_peer_access(ec)) {
        throw std::runtime_error("peer access is unavailable on this topology");
    }
    ops::PeerEvents eager_events(ec);
    ops::PeerEvents graph_events(ec);

    using Guard = std::unique_ptr<GuardedDeviceBuffer>;
    std::array<Guard, kMaximumDevices> value_buf, staging_buf;
    std::array<Tensor, kMaximumDevices> value, staging;
    for (int rank = 0; rank < tp; ++rank) {
        set_device(ec, rank);
        value_buf[rank]   = std::make_unique<GuardedDeviceBuffer>(kWidth * sizeof(std::uint16_t));
        staging_buf[rank] = std::make_unique<GuardedDeviceBuffer>(kWidth * sizeof(std::uint16_t));
        value[rank]   = Tensor(value_buf[rank]->data(), DType::BF16, {kWidth, 1});
        staging[rank] = Tensor(staging_buf[rank]->data(), DType::BF16, {kWidth, 1});
    }

    set_device(ec, 0);
    DecodeGraphPeerBridge bridge(ec.dev[0]->device,
                                 std::span<const int>(device_ids.data() + 1,
                                                      static_cast<std::size_t>(tp - 1)));
    std::array<cudaStream_t, kMaximumDevices> peer_streams{};
    for (int rank = 1; rank < tp; ++rank) { peer_streams[rank - 1] = ec.dev[rank]->stream; }

    DecodeGraphDefinition definition;
    definition.capture(
        ec.dev[0]->stream,
        [&] { ops::allreduce_sum(value, staging, ec, graph_events); },
        DecodeGraphPeerCapture{&bridge, peer_streams.data(), tp - 1});
    if (!definition.ready()) { throw std::runtime_error("multi-device capture did not produce a graph"); }
    std::cout << "tp=" << tp << " captured graph: peers=" << bridge.peer_count()
              << " nodes=" << definition.node_count() << '\n';
    if (definition.node_count() < static_cast<std::size_t>(tp)) {
        std::cout << "FAIL tp=" << tp << ": graph has fewer nodes than ranks\n";
        return 1;
    }

    DecodeGraphExecutable executable;
    executable.instantiate(definition);
    executable.upload(ec.dev[0]->stream);
    synchronize_all(ec);

    int failures = 0;
    for (std::uint32_t round = 0; round < 12; ++round) {
        const std::uint32_t seed = 700u + round * 17u;

        // Eager reference.
        preload(ec, seed, value, value_buf, staging_buf);
        ops::allreduce_sum(value, staging, ec, eager_events);
        synchronize_all(ec);
        const std::vector<std::uint16_t> eager = read_bits(ec, 0, value_buf);

        // Re-preload the same inputs and replay the captured graph on the origin stream.
        preload(ec, seed, value, value_buf, staging_buf);
        bridge.gate_launch(peer_streams.data(), ec.dev[0]->stream);
        executable.launch(ec.dev[0]->stream);
        synchronize_all(ec);
        const std::vector<std::uint16_t> graph = read_bits(ec, 0, value_buf);

        if (graph != eager) {
            std::cout << "FAIL tp=" << tp << " round " << round
                      << ": captured graph differs from eager allreduce\n";
            ++failures;
            continue;
        }
        for (int rank = 1; rank < tp; ++rank) {
            if (read_bits(ec, rank, value_buf) != graph) {
                std::cout << "FAIL tp=" << tp << " round " << round << ": rank " << rank
                          << " not bitwise-equal to rank 0 after graph replay\n";
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
        std::cout << "SKIP: multi-device graph bridge requires at least two devices\n";
        return 77;
    }

    try {
        int failures = 0;
        failures += run_parity({0, 1});
        if (count >= 4) {
            failures += run_parity({0, 1, 2, 3});
        } else {
            std::cout << "note: only " << count << " devices visible; skipping tp=4 bridge\n";
        }
        std::cout << (failures ? "FAIL" : "OK") << " decode-graph-peer\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "decode graph peer test failed: " << error.what() << '\n';
        return 1;
    }
}
