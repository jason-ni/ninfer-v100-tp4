#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

// Replays the prepared per-rank decode graphs, one launch per rank on that rank's own stream. The
// caller's existing "synchronize every peer, then synchronize rank 0" retires the round.
template <class Context, class Body>
void run_prepared(Context& state, DecodeGraphExecutable* executable, Body&& body) {
    if (executable != nullptr) {
        if (!executable->ready()) {
            throw std::logic_error("decode graph was not prepared at load time");
        }
        executable->launch();
        return;
    }
    body();
}

// Captures one CUDA Graph per rank in a single simultaneous-capture pass. Every rank's stream
// belongs to the same ThreadLocal-mode capture for the duration of `body`, so each rank's work is
// recorded independently and the NCCL collectives carry cross-rank ordering internally. Every
// workspace arena is reset first because the body allocates activations at deterministic arena
// offsets and those addresses are baked into the graph.
template <class Context, class Body>
void capture_graph(Context& state, DecodeGraphDefinition& definition, Body&& body) {
    const int tp = state.execution.tp;
    std::array<int, kMaximumDevices> devices{};
    std::array<cudaStream_t, kMaximumDevices> streams{};

    state.execution.work.reset();
    devices[0] = state.execution.device.device;
    streams[0] = state.execution.device.stream;
    for (int rank = 1; rank < tp; ++rank) {
        const TpPeerCore& peer = *state.execution.peers[static_cast<std::size_t>(rank)];
        peer.work->reset();
        devices[rank] = peer.device->device;
        streams[rank] = peer.device->stream;
    }

    definition.capture(std::span<const int>(devices.data(), tp),
                       std::span<const cudaStream_t>(streams.data(), tp), body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
