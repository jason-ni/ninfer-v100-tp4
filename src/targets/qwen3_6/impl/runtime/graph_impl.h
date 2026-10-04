#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

// A multi-device graph is launched ONCE, on rank 0's stream, and holds every peer rank's nodes
// inside it. Launching it also makes rank 0's stream wait for the whole graph, so the caller's
// existing "synchronize every peer, then synchronize rank 0" is still sufficient to retire the
// round.
template <class Context, class Body>
void run_prepared(Context& state, DecodeGraphExecutable* executable, Body&& body) {
    if (executable != nullptr) {
        if (!executable->ready()) {
            throw std::logic_error("decode graph was not prepared at load time");
        }
        if (state.execution.tp > 1) {
            if (state.execution.graph_bridge == nullptr) {
                throw std::logic_error("tensor-parallel graph launch requires a peer bridge");
            }
            // See DecodeGraphPeerBridge::gate_launch: the eager path issues each rank's kernels on
            // that rank's own stream and is therefore automatically ordered after the round's
            // mirrored KV page materialization; a graph launched on rank 0's stream is not.
            std::array<cudaStream_t, kMaximumDevices> peer_streams{};
            for (int rank = 1; rank < state.execution.tp; ++rank) {
                peer_streams[static_cast<std::size_t>(rank)] =
                    state.execution.peers[static_cast<std::size_t>(rank)]->device->stream;
            }
            state.execution.graph_bridge->gate_launch(peer_streams.data() + 1,
                                                      state.execution.device.stream);
        }
        executable->launch(state.execution.device.stream);
    } else {
        body();
    }
}

// One capture site for every decode family, at every tp.
//
// At tp > 1 the body issues work on EVERY device's stream and orders them with the collectives'
// cross-device events. Those events only become graph edges if every stream belongs to the same
// capture, so each peer rank's stream is forked into rank 0's capture for the duration of the body
// and joined back before it ends: one cudaGraph holding all devices' nodes, launched once on rank
// 0's stream. Independent per-device captures are not an option -- CUDA rejects an event wait that
// would cross two live captures (cudaErrorStreamCaptureMerge).
//
// Every workspace arena is reset first, for the same reason at every rank: the body allocates its
// activations at deterministic arena offsets, and those addresses are baked into the graph.
template <class Context, class Body>
void capture_graph(Context& state, DecodeGraphDefinition& definition, Body&& body) {
    state.execution.work.reset();
    if (state.execution.tp == 1) {
        definition.capture(state.execution.device.stream, body);
        return;
    }
    if (state.execution.graph_bridge == nullptr) {
        throw std::logic_error("tensor-parallel graph capture requires a peer capture bridge");
    }
    std::array<cudaStream_t, kMaximumDevices> peer_streams{};
    for (int rank = 1; rank < state.execution.tp; ++rank) {
        const TpPeerCore& peer = *state.execution.peers[static_cast<std::size_t>(rank)];
        peer.work->reset();
        peer_streams[static_cast<std::size_t>(rank)] = peer.device->stream;
    }
    definition.capture(state.execution.device.stream, body,
                       DecodeGraphPeerCapture{state.execution.graph_bridge,
                                              peer_streams.data() + 1,
                                              state.execution.tp - 1});
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
