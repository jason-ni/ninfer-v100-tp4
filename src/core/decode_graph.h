#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <functional>
#include <span>
#include <vector>

namespace ninfer {

// The events that enroll OTHER devices' streams in a capture that began on the first device's
// stream, and that order a replay of the result. Tensor-parallel decode issues work on every
// device's stream and orders them against each other with cross-device event edges; those edges
// only become graph edges if every stream belongs to the SAME capture. Two independent captures
// cannot be linked -- a cudaStreamWaitEvent across two live captures is rejected with
// cudaErrorStreamCaptureMerge -- so a tensor-parallel decode program is ONE graph holding all
// devices' nodes, not one graph per device.
//
//   record(fork[i]) on the origin stream  ->  wait(fork[i]) on peer i's stream   (peer joins)
//   ...the whole multi-device decode body...
//   record(join[i]) on peer i's stream     ->  wait(join[i]) on the origin stream (peer rejoins)
//
// The join is mandatory: cudaStreamEndCapture fails with cudaErrorStreamCaptureUnjoined if any
// forked stream is still outstanding. One gate event per peer serves gate_launch() below, which is
// about replay rather than capture. All events are created once (cudaEventCreate is not
// capturable) and one bridge instance serves an unbounded number of sequential captures/launches.
//
// The one-peer form is exactly the original two-device topology; the N-peer form is the tp4 (and
// general) extension. One implementation serves both.
class DecodeGraphPeerBridge {
public:
    // `peer_devices` are the non-origin devices, in rank order; must all differ from the origin and
    // from each other. The array is copied, so it need not outlive the constructor.
    DecodeGraphPeerBridge(int origin_device, std::span<const int> peer_devices);
    // Two-device convenience form.
    DecodeGraphPeerBridge(int origin_device, int peer_device);
    ~DecodeGraphPeerBridge();

    DecodeGraphPeerBridge(const DecodeGraphPeerBridge&)            = delete;
    DecodeGraphPeerBridge& operator=(const DecodeGraphPeerBridge&) = delete;
    DecodeGraphPeerBridge(DecodeGraphPeerBridge&& other) noexcept;
    DecodeGraphPeerBridge& operator=(DecodeGraphPeerBridge&& other) noexcept;

    [[nodiscard]] int origin_device() const noexcept { return origin_device_; }
    [[nodiscard]] int peer_count() const noexcept { return static_cast<int>(peer_devices_.size()); }
    [[nodiscard]] int peer_device(int index) const { return peer_devices_.at(static_cast<std::size_t>(index)); }
    [[nodiscard]] cudaEvent_t fork_event(int index) const { return fork_.at(static_cast<std::size_t>(index)); }
    [[nodiscard]] cudaEvent_t join_event(int index) const { return join_.at(static_cast<std::size_t>(index)); }
    [[nodiscard]] bool live() const noexcept;

    // REPLAY-side ordering, not capture-side. A multi-device graph is launched on the ORIGIN
    // device's stream; the graph's own edges then order its peer-device nodes after the graph
    // root, but they say nothing about work the caller already enqueued on each peer device's own
    // stream (at decode time: the mirrored KV page materialization). Eager execution gets that
    // ordering for free because it issues the peer's kernels on that same stream; a graph launch
    // does not, so the origin stream is explicitly ordered after every peer stream's outstanding
    // work before each launch, which transitively orders the whole graph after it.
    void gate_launch(const cudaStream_t* peer_streams, cudaStream_t origin_stream) const;
    // One-peer convenience form.
    void gate_launch(cudaStream_t peer_stream, cudaStream_t origin_stream) const;

private:
    void create_events();

    int origin_device_ = 0;
    std::vector<int> peer_devices_;
    std::vector<cudaEvent_t> fork_;
    std::vector<cudaEvent_t> join_;
    std::vector<cudaEvent_t> gate_;
};

// The peer half of a multi-device capture: which streams to enroll, and the bridge that enrolls
// them. `streams` holds one stream per bridge peer, in the same order as the bridge's peer devices.
struct DecodeGraphPeerCapture {
    const DecodeGraphPeerBridge* bridge = nullptr;
    const cudaStream_t* streams         = nullptr;
    int stream_count                    = 0;
};

class DecodeGraphDefinition {
public:
    DecodeGraphDefinition() = default;
    ~DecodeGraphDefinition();

    DecodeGraphDefinition(const DecodeGraphDefinition&)            = delete;
    DecodeGraphDefinition& operator=(const DecodeGraphDefinition&) = delete;
    DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept;
    DecodeGraphDefinition& operator=(DecodeGraphDefinition&& other) noexcept;

    // Single-device capture: `stream` is both the origin and the only stream captured.
    void capture(cudaStream_t stream, const std::function<void()>& body);
    // Multi-device capture: `stream` is the origin (device 0) and `peer` names each peer device's
    // stream, which is forked into the same capture for the duration of `body` and joined back
    // before the capture ends. A null `peer.bridge` is the single-device form above.
    void capture(cudaStream_t stream, const std::function<void()>& body,
                 const DecodeGraphPeerCapture& peer);
    [[nodiscard]] bool ready() const noexcept;
    // Node count of the captured graph, 0 when empty. Cross-device event edges are edges, not
    // nodes, so this counts real device work on BOTH devices.
    [[nodiscard]] std::size_t node_count() const;
    void reset() noexcept;

private:
    friend class DecodeGraphExecutable;
    cudaGraph_t graph_ = nullptr;
};

class DecodeGraphExecutable {
public:
    DecodeGraphExecutable() = default;
    ~DecodeGraphExecutable();

    DecodeGraphExecutable(const DecodeGraphExecutable&)            = delete;
    DecodeGraphExecutable& operator=(const DecodeGraphExecutable&) = delete;
    DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept;
    DecodeGraphExecutable& operator=(DecodeGraphExecutable&& other) noexcept;

    void instantiate(const DecodeGraphDefinition& definition);
    void update(const DecodeGraphDefinition& definition);
    void upload(cudaStream_t stream);
    void launch(cudaStream_t stream);
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

private:
    cudaGraphExec_t exec_ = nullptr;
};

} // namespace ninfer
