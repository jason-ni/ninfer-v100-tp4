#pragma once

#include "core/device.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <functional>
#include <span>

namespace ninfer {

// Tensor-parallel decode captures ONE CUDA Graph PER PARTICIPATING RANK, all in a single capture
// pass, and replays them concurrently on their own streams. This replaces the former
// single-graph-holding-all-devices design: a single graph serializes the rank critical path and
// starves the other GPUs (measured ~2.5x slower at tp4 on V100), whereas independent per-rank
// graphs keep the ranks overlapped exactly as the eager path does.
//
// The properties that make this capture legal and deterministic:
//
//  1. Simultaneous capture needs cudaStreamCaptureModeThreadLocal. The single engine thread begins
//     capture on every rank's stream, runs the body once (issuing each rank's work, and every
//     collective as a grouped per-rank NCCL call), then ends capture on every stream. ThreadLocal
//     keeps a capture on one stream from hijacking another stream's activation.
//  2. Cross-rank ordering is NCCL's own, not cudaStreamWaitEvent. A wait on an event recorded
//     inside a *different* live capture is rejected with cudaErrorStreamCaptureMerge, so any
//     per-call event edge would make simultaneous capture impossible. Grouped NCCL collectives
//     carry the ordering internally and capture cleanly.
//  3. Each graph is instantiated with flags 0. Device-launchable graphs are single-device only.
//  4. cudaGraphExecUpdate requires identical topology per rank; all profiles of a family are
//     captured from the same body against the same DeviceContexts, so a profile swap updates each
//     rank's executable in place.
//
// A DecodeGraphExecutable owns every rank's executable, device, and stream, so upload()/launch()
// take no arguments; replay simply launches each rank's graph on its own stream.
class DecodeGraphDefinition {
public:
    DecodeGraphDefinition() = default;
    ~DecodeGraphDefinition();

    DecodeGraphDefinition(const DecodeGraphDefinition&)            = delete;
    DecodeGraphDefinition& operator=(const DecodeGraphDefinition&) = delete;
    DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept;
    DecodeGraphDefinition& operator=(DecodeGraphDefinition&& other) noexcept;

    // Capture one graph per rank. `devices[r]` is the CUDA device and `streams[r]` the stream for
    // rank r; both spans must have the same, non-zero length (<= kMaximumDevices). `body` issues
    // each rank's work on its stream and is invoked exactly once.
    void capture(std::span<const int> devices, std::span<const cudaStream_t> streams,
                 const std::function<void()>& body);
    [[nodiscard]] bool ready() const noexcept;
    [[nodiscard]] int rank_count() const noexcept;
    // Total node count across every rank's graph; 0 when empty. NCCL collective edges are edges,
    // not nodes, so this counts real device work.
    [[nodiscard]] std::size_t node_count() const;
    void reset() noexcept;

private:
    friend class DecodeGraphExecutable;
    std::array<cudaGraph_t, kMaximumDevices> graphs_{};
    int ranks_ = 0;
};

class DecodeGraphExecutable {
public:
    DecodeGraphExecutable() = default;
    ~DecodeGraphExecutable();

    DecodeGraphExecutable(const DecodeGraphExecutable&)            = delete;
    DecodeGraphExecutable& operator=(const DecodeGraphExecutable&) = delete;
    DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept;
    DecodeGraphExecutable& operator=(DecodeGraphExecutable&& other) noexcept;

    // Instantiate one executable per rank from the matching definition, remembering each rank's
    // device and stream for later update/upload/launch.
    void instantiate(const DecodeGraphDefinition& definition, std::span<const int> devices,
                     std::span<const cudaStream_t> streams);
    void update(const DecodeGraphDefinition& definition);
    void upload();
    void launch();
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

private:
    std::array<cudaGraphExec_t, kMaximumDevices> execs_{};
    std::array<int, kMaximumDevices> devices_{};
    std::array<cudaStream_t, kMaximumDevices> streams_{};
    int ranks_ = 0;
};

} // namespace ninfer
