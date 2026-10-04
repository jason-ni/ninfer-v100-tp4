// Implements: src/core/decode_graph.h
//
// Per-rank CUDA Graph capture. See the header for the four binding constraints. This file owns the
// low-level capture/instantiate/update/upload/launch mechanics and the current-device save/restore
// every multi-device call needs.
#include "core/decode_graph.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

void cuda_check(cudaError_t result, const char* what) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(result));
    }
}

// Restores the caller's device on scope exit so per-rank cudaSetDevice calls do not leak.
class ScopedDevice {
public:
    ScopedDevice() { cuda_check(cudaGetDevice(&previous_), "cudaGetDevice"); }
    ~ScopedDevice() { cudaSetDevice(previous_); }
    ScopedDevice(const ScopedDevice&)            = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;
    static void set(int device) { cuda_check(cudaSetDevice(device), "cudaSetDevice"); }

private:
    int previous_ = 0;
};

void destroy_graph(cudaGraph_t& graph) {
    if (graph != nullptr) {
        cudaGraphDestroy(graph);
        graph = nullptr;
    }
}

void destroy_graph_exec(cudaGraphExec_t& exec) {
    if (exec != nullptr) {
        cudaGraphExecDestroy(exec);
        exec = nullptr;
    }
}

} // namespace

DecodeGraphDefinition::~DecodeGraphDefinition() { reset(); }

DecodeGraphDefinition::DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept
    : graphs_(other.graphs_), ranks_(other.ranks_) {
    other.graphs_.fill(nullptr);
    other.ranks_ = 0;
}

DecodeGraphDefinition& DecodeGraphDefinition::operator=(DecodeGraphDefinition&& other) noexcept {
    if (this != &other) {
        reset();
        graphs_ = other.graphs_;
        ranks_  = other.ranks_;
        other.graphs_.fill(nullptr);
        other.ranks_ = 0;
    }
    return *this;
}

void DecodeGraphDefinition::capture(std::span<const int> devices,
                                    std::span<const cudaStream_t> streams,
                                    const std::function<void()>& body) {
    if (devices.size() != streams.size() || streams.empty() ||
        streams.size() > kMaximumDevices) {
        throw std::invalid_argument("DecodeGraphDefinition::capture: rank count mismatch");
    }
    if (!body) { throw std::invalid_argument("DecodeGraphDefinition::capture: null body"); }

    reset();
    const int ranks = static_cast<int>(streams.size());
    const ScopedDevice restore;

    try {
        for (int rank = 0; rank < ranks; ++rank) {
            ScopedDevice::set(devices[rank]);
            cuda_check(cudaStreamBeginCapture(streams[rank], cudaStreamCaptureModeThreadLocal),
                       "cudaStreamBeginCapture");
        }
        // The body runs exactly as it does eagerly, on the origin device: the begin loop above
        // leaves the current device on the last rank, and the rank-0-only tail of a schedule (the
        // ingress copy, scatter, sample) launches on origin's stream and would otherwise be an
        // invalid-handle launch.
        ScopedDevice::set(devices[0]);
        body();
        for (int rank = 0; rank < ranks; ++rank) {
            ScopedDevice::set(devices[rank]);
            cudaGraph_t graph = nullptr;
            cuda_check(cudaStreamEndCapture(streams[rank], &graph), "cudaStreamEndCapture");
            graphs_[rank] = graph;
        }
        ranks_ = ranks;
    } catch (...) {
        // Best effort: end and destroy any capture still open so the streams are usable again.
        for (int rank = 0; rank < ranks; ++rank) {
            ScopedDevice::set(devices[rank]);
            cudaGraph_t dangling = nullptr;
            if (cudaStreamEndCapture(streams[rank], &dangling) == cudaSuccess) {
                destroy_graph(dangling);
            }
        }
        reset();
        throw;
    }
}

bool DecodeGraphDefinition::ready() const noexcept {
    if (ranks_ == 0) return false;
    for (int rank = 0; rank < ranks_; ++rank) {
        if (graphs_[rank] == nullptr) return false;
    }
    return true;
}

int DecodeGraphDefinition::rank_count() const noexcept { return ranks_; }

std::size_t DecodeGraphDefinition::node_count() const {
    std::size_t total = 0;
    for (int rank = 0; rank < ranks_; ++rank) {
        if (graphs_[rank] == nullptr) continue;
        std::size_t count = 0;
        if (cudaGraphGetNodes(graphs_[rank], nullptr, &count) != cudaSuccess) continue;
        total += count;
    }
    return total;
}

void DecodeGraphDefinition::reset() noexcept {
    for (cudaGraph_t& graph : graphs_) destroy_graph(graph);
    ranks_ = 0;
}

DecodeGraphExecutable::~DecodeGraphExecutable() { reset(); }

DecodeGraphExecutable::DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept
    : execs_(other.execs_), devices_(other.devices_), streams_(other.streams_),
      ranks_(other.ranks_) {
    other.execs_.fill(nullptr);
    other.ranks_ = 0;
}

DecodeGraphExecutable& DecodeGraphExecutable::operator=(DecodeGraphExecutable&& other) noexcept {
    if (this != &other) {
        reset();
        execs_   = other.execs_;
        devices_ = other.devices_;
        streams_ = other.streams_;
        ranks_   = other.ranks_;
        other.execs_.fill(nullptr);
        other.ranks_ = 0;
    }
    return *this;
}

void DecodeGraphExecutable::instantiate(const DecodeGraphDefinition& definition,
                                        std::span<const int> devices,
                                        std::span<const cudaStream_t> streams) {
    if (!definition.ready()) {
        throw std::logic_error("cannot instantiate an empty CUDA Graph definition");
    }
    if (devices.size() != streams.size() ||
        static_cast<int>(streams.size()) != definition.rank_count()) {
        throw std::invalid_argument("DecodeGraphExecutable::instantiate: rank count mismatch");
    }
    reset();

    const int ranks = definition.rank_count();
    const ScopedDevice restore;
    for (int rank = 0; rank < ranks; ++rank) {
        ScopedDevice::set(devices[rank]);
        cudaGraphExec_t exec = nullptr;
        const cudaError_t err = cudaGraphInstantiate(&exec, definition.graphs_[rank], 0);
        if (err != cudaSuccess) {
            destroy_graph_exec(exec);
            reset();
            cuda_check(err, "cudaGraphInstantiate");
        }
        execs_[rank]   = exec;
        devices_[rank] = devices[rank];
        streams_[rank] = streams[rank];
    }
    ranks_ = ranks;
}

void DecodeGraphExecutable::update(const DecodeGraphDefinition& definition) {
    if (!ready() || !definition.ready() || definition.rank_count() != ranks_) {
        throw std::logic_error("CUDA Graph update requires a definition and executable");
    }
    const ScopedDevice restore;
    for (int rank = 0; rank < ranks_; ++rank) {
        ScopedDevice::set(devices_[rank]);
        cudaGraphExecUpdateResultInfo result{};
        const cudaError_t err = cudaGraphExecUpdate(execs_[rank], definition.graphs_[rank], &result);
        if (err != cudaSuccess || result.result != cudaGraphExecUpdateSuccess) {
            throw std::runtime_error(
                "CUDA Graph executable update failed: " + std::string(cudaGetErrorName(err)) +
                " (update result " + std::to_string(static_cast<int>(result.result)) + ")");
        }
    }
}

void DecodeGraphExecutable::upload() {
    if (!ready()) { throw std::logic_error("cannot upload an empty CUDA Graph executable"); }
    const ScopedDevice restore;
    for (int rank = 0; rank < ranks_; ++rank) {
        ScopedDevice::set(devices_[rank]);
        cuda_check(cudaGraphUpload(execs_[rank], streams_[rank]), "cudaGraphUpload");
    }
}

void DecodeGraphExecutable::launch() {
    if (!ready()) { throw std::logic_error("cannot launch an empty CUDA Graph executable"); }
    const ScopedDevice restore;
    for (int rank = 0; rank < ranks_; ++rank) {
        ScopedDevice::set(devices_[rank]);
        cuda_check(cudaGraphLaunch(execs_[rank], streams_[rank]), "cudaGraphLaunch");
    }
}

bool DecodeGraphExecutable::ready() const noexcept {
    if (ranks_ == 0) return false;
    for (int rank = 0; rank < ranks_; ++rank) {
        if (execs_[rank] == nullptr) return false;
    }
    return true;
}

void DecodeGraphExecutable::reset() noexcept {
    for (cudaGraphExec_t& exec : execs_) destroy_graph_exec(exec);
    ranks_ = 0;
}

} // namespace ninfer
