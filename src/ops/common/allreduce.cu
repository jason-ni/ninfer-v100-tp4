// Implements: include/ninfer/ops/allreduce.h
//
// Host-side composition only: the transport is cudaMemcpyAsync with cudaMemcpyDeviceToDevice over
// UVA pointers (see pull_peer() below -- deliberately NOT cudaMemcpyPeerAsync, which stream
// capture rejects), and the local combine reuses the qualified residual_add computation body
// (x += y in BF16 with FP32 accumulation and a single round-to-nearest-even on store), which is
// exactly this Op's local step. Sharing that private launch body keeps one implementation of the
// BF16 sum instead of a second, separately qualified copy of the same arithmetic.
//
// The summing reduce is a star through rank 0 so every rank receives bitwise-identical bytes:
//
//   phase A, all ranks:  record(inputs_ready[r])
//   phase B, rank 0:     for s in 1..tp-1 { wait(inputs_ready[s]); partial[s] -> staging[0];
//                        staging[0] += buffer[0] }; record(pull_done[0])
//   phase C, rank r>0:   wait(pull_done[0]); buffer[0] -> buffer[r]; record(pull_done[r])
//   phase D, rank 0:     for r in 1..tp-1 { wait(pull_done[r]) }
//
// At tp == 2 this is exactly the historical two-party exchange; at tp == 1 it is a no-op.
//
// THE PULL ITSELF is cudaMemcpyAsync with cudaMemcpyDeviceToDevice over UVA pointers, NOT
// cudaMemcpyPeerAsync -- see pull_peer() below for why. The choreography, the streams each call
// is issued on, and the ordering proof are unchanged by that choice: it is the same transfer
// expressed through the API that CUDA graph capture accepts.
#include "ninfer/ops/allreduce.h"

#include "ops/launcher/residual_add.h" // detail::residual_add_launch

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <array>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

void require_tp(const ExecutionContext& ec, const char* message) {
    require(ec.tp >= 1 && ec.tp <= static_cast<int>(kMaximumDevices), message);
    for (int rank = 0; rank < ec.tp; ++rank) {
        require(ec.dev[rank].has_value(), message);
        for (int other = rank + 1; other < ec.tp; ++other) {
            require(ec.dev[rank]->device != ec.dev[other]->device, message);
        }
    }
}

std::uint8_t* byte_offset(void* base, std::size_t offset) {
    return static_cast<std::uint8_t*>(base) + offset;
}

// The inbound half of a pull: `bytes` from `source` (resident on a peer device) into
// `destination` (resident on the device `stream` belongs to), issued on the DESTINATION's stream.
//
// Deliberately NOT cudaMemcpyPeerAsync. That entry point is rejected inside a stream capture
// region with cudaErrorStreamCaptureUnsupported, which would make the entire tensor-parallel
// decode program uncapturable and cost the ~40-per-layer host launch overhead that CUDA Graphs
// exist to remove. Under unified virtual addressing -- which every 64-bit Linux CUDA context has
// -- a device pointer already names its device, so cudaMemcpyAsync with cudaMemcpyDeviceToDevice
// expresses exactly the same cross-device transfer: direct over PCIe when the driver granted peer
// access, transparently staged through host memory when it did not, identical either way in bytes
// moved and stream ordering.
cudaError_t pull_peer(void* destination, const void* source, std::size_t bytes,
                      cudaStream_t stream) {
    return cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, stream);
}

// Current-device save/restore. The collectives issue work for each device in turn and must not
// leave the caller's current device changed.
class CurrentDeviceGuard {
public:
    CurrentDeviceGuard() { CUDA_CHECK(cudaGetDevice(&previous_)); }

    ~CurrentDeviceGuard() {
        const cudaError_t status = cudaSetDevice(previous_);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during cudaSetDevice: %s: %s\n",
                         cudaGetErrorName(status), cudaGetErrorString(status));
        }
    }

    CurrentDeviceGuard(const CurrentDeviceGuard&)            = delete;
    CurrentDeviceGuard& operator=(const CurrentDeviceGuard&) = delete;

    static void set(int device) { CUDA_CHECK(cudaSetDevice(device)); }

private:
    int previous_ = 0;
};

void startup_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("peer transport startup: ") + operation + ": " +
                                 cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
    }
}

// Linux CUDA PCIe P2P is unsupported behind a translated IOMMU domain. A small
// allocation can nevertheless pass a copy probe while other mappings silently
// lose writes, so the domain restriction takes precedence over that probe.
std::string translated_iommu_domain(const ExecutionContext& ec) {
    std::string reason;
#if defined(__linux__)
    for (int rank = 0; rank < ec.tp; ++rank) {
        char pci_bus_id[32]{};
        startup_check(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), ec.dev[rank]->device),
                      "cudaDeviceGetPCIBusId");
        // CUDA may emit uppercase hexadecimal; Linux PCI sysfs names are lowercase.
        for (char& c : pci_bus_id) {
            if (c >= 'A' && c <= 'F') { c += 'a' - 'A'; }
        }
        std::ifstream domain_file(std::string("/sys/bus/pci/devices/") + pci_bus_id +
                                  "/iommu_group/type");
        std::string domain;
        domain_file >> domain;
        if (domain == "DMA" || domain == "DMA-FQ") {
            if (!reason.empty()) { reason += "; "; }
            reason += std::string("PCI ") + pci_bus_id + " uses translated IOMMU domain " + domain;
        }
    }
#else
    (void)ec;
#endif
    return reason;
}

// Startup-only storage. The exact payload catches drivers that advertise peer access
// but silently drop DMA writes (observed with V100s behind an IOMMU). Use the same
// pull API and destination compute stream as the collectives, without GPU peer loads.
class PeerTransferProbe {
public:
    explicit PeerTransferProbe(const ExecutionContext& ec) : ec_(ec) {
        startup_check(cudaGetDevice(&previous_), "cudaGetDevice");
    }

    ~PeerTransferProbe() {
        for (int rank = 0; rank < ec_.tp; ++rank) {
            cleanup(cudaSetDevice(ec_.dev[rank]->device), "cudaSetDevice");
            if (source_[rank] != nullptr) { cleanup(cudaFree(source_[rank]), "cudaFree source"); }
            if (destination_[rank] != nullptr) {
                cleanup(cudaFree(destination_[rank]), "cudaFree destination");
            }
        }
        cleanup(cudaSetDevice(previous_), "restore device");
    }

    PeerTransferProbe(const PeerTransferProbe&) = delete;
    PeerTransferProbe& operator=(const PeerTransferProbe&) = delete;

    void set_device(int rank) const {
        startup_check(cudaSetDevice(ec_.dev[rank]->device), "cudaSetDevice");
    }

    void initialize() {
        for (int rank = 0; rank < ec_.tp; ++rank) {
            set_device(rank);
            startup_check(cudaMalloc(&source_[rank], kBytes), "cudaMalloc source");
            startup_check(cudaMalloc(&destination_[rank], kBytes), "cudaMalloc destination");
            std::array<std::uint32_t, kWords> values{};
            for (std::size_t i = 0; i < kWords; ++i) { values[i] = pattern(rank, i); }
            startup_check(cudaMemcpyAsync(source_[rank], values.data(), kBytes,
                                           cudaMemcpyHostToDevice, ec_.dev[rank]->stream),
                          "initialize source");
            startup_check(cudaStreamSynchronize(ec_.dev[rank]->stream), "retire source");
        }
    }

    // Empty means every ordered pair's copy matched its independent host pattern exactly.
    std::string qualify() {
        std::string mismatch;
        for (int destination = 0; destination < ec_.tp; ++destination) {
            set_device(destination);
            cudaStream_t stream = ec_.dev[destination]->stream;
            for (int source = 0; source < ec_.tp; ++source) {
                if (source == destination) { continue; }
                startup_check(cudaMemsetAsync(destination_[destination], 0xcd, kBytes, stream),
                              "clear destination");
                startup_check(pull_peer(destination_[destination], source_[source], kBytes, stream),
                              "cross-device copy");
                startup_check(cudaStreamSynchronize(stream), "retire cross-device copy");
                std::array<std::uint32_t, kWords> actual{};
                startup_check(cudaMemcpy(actual.data(), destination_[destination], kBytes,
                                          cudaMemcpyDeviceToHost), "read destination");
                if (!mismatch.empty()) { continue; }
                for (std::size_t i = 0; i < kWords; ++i) {
                    if (actual[i] != pattern(source, i)) {
                        mismatch = "device " + std::to_string(ec_.dev[source]->device) + " -> " +
                                   std::to_string(ec_.dev[destination]->device) +
                                   " data mismatch at word " + std::to_string(i);
                        break;
                    }
                }
            }
        }
        return mismatch;
    }

    void disable_peer_access() const {
        for (int rank = 0; rank < ec_.tp; ++rank) {
            set_device(rank);
            for (int other = 0; other < ec_.tp; ++other) {
                if (other == rank) { continue; }
                const cudaError_t status = cudaDeviceDisablePeerAccess(ec_.dev[other]->device);
                if (status == cudaErrorPeerAccessNotEnabled) {
                    (void)cudaGetLastError();
                } else {
                    startup_check(status, "cudaDeviceDisablePeerAccess");
                }
            }
        }
    }

private:
    static constexpr std::size_t kWords = 4096;
    static constexpr std::size_t kBytes = kWords * sizeof(std::uint32_t);

    static std::uint32_t pattern(int rank, std::size_t index) {
        return 0x4f000000U ^ (std::uint32_t(rank) << 20U) ^
               (static_cast<std::uint32_t>(index) * 65537U);
    }

    static void cleanup(cudaError_t status, const char* operation) noexcept {
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during peer probe %s: %s: %s\n",
                         operation, cudaGetErrorName(status), cudaGetErrorString(status));
        }
    }

    const ExecutionContext& ec_;
    int previous_ = 0;
    std::array<void*, kMaximumDevices> source_{};
    std::array<void*, kMaximumDevices> destination_{};
};

#ifndef NDEBUG
// Debug-only residency and aliasing predicates. These cost a driver round trip per pointer, so
// they are compiled out of the Release build the product ships; a wrong-device or self-overlapping
// argument is a caller bug that surfaces here during development instead of as a silently wrong
// result or an opaque cudaErrorInvalidValue later.
void require_resident_on(const void* pointer, int device, const char* message) {
    cudaPointerAttributes attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointer));
    require(attributes.type == cudaMemoryTypeDevice && attributes.device == device, message);
}

void require_disjoint(const void* first, std::size_t first_bytes, const void* second,
                      std::size_t second_bytes, const char* message) {
    const auto* a = static_cast<const std::uint8_t*>(first);
    const auto* b = static_cast<const std::uint8_t*>(second);
    require(a + first_bytes <= b || b + second_bytes <= a, message);
}
#endif

} // namespace

bool enable_peer_access(const ExecutionContext& ec) {
    if (ec.tp < 2 || ec.tp > static_cast<int>(kMaximumDevices)) { return false; }
    for (int rank = 0; rank < ec.tp; ++rank) {
        if (!ec.dev[rank].has_value()) { return false; }
        for (int other = rank + 1; other < ec.tp; ++other) {
            if (ec.dev[rank]->device == ec.dev[other]->device) { return false; }
        }
    }

    PeerTransferProbe probe(ec);
    try {
        std::string direct_failure = translated_iommu_domain(ec);
        bool supported             = direct_failure.empty();
        if (supported) {
            for (int from = 0; from < ec.tp && supported; ++from) {
                for (int to = 0; to < ec.tp && supported; ++to) {
                    if (from == to) { continue; }
                    int can_access = 0;
                    startup_check(cudaDeviceCanAccessPeer(&can_access, ec.dev[from]->device,
                                                          ec.dev[to]->device),
                                  "cudaDeviceCanAccessPeer");
                    if (can_access == 0) { supported = false; }
                }
            }
        }
        if (supported) {
            for (int from = 0; from < ec.tp; ++from) {
                probe.set_device(from);
                for (int to = 0; to < ec.tp; ++to) {
                    if (from == to) { continue; }
                    const cudaError_t status =
                        cudaDeviceEnablePeerAccess(ec.dev[to]->device, 0);
                    if (status == cudaErrorPeerAccessAlreadyEnabled) {
                        (void)cudaGetLastError();
                    } else {
                        startup_check(status, "cudaDeviceEnablePeerAccess");
                    }
                }
            }
        } else {
            if (direct_failure.empty()) { direct_failure = "peer access unavailable"; }
            probe.disable_peer_access();
        }
        probe.initialize();
        if (supported) {
            direct_failure = probe.qualify();
            if (direct_failure.empty()) {
                return true;
            }
            probe.disable_peer_access();
        }

        // With peer access disabled (or prohibited by the IOMMU domain), the same UVA D2D
        // operation is handled by CUDA's driver-managed staging path. Qualify that exact API
        // rather than maintaining a second explicit D2H/H2D transport in every collective.
        const std::string fallback_failure = probe.qualify();
        if (!fallback_failure.empty()) {
            throw std::runtime_error("peer transport startup: UVA D2D fallback validation failed: " +
                                     fallback_failure);
        }
        std::fprintf(stderr,
                     "[ninfer] direct P2P disabled (%s); using verified CUDA UVA D2D staging\n",
                     direct_failure.c_str());
        return false;
    } catch (...) {
        // Failed startup must not leave a partially enabled set behind. Clear the
        // runtime's last error before cleanup; a fatal context error still prevents
        // further use, and the original startup exception remains authoritative.
        (void)cudaGetLastError();
        try { probe.disable_peer_access(); } catch (...) {}
        throw;
    }
}

PeerEvents::PeerEvents(const ExecutionContext& ec) {
    require_tp(ec, "PeerEvents: requires an ExecutionContext with 1..4 distinct devices");
    tp_ = ec.tp;
    const CurrentDeviceGuard guard;
    // Create through a local table so a mid-way failure destroys what was already created instead
    // of leaking it; only a fully constructed set is published into the members.
    std::array<cudaEvent_t, 2 * kMaximumDevices> created{};
    for (int slot = 0; slot < 2 * tp_; ++slot) {
        const int rank             = slot % tp_;
        const cudaError_t creation = cudaSetDevice(ec.dev[rank]->device);
        cudaError_t status         = creation;
        if (status == cudaSuccess) {
            status = cudaEventCreateWithFlags(&created[static_cast<std::size_t>(slot)],
                                              cudaEventDisableTiming);
        }
        if (status != cudaSuccess) {
            for (int done = 0; done < slot; ++done) {
                cudaEventDestroy(created[static_cast<std::size_t>(done)]);
            }
            tp_ = 0;
            throw std::runtime_error(std::string("PeerEvents: event creation failed: ") +
                                     cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
        }
    }
    for (int rank = 0; rank < tp_; ++rank) {
        inputs_ready_[static_cast<std::size_t>(rank)] = created[static_cast<std::size_t>(rank)];
        pull_done_[static_cast<std::size_t>(rank)] =
            created[static_cast<std::size_t>(tp_ + rank)];
    }
    // Seed pull_done with completed events so the first collective may use the same inter-call
    // lifetime edge as every later one.
    for (int rank = 0; rank < tp_; ++rank) {
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        CUDA_CHECK(cudaEventRecord(pull_done_[static_cast<std::size_t>(rank)],
                                   ec.dev[rank]->stream));
    }
    for (int rank = 0; rank < tp_; ++rank) {
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        CUDA_CHECK(cudaStreamSynchronize(ec.dev[rank]->stream));
    }
}

PeerEvents::~PeerEvents() {
    for (std::array<cudaEvent_t, kMaximumDevices>* group : {&inputs_ready_, &pull_done_}) {
        for (cudaEvent_t& event : *group) {
            if (event == nullptr) { continue; }
            const cudaError_t status = cudaEventDestroy(event);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "CUDA cleanup failed during cudaEventDestroy: %s: %s\n",
                             cudaGetErrorName(status), cudaGetErrorString(status));
            }
            event = nullptr;
        }
    }
}

PeerEvents::PeerEvents(PeerEvents&& other) noexcept
    : tp_(other.tp_), inputs_ready_(other.inputs_ready_), pull_done_(other.pull_done_) {
    other.tp_ = 0;
    other.inputs_ready_ = {};
    other.pull_done_    = {};
}

PeerEvents& PeerEvents::operator=(PeerEvents&& other) noexcept {
    // Swap rather than destroy-then-assign: `other`'s destructor releases whatever this instance
    // held, in exactly one place.
    std::swap(tp_, other.tp_);
    inputs_ready_.swap(other.inputs_ready_);
    pull_done_.swap(other.pull_done_);
    return *this;
}

void allreduce_sum(const std::array<Tensor, kMaximumDevices>& buffer,
                   const std::array<Tensor, kMaximumDevices>& staging, const ExecutionContext& ec,
                   const PeerEvents& events) {
    require_tp(ec, "allreduce_sum: requires an ExecutionContext with 1..4 distinct devices");
    const int tp = ec.tp;
    for (int rank = 0; rank < tp; ++rank) {
        require(buffer[rank].dtype == DType::BF16 && staging[rank].dtype == DType::BF16,
                "allreduce_sum: buffer/staging must be BF16");
        require(buffer[rank].data != nullptr && staging[rank].data != nullptr,
                "allreduce_sum: buffer/staging data must be non-null");
        require(buffer[rank].is_contiguous() && staging[rank].is_contiguous(),
                "allreduce_sum: buffer/staging must be contiguous");
        for (int d = 0; d < 4; ++d) {
            require(buffer[rank].ne[d] == buffer[0].ne[d] && staging[rank].ne[d] == buffer[0].ne[d],
                    "allreduce_sum: buffer/staging shapes must match on every device");
        }
    }
    require(events.live(), "allreduce_sum: events must be live");
    require(events.count() == tp, "allreduce_sum: events must be created for this tp");

    const std::size_t bytes = buffer[0].bytes();
    if (bytes == 0 || tp == 1) { return; }

#ifndef NDEBUG
    for (int rank = 0; rank < tp; ++rank) {
        require_resident_on(buffer[rank].data, ec.dev[rank]->device,
                            "allreduce_sum: buffer[r] must be resident on ec.dev[r]");
        require_resident_on(staging[rank].data, ec.dev[rank]->device,
                            "allreduce_sum: staging[r] must be resident on ec.dev[r]");
        require_disjoint(buffer[rank].data, bytes, staging[rank].data, bytes,
                         "allreduce_sum: staging[r] must not overlap buffer[r]");
    }
#endif

    const CurrentDeviceGuard guard;

    // Phase A: publish "my operand is complete" on each stream, before any wait observes it.
    for (int rank = 0; rank < tp; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
    }

    // Phase B: rank 0 gathers every peer partial into its own scratch and folds them into its
    // buffer in one fixed rank order, so the summed bytes are identical on every rank. The
    // scratch is rank 0's staging[0]; it is reused across peers and stream-ordered between uses.
    {
        const DeviceContext& rank0 = *ec.dev[0];
        CurrentDeviceGuard::set(rank0.device);
        Tensor accumulator = buffer[0];
        for (int source = 1; source < tp; ++source) {
            CUDA_CHECK(cudaStreamWaitEvent(rank0.stream, events.inputs_ready(source), 0));
            CUDA_CHECK(pull_peer(staging[0].data, buffer[source].data, bytes, rank0.stream));
            detail::residual_add_launch(staging[0], accumulator, rank0.stream);
        }
        CUDA_CHECK(cudaEventRecord(events.pull_done(0), rank0.stream));
    }

    // Phase C: each peer pulls rank 0's exact summed bytes into its own buffer.
    for (int rank = 1; rank < tp; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(0), 0));
        CUDA_CHECK(pull_peer(buffer[rank].data, buffer[0].data, bytes, local.stream));
        CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
    }

    // Phase D: rank 0 may only overwrite buffer[0] once every peer has finished reading it. That
    // same wait makes the next call's phase B safe.
    {
        const DeviceContext& rank0 = *ec.dev[0];
        CurrentDeviceGuard::set(rank0.device);
        for (int rank = 1; rank < tp; ++rank) {
            CUDA_CHECK(cudaStreamWaitEvent(rank0.stream, events.pull_done(rank), 0));
        }
    }
}

void allgather_rows(const std::array<Tensor, kMaximumDevices>& destination,
                    const std::array<Tensor, kMaximumDevices>& part, const ExecutionContext& ec,
                    const PeerEvents& events) {
    require_tp(ec, "allgather_rows: requires an ExecutionContext with 1..4 distinct devices");
    const int tp                  = ec.tp;
    const DType dtype             = destination[0].dtype;
    const std::int32_t row_length = destination[0].ne[0];
    const std::int32_t total_rows = destination[0].ne[1];
    for (int rank = 0; rank < tp; ++rank) {
        require(destination[rank].dtype == dtype && part[rank].dtype == dtype,
                "allgather_rows: destination/part must share one dtype");
        require(destination[rank].data != nullptr && part[rank].data != nullptr,
                "allgather_rows: destination/part data must be non-null");
        require(destination[rank].is_contiguous() && part[rank].is_contiguous(),
                "allgather_rows: destination/part must be contiguous");
        require(destination[rank].ne[0] == row_length && part[rank].ne[0] == row_length,
                "allgather_rows: destination/part must agree on row length ne[0]");
        require(destination[rank].ne[1] == total_rows,
                "allgather_rows: all destinations must have the same row count");
        require(destination[rank].ne[2] == 1 && destination[rank].ne[3] == 1 &&
                    part[rank].ne[2] == 1 && part[rank].ne[3] == 1,
                "allgather_rows: destination/part must be two-dimensional [C, R]");
    }
    std::int64_t owned_rows = 0;
    for (int rank = 0; rank < tp; ++rank) { owned_rows += part[rank].ne[1]; }
    require(owned_rows == total_rows,
            "allgather_rows: owned row counts must sum to the destination row count");
    require(events.live(), "allgather_rows: events must be live");
    require(events.count() == tp, "allgather_rows: events must be created for this tp");

    const std::size_t row_bytes = static_cast<std::size_t>(row_length) * dtype_size(dtype);
    std::array<std::size_t, kMaximumDevices> block{};
    std::array<std::size_t, kMaximumDevices> offset{};
    for (int rank = 0; rank < tp; ++rank) {
        block[rank] = row_bytes * static_cast<std::size_t>(part[rank].ne[1]);
    }
    for (int rank = 1; rank < tp; ++rank) { offset[rank] = offset[rank - 1] + block[rank - 1]; }

    if (row_bytes == 0 || tp == 1) { return; }

#ifndef NDEBUG
    for (int rank = 0; rank < tp; ++rank) {
        require_resident_on(destination[rank].data, ec.dev[rank]->device,
                            "allgather_rows: destination[r] must be resident on ec.dev[r]");
        require_resident_on(part[rank].data, ec.dev[rank]->device,
                            "allgather_rows: part[r] must be resident on ec.dev[r]");
        require_disjoint(destination[rank].data, destination[rank].bytes(), part[rank].data,
                         block[rank], "allgather_rows: part[r] must not overlap destination[r]");
    }
#endif

    const CurrentDeviceGuard guard;

    // Phase A: publish "my block is complete".
    for (int rank = 0; rank < tp; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
    }

    // Phase B: rank r writes its own block locally and pulls every peer block, all on its stream.
    for (int rank = 0; rank < tp; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaMemcpyAsync(byte_offset(destination[rank].data, offset[rank]),
                                   part[rank].data, block[rank], cudaMemcpyDeviceToDevice,
                                   local.stream));
        for (int source = 0; source < tp; ++source) {
            if (source == rank) { continue; }
            CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(source), 0));
            CUDA_CHECK(pull_peer(byte_offset(destination[rank].data, offset[source]),
                                 part[source].data, block[source], local.stream));
        }
        CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
    }

    // Phase C: the Op writes nothing else, but the caller (or the next call) will overwrite
    // part[rank]. Ordering each stream after every peer's read is what makes that safe without a
    // host synchronization.
    for (int rank = 0; rank < tp; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        for (int source = 0; source < tp; ++source) {
            if (source == rank) { continue; }
            CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(source), 0));
        }
    }
}

void gather_columns_rank0(const Tensor& destination,
                          const std::array<Tensor, kMaximumDevices>& part,
                          const ExecutionContext& ec, const PeerEvents& events) {
    require_tp(ec,
               "gather_columns_rank0: requires an ExecutionContext with 1..4 distinct devices");
    const int tp                  = ec.tp;
    const DType dtype             = destination.dtype;
    const std::int32_t full_width = destination.ne[0];
    const std::int32_t columns    = destination.ne[1];
    require(full_width > 0 && columns > 0,
            "gather_columns_rank0: dimensions must be positive");
    require(destination.data != nullptr && destination.is_contiguous(),
            "gather_columns_rank0: destination must be contiguous and non-null");
    require(destination.ne[2] == 1 && destination.ne[3] == 1,
            "gather_columns_rank0: destination must be two-dimensional [C,T]");
    for (int rank = 0; rank < tp; ++rank) {
        require(part[rank].dtype == dtype && part[rank].data != nullptr &&
                    part[rank].is_contiguous(),
                "gather_columns_rank0: parts must share dtype and be contiguous/non-null");
        require(part[rank].ne[1] == columns && part[rank].ne[0] > 0 &&
                    part[rank].ne[2] == 1 && part[rank].ne[3] == 1,
                "gather_columns_rank0: parts must be two-dimensional [C_r,T]");
    }
    std::int64_t owned_width = 0;
    for (int rank = 0; rank < tp; ++rank) { owned_width += part[rank].ne[0]; }
    require(owned_width == full_width,
            "gather_columns_rank0: owned widths must sum to destination width");
    require(events.live(), "gather_columns_rank0: events must be live");
    require(events.count() == tp, "gather_columns_rank0: events must be created for this tp");

    const std::size_t element_bytes = dtype_size(dtype);
    const std::size_t destination_pitch = static_cast<std::size_t>(full_width) * element_bytes;
    std::array<std::size_t, kMaximumDevices> block{};
    std::array<std::size_t, kMaximumDevices> offset{};
    for (int rank = 0; rank < tp; ++rank) {
        block[rank] = static_cast<std::size_t>(part[rank].ne[0]) * element_bytes;
    }
    for (int rank = 1; rank < tp; ++rank) { offset[rank] = offset[rank - 1] + block[rank - 1]; }

    if (element_bytes == 0 || columns == 0 || tp == 1) { return; }

#ifndef NDEBUG
    require_resident_on(destination.data, ec.dev[0]->device,
                        "gather_columns_rank0: destination must be resident on rank 0");
    for (int rank = 0; rank < tp; ++rank) {
        require_resident_on(part[rank].data, ec.dev[rank]->device,
                            "gather_columns_rank0: part must be resident on its rank");
        require_disjoint(destination.data, destination.bytes(), part[rank].data, part[rank].bytes(),
                         "gather_columns_rank0: part must not overlap destination");
    }
#endif

    const CurrentDeviceGuard guard;

    for (int rank = 0; rank < tp; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
    }

    const DeviceContext& rank0 = *ec.dev[0];
    CurrentDeviceGuard::set(rank0.device);
    for (int source = 0; source < tp; ++source) {
        CUDA_CHECK(cudaStreamWaitEvent(rank0.stream, events.inputs_ready(source), 0));
        CUDA_CHECK(cudaMemcpy2DAsync(byte_offset(destination.data, offset[source]),
                                     destination_pitch, part[source].data, block[source],
                                     block[source], static_cast<std::size_t>(columns),
                                     cudaMemcpyDeviceToDevice, rank0.stream));
    }
    CUDA_CHECK(cudaEventRecord(events.pull_done(0), rank0.stream));

    // Each peer may overwrite its proposal shard only after rank 0 has consumed it. No reciprocal
    // destination write is needed because the selector and all subsequent DFlash logic run on
    // rank 0.
    for (int rank = 1; rank < tp; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(0), 0));
    }
}

void broadcast_rank0(const Tensor& source,
                     const std::array<Tensor, kMaximumDevices>& destination,
                     const ExecutionContext& ec, const PeerEvents& events) {
    require_tp(ec, "broadcast_rank0: requires 1..4 distinct devices");
    require(events.live(), "broadcast_rank0: events must be live");
    require(events.count() == ec.tp, "broadcast_rank0: events must be created for this tp");
    require(source.data != nullptr && source.is_contiguous(),
            "broadcast_rank0: source must be contiguous and non-null");
    for (int rank = 1; rank < ec.tp; ++rank) {
        require(destination[rank].data != nullptr && destination[rank].dtype == source.dtype &&
                    destination[rank].is_contiguous(),
                "broadcast_rank0: invalid destination");
        for (int d = 0; d < 4; ++d) {
            require(source.ne[d] == destination[rank].ne[d],
                    "broadcast_rank0: shapes must match");
        }
    }
    const std::size_t bytes = source.bytes();
    if (bytes == 0 || ec.tp == 1) { return; }
#ifndef NDEBUG
    require_resident_on(source.data, ec.dev[0]->device, "broadcast_rank0: source is not on rank 0");
    for (int rank = 1; rank < ec.tp; ++rank) {
        require_resident_on(destination[rank].data, ec.dev[rank]->device,
                            "broadcast_rank0: destination[r] is not on rank r");
    }
#endif
    const CurrentDeviceGuard guard;
    const DeviceContext& origin = *ec.dev[0];
    CurrentDeviceGuard::set(origin.device);
    CUDA_CHECK(cudaEventRecord(events.inputs_ready(0), origin.stream));

    for (int rank = 1; rank < ec.tp; ++rank) {
        const DeviceContext& peer = *ec.dev[rank];
        CurrentDeviceGuard::set(peer.device);
        CUDA_CHECK(cudaStreamWaitEvent(peer.stream, events.inputs_ready(0), 0));
        CUDA_CHECK(pull_peer(destination[rank].data, source.data, bytes, peer.stream));
        CUDA_CHECK(cudaEventRecord(events.pull_done(rank), peer.stream));
    }

    CurrentDeviceGuard::set(origin.device);
    for (int rank = 1; rank < ec.tp; ++rank) {
        CUDA_CHECK(cudaStreamWaitEvent(origin.stream, events.pull_done(rank), 0));
    }
}

} // namespace ninfer::ops
