// ninfer::ops - collective wrappers over NCCL. Implements the public api, validates parameters,
// and issues grouped per-rank NCCL primitives on each rank's own stream. Host-compiled.
// See docs/op-development.md and docs/maintainer/tp4-design.md.
#include "ninfer/ops/allreduce.h"

#include "core/device.h"
#include "core/dtype.h"
#include "ops/common/tp_mailbox.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

[[noreturn]] void fail(const std::string& message) { throw std::invalid_argument(message); }

void require(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

void cuda_check(cudaError_t result, const char* what) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(result));
    }
}

void nccl_check(ncclResult_t result, const char* what) {
    if (result != ncclSuccess) {
        throw std::runtime_error(std::string(what) + ": " + ncclGetErrorString(result));
    }
}

// Moves bytes, so any element type NCCL names is valid. The collective call sites carry BF16
// activations, I32/I64 token and index lanes, and FP32 staging; the mapping is the one place the
// element type is translated, and an unmapped type fails loudly rather than moving wrong bytes.
ncclDataType_t nccl_element_type(DType dtype, const char* op) {
    switch (dtype) {
    case DType::BF16: return ncclBfloat16;
    case DType::FP16: return ncclFloat16;
    case DType::FP32: return ncclFloat32;
    case DType::I8: return ncclInt8;
    case DType::U8: return ncclUint8;
    case DType::I32: return ncclInt32;
    case DType::I64: return ncclInt64;
    case DType::FP8_E4M3FN: return ncclUint8;
    }
    throw std::invalid_argument(std::string(op) + ": unsupported collective element type");
}

// Restores the caller's current device on scope exit while letting each rank bind its own device
// around its NCCL call. Single-threaded engine, but capture-safe: cudaSetDevice is a host call.
class CurrentDeviceGuard {
public:
    CurrentDeviceGuard() { cuda_check(cudaGetDevice(&previous_), "cudaGetDevice"); }
    ~CurrentDeviceGuard() { cudaSetDevice(previous_); }
    CurrentDeviceGuard(const CurrentDeviceGuard&)            = delete;
    CurrentDeviceGuard& operator=(const CurrentDeviceGuard&) = delete;

    static void set(int device) { cuda_check(cudaSetDevice(device), "cudaSetDevice"); }

private:
    int previous_ = 0;
};

void require_tp(const ExecutionContext& ec, const char* op) {
    require(ec.tp >= 1 && ec.tp <= static_cast<int>(kMaximumDevices),
            std::string(op) + ": requires tp in [1, kMaximumDevices]");
    for (int rank = 0; rank < ec.tp; ++rank) {
        require(ec.dev[rank].has_value(),
                std::string(op) + ": device " + std::to_string(rank) + " is not present");
    }
    for (int a = 0; a < ec.tp; ++a) {
        for (int b = a + 1; b < ec.tp; ++b) {
            require(ec.dev[a]->device != ec.dev[b]->device,
                    std::string(op) + ": devices must be distinct");
        }
    }
}

void require_comms(const ExecutionContext& ec, const CollectiveComms& comms, const char* op) {
    require(comms.live(), std::string(op) + ": no live communicators");
    require(comms.count() == ec.tp, std::string(op) + ": communicator count does not match tp");
}

std::size_t element_count(const Tensor& tensor) {
    return tensor.bytes() / dtype_size(tensor.dtype);
}

void require_contiguous(const Tensor& tensor, const char* op, const char* name) {
    require(tensor.data != nullptr, std::string(op) + ": " + name + " is null");
    require(tensor.is_contiguous(), std::string(op) + ": " + name + " must be contiguous");
}

template <int N>
bool same_shape(const Tensor& a, const Tensor& b) {
    for (int i = 0; i < N; ++i) {
        if (a.ne[i] != b.ne[i]) return false;
    }
    return true;
}

} // namespace

CollectiveComms::CollectiveComms(const ExecutionContext& execution) : tp_(execution.tp) {
    require(tp_ >= 1 && tp_ <= static_cast<int>(kMaximumDevices),
            "CollectiveComms: requires tp in [1, kMaximumDevices]");
    for (int rank = 0; rank < tp_; ++rank) {
        require(execution.dev[rank].has_value(),
                "CollectiveComms: device " + std::to_string(rank) + " is not present");
    }
    if (tp_ == 1) return;

    std::array<int, kMaximumDevices> devices{};
    for (int rank = 0; rank < tp_; ++rank) devices[rank] = execution.dev[rank]->device;

    std::array<ncclComm_t, kMaximumDevices> comms{};
    const ncclResult_t init = ncclCommInitAll(comms.data(), tp_, devices.data());
    if (init != ncclSuccess) {
        for (int rank = 0; rank < tp_; ++rank) {
            if (comms[rank] != nullptr) ncclCommDestroy(comms[rank]);
        }
        throw std::runtime_error(std::string("ncclCommInitAll: ") + ncclGetErrorString(init));
    }
    for (int rank = 0; rank < tp_; ++rank) comms_[rank] = comms[rank];
    // The mailbox probe self-checks and races NCCL on a decode-sized message; any failure leaves
    // the NCCL path in charge.
    mailbox_ = detail::TpMailboxPlan::create(execution, comms_);
}

CollectiveComms::~CollectiveComms() {
    for (int rank = 0; rank < tp_; ++rank) {
        if (comms_[rank] != nullptr) {
            ncclCommDestroy(comms_[rank]);
            comms_[rank] = nullptr;
        }
    }
}

CollectiveComms::CollectiveComms(CollectiveComms&& other) noexcept
    : tp_(other.tp_), comms_(other.comms_), mailbox_(std::move(other.mailbox_)) {
    other.tp_ = 0;
    other.comms_.fill(nullptr);
}

CollectiveComms& CollectiveComms::operator=(CollectiveComms&& other) noexcept {
    if (this != &other) {
        for (int rank = 0; rank < tp_; ++rank) {
            if (comms_[rank] != nullptr) ncclCommDestroy(comms_[rank]);
        }
        tp_      = other.tp_;
        comms_   = other.comms_;
        mailbox_ = std::move(other.mailbox_);
        other.tp_ = 0;
        other.comms_.fill(nullptr);
    }
    return *this;
}

bool CollectiveComms::mailbox_allreduce_bf16(const std::array<Tensor, kMaximumDevices>& buffer,
                                             const ExecutionContext& ec) const {
    return mailbox_ != nullptr && mailbox_->allreduce_bf16(buffer, ec);
}

bool CollectiveComms::mailbox_allgather_rows_bf16(
    const std::array<Tensor, kMaximumDevices>& destination,
    const std::array<Tensor, kMaximumDevices>& part, const ExecutionContext& ec) const {
    return mailbox_ != nullptr && mailbox_->allgather_rows_bf16(destination, part, ec);
}

void allreduce_sum(const std::array<Tensor, kMaximumDevices>& buffer,
                   const std::array<Tensor, kMaximumDevices>& staging, const ExecutionContext& ec,
                   const CollectiveComms& comms) {
    static constexpr const char* kOp = "allreduce_sum";
    require_tp(ec, kOp);
    require_comms(ec, comms, kOp);

    const int tp = ec.tp;
    for (int rank = 0; rank < tp; ++rank) {
        require_contiguous(buffer[rank], kOp, "buffer");
        require(buffer[rank].dtype == DType::BF16, std::string(kOp) + ": buffer must be BF16");
        require(same_shape<2>(buffer[rank], buffer[0]),
                std::string(kOp) + ": buffer shapes must match");
        require_contiguous(staging[rank], kOp, "staging");
        require(staging[rank].bytes() >= buffer[rank].bytes(),
                std::string(kOp) + ": staging must hold one rank's buffer");
    }
    if (tp == 1) return;

    const std::size_t elems = element_count(buffer[0]);
    if (elems == 0) return;

    // Small BF16 messages take the probed mailbox transport; everything else stays on NCCL.
    if (comms.mailbox_allreduce_bf16(buffer, ec)) { return; }

    const CurrentDeviceGuard guard;
    nccl_check(ncclGroupStart(), kOp);
    for (int rank = 0; rank < tp; ++rank) {
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        nccl_check(ncclAllReduce(buffer[rank].data, buffer[rank].data, elems, ncclBfloat16,
                                 ncclSum, comms.comm(rank), ec.dev[rank]->stream),
                   kOp);
    }
    nccl_check(ncclGroupEnd(), kOp);
}

void allgather_rows(const std::array<Tensor, kMaximumDevices>& destination,
                    const std::array<Tensor, kMaximumDevices>& part, const ExecutionContext& ec,
                    const CollectiveComms& comms) {
    static constexpr const char* kOp = "allgather_rows";
    require_tp(ec, kOp);
    require_comms(ec, comms, kOp);

    const int tp = ec.tp;
    require(destination[0].data != nullptr && part[0].data != nullptr,
            std::string(kOp) + ": tensors are null");
    const DType elem      = part[0].dtype;
    const std::size_t es  = dtype_size(elem);
    const ncclDataType_t dt = nccl_element_type(elem, kOp);
    const std::int32_t row_length = destination[0].ne[0];
    const std::int32_t total_rows = destination[0].ne[1];
    std::int32_t gathered_rows      = 0;
    std::size_t total_elems         = 0;
    std::array<std::size_t, kMaximumDevices> block_elems{};
    std::array<std::size_t, kMaximumDevices> offset_elems{};
    for (int rank = 0; rank < tp; ++rank) {
        require_contiguous(part[rank], kOp, "part");
        require_contiguous(destination[rank], kOp, "destination");
        require(part[rank].dtype == elem && destination[rank].dtype == elem,
                std::string(kOp) + ": tensors must share one element type");
        require(part[rank].ne[0] == row_length,
                std::string(kOp) + ": part row length must match the destination");
        require(destination[rank].ne[0] == row_length &&
                    destination[rank].ne[1] == total_rows,
                std::string(kOp) + ": destination shape must be uniform across ranks");
        block_elems[rank]  = static_cast<std::size_t>(row_length) *
                            static_cast<std::size_t>(part[rank].ne[1]);
        offset_elems[rank] = total_elems;
        total_elems += block_elems[rank];
        gathered_rows += part[rank].ne[1];
    }
    require(gathered_rows == total_rows,
            std::string(kOp) + ": part row counts must sum to the destination rows");
    if (tp == 1) return;
    if (total_elems == 0) return;

    // The sharded-head gathers are BF16 block relocations inside the mailbox capacity; the
    // generic uneven case and oversized images stay on NCCL.
    if (comms.mailbox_allgather_rows_bf16(destination, part, ec)) { return; }

    const CurrentDeviceGuard guard;
    nccl_check(ncclGroupStart(), kOp);
    for (int rank = 0; rank < tp; ++rank) {
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        // Own block first, then receive every peer's block and send this rank's block to every peer.
        cuda_check(cudaMemcpyAsync(reinterpret_cast<char*>(destination[rank].data) +
                                       offset_elems[rank] * es,
                                   part[rank].data, block_elems[rank] * es,
                                   cudaMemcpyDeviceToDevice, ec.dev[rank]->stream),
                   "cudaMemcpyAsync");
        for (int peer = 0; peer < tp; ++peer) {
            if (peer == rank) continue;
            nccl_check(ncclRecv(reinterpret_cast<char*>(destination[rank].data) +
                                    offset_elems[peer] * es,
                                block_elems[peer], dt, peer, comms.comm(rank),
                                ec.dev[rank]->stream),
                       kOp);
            nccl_check(ncclSend(part[rank].data, block_elems[rank], dt, peer,
                                comms.comm(rank), ec.dev[rank]->stream),
                       kOp);
        }
    }
    nccl_check(ncclGroupEnd(), kOp);
}

void allgather_columns(const std::array<Tensor, kMaximumDevices>& destination,
                       const std::array<Tensor, kMaximumDevices>& part, const ExecutionContext& ec,
                       const CollectiveComms& comms) {
    static constexpr const char* kOp = "allgather_columns";
    require_tp(ec, kOp);
    require_comms(ec, comms, kOp);

    const int tp = ec.tp;
    require(destination[0].data != nullptr && part[0].data != nullptr,
            std::string(kOp) + ": tensors are null");
    const DType elem      = part[0].dtype;
    const std::size_t es  = dtype_size(elem);
    const ncclDataType_t dt         = nccl_element_type(elem, kOp);
    const std::int32_t columns      = destination[0].ne[1];
    const std::int32_t total_width  = destination[0].ne[0];
    std::int32_t gathered_width     = 0;
    std::size_t total_elems         = 0;
    std::array<std::size_t, kMaximumDevices> block_elems{};
    std::array<std::size_t, kMaximumDevices> offset_elems{};
    for (int rank = 0; rank < tp; ++rank) {
        require_contiguous(part[rank], kOp, "part");
        require_contiguous(destination[rank], kOp, "destination");
        require(part[rank].dtype == elem && destination[rank].dtype == elem,
                std::string(kOp) + ": tensors must share one element type");
        require(part[rank].ne[1] == columns,
                std::string(kOp) + ": part columns must match the destination");
        require(destination[rank].ne[0] == total_width && destination[rank].ne[1] == columns,
                std::string(kOp) + ": destination shape must be uniform across ranks");
        block_elems[rank]  = static_cast<std::size_t>(part[rank].ne[0]) *
                            static_cast<std::size_t>(columns);
        offset_elems[rank] = total_elems;
        total_elems += block_elems[rank];
        gathered_width += part[rank].ne[0];
    }
    require(gathered_width == total_width,
            std::string(kOp) + ": part widths must sum to the destination width");
    if (tp == 1) return;
    if (total_elems == 0) return;

    // The gathered axis is the CONTIGUOUS one (ne[0]), so a rank's block is NOT a contiguous span
    // of the destination: it occupies `widths[rank]` leading rows of every column. Move each rank's
    // contiguous [W,T] block into a stream-ordered staging image (own block first, then one NCCL
    // receive per peer, exactly as gather_columns_rank0 does), then place every block with a
    // per-column strided copy. A flat concatenation is only correct when the blocked axis is the
    // strided one (allgather_rows), not here. Vision eager work has no small-message fast path, so
    // this Op always runs the NCCL route.
    const CurrentDeviceGuard guard;
    std::array<void*, kMaximumDevices> staging{};
    for (int rank = 0; rank < tp; ++rank) {
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        cuda_check(cudaMallocAsync(&staging[rank], total_elems * es, ec.dev[rank]->stream),
                   "cudaMallocAsync");
    }

    nccl_check(ncclGroupStart(), kOp);
    for (int rank = 0; rank < tp; ++rank) {
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        cuda_check(cudaMemcpyAsync(static_cast<char*>(staging[rank]) + offset_elems[rank] * es,
                                   part[rank].data, block_elems[rank] * es,
                                   cudaMemcpyDeviceToDevice, ec.dev[rank]->stream),
                   "cudaMemcpyAsync");
        for (int peer = 0; peer < tp; ++peer) {
            if (peer == rank) continue;
            nccl_check(ncclRecv(static_cast<char*>(staging[rank]) + offset_elems[peer] * es,
                                block_elems[peer], dt, peer, comms.comm(rank),
                                ec.dev[rank]->stream),
                       kOp);
            nccl_check(ncclSend(part[rank].data, block_elems[rank], dt, peer,
                                comms.comm(rank), ec.dev[rank]->stream),
                       kOp);
        }
    }
    nccl_check(ncclGroupEnd(), kOp);

    const std::size_t destination_pitch = static_cast<std::size_t>(total_width) * es;
    for (int rank = 0; rank < tp; ++rank) {
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        std::size_t destination_bytes = 0;
        for (int source = 0; source < tp; ++source) {
            const std::size_t width_bytes = static_cast<std::size_t>(part[source].ne[0]) * es;
            if (width_bytes != 0) {
                cuda_check(
                    cudaMemcpy2DAsync(
                        reinterpret_cast<char*>(destination[rank].data) + destination_bytes,
                        destination_pitch,
                        static_cast<char*>(staging[rank]) + offset_elems[source] * es, width_bytes,
                        width_bytes, static_cast<std::size_t>(columns), cudaMemcpyDeviceToDevice,
                        ec.dev[rank]->stream),
                    "cudaMemcpy2DAsync");
            }
            destination_bytes += width_bytes;
        }
    }
    for (int rank = 0; rank < tp; ++rank) {
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        cuda_check(cudaFreeAsync(staging[rank], ec.dev[rank]->stream), "cudaFreeAsync");
    }
}

void gather_columns_rank0(const Tensor& destination,
                          const std::array<Tensor, kMaximumDevices>& part,
                          const ExecutionContext& ec, const CollectiveComms& comms) {
    static constexpr const char* kOp = "gather_columns_rank0";
    require_tp(ec, kOp);
    require_comms(ec, comms, kOp);

    const int tp = ec.tp;
    require_contiguous(destination, kOp, "destination");
    const DType elem             = destination.dtype;
    const std::size_t es         = dtype_size(elem);
    const ncclDataType_t dt      = nccl_element_type(elem, kOp);
    const std::int32_t full_width = destination.ne[0];
    const std::int32_t columns    = destination.ne[1];

    std::array<std::int32_t, kMaximumDevices> widths{};
    std::array<std::size_t, kMaximumDevices> prefix{};
    std::int32_t owned = 0;
    std::size_t total_elems = 0;
    for (int rank = 0; rank < tp; ++rank) {
        require_contiguous(part[rank], kOp, "part");
        require(part[rank].dtype == elem,
                std::string(kOp) + ": part must share the destination element type");
        require(part[rank].ne[1] == columns, std::string(kOp) + ": part columns must match");
        widths[rank] = part[rank].ne[0];
        prefix[rank] = total_elems;
        owned += widths[rank];
        total_elems += static_cast<std::size_t>(widths[rank]) * static_cast<std::size_t>(columns);
    }
    require(owned == full_width, std::string(kOp) + ": owned widths must sum to the destination");
    if (tp == 1) return;
    if (total_elems == 0) return;

    const CurrentDeviceGuard guard;

    // Rank 0 receives every peer's contiguous [columns, w] block into a staging image, then applies
    // the same strided per-row placement the previous UVA path used. The staging is stream-ordered
    // so it is valid (and capturable) without host synchronization.
    CurrentDeviceGuard::set(ec.dev[0]->device);
    void* staging = nullptr;
    cuda_check(cudaMallocAsync(&staging, total_elems * es, ec.dev[0]->stream),
               "cudaMallocAsync");

    nccl_check(ncclGroupStart(), kOp);
    CurrentDeviceGuard::set(ec.dev[0]->device);
    for (int rank = 1; rank < tp; ++rank) {
        const std::size_t count =
            static_cast<std::size_t>(widths[rank]) * static_cast<std::size_t>(columns);
        if (count != 0) {
            nccl_check(ncclRecv(static_cast<char*>(staging) + prefix[rank] * es, count,
                                dt, rank, comms.comm(0), ec.dev[0]->stream),
                       kOp);
        }
    }
    cuda_check(cudaMemcpyAsync(staging, part[0].data,
                               static_cast<std::size_t>(widths[0]) *
                                   static_cast<std::size_t>(columns) * es,
                               cudaMemcpyDeviceToDevice, ec.dev[0]->stream),
               "cudaMemcpyAsync");
    for (int rank = 1; rank < tp; ++rank) {
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        const std::size_t count =
            static_cast<std::size_t>(widths[rank]) * static_cast<std::size_t>(columns);
        if (count != 0) {
            nccl_check(ncclSend(part[rank].data, count, dt, 0, comms.comm(rank),
                                ec.dev[rank]->stream),
                       kOp);
        }
    }
    nccl_check(ncclGroupEnd(), kOp);

    CurrentDeviceGuard::set(ec.dev[0]->device);
    std::size_t destination_bytes = 0;
    for (int rank = 0; rank < tp; ++rank) {
        const std::size_t width_bytes = static_cast<std::size_t>(widths[rank]) * es;
        if (width_bytes != 0) {
            cuda_check(cudaMemcpy2DAsync(static_cast<char*>(destination.data) + destination_bytes,
                                         static_cast<std::size_t>(full_width) * es,
                                         static_cast<char*>(staging) + prefix[rank] * es,
                                         width_bytes, width_bytes,
                                         static_cast<std::size_t>(columns),
                                         cudaMemcpyDeviceToDevice, ec.dev[0]->stream),
                       "cudaMemcpy2DAsync");
        }
        destination_bytes += width_bytes;
    }
    cuda_check(cudaFreeAsync(staging, ec.dev[0]->stream), "cudaFreeAsync");
}

void broadcast_rank0(const Tensor& source, const std::array<Tensor, kMaximumDevices>& destination,
                     const ExecutionContext& ec, const CollectiveComms& comms) {
    static constexpr const char* kOp = "broadcast_rank0";
    require_tp(ec, kOp);
    require_comms(ec, comms, kOp);

    const int tp = ec.tp;
    require_contiguous(source, kOp, "source");
    const ncclDataType_t dt = nccl_element_type(source.dtype, kOp);
    for (int rank = 1; rank < tp; ++rank) {
        require_contiguous(destination[rank], kOp, "destination");
        require(destination[rank].dtype == source.dtype,
                std::string(kOp) + ": destination dtype must match the source");
        require(same_shape<2>(destination[rank], source),
                std::string(kOp) + ": destination shape must match the source");
    }
    if (tp == 1) return;

    const std::size_t elems = element_count(source);
    if (elems == 0) return;

    const CurrentDeviceGuard guard;
    nccl_check(ncclGroupStart(), kOp);
    for (int rank = 0; rank < tp; ++rank) {
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        const void* send_buffer   = source.data;
        void* receive_buffer      = (rank == 0) ? source.data : destination[rank].data;
        nccl_check(ncclBroadcast(send_buffer, receive_buffer, elems, dt, 0,
                                 comms.comm(rank), ec.dev[rank]->stream),
                   kOp);
    }
    nccl_check(ncclGroupEnd(), kOp);
}

} // namespace ninfer::ops
