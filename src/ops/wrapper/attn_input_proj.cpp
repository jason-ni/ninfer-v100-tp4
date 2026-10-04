#include "ninfer/ops/attn_input_proj.h"
#include "ops/linear/ggml_k/ggml_k.h"

#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"
#include "ops/attn_input_proj/w8/w8_attn_input_plan.h"
#include "ops/common/split_launch.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t cols, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != cols ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_rowsplit(const Weight& weight, QType qtype, std::int32_t rows, const char* label) {
    const bool q4_planes =
        qtype != QType::Q4G64_F16S || (weight.qhigh == nullptr && weight.high_plane_bytes == 0);
    const bool q5_planes =
        qtype != QType::Q5G64_F16S || (weight.qhigh != nullptr && weight.high_plane_bytes != 0);
    if (weight.qtype != qtype || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 64 || weight.group != 64 ||
        weight.ndim != 2 || weight.n != rows || weight.k != 5120 || weight.shape[0] != rows ||
        weight.shape[1] != 5120 || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != 5120 || !q4_planes || !q5_planes ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 4) ||
        (qtype == QType::Q5G64_F16S && !aligned_to(weight.qhigh, 16))) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_w8_rowsplit(const Weight& weight, std::int32_t rows, const char* label) {
    if (weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 32 || weight.group != 32 ||
        weight.ndim != 2 || weight.n != rows || weight.k != 2048 || weight.shape[0] != rows ||
        weight.shape[1] != 2048 || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != 2048 || weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_bf16_contiguous(const Weight& weight, std::int32_t rows, std::int32_t hidden,
                             const char* label) {
    const std::uint64_t payload_bytes = static_cast<std::uint64_t>(rows) *
                                        static_cast<std::uint64_t>(hidden) * sizeof(std::uint16_t);
    if (weight.qtype != QType::BF16_CTRL || weight.layout != QuantLayout::Contiguous ||
        weight.payload_bytes < payload_bytes || weight.high_plane_bytes != 0 || weight.ndim != 2 ||
        weight.n != rows || weight.k != hidden || weight.shape[0] != rows ||
        weight.shape[1] != hidden || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != hidden || weight.qhigh != nullptr || weight.scales != nullptr ||
        weight.group_size != 0 || weight.group != 0 || !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("attn_input_proj: invalid compute policy");
}

void dispatch_single_parent(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                            Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                            cudaStream_t stream) {
    validate_policy(policy);
    if (weight.qtype == QType::GGML_K) {
        const Tensor outputs[]{q, k, gate, v};
        detail::ggml_k_project_split(x, weight, outputs, 4, false, stream, false, workspace);
        return;
    }
    if (weight.qtype == QType::BF16_CTRL) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("BF16 attn_input_proj admits only A16");
        }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        require_bf16_contiguous(weight, kRows, kHidden, "query/key/gate/value weight");
        detail::bf16_attn_input_dispatch(x, weight, q, gate, k, v, stream);
        return;
    }

    if (weight.qtype == QType::NVFP4) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
            throw std::invalid_argument("NVFP4 attn_input_proj admits only A16 or A4");
        }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        detail::validate_nvfp4_weight(weight, "nvfp4 attn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("nvfp4 attn_input_proj: unsupported weight shape");
        }
        detail::nvfp4_attn_input_dispatch(x, weight, q, gate, k, v, policy, workspace, stream);
        return;
    }

    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
            throw std::invalid_argument("FP8 attn_input_proj admits only A16 or A8");
        }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        detail::validate_fp8_weight(weight, "fp8 attn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("fp8 attn_input_proj: unsupported weight shape");
        }
        detail::fp8_attn_input_dispatch(x, weight, q, gate, k, v, policy, workspace, stream);
        return;
    }

    constexpr std::int32_t kHidden = 2048;
    constexpr std::int32_t kQRows  = 4096;
    constexpr std::int32_t kKvRows = 512;
    constexpr std::int32_t kRows   = 9216;
    const std::int32_t cols        = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
    if (policy != LinearPolicy::A16Only) {
        throw std::invalid_argument("W8 attn_input_proj admits only A16");
    }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(gate, kQRows, cols, "gate");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_w8_rowsplit(weight, kRows, "query/key/gate/value weight");
    detail::w8_attn_input_dispatch(x, weight, q, gate, k, v, stream);
}

} // namespace

std::size_t attn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                                     std::int32_t input_rows, LinearPolicy policy,
                                                     std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("attn_input_proj workspace: invalid token interval");
    }

    switch (parent_qtype) {
    case QType::GGML_K:
        return linear_workspace_capacity_bytes(parent_qtype, parent_rows, input_rows,
                                                policy, min_tokens, max_tokens);
    case QType::BF16_CTRL:
        if (parent_rows != 14336 || input_rows != 5120 || policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported BF16 profile");
        }
        return 0;
    case QType::NVFP4:
        if (parent_rows != detail::Nvfp4AttnInputGeometry::kOutputRows ||
            input_rows != detail::Nvfp4AttnInputGeometry::kInputRows ||
            (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4)) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_attn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    case QType::FP8_E4M3FN_ROW_BF16S:
        if (parent_rows != detail::Fp8AttnInputGeometry::kOutputRows ||
            input_rows != detail::Fp8AttnInputGeometry::kInputRows) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported FP8 profile");
        }
        return detail::fp8_attn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    case QType::W8G32_F16S:
        if (parent_rows != 9216 || input_rows != 2048 || policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported W8 profile");
        }
        (void)detail::w8_attn_input_resolve_plan(
            {input_rows, 4096, 512, parent_rows, input_rows, min_tokens});
        (void)detail::w8_attn_input_resolve_plan(
            {input_rows, 4096, 512, parent_rows, input_rows, max_tokens});
        return 0;
    case QType::Q4G64_F16S:
    case QType::Q5G64_F16S:
    case QType::Q6G64_F16S:
    case QType::FP32_CTRL:
    case QType::I32_CTRL:
        break;
    }
    throw std::invalid_argument("attn_input_proj workspace: unsupported parent qtype");
}

void attn_input_proj(const Tensor& x, const Weight& query_key_weight,
                     const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                     WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    const std::int32_t cols        = x.ne[1];
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(gate, kQRows, cols, "gate");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_rowsplit(query_key_weight, QType::Q4G64_F16S, kQRows + kKvRows, "query/key weight");
    require_rowsplit(gate_value_weight, QType::Q5G64_F16S, kQRows + kKvRows, "gate/value weight");

    detail::q4_q5_attn_input_dispatch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                      workspace, stream);
}

std::size_t q4_q5_attn_input_proj_workspace_capacity_bytes(std::int32_t min_tokens,
                                                            std::int32_t max_tokens) {
    return detail::q4_q5_attn_input_capacity_workspace_bytes(min_tokens, max_tokens);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy,
                     WorkspaceArena& workspace, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_gate_value_weight, q, gate, k, v, policy, &workspace,
                           stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_gate_value_weight, q, gate, k, v, LinearPolicy::A16Only,
                           nullptr, stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_value_weight, Tensor& q, Tensor& k,
                     Tensor& v, cudaStream_t stream) {
    constexpr std::int32_t kHidden = 2048;
    constexpr std::int32_t kQRows  = 4096;
    constexpr std::int32_t kKvRows = 1024;
    constexpr std::int32_t kRows   = 6144;
    const std::int32_t cols        = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_w8_rowsplit(query_key_value_weight, kRows, "query/key/value weight");

    detail::w8_attn_input_dispatch(x, query_key_value_weight, q, k, v, stream);
}

// --- Tensor-parallel split form (tp == 2) -------------------------------------------------------
// See include/ninfer/ops/attn_input_proj.h for the full design note (ShardPlan section layout, the
// head-local output sub-tensor mapping, and which formats are registered).
namespace {

constexpr std::int32_t kShardHidden   = 5120;
constexpr std::int32_t kFullQueryRows = 6144;  // q (and gate) rows over all ranks
constexpr std::int32_t kFullKeyRows   = 1024;  // k (and v) rows over all ranks
constexpr std::int32_t kFullFusedRows = 14336; // q + gate + k + v
constexpr std::int32_t kFullSplitRows = 7168;  // query_key / gate_value rows over all ranks

void validate_fused_column_rank_semantics(const Tensor& x, const Weight& w, const Tensor& q,
                                          const Tensor& gate, const Tensor& k, const Tensor& v,
                                          LinearPolicy policy, const ExecutionContext& ec) {
    validate_policy(policy);
    const std::int32_t cols  = x.ne[1];
    const std::int32_t tp    = ec.tp;
    const std::int32_t query = kFullQueryRows / tp;
    const std::int32_t key   = kFullKeyRows / tp;
    if (cols <= 0) { throw std::invalid_argument("attn_input_proj column-parallel: T must be positive"); }
    require_matrix(x, kShardHidden, cols, "x");
    require_matrix(q, query, cols, "q");
    require_matrix(gate, query, cols, "gate");
    require_matrix(k, key, cols, "k");
    require_matrix(v, key, cols, "v");

    if (w.qtype == QType::GGML_K) {
        (void)linear_workspace_capacity_bytes(w.qtype, w.n, w.k, policy, cols, cols);
    } else if (w.qtype == QType::NVFP4) {
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
            throw std::invalid_argument(
                "attn_input_proj column-parallel: NVFP4 admits only A16 or A4");
        }
        detail::validate_nvfp4_weight(w, "nvfp4 attn_input_proj column-parallel");
    } else if (w.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
            throw std::invalid_argument(
                "attn_input_proj column-parallel: FP8 admits only A16 or A8");
        }
        detail::validate_fp8_weight(w, "fp8 attn_input_proj column-parallel");
    } else {
        throw std::invalid_argument(
            "attn_input_proj column-parallel: unsupported fused weight format");
    }
    if (w.n != kFullFusedRows / tp || w.k != kShardHidden) {
        throw std::invalid_argument(
            "attn_input_proj column-parallel: unsupported weight shard shape");
    }
}

// Cross-rank agreement only a pair can check; every per-rank invariant is validated separately by
// validate_fused_column_rank_semantics. Mirrors linear_swiglu's own validate_swiglu_split_pair
// (src/ops/wrapper/linear_swiglu.cpp).
void validate_fused_split_pair(const std::array<Tensor, kMaximumDevices>& x, const std::array<Weight, kMaximumDevices>& w,
                               const ExecutionContext& ec) {
    detail::require_split_context(
        ec,
        "attn_input_proj column-parallel: requires an ExecutionContext with 2..4 distinct devices");
    for (int rank = 1; rank < ec.tp; ++rank) {
        if (x[rank].ne[1] != x[0].ne[1]) {
            throw std::invalid_argument(
                "attn_input_proj column-parallel: every rank must carry the same token count");
        }
        if (w[rank].qtype != w[0].qtype || w[rank].layout != w[0].layout) {
            throw std::invalid_argument(
                "attn_input_proj column-parallel: every rank must carry the same weight format");
        }
        if (w[rank].k != w[0].k) {
            throw std::invalid_argument(
                "attn_input_proj column-parallel: every rank must consume the same input extent K");
        }
    }
}

void validate_split_storage_column_rank_semantics(const Tensor& x, const Weight& query_key_w,
                                                   const Weight& gate_value_w, const Tensor& q,
                                                   const Tensor& gate, const Tensor& k,
                                                   const Tensor& v, const ExecutionContext& ec) {
    const std::int32_t cols = x.ne[1];
    const std::int32_t tp   = ec.tp;
    if (cols <= 0) { throw std::invalid_argument("attn_input_proj column-parallel: T must be positive"); }
    require_matrix(x, kShardHidden, cols, "x");
    require_matrix(q, kFullQueryRows / tp, cols, "q");
    require_matrix(gate, kFullQueryRows / tp, cols, "gate");
    require_matrix(k, kFullKeyRows / tp, cols, "k");
    require_matrix(v, kFullKeyRows / tp, cols, "v");
    require_rowsplit(query_key_w, QType::Q4G64_F16S, kFullSplitRows / tp, "query/key weight shard");
    require_rowsplit(gate_value_w, QType::Q5G64_F16S, kFullSplitRows / tp, "gate/value weight shard");
}

void validate_split_storage_split_pair(const std::array<Tensor, kMaximumDevices>& x,
                                       const std::array<Weight, kMaximumDevices>& query_key_w,
                                       const std::array<Weight, kMaximumDevices>& gate_value_w,
                                       const ExecutionContext& ec) {
    detail::require_split_context(
        ec,
        "attn_input_proj column-parallel: requires an ExecutionContext with 2..4 distinct devices");
    for (int rank = 1; rank < ec.tp; ++rank) {
        if (x[rank].ne[1] != x[0].ne[1]) {
            throw std::invalid_argument(
                "attn_input_proj column-parallel: every rank must carry the same token count");
        }
        if (query_key_w[rank].qtype != query_key_w[0].qtype ||
            gate_value_w[rank].qtype != gate_value_w[0].qtype) {
            throw std::invalid_argument(
                "attn_input_proj column-parallel: every rank must carry the same weight format");
        }
        if (query_key_w[rank].k != query_key_w[0].k ||
            gate_value_w[rank].k != gate_value_w[0].k) {
            throw std::invalid_argument(
                "attn_input_proj column-parallel: every rank must consume the same input extent K");
        }
    }
}

} // namespace

std::size_t attn_input_proj_column_parallel_workspace_capacity_bytes(QType qtype, LinearPolicy policy,
                                                                      std::int32_t min_tokens,
                                                                      std::int32_t max_tokens) {
    if (qtype == QType::GGML_K) {
        return linear_workspace_capacity_bytes(qtype, kFullFusedRows, kShardHidden,
                                                policy, min_tokens, max_tokens);
    }
    // The W4A4/A8 activation-quantize workspace is a pure function of (tokens, K), and K=5120 is
    // unchanged by the shard (only the output row count N halves) -- the tp1 query is exact here.
    if (qtype == QType::NVFP4) {
        return detail::nvfp4_attn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    if (qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        return detail::fp8_attn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    if (qtype == QType::Q4G64_F16S || qtype == QType::Q5G64_F16S) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("Q4/Q5 attention shard admits only A16");
        }
        return detail::q4_q5_attn_input_shard_capacity_workspace_bytes(min_tokens, max_tokens);
    }
    throw std::invalid_argument(
        "attn_input_proj column-parallel workspace: unsupported weight format");
}

void attn_input_proj_column_parallel(const std::array<Tensor, kMaximumDevices>& x,
                                     const std::array<Weight, kMaximumDevices>& query_key_gate_value_weight,
                                     const std::array<Tensor, kMaximumDevices>& q, const std::array<Tensor, kMaximumDevices>& gate,
                                     const std::array<Tensor, kMaximumDevices>& k, const std::array<Tensor, kMaximumDevices>& v,
                                     LinearPolicy policy,
                                     const std::array<WorkspaceArena*, kMaximumDevices>& workspace,
                                     const ExecutionContext& ec) {
    validate_fused_split_pair(x, query_key_gate_value_weight, ec);
    // Validate both ranks before issuing either, so a rejected pair enqueues nothing.
    for (std::size_t slot = 0; slot < static_cast<std::size_t>(ec.tp); ++slot) {
        validate_fused_column_rank_semantics(x[slot], query_key_gate_value_weight[slot], q[slot],
                                             gate[slot], k[slot], v[slot], policy, ec);
    }
    for (int rank = 0; rank < ec.tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        detail::require_rank_residency(
            ec, rank, x[slot].data, query_key_gate_value_weight[slot].payload, q[slot].data,
            "attn_input_proj column-parallel: every per-rank argument must be resident on "
            "ec.dev[rank]");
    }
    std::array<Tensor, kMaximumDevices> q_dst = q;
    std::array<Tensor, kMaximumDevices> gate_dst = gate;
    std::array<Tensor, kMaximumDevices> k_dst = k;
    std::array<Tensor, kMaximumDevices> v_dst = v;
    detail::for_each_rank(ec, [&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        const Weight& w  = query_key_gate_value_weight[slot];
        if (w.qtype == QType::GGML_K) {
            const Tensor outputs[]{q_dst[slot], k_dst[slot], gate_dst[slot], v_dst[slot]};
            detail::ggml_k_project_split(x[slot], w, outputs, 4, false,
                                         ec.dev[slot]->stream, false, workspace[slot]);
        } else if (w.qtype == QType::NVFP4) {
            detail::nvfp4_attn_input_dispatch_shard(x[slot], w, q_dst[slot], gate_dst[slot],
                                                    k_dst[slot], v_dst[slot], policy,
                                                    workspace[slot], ec.dev[slot]->stream);
        } else {
            detail::fp8_attn_input_dispatch_shard(x[slot], w, q_dst[slot], gate_dst[slot],
                                                  k_dst[slot], v_dst[slot], policy, workspace[slot],
                                                  ec.dev[slot]->stream);
        }
    });
}

void attn_input_proj_column_parallel(const std::array<Tensor, kMaximumDevices>& x,
                                     const std::array<Weight, kMaximumDevices>& query_key_gate_value_weight,
                                     const std::array<Tensor, kMaximumDevices>& q, const std::array<Tensor, kMaximumDevices>& gate,
                                     const std::array<Tensor, kMaximumDevices>& k, const std::array<Tensor, kMaximumDevices>& v,
                                     const ExecutionContext& ec) {
    attn_input_proj_column_parallel(x, query_key_gate_value_weight, q, gate, k, v,
                                    LinearPolicy::A16Only, {nullptr, nullptr}, ec);
}

void attn_input_proj_column_parallel(const std::array<Tensor, kMaximumDevices>& x,
                                     const std::array<Weight, kMaximumDevices>& query_key_weight,
                                     const std::array<Weight, kMaximumDevices>& gate_value_weight,
                                     const std::array<Tensor, kMaximumDevices>& q, const std::array<Tensor, kMaximumDevices>& gate,
                                     const std::array<Tensor, kMaximumDevices>& k, const std::array<Tensor, kMaximumDevices>& v,
                                     const std::array<WorkspaceArena*, kMaximumDevices>& workspace,
                                     const ExecutionContext& ec) {
    validate_split_storage_split_pair(x, query_key_weight, gate_value_weight, ec);
    for (std::size_t slot = 0; slot < static_cast<std::size_t>(ec.tp); ++slot) {
        if (workspace[slot] == nullptr) {
            throw std::invalid_argument("Q4/Q5 attention shard requires a workspace arena");
        }
        validate_split_storage_column_rank_semantics(x[slot], query_key_weight[slot],
                                                      gate_value_weight[slot], q[slot], gate[slot],
                                                      k[slot], v[slot], ec);
    }
    for (int rank = 0; rank < ec.tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        detail::require_rank_residency(
            ec, rank, x[slot].data, query_key_weight[slot].payload, q[slot].data,
            "attn_input_proj column-parallel: every per-rank argument must be resident on "
            "ec.dev[rank]");
        detail::require_rank_residency(
            ec, rank, x[slot].data, gate_value_weight[slot].payload, gate[slot].data,
            "attn_input_proj column-parallel: every per-rank argument must be resident on "
            "ec.dev[rank]");
    }
    std::array<Tensor, kMaximumDevices> q_dst = q;
    std::array<Tensor, kMaximumDevices> gate_dst = gate;
    std::array<Tensor, kMaximumDevices> k_dst = k;
    std::array<Tensor, kMaximumDevices> v_dst = v;
    detail::for_each_rank(ec, [&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        detail::q4_q5_attn_input_dispatch_shard(x[slot], query_key_weight[slot],
                                                gate_value_weight[slot], q_dst[slot], gate_dst[slot],
                                                k_dst[slot], v_dst[slot], *workspace[slot],
                                                ec.dev[slot]->stream);
    });
}

} // namespace ninfer::ops
