#pragma once

#include "artifact/binder.h"
#include "core/tensor.h"

#include <array>
#include <cstddef>

namespace ninfer::artifact {
class MaterializedArtifact;
}

namespace ninfer::targets::qwen3_6 {

struct VisionBackboneConfig {
    static constexpr int layers              = 27;
    static constexpr int hidden              = 1152;
    static constexpr int intermediate        = 4304;
    static constexpr int heads               = 16;
    static constexpr int head_dim            = hidden / heads;
    static constexpr int patch_dim           = 3 * 2 * 16 * 16;
    static constexpr int merge               = 2;
    static constexpr int merge_unit          = merge * merge;
    static constexpr int merger_hidden       = hidden * merge_unit;
    static constexpr int position_embeddings = 48 * 48;
    static constexpr int rotary_dim          = head_dim;
    static constexpr float rope_theta        = 10'000.0F;
    static constexpr float norm_epsilon      = 1.0e-6F;
};

// Per-device extent of a mod-8 split of `total` into `tp` contiguous parts, each a multiple of 8
// with the remainder distributed in whole 8-units (device d gets base + (d < rem ? 1 : 0) units).
// The FP16 GEMM requires its contraction to be a multiple of 8, so a row-parallel K split and its
// matching column-parallel N split both use this. Shared by the 27B TP shard map and the per-device
// materialization so the two can never disagree. Requires total % 8 == 0 and tp >= 1. See
// docs/maintainer/fp16-vision-plan.md §5.4.
[[nodiscard]] constexpr std::uint64_t vision_mod8_shard_count(std::uint64_t total, int tp,
                                                              int device) {
    const std::uint64_t units = total / 8;
    const std::uint64_t base  = units / static_cast<std::uint64_t>(tp);
    const std::uint64_t rem   = units % static_cast<std::uint64_t>(tp);
    return (base + (static_cast<std::uint64_t>(device) < rem ? 1 : 0)) * 8;
}

struct VisionLayerPlan {
    artifact::ObjectHandle qkv;
    artifact::ObjectHandle qkv_bias;
    artifact::ObjectHandle output;
    artifact::ObjectHandle output_bias;
    artifact::ObjectHandle fc1;
    artifact::ObjectHandle fc1_bias;
    artifact::ObjectHandle fc2;
    artifact::ObjectHandle fc2_bias;
    artifact::ObjectHandle norm1_weight;
    artifact::ObjectHandle norm1_bias;
    artifact::ObjectHandle norm2_weight;
    artifact::ObjectHandle norm2_bias;
};

struct VisionBackbonePlan {
    artifact::ObjectHandle patch_embedding;
    artifact::ObjectHandle patch_embedding_bias;
    artifact::ObjectHandle position_embedding;
    std::array<VisionLayerPlan, VisionBackboneConfig::layers> layers;
};

struct VisionMergerInputPlan {
    artifact::ObjectHandle fc1;
    artifact::ObjectHandle fc1_bias;
};

struct VisionMergerNormPlan {
    artifact::ObjectHandle weight;
    artifact::ObjectHandle bias;
};

struct VisionLayerWeights {
    Weight qkv;
    Tensor qkv_bias;
    Weight output;
    Tensor output_bias;
    Weight fc1;
    Tensor fc1_bias;
    Weight fc2;
    Tensor fc2_bias;
    Tensor norm1_weight;
    Tensor norm1_bias;
    Tensor norm2_weight;
    Tensor norm2_bias;
};

struct VisionCommonWeights {
    Weight patch_embedding;
    Tensor patch_embedding_bias;
    Tensor position_embedding;
    std::array<VisionLayerWeights, VisionBackboneConfig::layers> layers;
    Weight merger_fc1;
    Tensor merger_fc1_bias;
    Tensor merger_norm_weight;
    Tensor merger_norm_bias;
};

struct VisionWeights {
    VisionCommonWeights common;
    Weight merger_fc2;
    Tensor merger_fc2_bias;
};

[[nodiscard]] VisionBackbonePlan bind_vision_backbone(artifact::Binder& binder,
                                                      artifact::TensorPlacement placement);
[[nodiscard]] VisionMergerInputPlan bind_vision_merger_input(artifact::Binder& binder,
                                                             artifact::TensorPlacement placement);
[[nodiscard]] VisionMergerNormPlan bind_vision_merger_norm(artifact::Binder& binder,
                                                           artifact::TensorPlacement placement);

// Materializes `device`'s own copy of the shared (replicated) vision weights plus the per-device
// SHARD of each sharded matrix. At tp == 1 every extent is the whole matrix, so this is identical
// to the single-device path; at tp > 1 the sharded extents come from `vision_mod8_shard_count` /
// the even head split, matching the 27B shard map exactly.
[[nodiscard]] VisionCommonWeights materialize_vision_common(
    const artifact::MaterializedArtifact& materialized, const VisionBackbonePlan& backbone,
    const VisionMergerInputPlan& merger_input, const VisionMergerNormPlan& merger_norm,
    int device, int tp);

} // namespace ninfer::targets::qwen3_6
