#include <ninfer/targets/qwen3_6/vision.h>

#include "artifact/materializer.h"
#include "artifact/typed_binding.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_6 {

VisionBackbonePlan bind_vision_backbone(artifact::Binder& binder,
                                        artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };

    VisionBackbonePlan out;
    // The Vision linear matrices are stored as native FP16 (a cast from the BF16 source) so the
    // projections run on the Volta FP16 tensor cores (`ops::linear` routes FP16_CTRL to the fused
    // mma.sync route) instead of a lossy Q-format. Biases, norms, and the position embedding stay
    // BF16: they are elementwise, not GEMM operands.
    out.patch_embedding = bind("vision/patch_embedding", NumericFormat::FP16,
                                {VisionBackboneConfig::hidden, VisionBackboneConfig::patch_dim});
    out.patch_embedding_bias =
        bind("vision/patch_embedding_bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
    out.position_embedding =
        bind("vision/position_embedding", NumericFormat::BF16,
             {VisionBackboneConfig::position_embeddings, VisionBackboneConfig::hidden});

    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        VisionLayerPlan& target  = out.layers[layer];
        const std::string prefix = "vision/layers/" + std::to_string(layer) + "/";
        target.qkv               = bind(prefix + "attention/qkv", NumericFormat::FP16,
                                        {3 * VisionBackboneConfig::hidden, VisionBackboneConfig::hidden});
        target.qkv_bias          = bind(prefix + "attention/qkv_bias", NumericFormat::BF16,
                                        {3 * VisionBackboneConfig::hidden});
        target.output            = bind(prefix + "attention/output", NumericFormat::FP16,
                                        {VisionBackboneConfig::hidden, VisionBackboneConfig::hidden});
        target.output_bias       = bind(prefix + "attention/output_bias", NumericFormat::BF16,
                                        {VisionBackboneConfig::hidden});
        target.fc1               = bind(prefix + "mlp/fc1", NumericFormat::FP16,
                                        {VisionBackboneConfig::intermediate, VisionBackboneConfig::hidden});
        target.fc1_bias          = bind(prefix + "mlp/fc1_bias", NumericFormat::BF16,
                                        {VisionBackboneConfig::intermediate});
        target.fc2               = bind(prefix + "mlp/fc2", NumericFormat::FP16,
                                        {VisionBackboneConfig::hidden, VisionBackboneConfig::intermediate});
        target.fc2_bias =
            bind(prefix + "mlp/fc2_bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_weight =
            bind(prefix + "norm1/weight", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_bias =
            bind(prefix + "norm1/bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_weight =
            bind(prefix + "norm2/weight", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_bias =
            bind(prefix + "norm2/bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
    }
    return out;
}

VisionMergerInputPlan bind_vision_merger_input(artifact::Binder& binder,
                                               artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    return VisionMergerInputPlan{
        .fc1      = bind("vision/merger/fc1", NumericFormat::FP16,
                         {VisionBackboneConfig::merger_hidden, VisionBackboneConfig::merger_hidden}),
        .fc1_bias = bind("vision/merger/fc1_bias", NumericFormat::BF16,
                         {VisionBackboneConfig::merger_hidden}),
    };
}

VisionMergerNormPlan bind_vision_merger_norm(artifact::Binder& binder,
                                             artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    return VisionMergerNormPlan{
        .weight =
            bind("vision/merger/norm/weight", NumericFormat::BF16, {VisionBackboneConfig::hidden}),
        .bias =
            bind("vision/merger/norm/bias", NumericFormat::BF16, {VisionBackboneConfig::hidden}),
    };
}

VisionCommonWeights materialize_vision_common(const artifact::MaterializedArtifact& materialized,
                                              const VisionBackbonePlan& backbone,
                                              const VisionMergerInputPlan& merger_input,
                                              const VisionMergerNormPlan& merger_norm,
                                              int device, int tp) {
    using artifact::NumericFormat;
    using V = VisionBackboneConfig;

    // Per-device extents of the sharded matrices. At tp == 1 every one of these is the whole
    // matrix, so this reduces to the single-device path. qkv is split by head (even: each plane
    // holds hidden/tp rows); fc1/fc2 and merger_fc1/fc2 share the mod-8 per-device extent so a
    // row-parallel contraction width matches its column-parallel producer on every rank.
    const std::int32_t qkv_rows     = 3 * V::hidden / tp;
    const std::int32_t o_proj_cols  = static_cast<std::int32_t>(vision_mod8_shard_count(V::hidden, tp, device));
    const std::int32_t fc1_rows     = static_cast<std::int32_t>(vision_mod8_shard_count(V::intermediate, tp, device));
    const std::int32_t fc2_cols     = fc1_rows; // fc2's contraction must equal fc1's output width.
    const std::int32_t merger_fc1_rows = static_cast<std::int32_t>(vision_mod8_shard_count(V::merger_hidden, tp, device));

    VisionCommonWeights out;
    // Replicated (full copy on every rank): patch_embedding (decision 8), position embedding, and
    // every bias/norm. Only the `device` argument moves.
    out.patch_embedding = artifact::materialized_weight(
        materialized, backbone.patch_embedding, NumericFormat::FP16,
        V::hidden, V::patch_dim, device);
    out.patch_embedding_bias =
        artifact::materialized_tensor(materialized, backbone.patch_embedding_bias,
                                      NumericFormat::BF16, {V::hidden}, device);
    out.position_embedding = artifact::materialized_tensor(
        materialized, backbone.position_embedding, NumericFormat::BF16,
        {V::hidden, V::position_embeddings}, device);

    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        const VisionLayerPlan& source = backbone.layers[layer];
        VisionLayerWeights& target    = out.layers[layer];
target.qkv = artifact::materialized_weight(
            materialized, source.qkv, NumericFormat::FP16, qkv_rows, V::hidden, device);
        // qkv_bias is column-parallel (N) -- the same head split as the qkv weight -- so each rank
        // materializes only its q|k|v head-block bias, matching the sharded qkv output width.
        target.qkv_bias = artifact::materialized_tensor(materialized, source.qkv_bias,
                                        NumericFormat::BF16, {qkv_rows}, device);
        target.output = artifact::materialized_weight(
            materialized, source.output, NumericFormat::FP16, V::hidden, o_proj_cols, device);
        target.output_bias = artifact::materialized_tensor(materialized, source.output_bias,
                                           NumericFormat::BF16, {V::hidden}, device);
        target.fc1 = artifact::materialized_weight(
            materialized, source.fc1, NumericFormat::FP16, fc1_rows, V::hidden, device);
        // fc1_bias is column-parallel (N), split exactly like the fc1 weight (mod-8 intermediate).
        target.fc1_bias =
            artifact::materialized_tensor(materialized, source.fc1_bias, NumericFormat::BF16,
                                          {fc1_rows}, device);
        target.fc2 = artifact::materialized_weight(
            materialized, source.fc2, NumericFormat::FP16, V::hidden, fc2_cols, device);
        target.fc2_bias = artifact::materialized_tensor(
            materialized, source.fc2_bias, NumericFormat::BF16, {V::hidden}, device);
        target.norm1_weight = artifact::materialized_tensor(
            materialized, source.norm1_weight, NumericFormat::BF16, {V::hidden}, device);
        target.norm1_bias = artifact::materialized_tensor(
            materialized, source.norm1_bias, NumericFormat::BF16, {V::hidden}, device);
        target.norm2_weight = artifact::materialized_tensor(
            materialized, source.norm2_weight, NumericFormat::BF16, {V::hidden}, device);
        target.norm2_bias = artifact::materialized_tensor(
            materialized, source.norm2_bias, NumericFormat::BF16, {V::hidden}, device);
    }

    out.merger_fc1 = artifact::materialized_weight(
        materialized, merger_input.fc1, NumericFormat::FP16,
        merger_fc1_rows, V::merger_hidden, device);
    out.merger_fc1_bias =
        artifact::materialized_tensor(materialized, merger_input.fc1_bias, NumericFormat::BF16,
                                      {merger_fc1_rows}, device);
    out.merger_norm_weight = artifact::materialized_tensor(
        materialized, merger_norm.weight, NumericFormat::BF16, {V::hidden}, device);
    out.merger_norm_bias = artifact::materialized_tensor(
        materialized, merger_norm.bias, NumericFormat::BF16, {V::hidden}, device);
    return out;
}

} // namespace ninfer::targets::qwen3_6
