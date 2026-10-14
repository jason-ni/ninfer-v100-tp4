#include <ninfer/targets/qwen3_6/vision.h>

#include "artifact/materializer.h"
#include "artifact/typed_binding.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_6 {

namespace {

using artifact::NumericFormat;

QType qtype_of(NumericFormat format) {
    switch (format) {
    case NumericFormat::FP16: return QType::FP16_CTRL;
    case NumericFormat::Q4G64_F16S: return QType::Q4G64_F16S;
    case NumericFormat::Q5G64_F16S: return QType::Q5G64_F16S;
    case NumericFormat::Q6G64_F16S: return QType::Q6G64_F16S;
    case NumericFormat::W8G32_F16S: return QType::W8G32_F16S;
    }
    throw std::logic_error("vision role format has no Weight descriptor");
}

// Null-safe Weight descriptor for a staged slot: the same field set artifact's materialized path
// fills, with every payload-relative pointer dropped when `base` is null (the between-encodes
// form held by the model view). Vision matrices never carry NVFP4/GGML divisors.
Weight weight_view(const void* base, NumericFormat format, std::int32_t rows, std::int32_t columns) {
    if (rows <= 0 || columns <= 0) { throw std::invalid_argument("vision weight view has an empty shape"); }
    Weight out{};
    out.n             = rows;
    out.k             = columns;
    out.ndim          = 2;
    out.shape[0]      = rows;
    out.shape[1]      = columns;
    const auto* bytes = static_cast<const std::byte*>(base);
    if (format == NumericFormat::FP16) {
        out.payload         = base;
        out.qdata           = base;
        out.payload_bytes   = static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(columns) * 2;
        out.qtype           = QType::FP16_CTRL;
        out.layout          = QuantLayout::Contiguous;
        out.padded_shape[0] = rows;
        out.padded_shape[1] = columns;
        return out;
    }
    const std::array<std::uint64_t, 2> matrix_shape = {
        static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(columns)};
    const artifact::RowSplitGeometry geometry = artifact::row_split_geometry(format, matrix_shape);
    out.payload          = bytes;
    out.payload_bytes    = geometry.encoded_bytes;
    out.high_plane_bytes = geometry.high_plane_bytes;
    out.qtype            = qtype_of(format);
    out.layout           = QuantLayout::RowSplit;
    out.group_size       = static_cast<std::uint32_t>(geometry.group_size);
    out.qdata            = base;
    out.qhigh            = (geometry.high_plane_bytes == 0 || base == nullptr)
                               ? nullptr
                               : static_cast<const void*>(bytes + geometry.high_plane_offset);
    out.scales           = base == nullptr
                               ? nullptr
                               : static_cast<const void*>(bytes + geometry.scale_plane_offset);
    out.group            = static_cast<std::int32_t>(geometry.group_size);
    out.scale_dtype      = DType::FP16;
    out.padded_shape[0]  = rows;
    out.padded_shape[1]  = static_cast<std::int32_t>(geometry.padded_columns);
    return out;
}

// Null-safe dense (BF16) view. `dim1 == 0` marks a 1-D vector.
Tensor dense_view(const void* base, std::int32_t dim0, std::int32_t dim1) {
    if (dim1 == 0) {
        return Tensor(const_cast<void*>(base), DType::BF16, {dim0});
    }
    return Tensor(const_cast<void*>(base), DType::BF16, {dim0, dim1});
}

struct SlotDims {
    bool matrix               = false;
    NumericFormat format      = NumericFormat::BF16;
    std::int32_t dim0         = 0;
    std::int32_t dim1         = 0;   // columns for matrices; 0 for vectors
};

// The view extents of every stage slot, in one table shared by the bank builder, the capacity
// planner, and the view filler. Matrix rows/columns are the SHARDED extents the device views see;
// the position embedding carries the runtime's internal [hidden, position_embeddings] orientation
// (the artifact stores the transposed table).
SlotDims slot_dims(VisionStageRole role, const VisionShardExtents& ext, const VisionRecipe& recipe,
                   std::int32_t out_hidden) {
    using V = VisionBackboneConfig;
    switch (role) {
    case VisionStageRole::PatchEmbedding:
        return {true, recipe.patch_embedding, V::hidden, V::patch_dim};
    case VisionStageRole::PatchEmbeddingBias: return {false, NumericFormat::BF16, V::hidden, 0};
    case VisionStageRole::PositionEmbedding:
        return {false, NumericFormat::BF16, V::hidden, V::position_embeddings};
    case VisionStageRole::Qkv: return {true, recipe.qkv, ext.qkv_rows, V::hidden};
    case VisionStageRole::QkvBias: return {false, NumericFormat::BF16, ext.qkv_rows, 0};
    case VisionStageRole::AttentionOutput:
        return {true, recipe.output, V::hidden, ext.o_proj_cols};
    case VisionStageRole::AttentionOutputBias:
        return {false, NumericFormat::BF16, V::hidden, 0};
    case VisionStageRole::MlpFc1: return {true, recipe.fc1, ext.mlp.n_count, V::hidden};
    case VisionStageRole::MlpFc1Bias:
        return {false, NumericFormat::BF16, ext.mlp.n_count, 0};
    case VisionStageRole::MlpFc2:
        return {true, recipe.fc2, V::hidden, ext.mlp.producer_width};
    case VisionStageRole::MlpFc2Bias: return {false, NumericFormat::BF16, V::hidden, 0};
    case VisionStageRole::Norm1Weight:
    case VisionStageRole::Norm1Bias:
    case VisionStageRole::Norm2Weight:
    case VisionStageRole::Norm2Bias:
        return {false, NumericFormat::BF16, V::hidden, 0};
    case VisionStageRole::MergerFc1:
        return {true, recipe.merger_fc1, ext.merger.n_count, V::merger_hidden};
    case VisionStageRole::MergerFc1Bias:
        return {false, NumericFormat::BF16, ext.merger.n_count, 0};
    case VisionStageRole::MergerFc2:
        return {true, recipe.merger_fc2, out_hidden, ext.merger.producer_width};
    case VisionStageRole::MergerFc2Bias:
        return {false, NumericFormat::BF16, out_hidden, 0};
    case VisionStageRole::MergerNormWeight:
    case VisionStageRole::MergerNormBias:
        return {false, NumericFormat::BF16, V::hidden, 0};
    }
    throw std::logic_error("vision stage slot has no descriptor");
}

std::uint64_t slot_element_bytes(SlotDims dims) {
    const std::uint64_t elements =
        static_cast<std::uint64_t>(dims.dim0) * (dims.dim1 == 0 ? 1 : static_cast<std::uint64_t>(dims.dim1));
    return elements * 2;
}

std::uint64_t slot_encoded_bytes(SlotDims dims) {
    if (!dims.matrix) { return slot_element_bytes(dims); }   // BF16 elementwise slot
    if (dims.format == NumericFormat::FP16) { return slot_element_bytes(dims); }
    const std::array<std::uint64_t, 2> matrix_shape = {
        static_cast<std::uint64_t>(dims.dim0), static_cast<std::uint64_t>(dims.dim1)};
    return artifact::row_split_geometry(dims.format, matrix_shape).encoded_bytes;
}

artifact::StorageLayout storage_of(SlotDims dims) {
    return (!dims.matrix || dims.format == NumericFormat::FP16)
               ? artifact::StorageLayout::ContiguousLeV1
               : artifact::StorageLayout::RowSplitK128V1;
}

// Walks the 333 canonical slots in order; the callback receives (role, layer).
template <class Visit>
void for_each_stage_slot(Visit&& visit) {
    visit(VisionStageRole::PatchEmbedding, 0);
    visit(VisionStageRole::PatchEmbeddingBias, 0);
    visit(VisionStageRole::PositionEmbedding, 0);
    for (std::int32_t layer = 0; layer < VisionBackboneConfig::layers; ++layer) {
        visit(VisionStageRole::Qkv, layer);
        visit(VisionStageRole::QkvBias, layer);
        visit(VisionStageRole::AttentionOutput, layer);
        visit(VisionStageRole::AttentionOutputBias, layer);
        visit(VisionStageRole::MlpFc1, layer);
        visit(VisionStageRole::MlpFc1Bias, layer);
        visit(VisionStageRole::MlpFc2, layer);
        visit(VisionStageRole::MlpFc2Bias, layer);
        visit(VisionStageRole::Norm1Weight, layer);
        visit(VisionStageRole::Norm1Bias, layer);
        visit(VisionStageRole::Norm2Weight, layer);
        visit(VisionStageRole::Norm2Bias, layer);
    }
    visit(VisionStageRole::MergerFc1, 0);
    visit(VisionStageRole::MergerFc1Bias, 0);
    visit(VisionStageRole::MergerFc2, 0);
    visit(VisionStageRole::MergerFc2Bias, 0);
    visit(VisionStageRole::MergerNormWeight, 0);
    visit(VisionStageRole::MergerNormBias, 0);
}

} // namespace

std::uint32_t vision_matrix_group_size(NumericFormat format) {
    switch (format) {
    case NumericFormat::Q4G64_F16S:
    case NumericFormat::Q5G64_F16S:
    case NumericFormat::Q6G64_F16S: return 64;
    case NumericFormat::W8G32_F16S: return 32;
    case NumericFormat::FP16:
    case NumericFormat::BF16: return 0;
    default: throw std::logic_error("vision_matrix_group_size: no group width for that format");
    }
}

std::size_t vision_stage_slot_of(VisionStageRole role, std::int32_t layer) {
    const int r = static_cast<int>(role);
    // Role numbers 0..2 are the common slots, 3..14 the per-layer set, 15..20 the merger set.
    // The layer band occupies kVisionStageLayerSlots slots per backbone layer, so the merger set
    // begins after all of them; `layer_role_end` is the ROLE number where the merger set starts.
    const int layer_role_end = kVisionStageCommonSlots + kVisionStageLayerSlots;
    if (r < kVisionStageCommonSlots) {
        if (layer != 0) { return 0; }   // common slots ignore the layer parameter
        return static_cast<std::size_t>(r);
    }
    if (r < layer_role_end) {
        if (layer < 0 || layer >= VisionBackboneConfig::layers) {
            throw std::out_of_range("vision stage slot: layer index is outside the backbone");
        }
        return static_cast<std::size_t>(
            kVisionStageCommonSlots + static_cast<std::uint64_t>(layer) * kVisionStageLayerSlots +
            (r - kVisionStageCommonSlots));
    }
    const std::size_t merger_slot_base =
        kVisionStageCommonSlots + static_cast<std::size_t>(VisionBackboneConfig::layers) *
                                      kVisionStageLayerSlots;
    if (r < layer_role_end + kVisionStageMergerSlots && layer == 0) {
        return merger_slot_base + static_cast<std::size_t>(r - layer_role_end);
    }
    throw std::out_of_range("vision stage slot: unknown role");
}

VisionShardExtents vision_shard_extents(const VisionRecipe& recipe, int tp, int rank) {
    using V = VisionBackboneConfig;
    if (tp < 1 || tp > V::heads || V::heads % tp != 0 || rank < 0 || rank >= tp) {
        throw std::invalid_argument("vision shard extents: invalid tensor-parallel width or rank");
    }
    VisionShardExtents out;
    out.tp       = tp;
    out.rank     = rank;
    out.qkv_rows = 3 * V::hidden / tp;   // head-aligned under every recipe (three planes)

    const auto couple = [&](std::uint64_t total, std::uint64_t unit, CoupledShard& shard) {
        std::uint64_t begin = 0, count = 0, width = 0, tail = 0;
        detail::vision_coupled_partition(total, unit, tp, rank, begin, count, width, tail);
        if (count == 0 || width < count) {
            throw std::logic_error("vision coupled split left an empty logical slice");
        }
        shard.n_begin        = static_cast<std::int32_t>(begin);
        shard.n_count        = static_cast<std::int32_t>(count);
        shard.producer_width = static_cast<std::int32_t>(width);
        // The K slice starts where the padded prefix begins; at the supported shapes the prefix
        // never reaches the logical end (every shard owns logical channels), so it equals begin.
        shard.k_begin = shard.n_begin;
        shard.tail    = static_cast<std::int32_t>(tail);
        (void)shard;
    };

    if (recipe.is_all_fp16()) {
        // The native-FP16 GEMM routes only need an 8-aligned contraction, so the unit-8 split
        // applies; at every supported tp it reproduces the historical mod-8 widths exactly.
        couple(V::intermediate, 8, out.mlp);
        couple(V::merger_hidden, 8, out.merger);
        out.o_proj_k_begin = static_cast<std::int32_t>(vision_mod8_shard_begin(V::hidden, tp, rank));
        out.o_proj_cols    = static_cast<std::int32_t>(vision_mod8_shard_count(V::hidden, tp, rank));
        return out;
    }

    // Quantized recipe: the coupled pairs split the PADDED spans on the K-sliced consumer's
    // group width so both slice bounds land on the group boundaries the layouts admit.
    const std::uint32_t o_group = vision_matrix_group_size(recipe.output);
    const bool splittable       = o_group > 0 && (V::hidden % tp) == 0 &&
                                  ((V::hidden / tp) % static_cast<int>(o_group)) == 0;
    if (splittable) {
        out.o_proj_k_begin = rank * (V::hidden / tp);
        out.o_proj_cols    = V::hidden / tp;
    } else {
        // The per-rank contraction width is not a group multiple: every rank takes the FULL
        // o_proj behind an allgather of the attention output (o_proj_replicated).
        out.o_proj_replicated = tp > 1;
        out.o_proj_cols       = V::hidden;
    }
    couple(V::intermediate, vision_matrix_group_size(recipe.fc2), out.mlp);
    couple(V::merger_hidden, vision_matrix_group_size(recipe.merger_fc2), out.merger);
    return out;
}

VisionBackbonePlan bind_vision_backbone(artifact::Binder& binder,
                                        artifact::TensorPlacement placement,
                                        const VisionRecipe& recipe) {
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };

    VisionBackbonePlan out;
    // The GEMM matrices store at the recipe's published formats (native FP16 for the text-nvfp4
    // towers, Q4/Q5/Q6/W8 for the Coder390 package); biases, norms, and the position embedding
    // stay BF16 in every published artifact -- they are elementwise, not GEMM operands.
    out.patch_embedding =
        bind("vision/patch_embedding", recipe.patch_embedding,
             {VisionBackboneConfig::hidden, VisionBackboneConfig::patch_dim});
    out.patch_embedding_bias =
        bind("vision/patch_embedding_bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
    out.position_embedding =
        bind("vision/position_embedding", NumericFormat::BF16,
             {VisionBackboneConfig::position_embeddings, VisionBackboneConfig::hidden});

    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        VisionLayerPlan& target  = out.layers[layer];
        const std::string prefix = "vision/layers/" + std::to_string(layer) + "/";
        target.qkv               = bind(prefix + "attention/qkv", recipe.qkv,
                                         {3 * VisionBackboneConfig::hidden, VisionBackboneConfig::hidden});
        target.qkv_bias          = bind(prefix + "attention/qkv_bias", NumericFormat::BF16,
                                         {3 * VisionBackboneConfig::hidden});
        target.output            = bind(prefix + "attention/output", recipe.output,
                                         {VisionBackboneConfig::hidden, VisionBackboneConfig::hidden});
        target.output_bias       = bind(prefix + "attention/output_bias", NumericFormat::BF16,
                                         {VisionBackboneConfig::hidden});
        target.fc1               = bind(prefix + "mlp/fc1", recipe.fc1,
                                         {VisionBackboneConfig::intermediate, VisionBackboneConfig::hidden});
        target.fc1_bias          = bind(prefix + "mlp/fc1_bias", NumericFormat::BF16,
                                         {VisionBackboneConfig::intermediate});
        target.fc2               = bind(prefix + "mlp/fc2", recipe.fc2,
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
                                               artifact::TensorPlacement placement,
                                               const VisionRecipe& recipe) {
    using V = VisionBackboneConfig;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    return VisionMergerInputPlan{
        .fc1      = bind("vision/merger/fc1", recipe.merger_fc1,
                         {V::merger_hidden, V::merger_hidden}),
        .fc1_bias = bind("vision/merger/fc1_bias", NumericFormat::BF16, {V::merger_hidden}),
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
        .weight = bind("vision/merger/norm/weight", NumericFormat::BF16,
                       {VisionBackboneConfig::hidden}),
        .bias = bind("vision/merger/norm/bias", NumericFormat::BF16,
                     {VisionBackboneConfig::hidden}),
    };
}

VisionCommonWeights materialize_vision_common(const artifact::MaterializedArtifact& materialized,
                                              const VisionBackbonePlan& backbone,
                                              const VisionMergerInputPlan& merger_input,
                                              const VisionMergerNormPlan& merger_norm,
                                              int device, int tp, const VisionRecipe& recipe) {
    using V = VisionBackboneConfig;
    // Per-device extents of the sharded matrices, single source of truth shared with the target's
    // shard map (`vision_shard_extents`); the FP16 recipe reproduces the historical even head
    // split / mod-8 widths exactly.
    const VisionShardExtents ext = vision_shard_extents(recipe, tp, device);

    VisionCommonWeights out;
    // Replicated (full copy on every rank): patch_embedding, position embedding, and every
    // bias/norm. Only the `device` argument moves.
    out.patch_embedding = artifact::materialized_weight(
        materialized, backbone.patch_embedding, recipe.patch_embedding,
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
            materialized, source.qkv, recipe.qkv, ext.qkv_rows, V::hidden, device);
        // qkv_bias is column-parallel (N) -- the same head split as the qkv weight -- so each rank
        // materializes only its q|k|v head-block bias, matching the sharded qkv output width.
        target.qkv_bias = artifact::materialized_tensor(materialized, source.qkv_bias,
                                    NumericFormat::BF16, {ext.qkv_rows}, device);
        target.output = artifact::materialized_weight(
            materialized, source.output, recipe.output, V::hidden, ext.o_proj_cols, device);
        target.output_bias = artifact::materialized_tensor(materialized, source.output_bias,
                                           NumericFormat::BF16, {V::hidden}, device);
        target.fc1 = artifact::materialized_weight(
            materialized, source.fc1, recipe.fc1, ext.mlp.n_count, V::hidden, device);
        // fc1_bias is column-parallel (N), split exactly like the fc1 weight.
        target.fc1_bias =
            artifact::materialized_tensor(materialized, source.fc1_bias, NumericFormat::BF16,
                                          {ext.mlp.n_count}, device);
        target.fc2 = artifact::materialized_weight(
            materialized, source.fc2, recipe.fc2, V::hidden, ext.mlp.producer_width, device);
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
        materialized, merger_input.fc1, recipe.merger_fc1, ext.merger.n_count,
        V::merger_hidden, device);
    out.merger_fc1_bias =
        artifact::materialized_tensor(materialized, merger_input.fc1_bias, NumericFormat::BF16,
                                      {ext.merger.n_count}, device);
    out.merger_norm_weight = artifact::materialized_tensor(
        materialized, merger_norm.weight, NumericFormat::BF16, {V::hidden}, device);
    out.merger_norm_bias = artifact::materialized_tensor(
        materialized, merger_norm.bias, NumericFormat::BF16, {V::hidden}, device);
    return out;
}

VisionStageRank materialize_vision_stage_rank(const artifact::MaterializedArtifact& materialized,
                                              const VisionBackbonePlan& backbone,
                                              const VisionMergerInputPlan& merger_input,
                                              artifact::ObjectHandle merger_fc2,
                                              artifact::ObjectHandle merger_fc2_bias,
                                              const VisionMergerNormPlan& merger_norm,
                                              const VisionRecipe& recipe, std::int32_t out_hidden,
                                              int device, int tp) {
    VisionStageRank out;
    out.tp              = tp;
    out.rank            = device;
    out.recipe          = recipe;
    out.extents         = vision_shard_extents(recipe, tp, device);
    const std::span<const std::byte> region = materialized.host_staging(device);
    out.region_bytes = region.size();
    out.host_source  = region.data();

    auto register_slot = [&](VisionStageRole role, int layer, artifact::ObjectHandle handle) {
        const SlotDims dims = slot_dims(role, out.extents, recipe, out_hidden);
        const std::uint64_t expected = slot_encoded_bytes(dims);
        const auto index             = vision_stage_slot_of(role, layer);
        VisionStageSlot& slot        = out.slots[index];
        slot.role    = role;
        slot.layer   = layer;
        slot.format  = dims.format;
        slot.matrix  = dims.matrix;
        slot.dim0    = dims.dim0;
        slot.dim1    = dims.dim1;
        const std::byte* data = materialized.host_staging_data(handle, device);
        const std::uint64_t placed = materialized.host_staging_bytes(handle, device);
        const std::uint64_t offset = static_cast<std::uint64_t>(data - region.data());
        if (placed != expected) {
            throw std::runtime_error("vision staged shard holds " + std::to_string(placed) +
                                     " bytes for device " + std::to_string(device) + ", but "
                                     "the recipe layout expects " + std::to_string(expected) +
                                     " bytes (slot role " + std::to_string((int)role) +
                                     " layer " + std::to_string(layer) + ")");
        }
        if (offset + placed > region.size() ||
            offset > std::numeric_limits<std::uint32_t>::max() ||
            placed > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("vision staged shard escapes its staging region");
        }
        slot.offset = static_cast<std::uint32_t>(offset);
        slot.bytes  = static_cast<std::uint32_t>(placed);
    };

    register_slot(VisionStageRole::PatchEmbedding, 0, backbone.patch_embedding);
    register_slot(VisionStageRole::PatchEmbeddingBias, 0, backbone.patch_embedding_bias);
    register_slot(VisionStageRole::PositionEmbedding, 0, backbone.position_embedding);
    for (int layer = 0; layer < VisionBackboneConfig::layers; ++layer) {
        const VisionLayerPlan& l = backbone.layers[static_cast<std::size_t>(layer)];
        register_slot(VisionStageRole::Qkv, layer, l.qkv);
        register_slot(VisionStageRole::QkvBias, layer, l.qkv_bias);
        register_slot(VisionStageRole::AttentionOutput, layer, l.output);
        register_slot(VisionStageRole::AttentionOutputBias, layer, l.output_bias);
        register_slot(VisionStageRole::MlpFc1, layer, l.fc1);
        register_slot(VisionStageRole::MlpFc1Bias, layer, l.fc1_bias);
        register_slot(VisionStageRole::MlpFc2, layer, l.fc2);
        register_slot(VisionStageRole::MlpFc2Bias, layer, l.fc2_bias);
        register_slot(VisionStageRole::Norm1Weight, layer, l.norm1_weight);
        register_slot(VisionStageRole::Norm1Bias, layer, l.norm1_bias);
        register_slot(VisionStageRole::Norm2Weight, layer, l.norm2_weight);
        register_slot(VisionStageRole::Norm2Bias, layer, l.norm2_bias);
    }
    register_slot(VisionStageRole::MergerFc1, 0, merger_input.fc1);
    register_slot(VisionStageRole::MergerFc1Bias, 0, merger_input.fc1_bias);
    register_slot(VisionStageRole::MergerFc2, 0, merger_fc2);
    register_slot(VisionStageRole::MergerFc2Bias, 0, merger_fc2_bias);
    register_slot(VisionStageRole::MergerNormWeight, 0, merger_norm.weight);
    register_slot(VisionStageRole::MergerNormBias, 0, merger_norm.bias);

    // The bank and the pre-model capacity planner must agree on the block size byte-for-byte; a
    // drift means the packing assumption (uniform 256-byte slot alignment) moved underneath one
    // of them.
    const std::size_t planned = vision_stage_rank_bytes(recipe, out_hidden, tp, device);
    if (planned != out.region_bytes) {
        throw std::runtime_error("vision staging block size disagrees between plan-time geometry "
                                 "(" + std::to_string(planned) + " bytes) and the materialized "
                                 "region (" + std::to_string(out.region_bytes) + " bytes)");
    }
    return out;
}

void fill_staged_vision_weights(const VisionStageRank& rank, const void* base, VisionWeights& out) {
    const auto slot = [&](VisionStageRole role, int layer = 0) -> const VisionStageSlot& {
        return rank.slots[vision_stage_slot_of(role, layer)];
    };
    const auto point_at = [&](const VisionStageSlot& s) -> const void* {
        return base == nullptr
                   ? nullptr
                   : static_cast<const std::byte*>(base) + s.offset;
    };
    const auto fill_matrix = [&](VisionStageRole role, int layer, Weight& target) {
        const VisionStageSlot& s = slot(role, layer);
        target = weight_view(point_at(s), s.format, s.dim0, s.dim1);
    };
    const auto fill_vector = [&](VisionStageRole role, int layer, Tensor& target) {
        const VisionStageSlot& s = slot(role, layer);
        target = dense_view(point_at(s), s.dim0, s.dim1);
    };

    fill_matrix(VisionStageRole::PatchEmbedding, 0, out.common.patch_embedding);
    fill_vector(VisionStageRole::PatchEmbeddingBias, 0, out.common.patch_embedding_bias);
    fill_vector(VisionStageRole::PositionEmbedding, 0, out.common.position_embedding);
    for (int layer = 0; layer < VisionBackboneConfig::layers; ++layer) {
        VisionLayerWeights& target = out.common.layers[static_cast<std::size_t>(layer)];
        fill_matrix(VisionStageRole::Qkv, layer, target.qkv);
        fill_vector(VisionStageRole::QkvBias, layer, target.qkv_bias);
        fill_matrix(VisionStageRole::AttentionOutput, layer, target.output);
        fill_vector(VisionStageRole::AttentionOutputBias, layer, target.output_bias);
        fill_matrix(VisionStageRole::MlpFc1, layer, target.fc1);
        fill_vector(VisionStageRole::MlpFc1Bias, layer, target.fc1_bias);
        fill_matrix(VisionStageRole::MlpFc2, layer, target.fc2);
        fill_vector(VisionStageRole::MlpFc2Bias, layer, target.fc2_bias);
        fill_vector(VisionStageRole::Norm1Weight, layer, target.norm1_weight);
        fill_vector(VisionStageRole::Norm1Bias, layer, target.norm1_bias);
        fill_vector(VisionStageRole::Norm2Weight, layer, target.norm2_weight);
        fill_vector(VisionStageRole::Norm2Bias, layer, target.norm2_bias);
    }
    fill_matrix(VisionStageRole::MergerFc1, 0, out.common.merger_fc1);
    fill_vector(VisionStageRole::MergerFc1Bias, 0, out.common.merger_fc1_bias);
    fill_vector(VisionStageRole::MergerNormWeight, 0, out.common.merger_norm_weight);
    fill_vector(VisionStageRole::MergerNormBias, 0, out.common.merger_norm_bias);
    fill_matrix(VisionStageRole::MergerFc2, 0, out.merger_fc2);
    fill_vector(VisionStageRole::MergerFc2Bias, 0, out.merger_fc2_bias);
}

std::size_t vision_stage_rank_bytes(const VisionRecipe& recipe, std::int32_t out_hidden, int tp,
                                     int rank) {
    const VisionShardExtents ext = vision_shard_extents(recipe, tp, rank);
    std::size_t total = 0;
    for_each_stage_slot([&](auto role, auto layer) {
        const SlotDims dims                  = slot_dims(role, ext, recipe, out_hidden);
        const std::uint64_t encoded          = slot_encoded_bytes(dims);
        const std::uint64_t alignment        = artifact::tensor_alignment(storage_of(dims));
        total += static_cast<std::size_t>((encoded + alignment - 1) / alignment * alignment);
    });
    return total;
}

} // namespace ninfer::targets::qwen3_6