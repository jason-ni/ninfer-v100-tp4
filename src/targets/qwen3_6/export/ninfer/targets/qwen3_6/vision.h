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

// Starting position of `device`'s mod-8 shard (the sum of the lower devices' shards).
[[nodiscard]] constexpr std::uint64_t vision_mod8_shard_begin(std::uint64_t total, int tp,
                                                               int device) {
    std::uint64_t begin = 0;
    for (int rank = 0; rank < device; ++rank) { begin += vision_mod8_shard_count(total, tp, rank); }
    return begin;
}

// ---------------------------------------------------------------------------
// Published Vision weight recipes.
//
// Every elementwise tensor inside the tower (all biases, all norms, the position embedding, the
// patch-embedding bias) is BF16 in every published artifact, so a recipe names only the storage
// format of the seven GEMM matrices. `kVisionFp16` is the native-FP16 cast published alongside
// the NVFP4 text towers; `kVisionRecipeCoder390Quantized` is the Q4/Q5/Q6/W8 set published by
// the 27B Coder390 efficient-think package. A WeightsProfile maps to exactly one recipe; the
// mapping is publisher evidence, not a target-side choice.
struct VisionRecipe {
    artifact::NumericFormat patch_embedding = artifact::NumericFormat::FP16;
    artifact::NumericFormat qkv             = artifact::NumericFormat::FP16;
    artifact::NumericFormat output          = artifact::NumericFormat::FP16;
    artifact::NumericFormat fc1             = artifact::NumericFormat::FP16;
    artifact::NumericFormat fc2             = artifact::NumericFormat::FP16;
    artifact::NumericFormat merger_fc1      = artifact::NumericFormat::FP16;
    artifact::NumericFormat merger_fc2      = artifact::NumericFormat::FP16;

    [[nodiscard]] constexpr bool is_all_fp16() const noexcept {
        using F = artifact::NumericFormat;
        return patch_embedding == F::FP16 && qkv == F::FP16 && output == F::FP16 && fc1 == F::FP16 &&
               fc2 == F::FP16 && merger_fc1 == F::FP16 && merger_fc2 == F::FP16;
    }
};

inline constexpr VisionRecipe kVisionFp16{};
inline constexpr VisionRecipe kVisionRecipeCoder390Quantized{
    .patch_embedding = artifact::NumericFormat::Q6G64_F16S,
    .qkv             = artifact::NumericFormat::Q4G64_F16S,
    .output          = artifact::NumericFormat::Q5G64_F16S,
    .fc1             = artifact::NumericFormat::Q4G64_F16S,
    .fc2            = artifact::NumericFormat::Q5G64_F16S,
    .merger_fc1      = artifact::NumericFormat::W8G32_F16S,
    .merger_fc2      = artifact::NumericFormat::W8G32_F16S,
};

// K-axis quantization-group width of the row-split-k128-v1 matrix formats (0 for dense formats).
// Column splits (the row-parallel joins) must land on these boundaries; `std::logic_error` names
// any other format.
[[nodiscard]] std::uint32_t vision_matrix_group_size(artifact::NumericFormat format);

namespace detail {

// Splits [0, logical_total) into `tp` contiguous shards whose PADDED spans cover
// align_up(logical_total, unit): the unit counts are distributed across ranks (leading ranks
// absorb the remainder), so every shard boundary lands on a unit multiple. A shard's LOGICAL slice
// is its intersection with [0, logical_total); where the padded span overshoots the logical end
// the shard owns fewer logical channels than its padded width and reports the difference as
// `tail`. The producer zeroes exactly those channels after its bias add, so the consumer's
// contraction over the padded width stays exact despite the weight object's pad-lane codes and
// scales. Preconditions: unit >= 1, tp >= 1, rank in [0, tp) (checked by the caller); at the
// supported shapes every shard owns at least one logical channel.
constexpr void vision_coupled_partition(std::uint64_t logical_total, std::uint64_t unit,
                                        int tp, int rank, std::uint64_t& begin,
                                        std::uint64_t& count, std::uint64_t& width,
                                        std::uint64_t& tail) noexcept {
    const std::uint64_t padded      = (logical_total + unit - 1) / unit * unit;
    const std::uint64_t r           = static_cast<std::uint64_t>(rank);
    const std::uint64_t base_units  = padded / unit / static_cast<std::uint64_t>(tp);
    const std::uint64_t rem_units   = padded / unit % static_cast<std::uint64_t>(tp);
    width                           = (base_units + (r < rem_units ? 1 : 0)) * unit;
    const std::uint64_t pref        = r * base_units * unit + (r < rem_units ? r : rem_units) * unit;
    begin                           = pref < logical_total ? pref : logical_total;
    const std::uint64_t owned       = pref >= logical_total ? 0 : (pref + width > logical_total
                                                                       ? logical_total - pref
                                                                       : width);
    count = owned;
    tail  = width - owned;
}

} // namespace detail

// One column-parallel producer (an N-split weight whose bias splits identically) coupled to its
// row-parallel consumer (a K-split weight contracting over the same index space):
//
//   * n_begin/n_count   -- the producer WEIGHT's row slice [n_begin, n_begin + n_count), and its
//                          N-bias vector's split extents (dense vectors: any boundary legal);
//   * producer_width    -- the padded width the producer materializes (the mlp_up / merger
//                          activation allocation) and the consumer weight declares as its K;
//   * k_begin           -- the consumer weight's column slice [k_begin, k_begin + producer_width)
//                          on the parent's K axis (both bounds unit-aligned per the layouts);
//   * tail              -- zero channels [n_count, producer_width) of the producer (last rank
//                          only when the logical width is not a unit multiple).
struct CoupledShard {
    std::int32_t n_begin       = 0;
    std::int32_t n_count       = 0;
    std::int32_t producer_width = 0;
    std::int32_t k_begin       = 0;
    std::int32_t tail          = 0;
};

// Per-rank widths of every sharded Vision matrix under `recipe`. tp == 1 reproduces the full
// single-device tower. The qkv planes (and their bias) split by head under every recipe
// (3 * hidden / tp rows); the coupled pairs split the PADDED intermediate/merger spans on the
// consumer's group size (group 8 for the FP16 recipe, which reduces to vision_mod8_shard_count
// exactly) or -- when the attention-output width is not a group multiple (Q-format at tp == 4) --
// keep the o_proj REPLICATED on every rank behind an input allgather (o_proj_replicated).
struct VisionShardExtents {
    int tp                  = 1;
    int rank                = 0;
    bool o_proj_replicated  = false;
    std::int32_t qkv_rows   = 0;
    std::int32_t o_proj_k_begin = 0;
    std::int32_t o_proj_cols    = 0;
    CoupledShard mlp{};
    CoupledShard merger{};
};

[[nodiscard]] VisionShardExtents vision_shard_extents(const VisionRecipe& recipe, int tp, int rank);

// ---------------------------------------------------------------------------
// Host-staged (streaming) Vision banks.
//
// A `VisionStageRank` describes ONE rank's pinned-host staging block: its 333 slots in canonical
// order, each naming the encoded shard's offset/byte extent plus the descriptor dims the views
// re-derive at upload time. The 27B target places every Vision shard HostStaging (see
// TensorPlacement) and builds one bank per rank at load; the encode path uploads the whole block
// to a per-encode workspace region and calls `fill_staged_vision_weights` to point an owned
// VisionWeights there. Nothing here allocates device memory; the bank is plain value data.
enum class VisionStageRole : std::uint8_t {
    // Common (slots 0..2).
    PatchEmbedding        = 0,
    PatchEmbeddingBias    = 1,
    PositionEmbedding     = 2,
    // Per-layer (slot 3 + layer * 12 + offset), fixed in-order set.
    Qkv                   = 3,
    QkvBias,
    AttentionOutput,
    AttentionOutputBias,
    MlpFc1,
    MlpFc1Bias,
    MlpFc2,
    MlpFc2Bias,
    Norm1Weight,
    Norm1Bias,
    Norm2Weight,
    Norm2Bias,
    // Merger (slots 3 + layers * 12 + offset), fixed in-order set.
    MergerFc1,
    MergerFc1Bias,
    MergerFc2,
    MergerFc2Bias,
    MergerNormWeight,
    MergerNormBias,
};

inline constexpr int kVisionStageCommonSlots = 3;
inline constexpr int kVisionStageLayerSlots  = 12;
inline constexpr int kVisionStageMergerSlots = 6;
inline constexpr std::size_t kVisionStageSlots =
    kVisionStageCommonSlots + static_cast<std::size_t>(VisionBackboneConfig::layers) *
                                  kVisionStageLayerSlots + kVisionStageMergerSlots;
static_assert(kVisionStageSlots == 333);

[[nodiscard]] std::size_t vision_stage_slot_of(VisionStageRole role, std::int32_t layer = 0);

// One slot of a rank's staged block. `dim0`/`dim1` describe the view the filler rebuilds:
// matrices [rows, columns], dense vectors [extent] with dim1 == 0.
struct VisionStageSlot {
    VisionStageRole role = VisionStageRole::PatchEmbedding;
    std::int32_t layer   = 0;
    artifact::NumericFormat format = artifact::NumericFormat::BF16;
    bool matrix                    = false;
    std::int32_t dim0 = 0;
    std::int32_t dim1 = 0;
    // Byte offset inside this rank's staged region and the encoded shard byte count. Offsets are
    // assigned by the binder (256-aligned packing); the gap bytes between slots are zero at load.
    std::uint32_t offset = 0;
    std::uint32_t bytes  = 0;
};

struct VisionStageRank {
    int tp   = 1;
    int rank = 0;
    // The recipe the bank was built from (identifies the slot formats and the shard widths used).
    VisionRecipe recipe{};
    // Full staged-block byte count for this rank (planned host capacity == aligned slot sum), and
    // the base of the rank's PinnedHostBuffer staging region (owned by the MaterializedArtifact,
    // which outlives the model view this bank rides in). The encode path uploads
    // [host_source, host_source + region_bytes) to a workspace region and rebases the views there.
    std::size_t region_bytes = 0;
    const std::byte* host_source = nullptr;
    VisionShardExtents extents{};
    std::array<VisionStageSlot, kVisionStageSlots> slots{};
};

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

// Binds the backbone (patch embedding, position embedding, every layer's attention/MLP norms and
// projections) at the recipe's matrix formats; biases and norms bind BF16 regardless.
[[nodiscard]] VisionBackbonePlan bind_vision_backbone(artifact::Binder& binder,
                                                      artifact::TensorPlacement placement,
                                                      const VisionRecipe& recipe);
// merger fc1 follows the recipe; its bias binds BF16.
[[nodiscard]] VisionMergerInputPlan bind_vision_merger_input(artifact::Binder& binder,
                                                             artifact::TensorPlacement placement,
                                                             const VisionRecipe& recipe);
[[nodiscard]] VisionMergerNormPlan bind_vision_merger_norm(artifact::Binder& binder,
                                                            artifact::TensorPlacement placement);

// Materializes `device`'s own copy of the shared (replicated) vision weights plus the per-device
// SHARD of each sharded matrix. At tp == 1 every extent is the whole matrix, so this is identical
// to the single-device path; at tp > 1 the sharded extents come from `vision_shard_extents`
// (the FP16 recipe reduces to the historical even head split / mod-8 widths).
[[nodiscard]] VisionCommonWeights materialize_vision_common(
    const artifact::MaterializedArtifact& materialized, const VisionBackbonePlan& backbone,
    const VisionMergerInputPlan& merger_input, const VisionMergerNormPlan& merger_norm,
    int device, int tp, const VisionRecipe& recipe);

// Builds `device`'s stage bank from an already-bound-and-materialized artifact: every Vision
// shard must carry a HostStaging placement, and each placed byte extent must equal the layout
// encoding of the recipe's declared extents (a mismatch throws before any encode could read a
// wrong-width plane). `out_hidden` is the target's text hidden width (the merger fc2 row count,
// which the family config does not own); the merger fc2 matrix and its bias are bound outside the
// three plan structs and arrive as explicit handles.
[[nodiscard]] VisionStageRank materialize_vision_stage_rank(
    const artifact::MaterializedArtifact& materialized, const VisionBackbonePlan& backbone,
    const VisionMergerInputPlan& merger_input, artifact::ObjectHandle merger_fc2,
    artifact::ObjectHandle merger_fc2_bias, const VisionMergerNormPlan& merger_norm,
    const VisionRecipe& recipe, std::int32_t out_hidden, int device, int tp);

// Points every view of an owned `out` VisionWeights at this rank's uploaded copy: payload/base
// pointers become `base` + slot.offset and the geometry re-derives per slot. `base == nullptr`
// yields the descriptor-only form (null payload pointers) a model view holds between encodes.
void fill_staged_vision_weights(const VisionStageRank& rank, const void* base, VisionWeights& out);

// Pure geometry: this rank's staged block byte count for `recipe` (aligned slot sum), computed
// without any artifact. Used by the pre-model workspace capacity planning.
[[nodiscard]] std::size_t vision_stage_rank_bytes(const VisionRecipe& recipe, std::int32_t out_hidden,
                                                  int tp, int rank);

} // namespace ninfer::targets::qwen3_6
