#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include <ninfer/ops/allreduce.h>
#include <ninfer/targets/qwen3_6/vision_control.h>
#include "runtime/contract/transient_region.h"
#include "targets/qwen3_6/impl/runtime/vision_prefill.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

struct VisionItemView {
    std::span<const std::uint16_t> patches;
    const qwen3_6::VisionItemControl* control = nullptr;
};

struct VisionScheduleConfig {
    static constexpr int layers              = VisionConfig::layers;
    static constexpr int hidden              = VisionConfig::hidden;
    static constexpr int intermediate        = VisionConfig::intermediate;
    static constexpr int out_hidden          = VisionConfig::output_hidden;
    static constexpr int heads               = VisionConfig::heads;
    static constexpr int head_dim            = VisionConfig::head_dim;
    static constexpr int patch_dim           = VisionConfig::patch_dim;
    static constexpr int merge_unit          = VisionConfig::merge_unit;
    static constexpr int merger_hidden       = VisionConfig::merger_hidden;
    static constexpr int position_embeddings = VisionConfig::position_embeddings;
    static constexpr int rotary_dim          = VisionConfig::rotary_dim;
    static constexpr float rope_theta        = VisionConfig::rope_theta;
    static constexpr float norm_eps          = VisionConfig::norm_epsilon;
};

class VisionContext {
public:
    // Single-device construction (tp == 1). The vision tower runs entirely on `device`; the
    // sharded members below stay empty and `encode` uses the one-arena overload.
    VisionContext(DeviceContext& device, const LoadedModelData& model);

    // Tensor-parallel construction (tp > 1). `ec` carries every rank's DeviceContext and the group
    // width; `events` is the shared allreduce collective; `peer_weights[r]` is rank r's own sharded
    // model view (index 0 is unused -- rank 0 binds from `model`). Rank 0's device is `ec.dev[0]`,
    // which must be the same object as `device`.
    VisionContext(DeviceContext& device, const LoadedModelData& model, const ExecutionContext& ec,
                  const ops::CollectiveComms& events,
                  const std::array<const LoadedModelData*, kMaximumDevices>& peer_weights);

    [[nodiscard]] static std::size_t output_transient_bytes(std::size_t merged_tokens);
    // Per-rank scratch the item needs. `tp` selects the shard width the encode will run at; the
    // default is the single-device value.
    [[nodiscard]] static std::size_t workspace_bytes(const qwen3_6::VisionItemControl& item,
                                                     std::size_t tp = 1);
    // Per-rank scratch capacity for the widest supported vision item. `tp` sizes for a sharded
    // encode (max over ranks, since tp == 4's mod-8 split is uneven); tp == 1 is the single-device
    // value and is unchanged from before this parameter existed.
    [[nodiscard]] static std::size_t workspace_capacity_bytes(std::uint32_t max_merged_tokens,
                                                               std::uint32_t max_segments,
                                                               std::size_t tp = 1);
    // Single-device encode (tp == 1).
    void encode(const VisionItemView& item, Tensor& output, WorkspaceArena& workspace) const;
    // Sharded encode (tp > 1). `out[r]` is rank r's [H,V] BF16 destination (the replicated result
    // after the final row-parallel reduce), resident on ec.dev[r]; `work[r]` is rank r's scratch
    // arena. Rank 0's destination is normally the caller's transient; peers use private buffers so
    // each rank can scatter its own local copy without a cross-device copy. All ranks are driven
    // from the same host thread (the collective orders them).
    void encode(const VisionItemView& item, const std::array<Tensor, kMaximumDevices>& out,
                const std::array<WorkspaceArena*, kMaximumDevices>& work) const;

    [[nodiscard]] int tp_width() const noexcept { return tp_count_; }
    [[nodiscard]] bool tensor_parallel() const noexcept { return tp_count_ > 1; }

private:
    struct BlockW {
        const Tensor* norm1_weight    = nullptr;
        const Tensor* norm1_bias      = nullptr;
        const Weight* qkv             = nullptr;
        const Tensor* qkv_bias        = nullptr;
        const Weight* projection      = nullptr;
        const Tensor* projection_bias = nullptr;
        const Tensor* norm2_weight    = nullptr;
        const Tensor* norm2_bias      = nullptr;
        const Weight* fc1             = nullptr;
        const Tensor* fc1_bias        = nullptr;
        const Weight* fc2             = nullptr;
        const Tensor* fc2_bias        = nullptr;
    };

    struct MergerW {
        const Tensor* norm_weight = nullptr;
        const Tensor* norm_bias   = nullptr;
        const Weight* fc1         = nullptr;
        const Tensor* fc1_bias    = nullptr;
        const Weight* fc2         = nullptr;
        const Tensor* fc2_bias    = nullptr;
    };

    // One rank's full set of vision weight bindings. For tp == 1 only index 0 is populated and
    // holds the tower at full width; for tp > 1 index r holds rank r's sharded view (qkv/o_proj/
    // fc2/merger_fc2 row-parallel, fc1/merger_fc1 column-parallel, everything else replicated).
    struct VisionWeights {
        const Weight* patch_embed      = nullptr;
        const Tensor* patch_embed_bias = nullptr;
        const Tensor* position_embed   = nullptr;
        std::array<BlockW, VisionScheduleConfig::layers> blocks{};
        MergerW merger{};
    };

    void bind_rank(int rank, const LoadedModelData& model);
    [[nodiscard]] const VisionWeights& weights_for(int rank) const {
        return ranks_[static_cast<std::size_t>(rank)];
    }

    DeviceContext& ctx_;
    std::array<VisionWeights, kMaximumDevices> ranks_{};
    // Tensor-parallel execution context (empty for tp == 1). `ec` owns every rank's DeviceContext;
    // `events` is the shared allreduce collective reused by every row-parallel projection.
    const ExecutionContext* ec_     = nullptr;
    const ops::CollectiveComms* events_ = nullptr;
    int tp_count_                    = 1;
};

struct VisionChunk {
    std::int32_t length                       = 0;
    const qwen3_6::VisionItemControl* control = nullptr;
    Tensor embeddings;
};

// Tensor-parallel setup for a VisionPrefillSession. `ec` owns every rank's DeviceContext and the
// group width; `events` is the shared allreduce collective; `peer_models[r]` is rank r's sharded
// model view (index 0 unused -- rank 0 binds from the session's primary model). Null means the
// single-device path.
struct VisionTpSetup {
    const ExecutionContext* ec = nullptr;
    const ops::CollectiveComms* events = nullptr;
    std::array<const LoadedModelData*, kMaximumDevices> peer_models{};
    // Per-rank preallocated scratch arena (rank 0 is the caller's `workspace`; peers are their
    // Program's own text workspace). The sharded encode suballocates from these instead of owning a
    // per-request device allocation, so a native vision request never touches cudaMalloc.
    std::array<WorkspaceArena*, kMaximumDevices> workspaces{};
    // Per-rank PEER destination for the replicated final [out_hidden, merged] embeddings (index 0
    // unused: rank 0 writes the caller's transient). Persists across chunks for the MTP bridge, so
    // it must outlive every workspace reset.
    std::array<DeviceSpan, kMaximumDevices> embed_destinations{};
};

class VisionPrefillSession {
public:
    VisionPrefillSession(DeviceContext& device, const LoadedModelData& model,
                         WorkspaceArena& workspace, qwen3_6::PreparedPromptData& prompt,
                         const VisionPrefillPlan& plan, runtime::TransientRegion transient,
                         const VisionTpSetup* tp = nullptr);

    [[nodiscard]] VisionChunk prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length);
    void release_encoded_media_payloads() noexcept;
    [[nodiscard]] double elapsed_seconds() const;
    [[nodiscard]] int tp_width() const noexcept { return tp_count_; }
    [[nodiscard]] bool tensor_parallel() const noexcept { return tp_count_ > 1; }
    // Per-rank [H,V] BF16 views of the last encoded item's (replicated) embeddings, valid after a
    // prepare_chunk that encoded an item. Rank 0 is the caller's transient; peers are this session's
    // private buffers, so a tensor-parallel text prefill scatters each rank's local copy without a
    // cross-device transfer. Only meaningful when tensor_parallel().
    [[nodiscard]] const std::array<Tensor, kMaximumDevices>& tp_embeddings() const noexcept {
        return tp_embeddings_;
    }

private:
    DeviceContext& device_;
    WorkspaceArena& workspace_;
    qwen3_6::PreparedPromptData& prompt_;
    const VisionPrefillPlan& plan_;
    runtime::TransientRegion transient_;
    // Optional because the single-device and tensor-parallel constructors differ in arity; exactly
    // one is emplaced at construction.
    std::optional<VisionContext> context_;
    std::optional<std::uint32_t> active_item_;
    std::vector<std::uint32_t> encoded_payloads_pending_release_;
    std::vector<CudaEventTimer> timers_;
    // Tensor-parallel state (empty for tp == 1). The scratch arenas and peer embedding
    // destinations are preallocated by the Program and handed in through VisionTpSetup; the session
    // only suballocates from them, so a vision request performs no cudaMalloc of its own.
    int tp_count_ = 1;
    std::array<WorkspaceArena*, kMaximumDevices> tp_workspaces_{};
    std::array<DeviceSpan, kMaximumDevices> tp_embed_spans_{};
    std::array<Tensor, kMaximumDevices> tp_embeddings_{};
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
