#pragma once

#include "targets/qwen3_6_27b/impl/config.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "ninfer/ops/allreduce.h" // ExecutionContext, ops::CollectiveComms (tp2 split forms)
#include <ninfer/targets/qwen3_6/runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::targets::qwen3_6_27b::detail {

using GraphExecutionProfile = qwen3_6::GraphExecutionProfile;

// Compile-time data and the three closed execution leaves supplied to the Qwen3.6 family runtime.
// It owns no request state, execution phase, graph object, or schedule callback.
struct Variant {
    using WeightsProfile                 = detail::WeightsProfile;
    using TextConfig                     = detail::TextConfig;
    using VisionConfig                   = detail::VisionConfig;
    using DFlashConfig                   = detail::DFlashConfig;
    using ModelView                      = detail::RuntimeModelView;
    using FullAttentionProjectionWeights = detail::FullAttentionProjectionPayload;
    using GdnProjectionWeights           = detail::GdnProjectionPayload;
    using PostMixerWeights               = detail::DensePostMixerPayload;
    using MtpAttentionProjectionWeights  = detail::MtpAttentionPayload;
    using MtpPostMixerWeights            = detail::DensePostMixerPayload;
    using VisionWeights                  = qwen3_6::VisionWeights;
    using GraphExecutionProfile          = detail::GraphExecutionProfile;

    static constexpr float attention_scale                     = kAttentionScale;
    static constexpr float gdn_scale                           = kGdnScale;
    static constexpr std::uint32_t prefill_chunk_alignment     = kPrefillChunkAlignment;
    static constexpr std::uint32_t maximum_mtp_draft_tokens    = kMaximumMtpDraftTokens;
    static constexpr std::uint32_t maximum_dflash_draft_tokens = kMaximumDFlashDraftTokens;
    static constexpr std::uint32_t maximum_context             = kNativeContext;
    // This checkpoint's text attention is head_dim 256 with a 64-dim partial rotary
    // subspace (32 frequency pairs), exactly the geometry `detail::yarn_scale()` computes a
    // corrected inverse-frequency table for, and the published qwen3.8 long-context deployment
    // config carries `rope_type: yarn` for it. `--rope yarn` is therefore admissible here.
    static constexpr bool supports_yarn_rope                   = true;
    static constexpr bool supports_dflash                      = DFlashConfig::supported;
    static constexpr std::int32_t draft_head_rows              = 131072;

    static void attention_projection(const Tensor& hidden,
                                     const FullAttentionProjectionWeights& weights, Tensor& query,
                                     Tensor& gate, Tensor& key, Tensor& value,
                                     qwen3_6::TextPhase phase, WorkspaceArena& workspace,
                                     cudaStream_t stream);
    static void attention_output_projection(const Tensor& attention, const Weight& weight,
                                            Tensor& residual, qwen3_6::TextPhase phase,
                                            WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_attention_projection(const Tensor& hidden,
                                         const MtpAttentionProjectionWeights& weights,
                                         Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                                         WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_kv_projection(const Tensor& hidden,
                                  const MtpAttentionProjectionWeights& weights, Tensor& key,
                                  Tensor& value, WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_q_gate_projection(const Tensor& hidden,
                                      const MtpAttentionProjectionWeights& weights, Tensor& query,
                                      Tensor& gate, WorkspaceArena& workspace, cudaStream_t stream);
    static void gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                     Tensor& qkv, Tensor& output_gate, qwen3_6::TextPhase phase,
                                     WorkspaceArena& workspace, cudaStream_t stream);
    static void
    gdn_input_projection_snapshot(const Tensor& hidden, const GdnProjectionWeights& weights,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_slot,
                                  const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& output_gate, qwen3_6::TextPhase phase,
                                  WorkspaceArena& workspace, cudaStream_t stream);
    static void gdn_input_projection_record(
        const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
        const Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slots,
        Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value, Tensor& output_gate,
        qwen3_6::TextPhase phase, WorkspaceArena& workspace, cudaStream_t stream);
    static void gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                      qwen3_6::TextPhase phase, WorkspaceArena& workspace,
                                      cudaStream_t stream);
    static void gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                            float eps, const GdnProjectionWeights& weights,
                                            Tensor& hidden, Tensor& g, Tensor& beta,
                                            WorkspaceArena& workspace, cudaStream_t stream);
    static void post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                           qwen3_6::TextPhase phase, WorkspaceArena& workspace,
                           cudaStream_t stream);
    static void mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                               Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream);
    [[nodiscard]] static std::size_t
    mtp_attention_projection_workspace_capacity_bytes(std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t mtp_kv_projection_workspace_capacity_bytes(std::int32_t first,
                                                                                std::int32_t last);
    [[nodiscard]] static std::size_t
    mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    attention_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                  qwen3_6::TextPhase phase, std::int32_t first,
                                                  std::int32_t last);
    [[nodiscard]] static std::size_t
    attention_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                         qwen3_6::TextPhase phase,
                                                         std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_input_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                  qwen3_6::TextPhase phase, std::int32_t first,
                                                  std::int32_t last);
    [[nodiscard]] static std::size_t gdn_input_projection_snapshot_workspace_capacity_bytes(
        WeightsProfile weights_profile, qwen3_6::TextPhase phase, std::int32_t batch_size,
        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t gdn_input_projection_record_workspace_capacity_bytes(
        WeightsProfile weights_profile, qwen3_6::TextPhase phase, std::int32_t batch_size,
        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                   qwen3_6::TextPhase phase, std::int32_t first,
                                                   std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_norm_control_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                         std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    post_mixer_workspace_capacity_bytes(WeightsProfile weights_profile, qwen3_6::TextPhase phase,
                                        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                                             std::int32_t last);

    // --- tp > 1 split forms --------------------------------------------------------------------
    //
    // One call drives EVERY rank: every argument is an array indexed by rank, holding that rank's
    // own shard-shaped tensor/weight resident on `ec.dev[rank]` and issued on that rank's stream.
    // The three column-parallel leaves need no communication at all; the two row-parallel leaves
    // (`attention_output_projection`, `gdn_output_projection`, and `post_mixer`'s trailing `down`)
    // carry the block's single all-reduce inside `ops::linear_add_row_parallel`, which also folds
    // the residual in exactly once, on rank 0, before the reduce.
    //
    // `staging[r]` is scratch of the residual's shape on `ec.dev[r]`; it receives the peer's
    // partial and its contents afterwards are unspecified.
    static void attention_projection(const std::array<Tensor, kMaximumDevices>& hidden,
                                     const std::array<const FullAttentionProjectionWeights*, kMaximumDevices>& w,
                                     const std::array<Tensor, kMaximumDevices>& query,
                                     const std::array<Tensor, kMaximumDevices>& gate,
                                     const std::array<Tensor, kMaximumDevices>& key,
                                     const std::array<Tensor, kMaximumDevices>& value, qwen3_6::TextPhase phase,
                                     const std::array<WorkspaceArena*, kMaximumDevices>& workspace,
                                     const ExecutionContext& ec);
    static void attention_output_projection(const std::array<Tensor, kMaximumDevices>& attention,
                                            const std::array<Weight, kMaximumDevices>& weight,
                                            const std::array<Tensor, kMaximumDevices>& residual,
                                            const std::array<Tensor, kMaximumDevices>& staging,
                                            qwen3_6::TextPhase phase,
                                            const std::array<WorkspaceArena*, kMaximumDevices>& workspace,
                                            const ExecutionContext& ec, const ops::CollectiveComms& ev);
    static void gdn_input_projection(const std::array<Tensor, kMaximumDevices>& hidden,
                                     const std::array<const GdnProjectionWeights*, kMaximumDevices>& w,
                                     const std::array<Tensor, kMaximumDevices>& qkv,
                                     const std::array<Tensor, kMaximumDevices>& output_gate,
                                     qwen3_6::TextPhase phase,
                                     const std::array<WorkspaceArena*, kMaximumDevices>& workspace,
                                     const ExecutionContext& ec);
    static void gdn_input_projection_snapshot(
        const std::array<Tensor, kMaximumDevices>& hidden,
        const std::array<const GdnProjectionWeights*, kMaximumDevices>& w,
        const std::array<Tensor, kMaximumDevices>& conv_weight, const std::array<Tensor, kMaximumDevices>& conv_states,
        const std::array<Tensor, kMaximumDevices>& valid_columns, const std::array<Tensor, kMaximumDevices>& initial_slot,
        const std::array<Tensor, kMaximumDevices>& snapshot_base_slot, const std::array<Tensor, kMaximumDevices>& query,
        const std::array<Tensor, kMaximumDevices>& key, const std::array<Tensor, kMaximumDevices>& value,
        const std::array<Tensor, kMaximumDevices>& output_gate, qwen3_6::TextPhase phase,
        const std::array<WorkspaceArena*, kMaximumDevices>& workspace, const ExecutionContext& ec);
    static void gdn_input_projection_record(
        const std::array<Tensor, kMaximumDevices>& hidden, const std::array<const GdnProjectionWeights*, kMaximumDevices>& w,
        const std::array<Tensor, kMaximumDevices>& conv_weight, const std::array<Tensor, kMaximumDevices>& conv_states,
        const std::array<Tensor, kMaximumDevices>& valid_columns, const std::array<Tensor, kMaximumDevices>& initial_slots,
        const std::array<Tensor, kMaximumDevices>& conv_record, const std::array<Tensor, kMaximumDevices>& query,
        const std::array<Tensor, kMaximumDevices>& key, const std::array<Tensor, kMaximumDevices>& value,
        const std::array<Tensor, kMaximumDevices>& output_gate, qwen3_6::TextPhase phase,
        const std::array<WorkspaceArena*, kMaximumDevices>& workspace, const ExecutionContext& ec);
    static void gdn_output_projection(const std::array<Tensor, kMaximumDevices>& hidden,
                                      const std::array<Weight, kMaximumDevices>& weight,
                                      const std::array<Tensor, kMaximumDevices>& residual,
                                      const std::array<Tensor, kMaximumDevices>& staging,
                                      qwen3_6::TextPhase phase,
                                      const std::array<WorkspaceArena*, kMaximumDevices>& workspace,
                                      const ExecutionContext& ec, const ops::CollectiveComms& ev);
    // MTP split leaves. `mtp_attention_projection` is column-parallel over the packed
    // [14336, 5120] parent (shard [7168, 5120], whose row order is q | k | gate | v at the
    // per-rank section widths) and then splits each rank's own packed block in place;
    // `mtp_post_mixer` is the MTP layer's swiglu pair, identical in shape to `post_mixer`.
    // `mtp_kv_projection` / `mtp_q_gate_projection` are the prefill-only section subsets, which
    // the tp1 leaf fuses with `linear_pair` and which split into independent column-parallel
    // calls because their two outputs are separate row views of the same shard.
    static void mtp_attention_projection(const std::array<Tensor, kMaximumDevices>& hidden,
                                         const std::array<const MtpAttentionProjectionWeights*, kMaximumDevices>& w,
                                         const std::array<Tensor, kMaximumDevices>& query,
                                         const std::array<Tensor, kMaximumDevices>& gate,
                                         const std::array<Tensor, kMaximumDevices>& key,
                                         const std::array<Tensor, kMaximumDevices>& value,
                                         const std::array<WorkspaceArena*, kMaximumDevices>& workspace,
                                         const ExecutionContext& ec);
    static void mtp_kv_projection(const std::array<Tensor, kMaximumDevices>& hidden,
                                  const std::array<const MtpAttentionProjectionWeights*, kMaximumDevices>& w,
                                  const std::array<Tensor, kMaximumDevices>& key,
                                  const std::array<Tensor, kMaximumDevices>& value,
                                  const std::array<WorkspaceArena*, kMaximumDevices>& workspace,
                                  const ExecutionContext& ec);
    static void mtp_q_gate_projection(const std::array<Tensor, kMaximumDevices>& hidden,
                                      const std::array<const MtpAttentionProjectionWeights*, kMaximumDevices>& w,
                                      const std::array<Tensor, kMaximumDevices>& query,
                                      const std::array<Tensor, kMaximumDevices>& gate,
                                      const std::array<WorkspaceArena*, kMaximumDevices>& workspace,
                                      const ExecutionContext& ec);
    static void mtp_post_mixer(const std::array<Tensor, kMaximumDevices>& hidden,
                               const std::array<const MtpPostMixerWeights*, kMaximumDevices>& w,
                               const std::array<Tensor, kMaximumDevices>& residual,
                               const std::array<Tensor, kMaximumDevices>& staging,
                               const std::array<WorkspaceArena*, kMaximumDevices>& workspace,
                               const ExecutionContext& ec, const ops::CollectiveComms& ev);
    static void gdn_control_projection(const std::array<Tensor, kMaximumDevices>& hidden,
                                       const std::array<const GdnProjectionWeights*, kMaximumDevices>& w,
                                       const std::array<Tensor, kMaximumDevices>& g,
                                       const std::array<Tensor, kMaximumDevices>& beta,
                                       const std::array<WorkspaceArena*, kMaximumDevices>& workspace,
                                       const ExecutionContext& ec);
    static void post_mixer(const std::array<Tensor, kMaximumDevices>& hidden,
                           const std::array<const PostMixerWeights*, kMaximumDevices>& w,
                           const std::array<Tensor, kMaximumDevices>& residual,
                           const std::array<Tensor, kMaximumDevices>& staging, qwen3_6::TextPhase phase,
                           const std::array<WorkspaceArena*, kMaximumDevices>& workspace,
                           const ExecutionContext& ec, const ops::CollectiveComms& ev);

    [[nodiscard]] static std::vector<GraphExecutionProfile>
    ordinary_graph_profiles(std::uint32_t capacity);
    [[nodiscard]] static std::vector<GraphExecutionProfile>
    mtp_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window);
    [[nodiscard]] static std::vector<GraphExecutionProfile>
    dflash_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window,
                          std::uint32_t batch_size);
};

} // namespace ninfer::targets::qwen3_6_27b::detail
