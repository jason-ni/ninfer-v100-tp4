#pragma once

#include <ninfer/targets/qwen3_6/startup_features.h>
#include <ninfer/targets/qwen3_6/vision.h>

#include "core/tensor.h"

#include <array>
#include <cstddef>
#include <optional>

namespace ninfer {

class DeviceArena;

namespace targets::qwen3_6 {

template <class ProjectionPayload, class PostMixerPayload>
struct FullAttentionWeights {
    Tensor input_norm;
    ProjectionPayload projection;
    Tensor query_norm;
    Tensor key_norm;
    Weight output;
    Tensor post_attention_norm;
    PostMixerPayload post_mixer;
};

template <class ProjectionPayload, class PostMixerPayload>
struct GdnWeights {
    Tensor input_norm;
    ProjectionPayload projection;
    Tensor convolution;
    Tensor norm;
    Weight output;
    Tensor post_attention_norm;
    PostMixerPayload post_mixer;
};

template <class AttentionPayload, class PostMixerPayload>
struct MtpWeights {
    Weight input_projection;
    Tensor embedding_norm;
    Tensor hidden_norm;
    Tensor input_norm;
    AttentionPayload attention;
    Tensor query_norm;
    Tensor key_norm;
    Weight output;
    Tensor post_attention_norm;
    PostMixerPayload post_mixer;
    Tensor final_norm;
};

struct OptimizedProposalWeights {
    Weight head;
    Tensor token_ids;
};

struct DFlashLayerWeights {
    Tensor input_norm;
    Weight query;
    Weight key;
    Weight value;
    Tensor query_norm;
    Tensor key_norm;
    Weight attention_output;
    Tensor post_attention_norm;
    Weight gate_up;
    Weight down;
    // Dedicated context-side K/V projections published by the separate-projection DFlash2 package;
    // empty in artifacts that alias the context matrices onto the QKV rows.
    Weight context_key;
    Weight context_value;
    // DFlash2 dynamic grouped-convolution payloads.  The family runtime keeps these as
    // first-class model data even when a target selects the inexpensive static-convolution
    // execution profile; this lets the target leaf evolve without changing the artifact ABI.
    Tensor attention_conv_base_kernel;
    Weight attention_conv_kernel_projection;
    Tensor mlp_conv_base_kernel;
    Weight mlp_conv_kernel_projection;
};

template <std::size_t Layers>
struct DFlashWeights {
    // True when the artifact publishes dedicated context K/V matrices; false for the alias form
    // where the context stream shares the QKV rows.
    bool separate_context_kv = false;
    Weight feature_projection;
    Tensor context_norm;
    std::array<DFlashLayerWeights, Layers> layers;
    Tensor final_norm;
    Weight selector_hidden_projection;
    Tensor selector_predecessor_codebook;
    Tensor selector_successor_codebook;
};

template <class FullProjectionPayload, class GdnProjectionPayload, class MainPostMixerPayload,
          class MtpAttentionPayload, class MtpPostMixerPayload, class DFlashPayload,
          std::size_t FullAttentionLayers, std::size_t GdnLayers>
struct ModelView {
    using FullLayer = FullAttentionWeights<FullProjectionPayload, MainPostMixerPayload>;
    using GdnLayer  = GdnWeights<GdnProjectionPayload, MainPostMixerPayload>;
    using MtpLayer  = MtpWeights<MtpAttentionPayload, MtpPostMixerPayload>;
    using DFlash    = DFlashPayload;

    DeviceArena* weights_arena = nullptr;
    Weight token_embedding;
    std::array<FullLayer, FullAttentionLayers> full_layers;
    std::array<GdnLayer, GdnLayers> gdn_layers;
    Tensor final_norm;
    Weight output_head;
    StartupFeatures features;
    std::optional<OptimizedProposalWeights> optimized_proposal;
    std::optional<MtpLayer> mtp;
    std::optional<DFlashPayload> dflash;
    std::optional<VisionWeights> vision;
    // Host-staged (streaming) Vision bank for this rank: present when the target streams its
    // vision tower from pinned host shards instead of a resident device copy. `vision` still holds
    // the descriptor-only weight views; the encode path uploads the staged block and rebases it.
    std::optional<VisionStageRank> vision_stage;
};

} // namespace targets::qwen3_6
} // namespace ninfer
