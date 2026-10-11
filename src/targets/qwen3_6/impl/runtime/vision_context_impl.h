#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"

#include "core/device.h"
#include "core/layout.h"
#include <ninfer/targets/qwen3_6/vision.h>
#include <ninfer/targets/qwen3_6/vision_control.h>
#include "ninfer/ops/add_bias.h"
#include "ninfer/ops/gelu.h"
#include "ninfer/ops/layer_norm.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/vision_attention.h"
#include "ninfer/ops/vision_pos_embed.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("Vision ") + label + " overflows size_t");
    }
    return a * b;
}

constexpr std::size_t kWorkspaceAlignment = 256;

struct VisionWorkspaceLayout {
    TensorRegion position_ids;
    TensorRegion cu_seqlens;
    TensorRegion pos_indices;
    TensorRegion pos_weights;
    TensorRegion x;
    TensorRegion patch_bf16;
    TensorRegion attended;
    TensorRegion qkv;
    TensorRegion attention_norm;
    std::optional<LayoutRegion> attention_workspace;
    TensorRegion projected;
    TensorRegion mlp_down;
    TensorRegion mlp_up;
    TensorRegion mlp_norm;
    TensorRegion normalized;
    TensorRegion merger_hidden;
    // Row-parallel allreduce scratch, present only for tp > 1 (see build_workspace_layout).
    std::optional<TensorRegion> stage_hidden;
    std::optional<TensorRegion> stage_out;
    std::size_t bytes = 0;
};

// `tp`/`rank` shard the per-rank intermediates to match the 27B shard map exactly: qkv is split by
// head (all three planes, uniform 3*hidden/tp), while the MLP up/down and merger fc1/fc2 widths use
// the SAME mod-8 per-device extent (`vision_mod8_shard_count`), which is even at tp == 2 but uneven
// at tp == 4. tp == 1 reproduces the single-device layout exactly (3*hidden, intermediate,
// merger_hidden, full head count), so the one-arena `encode` and the static capacity queries are
// unaffected.
VisionWorkspaceLayout build_workspace_layout(std::size_t patches64, std::size_t tokens64,
                                             std::size_t segment_count, std::size_t tp = 1,
                                             std::size_t rank = 0) {
    if (patches64 == 0 || tokens64 == 0 ||
        patches64 !=
            checked_mul(tokens64, VisionScheduleConfig::merge_unit, "patch/token relation")) {
        throw std::invalid_argument("Vision workspace requires P=4V>0");
    }
    if (patches64 > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        tokens64 > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        segment_count == 0 ||
        segment_count >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("Vision request dimensions exceed int32");
    }
    if (tp < 1 || tp > VisionScheduleConfig::heads || VisionScheduleConfig::heads % tp != 0 ||
        rank >= tp) {
        throw std::invalid_argument("Vision workspace tensor-parallel width is invalid");
    }
    // All three q|k|v planes are split by head (see the 27B shard map), so each plane gets
    // hidden/tp == (heads/tp)*head_dim rows and the per-rank qkv width is 3 * hidden / tp. The MLP
    // and merger widths use the shared mod-8 split so fc2's contraction equals fc1's output on every
    // rank; at tp == 1 that helper returns the full width, leaving the layout unchanged.
    const int heads_per_rank      = static_cast<int>(VisionScheduleConfig::heads / tp);
    // Each rank's attention output spans only its own heads (heads/tp * head_dim == hidden/tp);
    // that is the row-parallel K of o_proj, so `attended` is sharded too.
    const int attn_out_width      = heads_per_rank * VisionScheduleConfig::head_dim;
    const int qkv_width           = 3 * attn_out_width;
    const int mlp_up_width        = static_cast<int>(qwen3_6::vision_mod8_shard_count(
        VisionScheduleConfig::intermediate, static_cast<int>(tp), static_cast<int>(rank)));
    const int merger_hidden_width = static_cast<int>(qwen3_6::vision_mod8_shard_count(
        VisionScheduleConfig::merger_hidden, static_cast<int>(tp), static_cast<int>(rank)));
    const auto patches            = static_cast<std::int32_t>(patches64);
    const auto tokens             = static_cast<std::int32_t>(tokens64);

    LayoutBuilder builder;
    VisionWorkspaceLayout out;
    const auto add = [&](DType dtype, std::initializer_list<std::int32_t> shape,
                         const char* label) {
        return builder.add_tensor(dtype, shape, kWorkspaceAlignment, label);
    };
    out.position_ids = add(DType::I32, {patches, 2}, "vision position ids");
    out.cu_seqlens =
        add(DType::I32, {static_cast<std::int32_t>(segment_count + 1)}, "vision segment bounds");
    out.pos_indices = add(DType::I32, {4, patches}, "vision position indices");
    out.pos_weights = add(DType::FP32, {4, patches}, "vision position weights");
    out.x           = add(DType::BF16, {VisionScheduleConfig::hidden, patches}, "vision residual");
    out.patch_bf16 =
        add(DType::BF16, {VisionScheduleConfig::patch_dim, patches}, "vision BF16 patches");
    {
        auto attention_scope = builder.scope();
        out.attended = add(DType::BF16, {attn_out_width, patches}, "vision attended");
        {
            auto qkv_scope = builder.scope();
            out.qkv = add(DType::BF16, {qkv_width, patches}, "vision QKV");
            {
                auto norm_scope    = builder.scope();
                out.attention_norm = add(DType::BF16, {VisionScheduleConfig::hidden, patches},
                                         "vision attention norm");
            }
            const std::size_t attention_bytes = ops::vision_attention_workspace_capacity_bytes(
                patches, patches, static_cast<std::int32_t>(segment_count),
                static_cast<std::int32_t>(segment_count), heads_per_rank);
            if (attention_bytes != 0) {
                out.attention_workspace =
                    builder.add(attention_bytes, kWorkspaceAlignment, "vision attention workspace");
            }
        }
        out.projected =
            add(DType::BF16, {VisionScheduleConfig::hidden, patches}, "vision projected");
    }
    {
        auto mlp_scope = builder.scope();
        out.mlp_up = add(DType::BF16, {mlp_up_width, patches}, "vision MLP up");
        {
            auto norm_scope = builder.scope();
            out.mlp_norm =
                add(DType::BF16, {VisionScheduleConfig::hidden, patches}, "vision MLP norm");
        }
        out.mlp_down = add(DType::BF16, {VisionScheduleConfig::hidden, patches}, "vision MLP down");
    }
    out.normalized =
        add(DType::BF16, {VisionScheduleConfig::hidden, patches}, "vision merger norm");
    out.merger_hidden =
        add(DType::BF16, {merger_hidden_width, tokens}, "vision merger hidden");
    // Row-parallel allreduce staging (tp > 1 only). `stage_hidden` is reused for both the o_proj and
    // fc2 reduces (they never overlap in time within a layer); `stage_out` sizes the final merger
    // fc2 reduce. Both match their output's dtype/shape and are disjoint from it, as required.
    if (tp > 1) {
        out.stage_hidden =
            add(DType::BF16, {VisionScheduleConfig::hidden, patches}, "vision row-parallel stage");
        out.stage_out =
            add(DType::BF16, {VisionScheduleConfig::out_hidden, tokens}, "vision merger stage");
    }
    out.bytes = builder.finish(1, "vision workspace");
    return out;
}

void copy_host(const void* src, Tensor& dst, cudaStream_t stream) {
    if (dst.bytes() == 0) { return; }
    CUDA_CHECK(cudaMemcpyAsync(dst.data, src, dst.bytes(), cudaMemcpyHostToDevice, stream));
}

} // namespace

VisionContext::VisionContext(DeviceContext& ctx, const LoadedModelData& weights) : ctx_(ctx) {
    bind_rank(0, weights);
}

VisionContext::VisionContext(DeviceContext& device, const LoadedModelData& model,
                             const ExecutionContext& ec, const ops::CollectiveComms& events,
                             const std::array<const LoadedModelData*, kMaximumDevices>& peer_weights)
    : ctx_(device), ec_(&ec), events_(&events), tp_count_(ec.tp) {
    if (ec.tp < 2) {
        throw std::invalid_argument("tensor-parallel VisionContext requires at least two devices");
    }
    if (&*ec.dev[0] != &device) {
        throw std::invalid_argument("tensor-parallel VisionContext rank 0 device disagrees with ec");
    }
    if (!events.live()) {
        throw std::invalid_argument("tensor-parallel VisionContext requires a live collective");
    }
    for (int rank = 1; rank < tp_count_; ++rank) {
        if (peer_weights[static_cast<std::size_t>(rank)] == nullptr) {
            throw std::invalid_argument("tensor-parallel VisionContext is missing a peer view");
        }
    }
    bind_rank(0, model);
    for (int rank = 1; rank < tp_count_; ++rank) {
        bind_rank(rank, *peer_weights[static_cast<std::size_t>(rank)]);
    }
}

void VisionContext::bind_rank(int rank, const LoadedModelData& weights) {
    if (!weights.vision) {
        throw std::invalid_argument("Vision execution was requested without materialized weights");
    }
    const auto& vision = *weights.vision;
    VisionWeights& out = ranks_[static_cast<std::size_t>(rank)];
    out.patch_embed      = &vision.common.patch_embedding;
    out.patch_embed_bias = &vision.common.patch_embedding_bias;
    out.position_embed   = &vision.common.position_embedding;
    for (std::uint32_t layer = 0; layer < out.blocks.size(); ++layer) {
        const auto& source = vision.common.layers[layer];
        BlockW& block      = out.blocks[layer];
        block.norm1_weight    = &source.norm1_weight;
        block.norm1_bias      = &source.norm1_bias;
        block.qkv             = &source.qkv;
        block.qkv_bias        = &source.qkv_bias;
        block.projection      = &source.output;
        block.projection_bias = &source.output_bias;
        block.norm2_weight    = &source.norm2_weight;
        block.norm2_bias      = &source.norm2_bias;
        block.fc1             = &source.fc1;
        block.fc1_bias        = &source.fc1_bias;
        block.fc2             = &source.fc2;
        block.fc2_bias        = &source.fc2_bias;
    }
    out.merger.norm_weight = &vision.common.merger_norm_weight;
    out.merger.norm_bias   = &vision.common.merger_norm_bias;
    out.merger.fc1         = &vision.common.merger_fc1;
    out.merger.fc1_bias    = &vision.common.merger_fc1_bias;
    out.merger.fc2         = &vision.merger_fc2;
    out.merger.fc2_bias    = &vision.merger_fc2_bias;
}

std::size_t VisionContext::workspace_bytes(const qwen3_6::VisionItemControl& item,
                                           std::size_t tp) {
    std::size_t bytes = 0;
    for (std::size_t rank = 0; rank < tp; ++rank) {
        bytes = std::max(
            bytes, build_workspace_layout(item.patch_count, item.merged_count,
                                          static_cast<std::size_t>(item.segment_count), tp, rank)
                       .bytes);
    }
    return bytes;
}

std::size_t VisionContext::output_transient_bytes(std::size_t merged_tokens) {
    if (merged_tokens == 0 ||
        merged_tokens > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Vision output transient extent must fit positive int32");
    }
    LayoutBuilder layout;
    (void)layout.add_tensor(
        DType::BF16, {VisionScheduleConfig::out_hidden, static_cast<std::int32_t>(merged_tokens)},
        kWorkspaceAlignment, "Vision item output transient");
    return layout.finish(kWorkspaceAlignment, "Vision item output transient layout");
}

std::size_t VisionContext::workspace_capacity_bytes(std::uint32_t max_merged_tokens,
                                                    std::uint32_t max_segments, std::size_t tp) {
    if (max_merged_tokens == 0 || max_segments == 0) {
        throw std::invalid_argument("Vision workspace capacity bounds must be positive");
    }
    if (tp < 1 || tp > VisionScheduleConfig::heads || VisionScheduleConfig::heads % tp != 0) {
        throw std::invalid_argument("Vision workspace tensor-parallel width is invalid");
    }
    const std::uint32_t segments = std::min(max_merged_tokens, max_segments);
    const std::size_t patches = checked_mul(max_merged_tokens, VisionScheduleConfig::merge_unit,
                                            "capacity patch count");
    // Each rank's arena must fit its own layout; tp == 4's mod-8 split is uneven, so size to the
    // maximum over ranks. tp == 1 is a single rank and reproduces the historical value exactly.
    std::size_t bytes = 0;
    for (std::size_t rank = 0; rank < tp; ++rank) {
        bytes = std::max(
            bytes, build_workspace_layout(patches, max_merged_tokens, segments, tp, rank).bytes);
    }
    return bytes;
}

void VisionContext::encode(const VisionItemView& item, Tensor& output,
                           WorkspaceArena& workspace) const {
    if (item.control == nullptr) { throw std::invalid_argument("Vision item control is null"); }
    const qwen3_6::VisionItemControl& control = *item.control;
    const auto patches64                      = control.patch_count;
    const auto tokens64                       = control.merged_count;
    if (item.patches.size() !=
        checked_mul(patches64, VisionScheduleConfig::patch_dim, "patch elements")) {
        throw std::invalid_argument("Vision processor patch buffer has invalid shape");
    }
    if (output.dtype != DType::BF16 || output.ne[0] != VisionScheduleConfig::out_hidden ||
        output.ne[1] != static_cast<std::int32_t>(tokens64) || output.ne[2] != 1 ||
        output.ne[3] != 1 || !output.is_contiguous() || output.data == nullptr) {
        throw std::invalid_argument("Vision output must be contiguous BF16 [H,V]");
    }
    const VisionWorkspaceLayout layout = build_workspace_layout(
        patches64, tokens64, static_cast<std::size_t>(control.segment_count));
    if (workspace.capacity() < layout.bytes) {
        throw std::invalid_argument("Vision workspace capacity is too small for request");
    }
    const auto patches  = static_cast<std::int32_t>(patches64);
    const auto tokens   = static_cast<std::int32_t>(tokens64);
    cudaStream_t stream = ctx_.stream;
    const VisionWeights& w = weights_for(0);
    workspace.reset();
    const DeviceSpan backing = workspace.alloc_bytes(layout.bytes, kWorkspaceAlignment);

    Tensor position_ids = layout.position_ids.bind(backing);
    Tensor cu_seqlens   = layout.cu_seqlens.bind(backing);
    Tensor pos_indices  = layout.pos_indices.bind(backing);
    Tensor pos_weights  = layout.pos_weights.bind(backing);
    copy_host(control.position_ids.data(), position_ids, stream);
    copy_host(control.cu_seqlens.data(), cu_seqlens, stream);
    copy_host(control.position_table_indices.data(), pos_indices, stream);
    copy_host(control.position_table_weights.data(), pos_weights, stream);

    Tensor x          = layout.x.bind(backing);
    Tensor patch_bf16 = layout.patch_bf16.bind(backing);
    copy_host(item.patches.data(), patch_bf16, stream);
    ops::linear(patch_bf16, *w.patch_embed, x, stream);
    ops::add_bias(*w.patch_embed_bias, x, stream);
    // The artifact records the source table shape [rows,hidden], while Tensor's
    // contiguous matrix convention is [inner,columns]. The payload is already
    // row-major, so this is a zero-copy [hidden,rows] view, not a transpose.
    Tensor position_table = w.position_embed->reshape(
        {VisionScheduleConfig::hidden, VisionScheduleConfig::position_embeddings});
    ops::vision_pos_embed_add(position_table, pos_indices, pos_weights, x, stream);
    for (std::size_t layer = 0; layer < w.blocks.size(); ++layer) {
        const BlockW& block = w.blocks[layer];
        {
            Tensor attended = layout.attended.bind(backing);
            {
                Tensor qkv = layout.qkv.bind(backing);
                {
                    Tensor h = layout.attention_norm.bind(backing);
                    ops::layer_norm(x, *block.norm1_weight, *block.norm1_bias,
                                    VisionScheduleConfig::norm_eps, h, stream);
                    ops::linear(h, *block.qkv, qkv, stream);
                }
                ops::add_bias(*block.qkv_bias, qkv, stream);
                const std::int32_t plane      = VisionScheduleConfig::hidden;
                const std::size_t plane_bytes = static_cast<std::size_t>(plane) * 2;
                Tensor q(qkv.data, DType::BF16,
                         {VisionScheduleConfig::head_dim, VisionScheduleConfig::heads, patches});
                Tensor k(static_cast<unsigned char*>(qkv.data) + plane_bytes, DType::BF16,
                         {VisionScheduleConfig::head_dim, VisionScheduleConfig::heads, patches});
                Tensor v(static_cast<unsigned char*>(qkv.data) + 2 * plane_bytes, DType::BF16,
                         {VisionScheduleConfig::head_dim, VisionScheduleConfig::heads, patches});
                q.nb[2] = qkv.nb[1];
                k.nb[2] = qkv.nb[1];
                v.nb[2] = qkv.nb[1];
                ops::rope(position_ids, VisionScheduleConfig::rotary_dim,
                          VisionScheduleConfig::rope_theta, q, k, stream);
                Tensor attended_heads = attended.view(
                    {VisionScheduleConfig::head_dim, VisionScheduleConfig::heads, patches});
                const DeviceSpan attention_backing = layout.attention_workspace
                                                         ? layout.attention_workspace->bind(backing)
                                                         : backing;
                WorkspaceArena attention_workspace(attention_backing);
                ops::vision_attention(q, k, v, cu_seqlens, attention_workspace, attended_heads,
                                      stream);
            }
            Tensor projected = layout.projected.bind(backing);
            ops::linear(attended, *block.projection, projected, stream);
            ops::add_bias(*block.projection_bias, projected, stream);
            ops::residual_add(projected, x, stream);
        }
        {
            Tensor down = layout.mlp_down.bind(backing);
            Tensor up   = layout.mlp_up.bind(backing);
            {
                Tensor h = layout.mlp_norm.bind(backing);
                ops::layer_norm(x, *block.norm2_weight, *block.norm2_bias,
                                VisionScheduleConfig::norm_eps, h, stream);
                ops::linear(h, *block.fc1, up, stream);
            }
            ops::add_bias(*block.fc1_bias, up, stream);
            ops::gelu(up, ops::GeluMode::Tanh, stream);
            ops::linear(up, *block.fc2, down, stream);
            ops::add_bias(*block.fc2_bias, down, stream);
            ops::residual_add(down, x, stream);
        }
    }

    Tensor normalized = layout.normalized.bind(backing);
    ops::layer_norm(x, *w.merger.norm_weight, *w.merger.norm_bias, VisionScheduleConfig::norm_eps,
                    normalized, stream);
    Tensor merged = normalized.view({VisionScheduleConfig::merger_hidden, tokens});
    Tensor hidden = layout.merger_hidden.bind(backing);
    ops::linear(merged, *w.merger.fc1, hidden, stream);
    ops::add_bias(*w.merger.fc1_bias, hidden, stream);
    ops::gelu(hidden, ops::GeluMode::Exact, stream);
    ops::linear(hidden, *w.merger.fc2, output, stream);
    ops::add_bias(*w.merger.fc2_bias, output, stream);
}

void VisionContext::encode(const VisionItemView& item, const std::array<Tensor, kMaximumDevices>& out,
                           const std::array<WorkspaceArena*, kMaximumDevices>& work) const {
    if (tp_count_ <= 1 || ec_ == nullptr || events_ == nullptr) {
        throw std::logic_error("sharded Vision encode requires tensor-parallel construction");
    }
    if (item.control == nullptr) { throw std::invalid_argument("Vision item control is null"); }
    const qwen3_6::VisionItemControl& control = *item.control;
    const auto patches64                      = control.patch_count;
    const auto tokens64                       = control.merged_count;
    if (item.patches.size() !=
        checked_mul(patches64, VisionScheduleConfig::patch_dim, "patch elements")) {
        throw std::invalid_argument("Vision processor patch buffer has invalid shape");
    }
    for (int r = 0; r < tp_count_; ++r) {
        const Tensor& o = out[static_cast<std::size_t>(r)];
        if (o.dtype != DType::BF16 || o.ne[0] != VisionScheduleConfig::out_hidden ||
            o.ne[1] != static_cast<std::int32_t>(tokens64) || o.ne[2] != 1 || o.ne[3] != 1 ||
            !o.is_contiguous() || o.data == nullptr) {
            throw std::invalid_argument("Vision rank output must be contiguous BF16 [H,V]");
        }
    }
    const int tp             = tp_count_;
    const ExecutionContext& ec     = *ec_;
    const ops::CollectiveComms& events = *events_;
    const int heads_per_rank = VisionScheduleConfig::heads / tp;
    const int attn_out_width = heads_per_rank * VisionScheduleConfig::head_dim;
    const auto patches       = static_cast<std::int32_t>(patches64);
    const auto tokens        = static_cast<std::int32_t>(tokens64);

    // Per-rank state. Each rank builds its own (possibly uneven, tp == 4) layout in its own arena.
    // The replicated tensors (x, norms, position tables) hold identical values on every rank by
    // construction; only qkv / attended / mlp_up / merger_hidden are sharded.
    std::array<VisionWorkspaceLayout, kMaximumDevices> layouts{};
    std::array<DeviceSpan, kMaximumDevices> backing{};
    std::array<Tensor, kMaximumDevices> position_ids{}, cu_seqlens{}, pos_indices{}, pos_weights{};
    std::array<Tensor, kMaximumDevices> x{}, patch_bf16{};
    for_each_rank(ec, [&](int rank) {
        const auto r      = static_cast<std::size_t>(rank);
        const auto stream = ec.dev[r]->stream;
        layouts[r] = build_workspace_layout(patches64, tokens64, control.segment_count, tp, rank);
        if (work[r] == nullptr || work[r]->capacity() < layouts[r].bytes) {
            throw std::invalid_argument("Vision rank workspace capacity is too small");
        }
        work[r]->reset();
        backing[r] = work[r]->alloc_bytes(layouts[r].bytes, kWorkspaceAlignment);
        position_ids[r] = layouts[r].position_ids.bind(backing[r]);
        cu_seqlens[r]   = layouts[r].cu_seqlens.bind(backing[r]);
        pos_indices[r]  = layouts[r].pos_indices.bind(backing[r]);
        pos_weights[r]  = layouts[r].pos_weights.bind(backing[r]);
        copy_host(control.position_ids.data(), position_ids[r], stream);
        copy_host(control.cu_seqlens.data(), cu_seqlens[r], stream);
        copy_host(control.position_table_indices.data(), pos_indices[r], stream);
        copy_host(control.position_table_weights.data(), pos_weights[r], stream);
        x[r]          = layouts[r].x.bind(backing[r]);
        patch_bf16[r] = layouts[r].patch_bf16.bind(backing[r]);
        copy_host(item.patches.data(), patch_bf16[r], stream);
    });

    // Patch embedding + position embed: REPLICATED. Every rank runs the same full-width projection
    // on its own copy of the patches, so x[r] is identical across ranks (no collective needed).
    for_each_rank(ec, [&](int rank) {
        const auto r      = static_cast<std::size_t>(rank);
        const auto stream = ec.dev[r]->stream;
        const VisionWeights& w = weights_for(rank);
        ops::linear(patch_bf16[r], *w.patch_embed, x[r], stream);
        ops::add_bias(*w.patch_embed_bias, x[r], stream);
        Tensor position_table = w.position_embed->reshape(
            {VisionScheduleConfig::hidden, VisionScheduleConfig::position_embeddings});
        ops::vision_pos_embed_add(position_table, pos_indices[r], pos_weights[r], x[r], stream);
    });

    for (std::size_t layer = 0; layer < VisionScheduleConfig::layers; ++layer) {
        std::array<Weight, kMaximumDevices> qkv_w{}, oproj_w{}, fc1_w{}, fc2_w{};
        std::array<Tensor, kMaximumDevices> qkv_bias{}, oproj_bias{}, fc1_bias{}, fc2_bias{};
        std::array<Tensor, kMaximumDevices> norm1_w{}, norm1_b{}, norm2_w{}, norm2_b{};
        for (int r = 0; r < tp; ++r) {
            const BlockW& b = weights_for(r).blocks[layer];
            qkv_w[r]      = *b.qkv;
            oproj_w[r]    = *b.projection;
            fc1_w[r]      = *b.fc1;
            fc2_w[r]      = *b.fc2;
            qkv_bias[r]   = *b.qkv_bias;
            oproj_bias[r] = *b.projection_bias;
            fc1_bias[r]   = *b.fc1_bias;
            fc2_bias[r]   = *b.fc2_bias;
            norm1_w[r]    = *b.norm1_weight;
            norm1_b[r]    = *b.norm1_bias;
            norm2_w[r]    = *b.norm2_weight;
            norm2_b[r]    = *b.norm2_bias;
        }
        // Attention sub-block.
        {
            std::array<Tensor, kMaximumDevices> attended{}, qkv{}, h{};
            for (int r = 0; r < tp; ++r) {
                attended[r] = layouts[r].attended.bind(backing[r]);
                qkv[r]      = layouts[r].qkv.bind(backing[r]);
                h[r]        = layouts[r].attention_norm.bind(backing[r]);
            }
            for_each_rank(ec, [&](int rank) {
                const auto r      = static_cast<std::size_t>(rank);
                const auto stream = ec.dev[r]->stream;
                ops::layer_norm(x[r], norm1_w[r], norm1_b[r], VisionScheduleConfig::norm_eps, h[r],
                                stream);
            });
            // qkv: column-parallel (sharded N), no collective.
            ops::linear_column_parallel(h, qkv_w, qkv, ec);
            const std::size_t plane_bytes = static_cast<std::size_t>(attn_out_width) * 2;
            for_each_rank(ec, [&](int rank) {
                const auto r      = static_cast<std::size_t>(rank);
                const auto stream = ec.dev[r]->stream;
                ops::add_bias(qkv_bias[r], qkv[r], stream);
                Tensor q(qkv[r].data, DType::BF16,
                         {VisionScheduleConfig::head_dim, heads_per_rank, patches});
                Tensor k(static_cast<unsigned char*>(qkv[r].data) + plane_bytes, DType::BF16,
                         {VisionScheduleConfig::head_dim, heads_per_rank, patches});
                Tensor v(static_cast<unsigned char*>(qkv[r].data) + 2 * plane_bytes, DType::BF16,
                         {VisionScheduleConfig::head_dim, heads_per_rank, patches});
                q.nb[2] = qkv[r].nb[1];
                k.nb[2] = qkv[r].nb[1];
                v.nb[2] = qkv[r].nb[1];
                ops::rope(position_ids[r], VisionScheduleConfig::rotary_dim,
                          VisionScheduleConfig::rope_theta, q, k, stream);
                Tensor attended_heads =
                    attended[r].view({VisionScheduleConfig::head_dim, heads_per_rank, patches});
                const DeviceSpan attention_backing = layouts[r].attention_workspace
                                                          ? layouts[r].attention_workspace->bind(
                                                                backing[r])
                                                          : backing[r];
                WorkspaceArena attention_workspace(attention_backing);
                ops::vision_attention(q, k, v, cu_seqlens[r], attention_workspace, attended_heads,
                                      stream);
            });
            // o_proj: row-parallel (sharded K), single allreduce -> replicated projected[r].
            std::array<Tensor, kMaximumDevices> projected{}, stage{};
            for (int r = 0; r < tp; ++r) {
                projected[r] = layouts[r].projected.bind(backing[r]);
                stage[r]     = layouts[r].stage_hidden->bind(backing[r]);
            }
            ops::linear_row_parallel(attended, oproj_w, projected, stage, ec, events);
            for_each_rank(ec, [&](int rank) {
                const auto r      = static_cast<std::size_t>(rank);
                const auto stream = ec.dev[r]->stream;
                ops::add_bias(oproj_bias[r], projected[r], stream);
                ops::residual_add(projected[r], x[r], stream);
            });
        }
        // MLP sub-block.
        {
            std::array<Tensor, kMaximumDevices> down{}, up{}, h{};
            for (int r = 0; r < tp; ++r) {
                down[r] = layouts[r].mlp_down.bind(backing[r]);
                up[r]   = layouts[r].mlp_up.bind(backing[r]);
                h[r]    = layouts[r].mlp_norm.bind(backing[r]);
            }
            for_each_rank(ec, [&](int rank) {
                const auto r      = static_cast<std::size_t>(rank);
                const auto stream = ec.dev[r]->stream;
                ops::layer_norm(x[r], norm2_w[r], norm2_b[r], VisionScheduleConfig::norm_eps, h[r],
                                stream);
            });
            // fc1: column-parallel (sharded N), no collective.
            ops::linear_column_parallel(h, fc1_w, up, ec);
            for_each_rank(ec, [&](int rank) {
                const auto r      = static_cast<std::size_t>(rank);
                const auto stream = ec.dev[r]->stream;
                ops::add_bias(fc1_bias[r], up[r], stream);
                ops::gelu(up[r], ops::GeluMode::Tanh, stream);
            });
            // fc2: row-parallel (sharded K), single allreduce -> replicated down[r].
            std::array<Tensor, kMaximumDevices> stage{};
            for (int r = 0; r < tp; ++r) { stage[r] = layouts[r].stage_hidden->bind(backing[r]); }
            ops::linear_row_parallel(up, fc2_w, down, stage, ec, events);
            for_each_rank(ec, [&](int rank) {
                const auto r      = static_cast<std::size_t>(rank);
                const auto stream = ec.dev[r]->stream;
                ops::add_bias(fc2_bias[r], down[r], stream);
                ops::residual_add(down[r], x[r], stream);
            });
        }
    }

    // Merger: layer_norm (replicated) -> fc1 (column, sharded N) -> gelu(Exact) -> fc2 (row,
    // sharded K, allreduce). Rank 0 writes the caller's output; peers write their own copy.
    {
        std::array<Weight, kMaximumDevices> mfc1_w{}, mfc2_w{};
        std::array<Tensor, kMaximumDevices> mfc1_bias{}, mfc2_bias{}, mnorm_w{}, mnorm_b{};
        for (int r = 0; r < tp; ++r) {
            const MergerW& m = weights_for(r).merger;
            mfc1_w[r]    = *m.fc1;
            mfc2_w[r]    = *m.fc2;
            mfc1_bias[r] = *m.fc1_bias;
            mfc2_bias[r] = *m.fc2_bias;
            mnorm_w[r]   = *m.norm_weight;
            mnorm_b[r]   = *m.norm_bias;
        }
        std::array<Tensor, kMaximumDevices> normalized{}, merged_in{}, up{}, down{}, stage{};
        for (int r = 0; r < tp; ++r) {
            normalized[r] = layouts[r].normalized.bind(backing[r]);
            up[r]         = layouts[r].merger_hidden.bind(backing[r]);
            down[r]       = out[static_cast<std::size_t>(r)];
            stage[r]      = layouts[r].stage_out->bind(backing[r]);
        }
        for_each_rank(ec, [&](int rank) {
            const auto r      = static_cast<std::size_t>(rank);
            const auto stream = ec.dev[r]->stream;
            ops::layer_norm(x[r], mnorm_w[r], mnorm_b[r], VisionScheduleConfig::norm_eps,
                            normalized[r], stream);
            // The [hidden, patches] residual is re-read as the merged [merger_hidden, tokens] input.
            merged_in[r] = normalized[r].view({VisionScheduleConfig::merger_hidden, tokens});
        });
        ops::linear_column_parallel(merged_in, mfc1_w, up, ec);
        for_each_rank(ec, [&](int rank) {
            const auto r      = static_cast<std::size_t>(rank);
            const auto stream = ec.dev[r]->stream;
            ops::add_bias(mfc1_bias[r], up[r], stream);
            ops::gelu(up[r], ops::GeluMode::Exact, stream);
        });
        ops::linear_row_parallel(up, mfc2_w, down, stage, ec, events);
        for_each_rank(ec, [&](int rank) {
            const auto r      = static_cast<std::size_t>(rank);
            const auto stream = ec.dev[r]->stream;
            ops::add_bias(mfc2_bias[r], down[r], stream);
        });
    }
}

VisionPrefillSession::VisionPrefillSession(DeviceContext& device, const LoadedModelData& model,
                                           WorkspaceArena& workspace,
                                           qwen3_6::PreparedPromptData& prompt,
                                           const VisionPrefillPlan& plan,
                                           runtime::TransientRegion transient,
                                           const VisionTpSetup* tp)
    : device_(device), workspace_(workspace), prompt_(prompt), plan_(plan), transient_(transient) {
    if (plan_.control == nullptr || plan_.control->items.empty() || plan_.uses.empty()) {
        throw std::invalid_argument("Vision prefill plan has no suffix item spans");
    }
    if (transient_.data == nullptr || transient_.alignment < kWorkspaceAlignment) {
        throw std::invalid_argument("Vision item output transient is missing or misaligned");
    }
    if (tp != nullptr) {
        if (tp->ec == nullptr || tp->events == nullptr || tp->ec->tp < 2) {
            throw std::invalid_argument(
                "tensor-parallel Vision session requires a multi-device context");
        }
        tp_count_ = tp->ec->tp;
        // The Program owns every rank's scratch arena and peer embedding destination; the session
        // only checks them against the widest item in this plan (the mod-8 split is uneven at
        // tp == 4, so the required capacity is the max over ranks) and never allocates.
        std::uint32_t max_merged   = 0;
        std::uint32_t max_segments = 0;
        for (const auto& item : plan_.control->items) {
            max_merged   = std::max(max_merged, static_cast<std::uint32_t>(item.merged_count));
            max_segments = std::max(max_segments, static_cast<std::uint32_t>(item.segment_count));
        }
        if (max_merged == 0 || max_segments == 0) {
            throw std::invalid_argument("tensor-parallel Vision plan has no encodable item");
        }
        const std::size_t capacity =
            VisionContext::workspace_capacity_bytes(max_merged, max_segments, tp_count_);
        const std::size_t embed_bytes = checked_mul(
            checked_mul(static_cast<std::size_t>(VisionScheduleConfig::out_hidden), max_merged,
                        "peer embedding elements"),
            dtype_size(DType::BF16), "peer embedding bytes");
        for (int r = 0; r < tp_count_; ++r) {
            const auto slot = static_cast<std::size_t>(r);
            WorkspaceArena* arena = tp->workspaces[slot];
            if (arena == nullptr || arena->capacity() < capacity) {
                throw std::invalid_argument("Vision rank workspace capacity is too small");
            }
            tp_workspaces_[slot] = arena;
            if (r > 0) {
                const DeviceSpan& span = tp->embed_destinations[slot];
                if (span.data == nullptr || span.bytes < embed_bytes) {
                    throw std::invalid_argument("Vision peer embedding destination is too small");
                }
                tp_embed_spans_[slot] = span;
            }
        }
        context_.emplace(device, model, *tp->ec, *tp->events, tp->peer_models);
    } else {
        context_.emplace(device, model);
    }
    encoded_payloads_pending_release_.reserve(plan_.uses.size());
    timers_.reserve(plan_.uses.size());
}

VisionChunk VisionPrefillSession::prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length) {
    if (nominal_length == 0 || begin >= prompt_.token_ids.size()) {
        throw std::invalid_argument("Vision chunk range is empty or outside the prompt");
    }
    const std::uint64_t nominal_end64 =
        static_cast<std::uint64_t>(begin) + static_cast<std::uint64_t>(nominal_length);
    std::uint32_t end = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(nominal_end64, prompt_.token_ids.size()));

    const VisionUseSpan* active = nullptr;
    for (const VisionUseSpan& use : plan_.uses) {
        if (use.end <= begin) { continue; }
        if (use.begin >= end) { break; }
        if (active == nullptr) {
            active = &use;
        } else {
            end = std::min(end, use.begin);
            break;
        }
    }
    if (end <= begin) { throw std::logic_error("Vision chunk cap made no forward progress"); }
    if (active == nullptr) {
        return VisionChunk{static_cast<std::int32_t>(end - begin), nullptr, {}};
    }
    if (active->item_index >= plan_.control->items.size() ||
        active->item_index >= prompt_.vision_items.size() ||
        active->item_index >= prompt_.media_payloads.size()) {
        throw std::logic_error("Vision prefill item index is out of range");
    }
    const qwen3_6::VisionItemControl& control = plan_.control->items[active->item_index];
    const qwen3_6::VisionItem& source         = prompt_.vision_items[active->item_index];
    if (source.modality != control.modality || source.grid.temporal != control.grid.temporal ||
        source.grid.height != control.grid.height || source.grid.width != control.grid.width ||
        source.patch_begin != control.patch_begin || source.patch_count != control.patch_count) {
        throw std::invalid_argument("Vision prefill plan does not describe the prepared item");
    }
    if (control.merged_count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("Vision item output columns exceed int32");
    }
    const std::size_t output_bytes =
        checked_mul(checked_mul(static_cast<std::size_t>(VisionScheduleConfig::out_hidden),
                                control.merged_count, "item output elements"),
                    dtype_size(DType::BF16), "item output bytes");
    if (output_bytes > transient_.size) {
        throw std::invalid_argument("Vision item output transient is too small");
    }
    Tensor output(
        transient_.data, DType::BF16,
        {VisionScheduleConfig::out_hidden, static_cast<std::int32_t>(control.merged_count)});

    if (!active_item_ || *active_item_ != active->item_index) {
        if (active_item_ && active->item_index <= *active_item_) {
            throw std::logic_error("Vision items are not consumed in strictly increasing order");
        }
        const std::size_t patch_elements = checked_mul(
            control.patch_count, static_cast<std::size_t>(VisionScheduleConfig::patch_dim),
            "item patch elements");
        const auto& payload = prompt_.media_payloads[active->item_index];
        if (!payload || payload->patch_elements != patch_elements) {
            throw std::invalid_argument("Vision item patch payload has an invalid shape");
        }
        timers_.emplace_back(device_);
        timers_.back().start();
        if (tensor_parallel()) {
            std::array<WorkspaceArena*, kMaximumDevices> work{};
            std::array<Tensor, kMaximumDevices> out{};
            const int merged = static_cast<int>(control.merged_count);
            for (int r = 0; r < tp_count_; ++r) {
                const auto slot = static_cast<std::size_t>(r);
                work[slot] = tp_workspaces_[slot];
                if (r == 0) {
                    out[slot] = output;
                } else {
                    out[slot] = Tensor(tp_embed_spans_[slot].data, DType::BF16,
                                       {VisionScheduleConfig::out_hidden, merged});
                }
            }
            context_->encode(VisionItemView{payload->span(), &control}, out, work);
            tp_embeddings_ = out;
        } else {
            context_->encode(VisionItemView{payload->span(), &control}, output, workspace_);
        }
        timers_.back().record_stop();
        if (!tensor_parallel()) { workspace_.reset(); }
        active_item_ = active->item_index;
        encoded_payloads_pending_release_.push_back(active->item_index);
    }
    return VisionChunk{static_cast<std::int32_t>(end - begin), &control, output};
}

void VisionPrefillSession::release_encoded_media_payloads() noexcept {
    for (const std::uint32_t item_index : encoded_payloads_pending_release_) {
        if (item_index >= prompt_.media_payloads.size()) { std::terminate(); }
        prompt_.media_payloads[item_index].reset();
    }
    encoded_payloads_pending_release_.clear();
}

double VisionPrefillSession::elapsed_seconds() const {
    double milliseconds = 0.0;
    for (const CudaEventTimer& timer : timers_) { milliseconds += timer.elapsed_ms(); }
    return milliseconds / 1000.0;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
