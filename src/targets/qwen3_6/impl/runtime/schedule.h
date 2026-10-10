#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/ops/kv_cache_append_prefix.h"
#include "ninfer/ops/swa.h"
#include "core/decode_graph.h"
#include "runtime/contract/transient_region.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/dflash_context.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/vision_prefill.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <functional>
#include <optional>
#include <span>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

using qwen3_6::PreparedPromptData;
using qwen3_6::PromptModality;

// Current-device save/restore around per-rank issue. Kernel launches go to the CURRENT device, so
// every op issued for rank r must run with ec.dev[r]->device current; the calls are enqueue-only,
// so the two ranks' kernels still overlap even though the host issues them in sequence. This
// mirrors ops::detail::for_each_rank without reaching into an Op's private header, and lives here
// rather than in one impl header because both the schedule and the MTP round need it.
class CurrentDevice {
public:
    CurrentDevice() { CUDA_CHECK(cudaGetDevice(&previous_)); }

    ~CurrentDevice() { (void)cudaSetDevice(previous_); }

    CurrentDevice(const CurrentDevice&)            = delete;
    CurrentDevice& operator=(const CurrentDevice&) = delete;

private:
    int previous_ = 0;
};

template <class Body>
void for_each_rank(const ExecutionContext& ec, Body&& body) {
    const CurrentDevice restore;
    for (int rank = 0; rank < ec.tp; ++rank) {
        CUDA_CHECK(cudaSetDevice(ec.dev[rank]->device));
        body(rank);
    }
}

// One non-origin rank's half of a tensor-parallel execution, at index `rank`. At tp == 1 no entry
// is populated and every schedule entry point below behaves exactly as it always has.
struct TpPeerCore {
    const ExecutionContext* execution          = nullptr;
    const ops::CollectiveComms* events              = nullptr;
    DeviceContext* device                      = nullptr;
    const LoadedModelData* model               = nullptr;
    WorkspaceArena* work                       = nullptr;
    LinearAttentionStatePool* linear_attention = nullptr;
    qwen3_6::RoundState* io                    = nullptr;
    Tensor* prefill_hidden                     = nullptr;
    const qwen3_6::PagedKVCache* text_cache    = nullptr;
    // Present only when the sequence plan enables MTP.
    const qwen3_6::PagedKVCache* mtp_cache  = nullptr;
    const GdnReplayRecords* replay_records  = nullptr;
    // This rank's own pinned MTP ingress record (see PeerRuntime::token_counts). It differs from
    // rank 0's only in the per-row `sampling[row].token_counts` pointer, which must name this
    // rank's counter lane: `speculative_accept_greedy_drafts` READS and atomically WRITES that
    // pointer in sampling mode, and a pointer into another device's arena is an illegal access
    // without peer mapping and a silent double-increment with it.
    const qwen3_6::MtpDecodeIngress* mtp_host_ingress = nullptr;
    const qwen3_6::DFlashDecodeIngress* dflash_host_ingress = nullptr;
};

struct ExecutionCore {
    DeviceContext& device;
    const LoadedModelData& model;
    WorkspaceArena& work;
    LinearAttentionStatePool& linear_attention;
    const GdnReplayRecords* replay_records;
    qwen3_6::RoundState& io;
    Tensor& prefill_hidden;
    std::uint32_t prefill_chunk;
    ProposalHead proposal_head;
    // YaRN rotary override, indexed by RANK: `[r]` names the table resident on rank r's own device
    // (`ProgramImplCore::rope_frequency`). Every TextContext built from this core forwards it to
    // its text rope call sites, MTP ones included. All-null is the native constant-table path,
    // bit-for-bit, which is why the default value is the pre-YaRN behavior.
    std::array<ops::RopeFrequencyOverride, kMaximumDevices> rope_frequency{};
    // Non-origin rank bundles, indexed by rank. [0] is unused (rank 0 is the references above);
    // [1, tp) are set, and [tp, kMaximumDevices) stay null. Each rank captures and replays its own
    // graph on its own stream, so no cross-device capture descriptor is needed here.
    std::array<const TpPeerCore*, kMaximumDevices> peers{};
    int tp = 1;
};

// Assembles the TextContext-side view of every non-origin rank. Entry `rank` is populated for
// `1 <= rank < execution.tp`; every other entry is empty (tp == 1 therefore yields all-empty).
[[nodiscard]] inline std::array<std::optional<TpExecution>, kMaximumDevices>
tp_execution(const ExecutionCore& execution) {
    std::array<std::optional<TpExecution>, kMaximumDevices> out{};
    for (int rank = 1; rank < execution.tp; ++rank) {
        const TpPeerCore* peer = execution.peers[static_cast<std::size_t>(rank)];
        if (peer == nullptr) { continue; }
        TpExecution card;
        card.execution      = peer->execution;
        card.events         = peer->events;
        card.device         = peer->device;
        card.weights        = peer->model;
        card.work           = peer->work;
        card.state          = peer->linear_attention;
        card.io             = peer->io;
        card.prefill_hidden = peer->prefill_hidden;
        card.batch_kv       = peer->text_cache;
        card.batch_mtp_kv   = peer->mtp_cache;
        card.replay_records = peer->replay_records;
        out[static_cast<std::size_t>(rank)] = card;
    }
    return out;
}

struct PrefillContext {
    ExecutionCore execution;
    qwen3_6::PagedKVCacheView text_kv;
    qwen3_6::PagedKVCacheView mtp_kv;
    // Per-rank peer views of each non-origin rank's per-sequence MTP KV window, indexed by rank;
    // empty at tp == 1 and when MTP is off. The text prefill needs no peer twin because it drives
    // the BATCH cache view plus table rows, but the MTP prefill appends and reads through the
    // per-sequence execution view.
    std::array<qwen3_6::PagedKVCacheView, kMaximumDevices> mtp_kv_peer{};
    const qwen3_6::PagedKVCache& text_cache;
    const qwen3_6::PagedKVCache* mtp_cache;
    DFlashPersistentState* dflash;
    std::uint32_t text_kv_base;
    const ops::SamplingConfig* sampling;
    Tensor* rewrite_checkpoint_hidden;
    std::int32_t current_state_slot                         = 0;
    std::int32_t rewrite_checkpoint_state_slot              = 0;
    std::uint32_t mtp_proposal_extent                       = 0;
    const qwen3_6::DFlashDecodeIngress* dflash_host_ingress = nullptr;
};

struct OrdinaryBatchContext {
    ExecutionCore execution;
    const qwen3_6::PagedKVCache& text_cache;
    qwen3_6::OrdinaryDecodeState& frame;
    const qwen3_6::OrdinaryDecodeIngress& host_ingress;
    qwen3_6::OrdinaryDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
    // Each non-origin rank's own pinned copy of `host_ingress`, indexed by rank, with every row's
    // sampling counter pointer nulled (ProgramImplCore::publish_peer_ordinary_ingress). Required
    // whenever `execution.tp` is engaged; null entries at tp1.
    std::array<qwen3_6::OrdinaryDecodeIngress*, kMaximumDevices> peer_host_ingress{};
};

struct MixedBatchContext {
    ExecutionCore execution;
    const qwen3_6::PagedKVCache& text_cache;
    qwen3_6::OrdinaryDecodeState& frame;
    const qwen3_6::MixedDecodeControl& host_control;
    qwen3_6::OrdinaryDecodeEgress& host_egress;
    std::int32_t* host_bonus_token;
    Tensor& continuation_hidden_store;
    // The flat [T] token stream and every parallel control tensor, in one host copy each. All of
    // them are rank-invariant logical values (positions, table-row indices, linear-state slots), so
    // unlike the ordinary ingress there is no per-rank host mirror: the schedule uploads this one
    // copy into EVERY rank's own mixed frame. `host_tokens`/`host_cache_positions`/
    // `host_rope_positions` are sized T; the rest are sized B (decode rows).
    std::span<const std::int32_t> host_tokens;
    std::span<const std::int32_t> host_cache_positions;
    std::span<const std::int32_t> host_rope_positions;
    std::span<const std::int32_t> host_decode_kv_rows;
    std::span<const std::int32_t> host_decode_linear_slots;
    // The owner's own text KV-table row, identical on every rank.
    std::int32_t prefill_kv_row = -1;
    // Owner rewrite-checkpoint capture (M2). When `rewrite_checkpoint_column >= 0`, the mixed pass
    // copies the raw pre-final-norm hidden at that flat column into `rewrite_checkpoint_hidden`
    // (rank 0) and copies the owner's GDN state into `rewrite_checkpoint_state_slot` (all ranks).
    Tensor* rewrite_checkpoint_hidden          = nullptr;
    std::int32_t rewrite_checkpoint_column     = -1;
    std::int32_t rewrite_checkpoint_state_slot = -1;
};

struct MtpBatchContext {
    ExecutionCore execution;
    const qwen3_6::PagedKVCache& text_cache;
    const qwen3_6::PagedKVCache& mtp_cache;
    qwen3_6::MtpDecodeState& frame;
    const qwen3_6::MtpDecodeIngress& host_ingress;
    qwen3_6::MtpDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
};

struct DFlashBatchContext {
    ExecutionCore execution;
    const qwen3_6::PagedKVCache& text_cache;
    DFlashPersistentState& dflash;
    qwen3_6::DFlashDecodeState& frame;
    const qwen3_6::DFlashDecodeIngress& host_ingress;
    qwen3_6::DFlashDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
    bool greedy_target = false;
};

struct DFlashAppendContext {
    ExecutionCore execution;
    DFlashPersistentState& dflash;
};

struct MtpGqaEnvelopes {
    ops::GqaExecutionEnvelope target_verify;
    ops::GqaExecutionEnvelope batch;
    std::array<ops::GqaExecutionEnvelope, kMaximumMtpDraftTokens - 1> ar;
};

struct DFlashEnvelopes {
    ops::SwaContextExecutionEnvelope local;
    ops::GqaContextExecutionEnvelope full;
    ops::KVCacheAppendPrefixExecutionEnvelope append;
};

struct TargetVerifyFrameView {
    Tensor ids;
    Tensor cache_positions;
    Tensor rope_positions;
    Tensor valid_columns;
    Tensor kv_table_rows;
    Tensor lanes;
    Tensor target_hidden;
    Tensor target_logits;
    Tensor target_tokens;
    Tensor drafts;
    Tensor current_extents;
    Tensor frontiers;
    Tensor anchors;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    Tensor selected_hidden;
    const GdnReplayRecords* replay_records = nullptr;
    const ops::SamplingConfig* sampling    = nullptr;
    DFlashFeatureSink* feature_sink        = nullptr;
};

void configure_text_card(TextContext& card, const ExecutionCore& execution,
                         const ops::SamplingConfig* sampling, std::int32_t current_state_slot,
                         std::int32_t rewrite_checkpoint_state_slot,
                         std::uint32_t mtp_proposal_extent);
void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::GqaExecutionEnvelope envelope, bool greedy_target = false);
// tp >= 2 form. `peer[rank]` is rank `rank`'s identically-shaped view of ITS OWN frame, indexed by
// rank (entry 0 unused); the acceptance arithmetic is replicated there rather than transferred,
// because every one of its inputs is either the ingress record (copied to every frame) or the
// gathered logits (bit-identical on every rank). What is NOT replicated is rank 0's bookkeeping:
// the continuation-hidden scatter and the egress transfer stay on rank 0 alone.
void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          std::array<TargetVerifyFrameView, kMaximumDevices> peer,
                          ops::GqaExecutionEnvelope envelope, bool greedy_target = false);

[[nodiscard]] PrefillChunkResult prefill_text_chunk(
    PrefillContext& state, std::span<const TokenId> ids, std::uint32_t nominal_length,
    std::optional<std::uint32_t> rewrite_checkpoint_capture_frontier, bool finalize_at_end);

// L2 mixed unit: one eager flat pass over T = B decode rows + the owner's prefill chunk. Fills the
// ordinary frame's decode hidden/logits/sampled (+ the mixed bonus token on finalize).
void mixed_prefill_decode_batch(MixedBatchContext& state);

[[nodiscard]] PrefillChunkResult
prefill_multimodal_chunk(PrefillContext& state, const PreparedPromptData& prompt,
                         VisionPrefillSession& vision, std::uint32_t nominal_length,
                         std::optional<std::uint32_t> rewrite_checkpoint_capture_frontier,
                         bool finalize_at_end);

struct MtpBridgeInput {
    const Tensor* previous_hidden = nullptr;
    std::int32_t position         = 0;
    std::array<std::int32_t, 3> rope_position{};
};

void sample_from_hidden(PrefillContext& state, const Tensor& hidden, std::int32_t absolute_position,
                        std::int32_t purpose);
// Retained hidden is authoritative on rank 0, including partial MTP commit corrections. Copy it
// into every peer's prefill scratch only on resume; every stream protects its producer/read
// lifetime. Index 0 is the rank-0 view itself.
[[nodiscard]] std::array<Tensor, kMaximumDevices> resume_hidden(ExecutionCore& execution,
                                                                const Tensor& hidden);
void mtp_bridge_and_propose(PrefillContext& state, const Tensor& next_token,
                            const Tensor& previous_hidden, std::int32_t position,
                            std::span<const std::int32_t> rope_position, bool build_proposal,
                            const std::array<Tensor, kMaximumDevices>* next_embeddings = nullptr);
void mtp_bridge_multimodal(PrefillContext& state, const PreparedPromptData& prompt,
                           VisionPrefillSession& vision, const MtpBridgeInput& bridge);

// Executes one exact-B ordinary decode traversal. All request rows enter through the stable
// ordinary ingress, share one model schedule, publish continuation hidden by selector, and leave
// through one compact egress transfer.
void capture_ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                                   ops::GqaExecutionEnvelope envelope,
                                   DecodeGraphDefinition& definition);
void ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                           ops::GqaExecutionEnvelope envelope, DecodeGraphExecutable* executable);

// Executes one exact-B MTP verification/alignment/proposal transaction. Each row may carry a
// different current and next proposal extent while the model traversal remains batched.
void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              MtpGqaEnvelopes envelopes, DecodeGraphDefinition& definition);
void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                      MtpGqaEnvelopes envelopes, DecodeGraphExecutable* executable);

[[nodiscard]] DFlashFeatureSink
dflash_feature_sink(PrefillContext& state, DFlashFeatureSink::PrefillConsumer consume_prefill = {});
void dflash_append_context(DFlashAppendContext& state, const Tensor& features,
                           const Tensor& positions, const Tensor& commit_counts,
                           const Tensor& lanes, const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope);
void dflash_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_counts, const Tensor& lanes,
                           const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope);
void capture_dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size,
                                 std::uint32_t k, DFlashEnvelopes envelopes,
                                 ops::GqaExecutionEnvelope target_envelope,
                                 DecodeGraphDefinition& definition);
void dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                         DFlashEnvelopes envelopes, ops::GqaExecutionEnvelope target_envelope,
                         DecodeGraphExecutable* executable);

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
