#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/mtp_round.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

// One rank's window into ITS OWN MtpDecodeState, sliced to the round's batch. Both ranks are
// sliced by the same function so a shape mistake cannot apply to one device only -- which matters
// because rank 1's buffers live on the other GPU, where an out-of-bounds write is silent: a
// peer-side write that is exactly in bounds at batch 1 runs off the end of the buffer at batch 2
// with nothing on the local device noticing.
struct MtpRoundView {
    Tensor anchors;
    Tensor frontiers;
    Tensor budgets;
    Tensor current_extents;
    Tensor target_valid;
    Tensor current_drafts;
    Tensor target_rope;
    Tensor text_rows;
    Tensor mtp_rows;
    Tensor lanes;
    Tensor rope_deltas;
    Tensor verify_ids;
    Tensor target_positions;
    Tensor target_tokens;
    Tensor target_logits;
    Tensor target_hidden;
    Tensor selected_hidden;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted;
    Tensor next_extents;
    Tensor alignment_ids;
    Tensor alignment_hidden;
    Tensor ar_hidden;
    Tensor next_hidden;
    Tensor ar_positions;
    Tensor ar_rope_positions;
    Tensor ar_valid_columns;
    Tensor next_drafts;
    Tensor proposal_logits;
    const ops::SamplingConfig* sampling = nullptr;
};

MtpRoundView slice_mtp_frame(qwen3_6::MtpDecodeState& frame, std::int32_t batch_size) {
    MtpRoundView out;
    out.anchors           = frame.anchors.slice(0, 0, batch_size);
    out.frontiers         = frame.base_frontiers.slice(0, 0, batch_size);
    out.budgets           = frame.remaining_budgets.slice(0, 0, batch_size);
    out.current_extents   = frame.current_extents.slice(0, 0, batch_size);
    out.target_valid      = frame.target_valid_columns.slice(0, 0, batch_size);
    out.current_drafts    = frame.current_drafts.slice(1, 0, batch_size);
    out.target_rope       = frame.target_rope_positions.slice(1, 0, batch_size);
    out.text_rows         = frame.text_kv_table_rows.slice(0, 0, batch_size);
    out.mtp_rows          = frame.mtp_kv_table_rows.slice(0, 0, batch_size);
    out.lanes             = frame.lanes.slice(0, 0, batch_size);
    out.rope_deltas       = frame.rope_deltas.slice(0, 0, batch_size);
    out.verify_ids        = frame.verify_ids.slice(1, 0, batch_size);
    out.target_positions  = frame.target_positions.slice(1, 0, batch_size);
    out.target_tokens     = frame.target_argmax.slice(1, 0, batch_size);
    out.target_logits     = frame.target_logits.slice(2, 0, batch_size);
    out.target_hidden     = frame.target_hidden.slice(2, 0, batch_size);
    out.selected_hidden   = frame.target_continuation_hidden.slice(1, 0, batch_size);
    out.licensed_tokens   = frame.licensed_tokens.slice(1, 0, batch_size);
    out.licensed_counts   = frame.licensed_counts.slice(0, 0, batch_size);
    out.accepted          = frame.accepted_drafts.slice(0, 0, batch_size);
    out.next_extents      = frame.next_extents.slice(0, 0, batch_size);
    out.alignment_ids     = frame.alignment_ids.slice(1, 0, batch_size);
    out.alignment_hidden  = frame.alignment_hidden.slice(2, 0, batch_size);
    out.ar_hidden         = frame.ar_hidden.slice(1, 0, batch_size);
    out.next_hidden       = frame.next_hidden.slice(1, 0, batch_size);
    out.ar_positions      = frame.ar_positions.slice(0, 0, batch_size);
    out.ar_rope_positions = frame.ar_rope_positions.slice(0, 0, batch_size);
    out.ar_valid_columns  = frame.ar_valid_columns.slice(0, 0, batch_size);
    out.next_drafts       = frame.next_drafts.slice(0, 0, batch_size);
    out.proposal_logits   = frame.proposal_logits.slice(1, 0, batch_size);
    out.sampling          = frame.sampling;
    return out;
}

TargetVerifyFrameView verify_view(const MtpRoundView& v, const GdnReplayRecords* records) {
    return TargetVerifyFrameView{
        .ids             = v.verify_ids,
        .cache_positions = v.target_positions,
        .rope_positions  = v.target_rope,
        .valid_columns   = v.target_valid,
        .kv_table_rows   = v.text_rows,
        .lanes           = v.lanes,
        .target_hidden   = v.target_hidden,
        .target_logits   = v.target_logits,
        .target_tokens   = v.target_tokens,
        .drafts          = v.current_drafts,
        .current_extents = v.current_extents,
        .frontiers       = v.frontiers,
        .anchors         = v.anchors,
        .licensed_tokens = v.licensed_tokens,
        .licensed_counts = v.licensed_counts,
        .accepted_drafts = v.accepted,
        .selected_hidden = v.selected_hidden,
        .replay_records  = records,
        .sampling        = v.sampling,
    };
}

void mtp_bridge_parallel(PrefillContext& state, const Tensor& next_token,
                     const Tensor& previous_hidden, std::int32_t position,
                     std::span<const std::int32_t> rope_position, bool build_proposal,
                     const std::array<Tensor, kMaximumDevices>* next_embeddings) {
    auto tp = tp_execution(state.execution);
    for (int rank = 1; rank < state.execution.tp; ++rank) {
        TpExecution& peer = *tp[rank];
        peer.mtp_kv = state.mtp_kv_peer[rank];
        if (!peer.mtp_kv.valid() || !peer.io->mtp) {
            throw std::logic_error("tensor-parallel MTP bridge requires both KV windows");
        }
    }
    state.execution.work.reset();
    for (int rank = 1; rank < state.execution.tp; ++rank) {
        state.execution.peers[rank]->work->reset();
    }
    const auto restored = resume_hidden(state.execution, previous_hidden);
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.execution.rope_frequency, state.text_kv,
                     state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache, &tp);
    configure_text_card(card, state.execution, state.sampling, state.current_state_slot,
                        state.rewrite_checkpoint_state_slot, state.mtp_proposal_extent);

    std::array<WorkspaceArena*, kMaximumDevices> work{};
    std::array<qwen3_6::RoundState*, kMaximumDevices> io{};
    std::array<Tensor, kMaximumDevices> positions, rope, ar_hidden, logits, ar_positions;
    const ExecutionContext& ec = *tp[1]->execution;
    for (int rank = 0; rank < state.execution.tp; ++rank) {
        work[rank] = rank == 0 ? &state.execution.work : tp[rank]->work;
        io[rank]   = rank == 0 ? &state.execution.io : tp[rank]->io;
    }
    for_each_rank(ec, [&](int rank) {
        const auto r       = static_cast<std::size_t>(rank);
        const auto stream  = ec.dev[rank]->stream;
        positions[r] = io[r]->mtp->target_positions.slice(0, 0, 1);
        ops::set_i32_scalar(positions[r], position, stream);
        rope[r] = work[r]->alloc(DType::I32, {1, 3});
        CUDA_CHECK(cudaMemcpyAsync(rope[r].data, rope_position.data(), rope_position.size_bytes(),
                                   cudaMemcpyHostToDevice, stream));
        ar_hidden[r]    = io[r]->mtp->ar_hidden;
        logits[r]       = io[r]->logits.slice(1, 0, 1);
        ar_positions[r] = io[r]->mtp->position.slice(0, 0, 1);
    });
    Tensor draft0 = state.execution.io.mtp->draft_tokens.slice(0, 0, 1);
    const auto visible = static_cast<std::uint32_t>(position + 1);
    card.mtp_forward_batch(next_token, restored, positions, rope, {visible, visible}, ar_hidden,
                           build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
                           build_proposal ? &draft0 : nullptr, next_embeddings);
    if (build_proposal) {
        for_each_rank(ec, [&](int rank) {
            const auto r = static_cast<std::size_t>(rank);
            ops::set_i32_scalar(ar_positions[r], position + 1, ec.dev[rank]->stream);
        });
        for (int i = 1; i < static_cast<int>(state.mtp_proposal_extent); ++i) {
            Tensor previous_token = state.execution.io.mtp->draft_tokens.slice(0, i - 1, 1);
            Tensor next_draft = state.execution.io.mtp->draft_tokens.slice(0, i, 1);
            std::array<Tensor, kMaximumDevices> next_hidden{};
            for (int rank = 0; rank < state.execution.tp; ++rank) {
                next_hidden[rank] = rank == 0
                                        ? state.execution.prefill_hidden.slice(1, i, 1)
                                        : tp[rank]->prefill_hidden->slice(1, i, 1);
            }
            const auto ar_visible = static_cast<std::uint32_t>(position + i + 1);
            card.mtp_forward_ar_step(previous_token, ar_hidden, ar_positions,
                                     {ar_visible, ar_visible}, next_hidden, logits, next_draft);
            for_each_rank(ec, [&](int rank) {
                const auto r       = static_cast<std::size_t>(rank);
                const auto stream  = ec.dev[rank]->stream;
                CUDA_CHECK(cudaMemcpyAsync(ar_hidden[r].data, next_hidden[r].data,
                                           ar_hidden[r].bytes(), cudaMemcpyDeviceToDevice, stream));
                ops::increment_i32_scalar(ar_positions[r], stream);
            });
        }
    }
    // The following suffix prefill resets every arena and may replace the staging hidden. Retire
    // all bridge streams here, including a bridge with no proposal or cross-rank logit gather.
    state.execution.device.synchronize();
    for (int rank = 1; rank < state.execution.tp; ++rank) {
        state.execution.peers[rank]->device->synchronize();
    }
    state.execution.work.reset();
    for (int rank = 1; rank < state.execution.tp; ++rank) {
        state.execution.peers[rank]->work->reset();
    }
}

} // namespace

void mtp_bridge_and_propose(PrefillContext& state, const Tensor& next_token,
                            const Tensor& previous_hidden, std::int32_t position,
                            std::span<const std::int32_t> rope_position, bool build_proposal,
                            const std::array<Tensor, kMaximumDevices>* next_embeddings) {
    if (!state.mtp_kv.valid() || !state.execution.io.mtp) {
        throw std::logic_error("MTP bridge requires MTP storage");
    }
    if (rope_position.size() != 3) {
        throw std::invalid_argument("MTP bridge requires one three-axis rope position");
    }
    if (build_proposal &&
        (state.mtp_proposal_extent == 0 ||
         state.mtp_proposal_extent >
             static_cast<std::uint32_t>(state.execution.io.mtp->draft_tokens.ne[0]))) {
        throw std::logic_error("MTP bridge proposal extent is outside the configured window");
    }
    if (state.execution.tp > 1) {
        mtp_bridge_parallel(state, next_token, previous_hidden, position, rope_position,
                            build_proposal, next_embeddings);
        return;
    }
    const Tensor* next_embedding = next_embeddings != nullptr ? &(*next_embeddings)[0] : nullptr;
    state.execution.work.reset();
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.execution.rope_frequency, state.text_kv,
                     state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.current_state_slot,
                        state.rewrite_checkpoint_state_slot, state.mtp_proposal_extent);

    Tensor position_view = state.execution.io.mtp->target_positions.slice(0, 0, 1);
    ops::set_i32_scalar(position_view, position, state.execution.device.stream);
    Tensor mtp_hidden         = state.execution.io.mtp->ar_hidden;
    Tensor logits             = state.execution.io.logits.slice(1, 0, 1);
    Tensor draft0             = state.execution.io.mtp->draft_tokens.slice(0, 0, 1);
    Tensor rope_position_view = state.execution.work.alloc(DType::I32, {1, 3});
    CUDA_CHECK(cudaMemcpyAsync(rope_position_view.data, rope_position.data(),
                               rope_position.size_bytes(), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    const auto bridge_visible = static_cast<std::uint32_t>(position + 1);
    const ops::GqaExecutionEnvelope bridge_envelope{bridge_visible, bridge_visible};
    card.mtp_forward_batch(next_token, previous_hidden, position_view, bridge_envelope, mtp_hidden,
                           build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
                           build_proposal ? &draft0 : nullptr, &rope_position_view, next_embedding);
    if (!build_proposal) { return; }

    Tensor ar_position = state.execution.io.mtp->position.slice(0, 0, 1);
    ops::set_i32_scalar(ar_position, position + 1, state.execution.device.stream);
    for (int i = 1; i < static_cast<int>(state.mtp_proposal_extent); ++i) {
        Tensor previous_token = state.execution.io.mtp->draft_tokens.slice(0, i - 1, 1);
        Tensor next_draft     = state.execution.io.mtp->draft_tokens.slice(0, i, 1);
        Tensor next_hidden    = state.execution.prefill_hidden.slice(1, i, 1);
        const auto visible    = static_cast<std::uint32_t>(position + i + 1);
        const ops::GqaExecutionEnvelope envelope{visible, visible};
        card.mtp_forward_ar_step(previous_token, state.execution.io.mtp->ar_hidden, ar_position,
                                 envelope, next_hidden, logits, next_draft);
        CUDA_CHECK(cudaMemcpyAsync(state.execution.io.mtp->ar_hidden.data, next_hidden.data,
                                   state.execution.io.mtp->ar_hidden.bytes(),
                                   cudaMemcpyDeviceToDevice, state.execution.device.stream));
        ops::increment_i32_scalar(ar_position, state.execution.device.stream);
    }
}

namespace {

// The alignment + proposal pipeline after a target verify (or after the M3 mixed shared pass, whose
// target hidden/logits/argmax have already been produced). The acceptance has run; this prepares
// the next round, runs the MTP alignment forward, selects the accepted hidden, proposes the draft
// window, and walks the autoregressive proposal steps. Shared verbatim by the isolated MTP round
// and the mixed MTP unit so the two cannot drift.
void mtp_propose_round(ExecutionCore& execution, TextContext& card, MtpRoundView& v,
                       std::int32_t batch_size, std::uint32_t k, MtpGqaEnvelopes envelopes,
                       std::int32_t max_context) {
    auto tp = tp_execution(execution);
    if (execution.tp <= 1) {
        ops::mtp_prepare_next_round(v.verify_ids, v.anchors, v.accepted, v.frontiers, v.budgets,
                                    v.licensed_counts, v.rope_deltas, v.alignment_ids,
                                    v.next_extents, v.ar_positions, v.ar_rope_positions,
                                    v.ar_valid_columns, max_context, execution.device.stream);
        card.mtp_forward_decode_batch(v.alignment_ids, v.target_hidden, v.target_positions,
                                      v.target_rope, v.licensed_counts, v.mtp_rows,
                                      envelopes.batch, v.alignment_hidden);
        ops::speculative_select_accepted_hidden(v.alignment_hidden, v.accepted, v.ar_hidden,
                                                execution.device.stream);
        Tensor draft0 = v.next_drafts.slice(1, 0, 1).view({batch_size});
        card.mtp_propose_batch(v.ar_hidden, v.proposal_logits, draft0);
        for (std::uint32_t step = 0; step + 1 < k; ++step) {
            Tensor previous =
                v.next_drafts.slice(1, static_cast<std::int32_t>(step), 1).view({batch_size});
            Tensor next = v.next_drafts.slice(1, static_cast<std::int32_t>(step + 1), 1)
                              .view({batch_size});
            Tensor position =
                v.ar_positions.slice(1, static_cast<std::int32_t>(step), 1).view({1, batch_size});
            Tensor rope = v.ar_rope_positions.slice(1, static_cast<std::int32_t>(step), 1)
                              .view({1, batch_size});
            Tensor valid = v.ar_valid_columns.slice(1, static_cast<std::int32_t>(step), 1)
                               .view({batch_size});
            Tensor previous_batch    = previous.view({1, batch_size});
            Tensor hidden_batch      = v.ar_hidden.view({TextConfig::hidden, 1, batch_size});
            Tensor next_hidden_batch = v.next_hidden.view({TextConfig::hidden, 1, batch_size});
            card.mtp_forward_decode_batch(previous_batch, hidden_batch, position, rope, valid,
                                          v.mtp_rows, envelopes.ar[step], next_hidden_batch);
            card.mtp_propose_batch(v.next_hidden, v.proposal_logits, next);
            CUDA_CHECK(cudaMemcpyAsync(v.ar_hidden.data, v.next_hidden.data, v.ar_hidden.bytes(),
                                       cudaMemcpyDeviceToDevice, execution.device.stream));
        }
        return;
    }

    const ExecutionContext& ec = *tp[1]->execution;
    std::array<MtpRoundView, kMaximumDevices> peer_round{};
    std::array<MtpRoundView*, kMaximumDevices> views{};
    views[0] = &v;
    for (int rank = 1; rank < execution.tp; ++rank) {
        peer_round[rank] = slice_mtp_frame(*tp[rank]->io->mtp_decode, batch_size);
        views[rank]      = &peer_round[rank];
    }
    for_each_rank(ec, [&](int rank) {
        MtpRoundView& r = *views[static_cast<std::size_t>(rank)];
        ops::mtp_prepare_next_round(r.verify_ids, r.anchors, r.accepted, r.frontiers, r.budgets,
                                    r.licensed_counts, r.rope_deltas, r.alignment_ids,
                                    r.next_extents, r.ar_positions, r.ar_rope_positions,
                                    r.ar_valid_columns, max_context, ec.dev[rank]->stream);
    });
    std::array<Tensor, kMaximumDevices> target_hidden{};
    std::array<Tensor, kMaximumDevices> target_positions{};
    std::array<Tensor, kMaximumDevices> target_rope{};
    std::array<Tensor, kMaximumDevices> licensed_counts{};
    std::array<Tensor, kMaximumDevices> mtp_rows{};
    std::array<Tensor, kMaximumDevices> alignment_hidden{};
    for (int rank = 0; rank < execution.tp; ++rank) {
        MtpRoundView& r        = *views[rank];
        target_hidden[rank]    = r.target_hidden;
        target_positions[rank] = r.target_positions;
        target_rope[rank]      = r.target_rope;
        licensed_counts[rank]  = r.licensed_counts;
        mtp_rows[rank]         = r.mtp_rows;
        alignment_hidden[rank] = r.alignment_hidden;
    }
    card.mtp_forward_decode_batch(v.alignment_ids, target_hidden, target_positions, target_rope,
                                  licensed_counts, mtp_rows, envelopes.batch, alignment_hidden);
    for_each_rank(ec, [&](int rank) {
        MtpRoundView& r = *views[static_cast<std::size_t>(rank)];
        ops::speculative_select_accepted_hidden(r.alignment_hidden, r.accepted, r.ar_hidden,
                                                ec.dev[rank]->stream);
    });

    std::array<Tensor, kMaximumDevices> proposal_logits{};
    std::array<Tensor, kMaximumDevices> proposal_hidden{};
    for (int rank = 0; rank < execution.tp; ++rank) {
        proposal_logits[rank] = views[rank]->proposal_logits;
        proposal_hidden[rank] = views[rank]->ar_hidden;
    }
    Tensor draft0 = v.next_drafts.slice(1, 0, 1).view({batch_size});
    card.mtp_propose_batch(proposal_hidden, proposal_logits, draft0);
    for (std::uint32_t step = 0; step + 1 < k; ++step) {
        Tensor previous =
            v.next_drafts.slice(1, static_cast<std::int32_t>(step), 1).view({batch_size});
        Tensor next = v.next_drafts.slice(1, static_cast<std::int32_t>(step + 1), 1)
                          .view({batch_size});
        std::array<Tensor, kMaximumDevices> position;
        std::array<Tensor, kMaximumDevices> rope;
        std::array<Tensor, kMaximumDevices> valid;
        std::array<Tensor, kMaximumDevices> hidden_batch;
        std::array<Tensor, kMaximumDevices> next_hidden_batch;
        std::array<Tensor, kMaximumDevices> next_hidden_values;
        for (int rank = 0; rank < execution.tp; ++rank) {
            MtpRoundView& view = *views[rank];
            position[rank] = view.ar_positions.slice(1, static_cast<std::int32_t>(step), 1)
                                 .view({1, batch_size});
            rope[rank] = view.ar_rope_positions.slice(1, static_cast<std::int32_t>(step), 1)
                             .view({1, batch_size});
            valid[rank] = view.ar_valid_columns.slice(1, static_cast<std::int32_t>(step), 1)
                              .view({batch_size});
            hidden_batch[rank] = view.ar_hidden.view({TextConfig::hidden, 1, batch_size});
            next_hidden_batch[rank] =
                view.next_hidden.view({TextConfig::hidden, 1, batch_size});
            next_hidden_values[rank] = view.next_hidden;
        }
        Tensor previous_batch = previous.view({1, batch_size});
        card.mtp_forward_decode_batch(previous_batch, hidden_batch, position, rope, valid, mtp_rows,
                                      envelopes.ar[step], next_hidden_batch);
        card.mtp_propose_batch(next_hidden_values, proposal_logits, next);
        for_each_rank(ec, [&](int rank) {
            MtpRoundView& r = *views[static_cast<std::size_t>(rank)];
            CUDA_CHECK(cudaMemcpyAsync(r.ar_hidden.data, r.next_hidden.data, r.ar_hidden.bytes(),
                                       cudaMemcpyDeviceToDevice, ec.dev[rank]->stream));
        });
    }
}

} // namespace

auto mtp_decode_batch_body(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                           MtpGqaEnvelopes envelopes) {
    return [&state, batch_size, k, envelopes] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kMtpDecodeMaximumDrafts) {
            throw std::logic_error("MTP decode batch state is incomplete");
        }

        qwen3_6::MtpDecodeState& frame = state.frame;
        const std::int32_t width       = static_cast<std::int32_t>(k) + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::MtpDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));
        auto tp = tp_execution(state.execution);
        if (state.execution.tp > 1) {
            // Each peer rank runs the round from ITS OWN copy of the same ingress record.
            // Everything a peer needs that is not in the ingress -- verify ids, target positions,
            // the accepted count, the next round's AR positions -- is DERIVED from it by the same
            // deterministic Ops, run again on that device, rather than transferred: the only other
            // inputs are the gathered logits, which are bit-identical on every rank.
            //
            // A peer uploads ITS OWN ingress record, not rank 0's. The copies differ in exactly one
            // field per row -- `sampling[row].token_counts`, which must name that rank's penalty
            // counter lane. `speculative_accept_greedy_drafts` reads and atomically writes that
            // pointer in sampling mode, so handing a peer a pointer into rank 0's arena is an
            // illegal access without peer mapping and a silent double-increment with it. Every
            // other byte is identical, which keeps the replicated accepts in step.
            const CurrentDevice restore;
            for (int rank = 1; rank < state.execution.tp; ++rank) {
                TpExecution& peer = *tp[rank];
                if (!peer.io->mtp_decode.has_value()) {
                    throw std::logic_error("tensor-parallel MTP decode requires a peer frame");
                }
                const qwen3_6::MtpDecodeIngress* peer_ingress =
                    state.execution.peers[rank]->mtp_host_ingress;
                if (peer_ingress == nullptr) {
                    throw std::logic_error(
                        "tensor-parallel MTP decode requires a peer ingress record");
                }
                CUDA_CHECK(cudaSetDevice(peer.device->device));
                CUDA_CHECK(cudaMemcpyAsync(peer.io->mtp_decode->ingress.data, peer_ingress,
                                           sizeof(qwen3_6::MtpDecodeIngress),
                                           cudaMemcpyHostToDevice, peer.device->stream));
            }
        }

        TextContext card(state.execution.device, state.execution.model, state.execution.work,
                         state.execution.rope_frequency, {}, state.execution.linear_attention,
                         state.execution.io, state.execution.prefill_hidden,
                         state.execution.prefill_chunk, 0, {}, &state.text_cache,
                         &state.mtp_cache, state.execution.tp > 1 ? &tp : nullptr);

        MtpRoundView v = slice_mtp_frame(frame, batch_size);

        if (state.execution.tp <= 1) {
            ops::speculative_prepare_verify_inputs(v.anchors, v.current_drafts, v.frontiers,
                                                   v.current_extents, v.verify_ids,
                                                   v.target_positions,
                                                   state.execution.device.stream);
            target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                 verify_view(v, state.execution.replay_records),
                                 envelopes.target_verify);

            mtp_propose_round(state.execution, card, v, batch_size, k, envelopes,
                              static_cast<std::int32_t>(state.text_cache.max_context()));
        } else {
            const ExecutionContext& ec = *tp[1]->execution;
            std::array<MtpRoundView, kMaximumDevices> peer_round{};
            std::array<MtpRoundView*, kMaximumDevices> views{};
            views[0] = &v;
            for (int rank = 1; rank < state.execution.tp; ++rank) {
                peer_round[rank] = slice_mtp_frame(*tp[rank]->io->mtp_decode, batch_size);
                views[rank]      = &peer_round[rank];
            }
            for_each_rank(ec, [&](int rank) {
                MtpRoundView& r = *views[static_cast<std::size_t>(rank)];
                ops::speculative_prepare_verify_inputs(r.anchors, r.current_drafts, r.frontiers,
                                                       r.current_extents, r.verify_ids,
                                                       r.target_positions, ec.dev[rank]->stream);
            });
            std::array<TargetVerifyFrameView, kMaximumDevices> peer_views{};
            for (int rank = 1; rank < state.execution.tp; ++rank) {
                peer_views[rank] = verify_view(*views[rank], tp[rank]->replay_records);
            }
            target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                 verify_view(v, state.execution.replay_records), peer_views,
                                 envelopes.target_verify);
            mtp_propose_round(state.execution, card, v, batch_size, k, envelopes,
                              static_cast<std::int32_t>(state.text_cache.max_context()));
        }

        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_6::MtpDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              MtpGqaEnvelopes envelopes, DecodeGraphDefinition& definition) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    capture_graph(state, definition, body);
}

void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                      MtpGqaEnvelopes envelopes, DecodeGraphExecutable* executable) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    run_prepared(state, executable, body);
}

auto mixed_mtp_batch_body(MtpMixedBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                          MtpGqaEnvelopes envelopes) {
    return [&state, batch_size, k, envelopes] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kMtpDecodeMaximumDrafts) {
            throw std::logic_error("mixed MTP batch state is incomplete");
        }
        qwen3_6::MtpDecodeState& frame = state.frame;
        if (!frame.mixed) { throw std::logic_error("mixed MTP unit requires the mixed frame"); }
        qwen3_6::MtpMixedControl& control = state.host_control;
        const std::int32_t width          = static_cast<std::int32_t>(k) + 1;
        const std::int32_t B              = control.decode_rows;
        const std::int32_t c              = control.prefill_len;
        if (B != batch_size || control.decode_width != width || B <= 0 || c <= 0) {
            throw std::logic_error("mixed MTP control is invalid");
        }
        const std::int32_t decode_cols = B * width;
        const std::int32_t T           = decode_cols + c;
        auto& mixed                    = *frame.mixed;
        if (T > mixed.ids.ne[0]) {
            throw std::logic_error("mixed MTP exceeds its token capacity");
        }
        if (state.host_owner_tokens.size() != static_cast<std::size_t>(c) ||
            state.host_owner_positions.size() != static_cast<std::size_t>(c) ||
            state.host_owner_rope_positions.size() != static_cast<std::size_t>(c)) {
            throw std::logic_error("mixed MTP owner staging does not match the control");
        }
        auto tp = tp_execution(state.execution, state.mtp_kv_peer);
        const auto i32 = [](void* base, std::int32_t offset) {
            return static_cast<std::int32_t*>(base) + offset;
        };
        // Stage every rank's own frame from ITS OWN ingress record, then derive the slab ids and
        // positions with the same deterministic Ops the isolated MTP round uses. The flat stream's
        // slab portion is a straight copy of the prepared `verify_ids`/`target_positions`/
        // `target_rope_positions` regions, so the shared pass reads the slab columns in exactly the
        // order the isolated target verify would. The owner columns are uploaded from rank 0's
        // staging into every rank (rank-invariant values).
        const auto stage_rank = [&](int rank) {
            qwen3_6::MtpDecodeState& f =
                rank == 0 ? frame : *tp[static_cast<std::size_t>(rank)]->io->mtp_decode;
            const qwen3_6::MtpDecodeIngress* ing =
                rank == 0 ? &state.host_ingress
                          : state.execution.peers[rank]->mtp_host_ingress;
            if (ing == nullptr) {
                throw std::logic_error("mixed MTP requires every rank's ingress record");
            }
            cudaStream_t stream =
                rank == 0 ? state.execution.device.stream
                          : tp[static_cast<std::size_t>(rank)]->device->stream;
            CUDA_CHECK(cudaMemcpyAsync(f.ingress.data, ing, sizeof(qwen3_6::MtpDecodeIngress),
                                       cudaMemcpyHostToDevice, stream));
            if (!f.mixed) {
                throw std::logic_error("mixed MTP requires every rank's mixed frame");
            }
            qwen3_6::MtpDecodeState::MixedState& mx = *f.mixed;
            ops::speculative_prepare_verify_inputs(f.anchors, f.current_drafts, f.base_frontiers,
                                                   f.current_extents, f.verify_ids,
                                                   f.target_positions, stream);
            CUDA_CHECK(cudaMemcpyAsync(mx.ids.data, f.verify_ids.data,
                                       static_cast<std::size_t>(decode_cols) * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(mx.positions.data, f.target_positions.data,
                                       static_cast<std::size_t>(decode_cols) * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                mx.rope_positions.data, f.target_rope_positions.data,
                static_cast<std::size_t>(decode_cols) * sizeof(std::int32_t),
                cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(i32(mx.ids.data, decode_cols), state.host_owner_tokens.data(),
                                       static_cast<std::size_t>(c) * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(i32(mx.positions.data, decode_cols),
                                       state.host_owner_positions.data(),
                                       static_cast<std::size_t>(c) * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(i32(mx.rope_positions.data, decode_cols),
                                       state.host_owner_rope_positions.data(),
                                       static_cast<std::size_t>(c) * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(mx.owner_kv_table_row.data, &state.owner_kv_row,
                                       sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
        };

        TextContext card(state.execution.device, state.execution.model, state.execution.work,
                         state.execution.rope_frequency, {}, state.execution.linear_attention,
                         state.execution.io, state.execution.prefill_hidden,
                         state.execution.prefill_chunk, 0, state.mtp_kv, &state.text_cache,
                         &state.mtp_cache, state.execution.tp > 1 ? &tp : nullptr);
        if (state.execution.replay_records == nullptr) {
            throw std::logic_error("mixed MTP target verify has no ReplaySSM record storage");
        }
        card.set_gdn_state_action(GdnStateAction::RecordForReplay, state.execution.replay_records);
        card.set_mtp_proposal_extent(state.mtp_proposal_extent);

        {
            const CurrentDevice restore;
            for (int rank = 0; rank < state.execution.tp; ++rank) {
                if (rank != 0) { CUDA_CHECK(cudaSetDevice(tp[rank]->device->device)); }
                stage_rank(rank);
            }
            // The control block carries rank-0 device pointers (owner sampling config, bonus
            // position); only rank 0's frame receives it.
            CUDA_CHECK(cudaMemcpyAsync(mixed.control.data, &control,
                                       sizeof(qwen3_6::MtpMixedControl), cudaMemcpyHostToDevice,
                                       state.execution.device.stream));
        }

        const ops::GqaExecutionEnvelope decode_envelope = envelopes.target_verify;
        const ops::GqaExecutionEnvelope prefill_envelope{
            static_cast<std::uint32_t>(control.prefill_bonus_position),
            static_cast<std::uint32_t>(control.prefill_bonus_position)};
        Tensor slab_hidden = frame.target_hidden.slice(2, 0, B);
        Tensor slab_logits = frame.target_logits.slice(2, 0, B);
        Tensor slab_tokens = frame.target_argmax.slice(1, 0, B);
        card.mixed_mtp_batch(mixed.ids.slice(0, 0, T), mixed.positions.slice(0, 0, T),
                             mixed.rope_positions.slice(0, 0, T),
                             frame.text_kv_table_rows.slice(0, 0, B),
                             frame.lanes.slice(0, 0, B),
                             frame.target_valid_columns.slice(0, 0, B), decode_envelope,
                             mixed.owner_kv_table_row, control.prefill_linear_slot,
                             prefill_envelope, control.prefill_finalize != 0, width,
                             slab_hidden, slab_logits, slab_tokens, mixed.owner_bonus_logits,
                             mixed.owner_tail_hidden, state.rewrite_checkpoint_hidden,
                             state.rewrite_checkpoint_column, state.rewrite_checkpoint_state_slot);

        MtpRoundView v = slice_mtp_frame(frame, batch_size);
        if (state.execution.tp <= 1) {
            target_accept(state.execution, state.continuation_hidden_store,
                          verify_view(v, state.execution.replay_records));
        } else {
            const ExecutionContext& ec = *tp[1]->execution;
            std::array<MtpRoundView, kMaximumDevices> peer_round{};
            std::array<MtpRoundView*, kMaximumDevices> views{};
            views[0] = &v;
            for (int rank = 1; rank < state.execution.tp; ++rank) {
                peer_round[rank] = slice_mtp_frame(*tp[rank]->io->mtp_decode, batch_size);
                views[rank]      = &peer_round[rank];
            }
            std::array<TargetVerifyFrameView, kMaximumDevices> peer_views{};
            for (int rank = 1; rank < state.execution.tp; ++rank) {
                peer_views[rank] = verify_view(*views[rank], tp[rank]->replay_records);
            }
            target_accept(state.execution, state.continuation_hidden_store,
                          verify_view(v, state.execution.replay_records), peer_views);
        }
        mtp_propose_round(state.execution, card, v, batch_size, k, envelopes,
                          static_cast<std::int32_t>(state.text_cache.max_context()));

        if (control.prefill_finalize != 0) {
            // Owner tail: sample the bonus token (rank 0), then seed the owner's MTP draft window by
            // running the MTP alignment over the owner's final chunk. The shared target forward has
            // already produced the owner's post-final-norm hidden (in each rank's prefill hidden
            // buffer) and the bonus logits.
            if (control.prefill_sampling_valid != 0) {
                ops::sample(mixed.owner_bonus_logits, mixed.owner_bonus_token,
                            TextConfig::token_domain, mixed.owner_sampling, mixed.bonus_position,
                            ops::kSamplePurposePrefill, state.execution.work,
                            state.execution.device.stream);
            } else {
                ops::argmax(mixed.owner_bonus_logits, mixed.owner_bonus_token,
                            TextConfig::token_domain, state.execution.device.stream);
            }
            std::int32_t bonus = 0;
            CUDA_CHECK(cudaMemcpyAsync(&bonus, mixed.owner_bonus_token.data, sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, state.execution.device.stream));
            CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream));
            if (state.host_owner_bonus_token != nullptr) { *state.host_owner_bonus_token = bonus; }
            // Alignment ids: hidden t predicts token t+1, so the owner chunk's ids shift by one and
            // the sampled bonus is the final column.
            std::vector<std::int32_t> align_host(static_cast<std::size_t>(c));
            for (std::int32_t j = 0; j + 1 < c; ++j) {
                align_host[static_cast<std::size_t>(j)] = state.host_owner_tokens[j + 1];
            }
            align_host[static_cast<std::size_t>(c) - 1] = bonus;
            state.execution.work.reset();
            Tensor align_ids = state.execution.work.alloc(DType::I32, {c});
            CUDA_CHECK(cudaMemcpyAsync(align_ids.data, align_host.data(),
                                       static_cast<std::size_t>(c) * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, state.execution.device.stream));
            std::array<Tensor, kMaximumDevices> align_hidden;
            std::array<Tensor, kMaximumDevices> align_positions;
            std::array<Tensor, kMaximumDevices> align_rope;
            for (int rank = 0; rank < state.execution.tp; ++rank) {
                const auto r = static_cast<std::size_t>(rank);
                Tensor* ph   = rank == 0 ? &state.execution.prefill_hidden : tp[r]->prefill_hidden;
                align_hidden[r] = ph->slice(1, 0, c);
                qwen3_6::MtpDecodeState::MixedState& mx =
                    rank == 0 ? *frame.mixed : *tp[r]->io->mtp_decode->mixed;
                align_positions[r] = mx.positions.slice(0, decode_cols, c);
                align_rope[r]      = mx.rope_positions.slice(0, decode_cols, c);
            }
            const auto owner_visible = static_cast<std::uint32_t>(state.owner_position);
            const ops::GqaExecutionEnvelope owner_envelope{owner_visible, owner_visible};
            card.owner_mtp_align(align_ids, align_hidden, align_positions, align_rope, owner_envelope,
                                 state.owner_position);
        }

        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_6::MtpDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

void mixed_mtp_batch(MtpMixedBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                     MtpGqaEnvelopes envelopes) {
    auto body = mixed_mtp_batch_body(state, batch_size, k, envelopes);
    body();
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
