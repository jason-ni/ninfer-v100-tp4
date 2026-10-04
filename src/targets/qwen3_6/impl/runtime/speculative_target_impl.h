#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::GqaExecutionEnvelope envelope, bool greedy_target) {
    if (frame.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    if (frame.feature_sink != nullptr) {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.lanes, envelope,
                                 frame.target_hidden, frame.target_logits, frame.target_tokens,
                                 *frame.feature_sink, greedy_target);
    } else {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.lanes, envelope,
                                 frame.target_hidden, frame.target_logits, frame.target_tokens,
                                 greedy_target);
    }
    if (greedy_target) {
        ops::speculative_accept_greedy_tokens(
            frame.target_tokens, frame.drafts, frame.current_extents, frame.frontiers,
            frame.anchors, frame.licensed_tokens, frame.licensed_counts, frame.accepted_drafts,
            TextConfig::token_domain, frame.sampling, execution.device.stream);
    } else {
        ops::speculative_accept_greedy_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.current_extents,
            frame.frontiers, frame.anchors, frame.licensed_tokens, frame.licensed_counts,
            frame.accepted_drafts, TextConfig::token_domain, frame.sampling, execution.work,
            execution.device.stream);
    }
    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,
                                            frame.selected_hidden, execution.device.stream);
    ops::scatter(frame.selected_hidden, frame.lanes, continuation_hidden_store,
                 execution.device.stream);
}

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          std::array<TargetVerifyFrameView, kMaximumDevices> peer,
                          ops::GqaExecutionEnvelope envelope, bool greedy_target) {
    if (execution.tp < 2) {
        throw std::logic_error("tensor-parallel target verify requires a peer");
    }
    if (frame.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    for (int rank = 1; rank < execution.tp; ++rank) {
        if (peer[rank].replay_records == nullptr) {
            throw std::logic_error("speculative target verify has no ReplaySSM record storage");
        }
        if (peer[rank].feature_sink != nullptr) {
            throw std::logic_error("tensor-parallel DFlash peer feature sink is unexpected");
        }
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);

    std::array<Tensor, kMaximumDevices> ids{};
    std::array<Tensor, kMaximumDevices> cache_positions{};
    std::array<Tensor, kMaximumDevices> rope_positions{};
    std::array<Tensor, kMaximumDevices> valid_columns{};
    std::array<Tensor, kMaximumDevices> kv_table_rows{};
    std::array<Tensor, kMaximumDevices> lanes{};
    std::array<Tensor, kMaximumDevices> target_hidden{};
    std::array<Tensor, kMaximumDevices> target_logits{};
    std::array<Tensor, kMaximumDevices> target_tokens{};
    for (int rank = 0; rank < execution.tp; ++rank) {
        const TargetVerifyFrameView& f = rank == 0 ? frame : peer[rank];
        ids[rank]            = f.ids;
        cache_positions[rank] = f.cache_positions;
        rope_positions[rank] = f.rope_positions;
        valid_columns[rank]  = f.valid_columns;
        kv_table_rows[rank]  = f.kv_table_rows;
        lanes[rank]          = f.lanes;
        target_hidden[rank]  = f.target_hidden;
        target_logits[rank]  = f.target_logits;
        target_tokens[rank]  = f.target_tokens;
    }
    if (frame.feature_sink != nullptr) {
        card.target_verify_batch(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                                 lanes, envelope, target_hidden, target_logits, target_tokens,
                                 *frame.feature_sink, greedy_target);
    } else {
        card.target_verify_batch(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                                 lanes, envelope, target_hidden, target_logits, target_tokens,
                                 greedy_target);
    }

    const ExecutionContext& ec = *execution.peers[1]->execution;
    std::array<WorkspaceArena*, kMaximumDevices> work{};
    std::array<TargetVerifyFrameView*, kMaximumDevices> views{};
    for (int rank = 0; rank < execution.tp; ++rank) {
        work[rank]  = rank == 0 ? &execution.work : execution.peers[rank]->work;
        views[rank] = rank == 0 ? &frame : &peer[rank];
    }
    for_each_rank(ec, [&](int rank) {
        const auto r              = static_cast<std::size_t>(rank);
        TargetVerifyFrameView& v  = *views[r];
        cudaStream_t stream       = ec.dev[rank]->stream;
        if (greedy_target) {
            ops::speculative_accept_greedy_tokens(
                v.target_tokens, v.drafts, v.current_extents, v.frontiers, v.anchors,
                v.licensed_tokens, v.licensed_counts, v.accepted_drafts, TextConfig::token_domain,
                v.sampling, stream);
        } else {
            ops::speculative_accept_greedy_drafts(
                v.target_tokens, v.target_logits, v.drafts, v.current_extents, v.frontiers,
                v.anchors, v.licensed_tokens, v.licensed_counts, v.accepted_drafts,
                TextConfig::token_domain, v.sampling, *work[r], stream);
        }
        ops::speculative_select_accepted_hidden(v.target_hidden, v.accepted_drafts,
                                                v.selected_hidden, stream);
    });
    ops::scatter(frame.selected_hidden, frame.lanes, continuation_hidden_store,
                 execution.device.stream);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
