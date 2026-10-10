#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scatter.h"

#include <optional>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

auto ordinary_batch_body(OrdinaryBatchContext& state, std::int32_t batch_size,
                         ops::GqaExecutionEnvelope envelope) {
    return [&state, batch_size, envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency)) {
            throw std::logic_error("ordinary decode batch state is incomplete");
        }

        qwen3_6::OrdinaryDecodeState& ordinary = state.frame;
        CUDA_CHECK(cudaMemcpyAsync(ordinary.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::OrdinaryDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));
        auto tp = tp_execution(state.execution);
        if (state.execution.tp > 1) {
            // Each peer rank decodes from ITS OWN copy of the same ingress record, so all ranks
            // read identical tokens, positions, KV rows and lane ids without a cross-device copy.
            // The copy is each peer's OWN pinned record, not rank 0's: the record ends with a
            // SamplingConfig per row whose `token_counts` is a rank-0 DEVICE address, and a peer
            // frame must never hold one: dereferencing a rank-0 device pointer from a peer is a
            // silent cross-device fault. Peers never sample -- the vocabulary-split output head
            // gathers to rank 0 and `ops::sample` below runs there alone -- so their copies carry
            // the configs with the counter pointers nulled.
            int previous = 0;
            CUDA_CHECK(cudaGetDevice(&previous));
            for (int rank = 1; rank < state.execution.tp; ++rank) {
                TpExecution& peer = *tp[rank];
                if (!peer.io->ordinary.has_value()) {
                    throw std::logic_error("tensor-parallel decode requires a peer ordinary frame");
                }
                if (state.peer_host_ingress[rank] == nullptr) {
                    throw std::logic_error("tensor-parallel decode requires a peer ingress record");
                }
                CUDA_CHECK(cudaSetDevice(peer.device->device));
                CUDA_CHECK(cudaMemcpyAsync(peer.io->ordinary->ingress.data,
                                           state.peer_host_ingress[rank],
                                           sizeof(qwen3_6::OrdinaryDecodeIngress),
                                           cudaMemcpyHostToDevice, peer.device->stream));
            }
            CUDA_CHECK(cudaSetDevice(previous));
        }

        TextContext card(state.execution.device, state.execution.model, state.execution.work,
                         state.execution.rope_frequency, {}, state.execution.linear_attention,
                         state.execution.io, state.execution.prefill_hidden,
                         state.execution.prefill_chunk, 0, {}, &state.text_cache, nullptr,
                         state.execution.tp > 1 ? &tp : nullptr);

        Tensor tokens          = ordinary.tokens.slice(0, 0, batch_size);
        Tensor cache_positions = ordinary.cache_positions.slice(0, 0, batch_size);
        Tensor rope_positions  = ordinary.rope_positions.slice(0, 0, batch_size);
        Tensor kv_rows         = ordinary.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor lanes           = ordinary.lanes.slice(0, 0, batch_size);
        Tensor hidden          = ordinary.hidden.slice(1, 0, batch_size);
        Tensor logits          = ordinary.logits.slice(1, 0, batch_size);
        Tensor sampled         = ordinary.sampled_tokens.slice(0, 0, batch_size);

        card.ordinary_decode_batch(tokens, cache_positions, rope_positions, kv_rows, lanes,
                                   envelope, hidden, logits);
        ops::scatter(hidden, lanes, state.continuation_hidden_store, state.execution.device.stream);
        ops::sample(logits, sampled, TextConfig::token_domain, ordinary.sampling, cache_positions,
                    ops::kSamplePurposeDecode, state.execution.work, state.execution.device.stream);
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, ordinary.egress.data,
                                   sizeof(qwen3_6::OrdinaryDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

} // namespace

void capture_ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                                   ops::GqaExecutionEnvelope envelope,
                                   DecodeGraphDefinition& definition) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    capture_graph(state, definition, body);
}

void ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                           ops::GqaExecutionEnvelope envelope, DecodeGraphExecutable* executable) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    run_prepared(state, executable, body);
}

void mixed_prefill_decode_batch(MixedBatchContext& state) {
    qwen3_6::OrdinaryDecodeState& ordinary = state.frame;
    if (!ordinary.mixed) { throw std::logic_error("mixed unit requires the mixed frame"); }
    auto& mixed = *ordinary.mixed;
    const qwen3_6::MixedDecodeControl& control = state.host_control;
    const std::int32_t B = control.decode_rows;
    const std::int32_t c = control.prefill_len;
    if (B <= 0 || B > static_cast<std::int32_t>(kMaximumConcurrency) || c <= 0) {
        throw std::logic_error("mixed unit control is invalid");
    }
    const std::int32_t T = B + c;
    if (T > mixed.tokens.ne[0] || B > mixed.decode_kv_table_rows.ne[0]) {
        throw std::logic_error("mixed unit exceeds its token capacity");
    }
    if (state.host_tokens.size() != static_cast<std::size_t>(T) ||
        state.host_cache_positions.size() != static_cast<std::size_t>(T) ||
        state.host_rope_positions.size() != static_cast<std::size_t>(T) ||
        state.host_decode_kv_rows.size() != static_cast<std::size_t>(B) ||
        state.host_decode_linear_slots.size() != static_cast<std::size_t>(B)) {
        throw std::logic_error("mixed unit host staging does not match the control");
    }
    auto tp = tp_execution(state.execution);
    {
        // Every rank's own mixed frame. The control tensors carry rank-invariant values, so one
        // host copy feeds them all; only rank 0's frame receives the control block, whose sampling
        // rows carry rank-0 device pointers that a peer frame must never hold.
        CurrentDevice restore;
        for (int rank = 0; rank < state.execution.tp; ++rank) {
            qwen3_6::OrdinaryDecodeState::MixedState* frame = &mixed;
            cudaStream_t stream = state.execution.device.stream;
            if (rank != 0) {
                TpExecution& peer = *tp[rank];
                if (!peer.io->ordinary || !peer.io->ordinary->mixed) {
                    throw std::logic_error("tensor-parallel mixed unit requires a peer mixed frame");
                }
                frame = &*peer.io->ordinary->mixed;
                CUDA_CHECK(cudaSetDevice(peer.device->device));
                stream = peer.device->stream;
            }
            const auto upload = [&](void* destination, const void* source, std::size_t bytes) {
                CUDA_CHECK(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice,
                                           stream));
            };
            upload(frame->tokens.data, state.host_tokens.data(),
                   static_cast<std::size_t>(T) * sizeof(std::int32_t));
            upload(frame->cache_positions.data, state.host_cache_positions.data(),
                   static_cast<std::size_t>(T) * sizeof(std::int32_t));
            upload(frame->rope_positions.data, state.host_rope_positions.data(),
                   static_cast<std::size_t>(T) * sizeof(std::int32_t));
            upload(frame->decode_kv_table_rows.data, state.host_decode_kv_rows.data(),
                   static_cast<std::size_t>(B) * sizeof(std::int32_t));
            upload(frame->decode_linear_state_slots.data, state.host_decode_linear_slots.data(),
                   static_cast<std::size_t>(B) * sizeof(std::int32_t));
            upload(frame->prefill_kv_table_row.data, &state.prefill_kv_row,
                   sizeof(std::int32_t));
        }
    }
    cudaStream_t stream = state.execution.device.stream;
    CUDA_CHECK(cudaMemcpyAsync(mixed.control.data, &control, sizeof(qwen3_6::MixedDecodeControl),
                               cudaMemcpyHostToDevice, stream));

    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.execution.rope_frequency, {}, state.execution.linear_attention,
                     state.execution.io, state.execution.prefill_hidden,
                     state.execution.prefill_chunk, 0, {}, &state.text_cache, nullptr,
                     state.execution.tp > 1 ? &tp : nullptr);

    Tensor ids     = mixed.tokens.slice(0, 0, T);
    Tensor cpos    = mixed.cache_positions.slice(0, 0, T);
    Tensor rpos    = mixed.rope_positions.slice(0, 0, T);
    Tensor dkv     = mixed.decode_kv_table_rows.slice(0, 0, B);
    Tensor dslots  = mixed.decode_linear_state_slots.slice(0, 0, B);
    Tensor hidden  = ordinary.hidden.slice(1, 0, B);
    Tensor logits  = ordinary.logits.slice(1, 0, B);
    Tensor sampled = ordinary.sampled_tokens.slice(0, 0, B);
    Tensor lanes   = mixed.lanes.slice(0, 0, B);

    const ops::GqaExecutionEnvelope decode_envelope{control.decode_min_visible,
                                                    control.decode_max_visible};
    const ops::GqaExecutionEnvelope prefill_envelope{control.prefill_min_visible,
                                                     control.prefill_max_visible};
    card.mixed_prefill_decode_batch(ids, cpos, rpos, dkv, dslots, decode_envelope,
                                    mixed.prefill_kv_table_row, control.prefill_linear_slot,
                                    prefill_envelope, control.prefill_finalize != 0, hidden, logits,
                                    mixed.bonus_logits, mixed.prefill_tail_hidden);
    ops::scatter(hidden, lanes, state.continuation_hidden_store, stream);
    ops::sample(logits, sampled, TextConfig::token_domain, mixed.sampling, cpos.slice(0, 0, B),
                ops::kSamplePurposeDecode, state.execution.work, stream);
    if (control.prefill_finalize != 0) {
        if (control.prefill_sampling_valid != 0) {
            ops::sample(mixed.bonus_logits, mixed.bonus_token, TextConfig::token_domain,
                        mixed.prefill_sampling, mixed.bonus_position, ops::kSamplePurposePrefill,
                        state.execution.work, stream);
        } else {
            ops::argmax(mixed.bonus_logits, mixed.bonus_token, TextConfig::token_domain, stream);
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, ordinary.egress.data,
                               sizeof(qwen3_6::OrdinaryDecodeEgress), cudaMemcpyDeviceToHost,
                               stream));
    if (control.prefill_finalize != 0 && state.host_bonus_token != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(state.host_bonus_token, mixed.bonus_token.data,
                                   sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream));
    }
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
