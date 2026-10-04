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

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
