#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/program.h"

#include "targets/qwen3_6/impl/runtime/schedule.h"
#include "targets/qwen3_6/impl/runtime/yarn_rope.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

// Evicted prefix-cache pages return to every rank's text-KV pool; snapshot slots are statically
// partitioned from the linear-attention pool, so releasing one is bookkeeping only.
struct ProgramImplCore::PrefixCacheSink final : ninfer::prefix_cache::PrefixPageSink {
    explicit PrefixCacheSink(ProgramImplCore& core) noexcept : core_(core) {}
    void release_block_pages(const ninfer::prefix_cache::PoolPages& pages) noexcept override {
        core_.release_cached_pages(pages);
    }
    void release_tail_pages(const ninfer::prefix_cache::PoolPages& pages) noexcept override {
        core_.release_cached_pages(pages);
    }
    void release_snapshot_slot(ninfer::prefix_cache::SlotId) noexcept override {}

private:
    ProgramImplCore& core_;
};

namespace {

using Clock = std::chrono::steady_clock;

std::int32_t checked_i32(std::uint32_t value, const char* label) {
    if (value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

std::array<std::int32_t, 3> prompt_rope_position(const PreparedPromptData& prompt,
                                                 std::uint32_t token) {
    const std::size_t tokens = prompt.token_ids.size();
    if (token >= tokens || prompt.positions.size() != 3 * tokens) {
        throw std::invalid_argument("MTP bridge position is outside prepared prompt metadata");
    }
    return {prompt.positions[token], prompt.positions[tokens + token],
            prompt.positions[2 * tokens + token]};
}

// Makes one device current for the duration of a statement and restores the caller's. Peer-side
// pool operations issue memsets and copies on the peer's stream, which must be done with the
// peer's device current.
// Sums this device's GDN recurrent + convolution state. At tp 2 the pool is planned with halved
// value heads and halved conv channels, so this is already the per-device number.
// The single construction of this Program's YaRN parameters. Both readers -- the `yarn_mscale`
// member initializer and the per-device table upload -- go through here, so the factor/origin/theta
// /rotary-pair tuple the descriptor's mscale describes is by construction the tuple the uploaded
// frequencies were computed from. Only meaningful when `plan.rope_mode == RopeMode::Yarn`.
qwen3_6::detail::YarnParams plan_yarn_params(const SequencePlanImpl& plan) {
    return qwen3_6::detail::YarnParams{
        .factor       = static_cast<float>(plan.yarn_factor),
        .original_max = static_cast<int>(plan.yarn_origin),
        .theta        = TextConfig::rope_theta,
        .rotary_pairs = TextConfig::rotary_dim / 2,
    };
}

std::size_t linear_attention_state_bytes(const LinearAttentionStatePoolLayout& layout) {
    std::size_t total = 0;
    for (const LayoutRegion& region : layout.conv) { total += region.bytes; }
    for (const LayoutRegion& region : layout.recurrent) { total += region.bytes; }
    return total;
}

class ScopedDevice {
public:
    explicit ScopedDevice(int device) {
        CUDA_CHECK(cudaGetDevice(&previous_));
        CUDA_CHECK(cudaSetDevice(device));
    }

    ~ScopedDevice() { (void)cudaSetDevice(previous_); }

    ScopedDevice(const ScopedDevice&)            = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;

private:
    int previous_ = 0;
};

schedule::MtpGqaEnvelopes mtp_gqa_envelopes(std::uint32_t max_frontier, std::uint32_t k,
                                            std::uint32_t capacity) {
    const auto visible = [capacity](std::uint64_t value) {
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(capacity, value));
    };
    schedule::MtpGqaEnvelopes out;
    out.target_verify = {1, visible(static_cast<std::uint64_t>(max_frontier) + k + 1ULL)};
    out.batch         = out.target_verify;
    for (std::uint32_t step = 0; step + 1 < k; ++step) {
        out.ar[step] = {1, visible(static_cast<std::uint64_t>(max_frontier) + k + step + 2ULL)};
    }
    return out;
}

schedule::DFlashEnvelopes dflash_envelopes(std::uint32_t min_frontier, std::uint32_t max_frontier,
                                           std::uint32_t k) {
    (void)min_frontier;
    return schedule::DFlashEnvelopes{
        .local  = {0, max_frontier},
        .full   = {0, max_frontier},
        .append = {0, k + 1},
    };
}

DecodeGraphProfile& select_graph_profile(DecodeGraphFamily& family, std::uint32_t batch_size,
                                         std::uint32_t frontier, const char* label,
                                         bool greedy_target = false) {
    const auto it = std::find_if(
        family.profiles.begin(), family.profiles.end(), [&](const DecodeGraphProfile& profile) {
            return profile.batch_size == batch_size && profile.min_execution_frontier <= frontier &&
                   frontier <= profile.max_execution_frontier &&
                   profile.greedy_target == greedy_target;
        });
    if (it == family.profiles.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage is incomplete");
    }
    return *it;
}

void validate_graph_profiles(const std::vector<GraphExecutionProfile>& profiles,
                             std::uint32_t max_frontier, const char* label) {
    if (profiles.empty() || profiles.front().min != 0 || profiles.back().max != max_frontier) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage has invalid endpoints");
    }
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        if (profiles[i].min > profiles[i].max ||
            (i != 0 && profiles[i].min != profiles[i - 1].max + 1)) {
            throw std::logic_error(std::string(label) + " CUDA Graph coverage has a gap");
        }
    }
}

DecodeGraphTopology& select_graph_topology(DecodeGraphFamily& family, std::uint32_t topology_class,
                                           const char* label) {
    const auto it = std::find_if(family.topologies.begin(), family.topologies.end(),
                                 [topology_class](const DecodeGraphTopology& topology) {
                                     return topology.topology_class == topology_class;
                                 });
    if (it == family.topologies.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph topology is unavailable");
    }
    return *it;
}

DecodeGraphExecutable& install_graph_profile(DecodeGraphFamily& family, DecodeGraphProfile& profile,
                                             const char* label) {
    DecodeGraphTopology& topology   = select_graph_topology(family, profile.topology_class, label);
    const std::size_t profile_index = static_cast<std::size_t>(&profile - family.profiles.data());
    if (topology.installed_profile != profile_index) {
        topology.executable.update(profile.definition);
        topology.installed_profile = profile_index;
    }
    return topology.executable;
}

template <class Prepare, class Synchronize>
void instantiate_graph_family(DecodeGraphFamily& family, const char* label,
                              std::span<const int> devices, std::span<const cudaStream_t> streams,
                              Prepare&& prepare, Synchronize&& synchronize_all) {
    if (family.profiles.empty()) {
        throw std::logic_error(std::string(label) + " CUDA Graph family has no profiles");
    }

    for (std::size_t i = 0; i < family.profiles.size(); ++i) {
        DecodeGraphProfile& profile = family.profiles[i];
        if (!profile.definition.ready()) {
            throw std::logic_error(std::string(label) + " CUDA Graph definition is empty");
        }
        const auto existing =
            std::find_if(family.topologies.begin(), family.topologies.end(),
                         [&](const DecodeGraphTopology& topology) {
                             return topology.topology_class == profile.topology_class;
                         });
        if (existing != family.topologies.end()) { continue; }

        family.topologies.emplace_back();
        DecodeGraphTopology& topology = family.topologies.back();
        topology.topology_class       = profile.topology_class;
        topology.executable.instantiate(profile.definition, devices, streams);
        topology.installed_profile = i;
    }

    const auto install_and_upload = [&](DecodeGraphTopology& topology, std::size_t profile_index) {
        DecodeGraphProfile& profile = family.profiles[profile_index];
        if (topology.installed_profile != profile_index) {
            topology.executable.update(profile.definition);
            topology.installed_profile = profile_index;
        }
        // Each rank owns its graph and uploads/launches it on that rank's own stream; the
        // synchronize covers every device because every rank's nodes retire on its own stream.
        topology.executable.upload();
        synchronize_all();
    };

    for (DecodeGraphTopology& topology : family.topologies) {
        std::optional<std::size_t> first_profile;
        for (std::size_t i = 0; i < family.profiles.size(); ++i) {
            if (family.profiles[i].topology_class == topology.topology_class) {
                if (!first_profile) {
                    first_profile = i;
                    install_and_upload(topology, i);

                    DecodeGraphProfile& profile = family.profiles[i];
                    prepare(profile.min_execution_frontier, profile.batch_size);
                    synchronize_all();
                    topology.executable.launch();
                    synchronize_all();
                    continue;
                }
                install_and_upload(topology, i);
            }
        }
        if (!first_profile) {
            throw std::logic_error(std::string(label) + " CUDA Graph topology has no definitions");
        }
        if (topology.installed_profile != *first_profile) {
            install_and_upload(topology, *first_profile);
        }
    }
}

} // namespace

PeerRuntime::PeerRuntime(DeviceContext& peer_device, const LoadedModelData& peer_model,
                         const SequencePlanImpl& plan)
    : device(peer_device), model(peer_model), persistent(plan.persistent.bytes),
      workspace_storage(plan.workspace.capacity),
      work(DeviceSpan{workspace_storage.base(), workspace_storage.capacity()}) {
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    decoder = std::make_unique<qwen3_6::DecoderState>(backing, plan.persistent.decoder);
    if (plan.persistent.dflash) { dflash.emplace(backing, *plan.persistent.dflash); }
    if (plan.persistent.replay_records) {
        replay_records.emplace(backing, *plan.persistent.replay_records);
    }
    io             = qwen3_6::RoundState(backing, plan.persistent.round);
    prefill_hidden = plan.persistent.prefill_hidden.bind(backing);
    token_counts   = plan.persistent.token_counts.bind(backing);
}

ProgramImplCore::ProgramImplCore(const std::array<const LoadedModelData*, kMaximumDevices>& models,
                                 const SequencePlanImpl& plan, ExecutionContext& execution_in)
    : model(*models[0]), execution(execution_in), device(execution_in.primary()), tp(execution_in.tp),
      capacity(plan.capacity), kv_capacity(plan.kv_capacity),
      max_concurrency(plan.max_concurrency), prefill_chunk(plan.prefill_chunk),
      draft_window(plan.draft_window), speculative_backend(plan.speculative_backend),
      kv_dtype(plan.kv_dtype), kv_quant_group(plan.kv_quant_group),
      proposal_head(plan.proposal_head), vision_enabled(plan.features.vision),
      use_cuda_graph(plan.use_cuda_graph), kv_payload_bytes(plan.persistent.kv_payload_bytes),
      gdn_state_bytes(linear_attention_state_bytes(plan.persistent.decoder.linear_attention)),
      graph_allowance_bytes(plan.graph_allowance_bytes), workspace_plan(plan.workspace),
      persistent(plan.persistent.bytes), workspace_storage(plan.workspace.capacity),
      work(DeviceSpan{workspace_storage.base(), workspace_storage.capacity()}),
      rope_mode(plan.rope_mode), effective_max_context(plan.effective_max_context),
      yarn_mscale(plan.rope_mode == RopeMode::Yarn
                      ? static_cast<double>(
                            qwen3_6::detail::yarn_rope_mscale(plan_yarn_params(plan)))
                      : 1.0),
      round_host(sizeof(TokenId)),
      ordinary_host(
          plan.speculative_backend == SpeculativeBackend::None
              ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_6::OrdinaryDecodeIngress) +
                                                     sizeof(qwen3_6::OrdinaryDecodeEgress))
              : std::nullopt),
      mtp_host(plan.speculative_backend == SpeculativeBackend::Mtp
                   ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_6::MtpDecodeIngress) +
                                                          sizeof(qwen3_6::MtpDecodeEgress))
                   : std::nullopt),
      dflash_host(plan.speculative_backend == SpeculativeBackend::DFlash
                      ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_6::DFlashDecodeIngress) +
                                                             sizeof(qwen3_6::DFlashDecodeEgress))
                      : std::nullopt) {
    if (model.weights_arena == nullptr) {
        throw std::invalid_argument("Qwen3.6 model view has no owning weight arena");
    }
    if (model.features != plan.features || model.mtp.has_value() != plan.features.mtp() ||
        model.dflash.has_value() != plan.features.dflash() ||
        model.optimized_proposal.has_value() != plan.features.optimized_proposal() ||
        model.vision.has_value() != plan.features.vision) {
        throw std::invalid_argument(
            "Qwen3.6 loaded weights do not match the frozen startup features");
    }
    if (model.mtp.has_value() && model.dflash.has_value()) {
        throw std::invalid_argument("MTP and DFlash model views are mutually exclusive");
    }
    if (model.dflash.has_value() && model.vision.has_value()) {
        throw std::invalid_argument("DFlash and Vision model views are mutually exclusive");
    }
    if (tp < 1 || tp > static_cast<int>(kMaximumDevices) || (tp > 1) != (plan.tp > 1)) {
        throw std::invalid_argument("Qwen3.6 program tensor-parallel width is inconsistent");
    }
    if (tp > 1) {
        for (int rank = 1; rank < tp; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            if (!execution.dev[slot].has_value() || models[slot] == nullptr) {
                throw std::invalid_argument(
                    "tensor-parallel program requires a device context and model view per rank");
            }
        }
        // Peer arenas must be allocated with the peer device current. cudaMalloc is not
        // stream-ordered and not capturable, so this happens once, here, and never in a hot path.
        int previous = 0;
        CUDA_CHECK(cudaGetDevice(&previous));
        for (int rank = 1; rank < tp; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            CUDA_CHECK(cudaSetDevice(execution.dev[slot]->device));
            try {
                peer[slot].emplace(*execution.dev[slot], *models[slot], plan);
            } catch (...) {
                (void)cudaSetDevice(previous);
                throw;
            }
        }
        CUDA_CHECK(cudaSetDevice(previous));
        collective_comms.emplace(execution);
    }
    if (rope_mode == RopeMode::Yarn) {
        // ONE resident 32-float corrected inverse-frequency table PER DEVICE, uploaded once, here.
        // cudaMalloc is neither stream-ordered nor capturable, and CUDA Graph capture bakes this
        // pointer into the replayed rope launch node, so the allocation has to happen exactly once
        // at construction and stay valid for the life of the Program. At tp 2 each rank ropes its
        // own head-local q/k on its own device and stream: rank 1 cannot dereference rank 0's
        // table, so each device gets its own copy of the same 128 bytes.
        const std::vector<float> table =
            qwen3_6::detail::yarn_scale(plan_yarn_params(plan)).first;
        if (table.size() != static_cast<std::size_t>(TextConfig::rotary_dim / 2)) {
            throw std::logic_error("YaRN frequency table does not match the rotary geometry");
        }
        const std::size_t table_bytes = table.size() * sizeof(float);
        for (int rank = 0; rank < tp; ++rank) {
            const auto slot            = static_cast<std::size_t>(rank);
            DeviceContext& rank_device = *execution.dev[slot];
            const ScopedDevice scope(rank_device.device);
            rope_frequency_storage[slot] = DeviceBuffer(table_bytes);
            CUDA_CHECK(cudaMemcpyAsync(rope_frequency_storage[slot].p, table.data(), table_bytes,
                                       cudaMemcpyHostToDevice, rank_device.stream));
            // Settled here rather than left as an ordering assumption: this copy is the only write
            // this buffer ever sees, and every later reader is a captured or eager rope launch.
            CUDA_CHECK(cudaStreamSynchronize(rank_device.stream));
            rope_frequency[slot] = ops::RopeFrequencyOverride{
                static_cast<const float*>(rope_frequency_storage[slot].p),
                static_cast<float>(yarn_mscale)};
        }
    }
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    decoder = std::make_unique<qwen3_6::DecoderState>(backing, plan.persistent.decoder);
    if (plan.persistent.replay_records) {
        replay_records.emplace(backing, *plan.persistent.replay_records);
    }
    if (replay_records.has_value() != (speculative_backend != SpeculativeBackend::None)) {
        throw std::logic_error("ReplaySSM records do not match the sequence plan");
    }
    if (plan.persistent.dflash) { dflash.emplace(backing, *plan.persistent.dflash); }
    if (dflash.has_value() != plan.features.dflash()) {
        throw std::logic_error("DFlash state does not match the frozen sequence plan");
    }

    io = qwen3_6::RoundState(backing, plan.persistent.round);
    if (io.mtp.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("round-state MTP extension does not match the sequence plan");
    }
    if (io.mtp_decode.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("MTP decode frame does not match the sequence plan");
    }
    if (io.ordinary.has_value() != (speculative_backend == SpeculativeBackend::None)) {
        throw std::logic_error("ordinary decode frame does not match the sequence plan");
    }
    if (io.dflash_prefill.has_value() != (speculative_backend == SpeculativeBackend::DFlash)) {
        throw std::logic_error("DFlash prefill scratch does not match the sequence plan");
    }
    if (io.dflash_decode.has_value() != (speculative_backend == SpeculativeBackend::DFlash)) {
        throw std::logic_error("DFlash decode frame does not match the sequence plan");
    }
    prefill_hidden                  = plan.persistent.prefill_hidden.bind(backing);
    token_counts                    = plan.persistent.token_counts.bind(backing);
    sampling_config                 = plan.persistent.sampling_config.bind(backing);
    tail_hidden_store               = plan.persistent.tail_hidden.bind(backing);
    rewrite_checkpoint_hidden_store = plan.persistent.rewrite_checkpoint_hidden.bind(backing);
    if (plan.prefix_cache.enabled) {
        prefix_snapshot_hidden_store = plan.persistent.prefix_snapshot_hidden.bind(backing);
    }
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        SequenceState& sequence = sequences[lane];
        sequence.lane           = lane;
        sequence.tail_hidden    = tail_hidden_store.slice(1, static_cast<std::int32_t>(lane), 1);
        sequence.rewrite_checkpoint_hidden =
            rewrite_checkpoint_hidden_store.slice(1, static_cast<std::int32_t>(lane), 1);
        sequence.ledger.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_identity.reserve(static_cast<std::size_t>(capacity) + 1ULL);
    }

    set_device_i32(io.text_kv_table_row, 0);
    set_device_i32(io.backend_kv_table_row, 0);

    host_tokens = static_cast<TokenId*>(round_host.data());
    if (ordinary_host) {
        ordinary_host_ingress = static_cast<qwen3_6::OrdinaryDecodeIngress*>(ordinary_host->data());
        ordinary_host_egress  = reinterpret_cast<qwen3_6::OrdinaryDecodeEgress*>(
            static_cast<unsigned char*>(ordinary_host->data()) +
            sizeof(qwen3_6::OrdinaryDecodeIngress));
        *ordinary_host_ingress = {};
        *ordinary_host_egress  = {};
        for (int rank = 1; rank < tp; ++rank) {
            const auto slot                 = static_cast<std::size_t>(rank);
            ordinary_peer_host[slot].emplace(sizeof(qwen3_6::OrdinaryDecodeIngress));
            ordinary_peer_host_ingress[slot] =
                static_cast<qwen3_6::OrdinaryDecodeIngress*>(ordinary_peer_host[slot]->data());
            *ordinary_peer_host_ingress[slot] = {};
        }
    }
    if (io.ordinary && io.ordinary->mixed) {
        const qwen3_6::OrdinaryDecodeState::MixedState& mixed = *io.ordinary->mixed;
        mixed_host_tokens.assign(static_cast<std::size_t>(mixed.tokens.ne[0]), 0);
        mixed_host_cache_positions.assign(static_cast<std::size_t>(mixed.cache_positions.ne[0]), 0);
        mixed_host_rope_positions.assign(static_cast<std::size_t>(mixed.rope_positions.ne[0]), 0);
        mixed_host_decode_kv_rows.assign(
            static_cast<std::size_t>(mixed.decode_kv_table_rows.ne[0]), 0);
        mixed_host_decode_linear_slots.assign(
            static_cast<std::size_t>(mixed.decode_linear_state_slots.ne[0]), 0);
        mixed_host_egress = {};
    }
    if (mtp_host) {
        mtp_host_ingress = static_cast<qwen3_6::MtpDecodeIngress*>(mtp_host->data());
        mtp_host_egress  = reinterpret_cast<qwen3_6::MtpDecodeEgress*>(
            static_cast<unsigned char*>(mtp_host->data()) + sizeof(qwen3_6::MtpDecodeIngress));
        *mtp_host_ingress = {};
        *mtp_host_egress  = {};
        for (int rank = 1; rank < tp; ++rank) {
            const auto slot           = static_cast<std::size_t>(rank);
            mtp_peer_host[slot].emplace(sizeof(qwen3_6::MtpDecodeIngress));
            mtp_peer_host_ingress[slot] =
                static_cast<qwen3_6::MtpDecodeIngress*>(mtp_peer_host[slot]->data());
            *mtp_peer_host_ingress[slot] = {};
        }
    }
    if (dflash_host) {
        dflash_host_ingress = static_cast<qwen3_6::DFlashDecodeIngress*>(dflash_host->data());
        dflash_host_egress  = reinterpret_cast<qwen3_6::DFlashDecodeEgress*>(
            static_cast<unsigned char*>(dflash_host->data()) +
            sizeof(qwen3_6::DFlashDecodeIngress));
        *dflash_host_ingress = {};
        *dflash_host_egress  = {};
        for (int rank = 1; rank < tp; ++rank) {
            const auto slot              = static_cast<std::size_t>(rank);
            dflash_peer_host[slot].emplace(sizeof(qwen3_6::DFlashDecodeIngress));
            dflash_peer_host_ingress[slot] =
                static_cast<qwen3_6::DFlashDecodeIngress*>(dflash_peer_host[slot]->data());
            *dflash_peer_host_ingress[slot] = {};
        }
    }
    if (io.mtp_decode && io.mtp_decode->mixed) {
        const qwen3_6::MtpDecodeState::MixedState& mixed = *io.mtp_decode->mixed;
        mtp_mixed_host_owner_tokens.assign(static_cast<std::size_t>(prefill_chunk), 0);
        mtp_mixed_host_owner_positions.assign(static_cast<std::size_t>(prefill_chunk), 0);
        mtp_mixed_host_owner_rope_positions.assign(static_cast<std::size_t>(prefill_chunk), 0);
        (void)mixed;
    }
    if (io.dflash_prefill) {
        CUDA_CHECK(cudaMemsetAsync(io.dflash_prefill->produced_count.data, 0,
                                   io.dflash_prefill->produced_count.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(io.rope_delta.data, 0, io.rope_delta.bytes(), device.stream));
    if (io.mtp) {
        CUDA_CHECK(
            cudaMemsetAsync(io.mtp->position.data, 0, io.mtp->position.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(sampling_config.data, 0, sampling_config.bytes(), device.stream));
    if (tp > 1) {
        int previous = 0;
        CUDA_CHECK(cudaGetDevice(&previous));
        for (int rank = 1; rank < tp; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            PeerRuntime& p  = *peer[slot];
            CUDA_CHECK(cudaSetDevice(p.device.device));
            CUDA_CHECK(cudaMemsetAsync(p.io.rope_delta.data, 0, p.io.rope_delta.bytes(),
                                       p.device.stream));
            if (p.io.mtp) {
                CUDA_CHECK(cudaMemsetAsync(p.io.mtp->position.data, 0,
                                           p.io.mtp->position.bytes(), p.device.stream));
            }
            CUDA_CHECK(cudaMemsetAsync(p.token_counts.data, 0, p.token_counts.bytes(),
                                       p.device.stream));
            set_peer_i32(p.device, p.io.text_kv_table_row, 0);
            set_peer_i32(p.device, p.io.backend_kv_table_row, 0);
        }
        CUDA_CHECK(cudaSetDevice(previous));
        for (int rank = 1; rank < tp; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            PeerRuntime& p  = *peer[slot];
            peer_core[slot].emplace(schedule::TpPeerCore{
                .execution        = &execution,
                .events           = &*collective_comms,
                .device           = &p.device,
                .model            = &p.model,
                .work             = &p.work,
                .linear_attention = &p.decoder->linear_attention,
                .io               = &p.io,
                .prefill_hidden   = &p.prefill_hidden,
                .text_cache       = &p.decoder->text_kv,
                .mtp_cache        = p.decoder->mtp_cache(),
                .replay_records   = p.replay_records ? &*p.replay_records : nullptr,
                .mtp_host_ingress    = mtp_peer_host_ingress[slot],
                .dflash_host_ingress = dflash_peer_host_ingress[slot]});
            if (p.replay_records.has_value() != replay_records.has_value()) {
                throw std::logic_error("peer ReplaySSM records do not match rank 0's");
            }
            if ((p.decoder->mtp_cache() != nullptr) != (decoder->mtp_cache() != nullptr)) {
                throw std::logic_error("peer MTP KV cache does not match rank 0's");
            }
            if (p.io.mtp.has_value() != io.mtp.has_value() ||
                p.io.mtp_decode.has_value() != io.mtp_decode.has_value()) {
                throw std::logic_error("peer MTP round state does not match rank 0's");
            }
        }
        for (int rank = 1; rank < tp; ++rank) {
            peer[static_cast<std::size_t>(rank)]->device.synchronize();
        }
    }
    device.synchronize();
    prepare_graphs();
    work.reset();
    work.reset_peak();
    workspace_logical_peak_bytes = 0;
    if (plan.prefix_cache.enabled) {
        prefix_snapshot_slots = plan.prefix_cache.snapshot_slots != 0
                                    ? plan.prefix_cache.snapshot_slots
                                    : max_concurrency + 1U;
        prefix_sink  = std::make_unique<PrefixCacheSink>(*this);
        prefix_index = std::make_unique<ninfer::prefix_cache::PrefixIndex>(
            ninfer::prefix_cache::PrefixIndexOptions{.device_snapshot_slots = prefix_snapshot_slots},
            prefix_sink.get());
        // M2: create host slab pool if requested. Slabs are rank-sliced: a block slab holds
        // tp x per-rank block bytes (rank r at offset r * per_rank_block_bytes), a snapshot slab
        // holds tp x per-rank snapshot image bytes. Each rank transfers only its own slice.
        if (plan.prefix_cache.host_cache_mib > 0) {
            const std::size_t total_bytes =
                static_cast<std::size_t>(plan.prefix_cache.host_cache_mib) * 1024U * 1024U;
            // Per-rank block bytes: this rank's text-pool page plus its backend-pool page (all
            // planes of both pools), exactly what one cache node stores per rank.
            const std::size_t per_rank_block_bytes =
                decoder->text_kv.pool().per_page_bytes() +
                (backend_kv_cache() != nullptr ? backend_kv_cache()->pool().per_page_bytes() : 0U);
            // Per-rank snapshot image bytes: one linear-attention slot = every GDN layer's conv
            // history plus recurrent state, at this rank's head/channel shard.
            const LinearAttentionStatePoolSpec& la_spec = decoder->linear_attention.spec;
            const std::size_t per_layer_conv_bytes =
                static_cast<std::size_t>(la_spec.conv_channels) *
                static_cast<std::size_t>(la_spec.conv_width) * ninfer::dtype_size(la_spec.conv_dtype);
            const std::size_t per_layer_recurrent_bytes =
                static_cast<std::size_t>(la_spec.key_head_dim) *
                static_cast<std::size_t>(la_spec.value_head_dim) *
                static_cast<std::size_t>(la_spec.value_heads) * 4U;
            const std::size_t per_rank_snapshot_bytes =
                static_cast<std::size_t>(la_spec.layers) *
                (per_layer_conv_bytes + per_layer_recurrent_bytes);
            prefix_block_rank_bytes    = per_rank_block_bytes;
            prefix_snapshot_rank_bytes = per_rank_snapshot_bytes;
            const std::size_t block_slab_bytes    = static_cast<std::size_t>(tp) * per_rank_block_bytes;
            const std::size_t snapshot_slab_bytes =
                static_cast<std::size_t>(tp) * per_rank_snapshot_bytes;
            // Split budget: 80% for KV blocks, 20% for snapshot images.
            const std::size_t block_budget = (total_bytes * 8) / 10;
            const std::size_t snap_budget  = total_bytes - block_budget;
            const std::size_t block_count = block_slab_bytes > 0 ? (block_budget / block_slab_bytes) : 0;
            const std::size_t snap_count  = snapshot_slab_bytes > 0 ? (snap_budget / snapshot_slab_bytes) : 0;
            if (block_count == 0 || snap_count == 0) {
                // Budget too small for at least one slab of each kind; skip host tier.
                std::fprintf(stderr,
                             "[host-cache] requested %u MiB is too small for one block slab "
                             "(%zu MiB) plus one snapshot slab (%zu MiB); host tier disabled\n",
                             plan.prefix_cache.host_cache_mib, block_slab_bytes >> 20U,
                             snapshot_slab_bytes >> 20U);
            } else {
                prefix_host_pool = std::make_unique<ninfer::prefix_cache::HostSlabPool>(
                    ninfer::prefix_cache::HostSlabPoolOptions{
                        .total_bytes         = total_bytes,
                        .block_slab_bytes    = block_slab_bytes,
                        .snapshot_slab_bytes = snapshot_slab_bytes,
                        .block_slab_count    = block_count,
                        .snapshot_slab_count = snap_count,
                    });
                std::fprintf(stderr,
                             "[host-cache] pinned %zu MiB (requested %u MiB): %zu block + %zu "
                             "snapshot slabs\n",
                             prefix_host_pool->pinned_bytes() >> 20U,
                             plan.prefix_cache.host_cache_mib,
                             prefix_host_pool->block_slab_count(),
                             prefix_host_pool->snapshot_slab_count());
                prefix_index->set_host_pool(prefix_host_pool.get());
                // M2: dedicated cache transfer stream per rank for D2H/H2D copies. Each rank's
                // stream and events must be created with THAT rank's device current, because they
                // operate on that rank's KV/GDN memory under ScopedDevice(rank_device.device); a
                // stream or event bound to the wrong device is an invalid resource handle.
                for (int rank = 0; rank < tp; ++rank) {
                    const auto slot = static_cast<std::size_t>(rank);
                    DeviceContext& rank_device = rank == 0 ? device : peer[slot]->device;
                    const ScopedDevice scope(rank_device.device);
                    cudaStream_t stream = nullptr;
                    cudaEvent_t event   = nullptr;
                    cudaEvent_t src     = nullptr;
                    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
                    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
                    CUDA_CHECK(cudaEventCreateWithFlags(&src, cudaEventDisableTiming));
                    prefix_cache_stream[slot]    = stream;
                    prefix_cache_event[slot]     = event;
                    prefix_cache_src_event[slot] = src;
                }
            }
        }
        prefix_max_new_taps =
            plan.prefix_cache.max_new_taps != 0 ? plan.prefix_cache.max_new_taps : 2U;
        prefix_ladder_tokens = plan.prefix_cache.ladder_tokens != 0
                                    ? plan.prefix_cache.ladder_tokens
                                    : std::max(4096U, 2U * plan.prefill_chunk);
        prefix_min_gap = plan.prefix_cache.min_gap != 0 ? plan.prefix_cache.min_gap
                                                        : std::max(1024U, plan.prefill_chunk);
    }
}

void ProgramImplCore::release_cached_pages(const ninfer::prefix_cache::PoolPages& pages) noexcept {
    if (pages.page[0] >= 0) {
        const std::int32_t ids[1] = {pages.page[0]};
        decoder->text_kv.pool().release_cache_pages(ids);
        for (int rank = 1; rank < tp; ++rank) {
            peer[static_cast<std::size_t>(rank)]->decoder->text_kv.pool().release_cache_pages(ids);
        }
    }
    if (pages.page[1] >= 0 && backend_kv_cache() != nullptr) {
        const std::int32_t ids[1] = {pages.page[1]};
        backend_kv_cache()->pool().release_cache_pages(ids);
        for (int rank = 1; rank < tp; ++rank) {
            qwen3_6::PagedKVCache* backend = rank_backend_cache(rank);
            if (backend != nullptr) { backend->pool().release_cache_pages(ids); }
        }
    }
}

ProgramImplCore::~ProgramImplCore() noexcept {
    if (device.stream != nullptr) { (void)cudaStreamSynchronize(device.stream); }
    for (int rank = 1; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (peer[slot] && peer[slot]->device.stream != nullptr) {
            (void)cudaStreamSynchronize(peer[slot]->device.stream);
        }
    }
    // M2: cleanup cache transfer streams and events. The sync above guarantees every enqueued
    // copy has landed, so any write-through still tracked as in-flight completes successfully.
    if (prefix_index != nullptr && !prefix_inflight_writes.empty()) {
        for (const InflightCacheWrite& write : prefix_inflight_writes) {
            if (write.is_snapshot) {
                prefix_index->finish_snapshot_write(write.snapshot, true);
            } else {
                prefix_index->finish_host_write(write.node, true);
            }
        }
        prefix_inflight_writes.clear();
    }
    for (int rank = 0; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (prefix_cache_stream[slot] != nullptr) {
            (void)cudaStreamSynchronize(prefix_cache_stream[slot]);
            (void)cudaStreamDestroy(prefix_cache_stream[slot]);
        }
        if (prefix_cache_event[slot] != nullptr) {
            (void)cudaEventDestroy(prefix_cache_event[slot]);
        }
        if (prefix_cache_src_event[slot] != nullptr) {
            (void)cudaEventDestroy(prefix_cache_src_event[slot]);
        }
    }
}

bool ProgramImplCore::can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept {
    if (lane >= max_concurrency || plan.impl_ == nullptr) { return false; }
    const RequestControl& request = requests[lane];
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        return false;
    }
    const SequenceState& sequence = sequences[lane];
    // Pool invariant: free = page_group_count - entitled - cached. Cache pages held by LRU-linked
    // nodes are reclaimable through evict_device_lru(), so they count as capacity here.
    const std::array<std::uint32_t, ninfer::prefix_cache::kMaxPools> evictable =
        prefix_index != nullptr ? prefix_index->evictable_page_counts()
                                : std::array<std::uint32_t, ninfer::prefix_cache::kMaxPools>{0U, 0U};
    const auto can_replace = [&evictable](const PagedKVPool& pool, std::uint32_t old_pages,
                                 std::uint32_t new_pages, std::uint32_t evictable_pages) {
        if (old_pages > pool.entitled_pages() || new_pages > pool.logical_page_capacity()) {
            return false;
        }
        const std::int64_t headroom =
            static_cast<std::int64_t>(pool.page_group_count()) -
            static_cast<std::int64_t>(pool.entitled_pages() - old_pages) -
            static_cast<std::int64_t>(pool.cached_pages()) +
            static_cast<std::int64_t>(evictable_pages);
        return headroom >= 0 && static_cast<std::int64_t>(new_pages) <= headroom;
    };
    const std::uint32_t old_text = sequence.kv ? sequence.kv->text.page_entitlement() : 0;
    if (!can_replace(decoder->text_kv.pool(), old_text, plan.impl_->text_kv_page_entitlement,
                     evictable[0])) {
        return false;
    }
    const qwen3_6::PagedKVCache* backend = backend_kv_cache();
    if (backend == nullptr) { return plan.impl_->backend_kv_page_entitlement == 0; }
    const std::uint32_t old_backend =
        sequence.kv && sequence.kv->backend ? sequence.kv->backend->page_entitlement() : 0;
    if (!can_replace(backend->pool(), old_backend, plan.impl_->backend_kv_page_entitlement,
                     evictable[1])) {
        return false;
    }
    return true;
}

bool ProgramImplCore::can_admit_lane_after_retained_eviction(
    std::uint32_t lane, const RequestPlan& plan) const noexcept {
    if (lane >= max_concurrency || plan.impl_ == nullptr) { return false; }
    const RequestControl& request = requests[lane];
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        return false;
    }

    std::uint32_t reclaimable_text    = 0;
    std::uint32_t reclaimable_backend = 0;
    for (std::uint32_t other = 0; other < max_concurrency; ++other) {
        if (other == lane || !sequences[other].retained || !sequences[other].kv) { continue; }
        reclaimable_text += sequences[other].kv->text.page_entitlement();
        if (sequences[other].kv->backend) {
            reclaimable_backend += sequences[other].kv->backend->page_entitlement();
        }
    }

    // Same cached-page accounting as can_admit_lane: cache pages are held by the pool, and only
    // the LRU-linked ones are reclaimable.
    const std::array<std::uint32_t, ninfer::prefix_cache::kMaxPools> evictable =
        prefix_index != nullptr ? prefix_index->evictable_page_counts()
                                : std::array<std::uint32_t, ninfer::prefix_cache::kMaxPools>{0U, 0U};
    const auto can_replace = [&evictable](const PagedKVPool& pool, std::uint32_t old_pages,
                                 std::uint32_t reclaimable_pages, std::uint32_t new_pages,
                                 std::uint32_t evictable_pages) {
        if (old_pages > pool.entitled_pages() ||
            reclaimable_pages > pool.entitled_pages() - old_pages ||
            new_pages > pool.logical_page_capacity()) {
            return false;
        }
        const std::uint32_t committed = pool.entitled_pages() - old_pages - reclaimable_pages;
        const std::int64_t headroom =
            static_cast<std::int64_t>(pool.page_group_count()) -
            static_cast<std::int64_t>(committed) -
            static_cast<std::int64_t>(pool.cached_pages()) +
            static_cast<std::int64_t>(evictable_pages);
        return headroom >= 0 && static_cast<std::int64_t>(new_pages) <= headroom;
    };

    const SequenceState& sequence = sequences[lane];
    const std::uint32_t old_text  = sequence.kv ? sequence.kv->text.page_entitlement() : 0;
    if (!can_replace(decoder->text_kv.pool(), old_text, reclaimable_text,
                     plan.impl_->text_kv_page_entitlement, evictable[0])) {
        return false;
    }

    const qwen3_6::PagedKVCache* backend = backend_kv_cache();
    if (backend == nullptr) { return plan.impl_->backend_kv_page_entitlement == 0; }
    const std::uint32_t old_backend =
        sequence.kv && sequence.kv->backend ? sequence.kv->backend->page_entitlement() : 0;
    return can_replace(backend->pool(), old_backend, reclaimable_backend,
                       plan.impl_->backend_kv_page_entitlement, evictable[1]);
}

runtime::AdmissionResources ProgramImplCore::admission_capacity() const noexcept {
    const qwen3_6::PagedKVCache* backend = backend_kv_cache();
    return runtime::AdmissionResources{
        .active_lanes     = max_concurrency,
        .main_kv_pages    = decoder->text_kv.pool().page_group_count(),
        .backend_kv_pages = backend != nullptr ? backend->pool().page_group_count() : 0U,
    };
}

runtime::AdmissionResources ProgramImplCore::admission_cache_residue() const noexcept {
    // Device prefix-cache residency that an admission-time eviction cannot reclaim. A device node
    // counts here when it is cached but held off the eviction LRU by a live reference: a borrowed
    // or promoted prefix page of a running sequence, or an in-flight write-through/restore pin.
    // `can_admit_lane` already credits only the LRU-linked pages as capacity, so protected-admission
    // feasibility has to charge this residue too -- otherwise a head that is genuinely blocked by
    // pinned cache pages looks like it is "not blocked by frozen incumbents". Over-counting is
    // safe: it can only make the protected projection conservative, never admit a request.
    if (prefix_index == nullptr) { return {}; }
    const std::array<std::uint32_t, ninfer::prefix_cache::kMaxPools> evictable =
        prefix_index->evictable_page_counts();
    const auto residual = [](std::uint32_t cached, std::uint32_t reclaimable) {
        return cached > reclaimable ? cached - reclaimable : 0U;
    };
    const qwen3_6::PagedKVCache* backend = backend_kv_cache();
    return runtime::AdmissionResources{
        .active_lanes     = 0,
        .main_kv_pages    = residual(decoder->text_kv.pool().cached_pages(), evictable[0]),
        .backend_kv_pages = backend != nullptr
                                ? residual(backend->pool().cached_pages(), evictable[1])
                                : 0U,
    };
}

runtime::PrefillStepResult ProgramImplCore::start_prefill_lane(std::uint32_t lane,
                                                               PreparedPromptData&& prompt,
                                                               RequestPlan&& plan,
                                                                runtime::TransientRegion transient) {
    tick_prefix_cache();
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    SequenceState& sequence = sequences[lane];
    RequestControl& request = requests[lane];
    if (plan.impl_ == nullptr) { throw std::invalid_argument("request plan is empty"); }
    RequestPlanImpl& request_plan = *plan.impl_;
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        throw std::logic_error("staged prefill requires a free request lane");
    }

    const std::uint32_t prompt_tokens = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (prompt_tokens != request_plan.summary.prompt_tokens ||
        (request_plan.vision.has_value() && !prompt.has_media())) {
        throw std::invalid_argument("request plan does not describe the prepared prompt");
    }
    if (prompt.identity.rewrite_checkpoint &&
        (prompt.identity.rewrite_checkpoint->frontier == 0 ||
         prompt.identity.rewrite_checkpoint->frontier > prompt_tokens)) {
        throw std::invalid_argument("prepared prompt has an invalid rewrite checkpoint");
    }
    const bool suffix_has_visual = std::any_of(
        prompt.token_types.begin() + static_cast<std::ptrdiff_t>(request_plan.reuse_base),
        prompt.token_types.end(), [](std::uint8_t type) { return type != 0; });
    if (suffix_has_visual != request_plan.vision.has_value()) {
        throw std::invalid_argument("request plan does not describe the prompt suffix modality");
    }
    if (request_plan.summary.transient_bytes != 0 &&
        (transient.data == nullptr || transient.size < request_plan.summary.transient_bytes ||
         transient.alignment < request_plan.summary.transient_alignment)) {
        throw std::invalid_argument("request transient region does not satisfy the plan");
    }
    if (request_plan.reuse != ReusePath::FullReset &&
        request_plan.reuse != ReusePath::BlockSnapshot &&
        (!sequence.retained ||
         !qwen3_6::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                          request_plan.reuse_base))) {
        throw std::logic_error("planned resident prefix is no longer reusable");
    }
    if (is_rewrite_checkpoint_restore(request_plan.reuse) &&
        (!sequence.rewrite_checkpoint.valid ||
         sequence.rewrite_checkpoint.frontier != request_plan.reuse_base ||
         request_plan.reuse != restore_path(sequence.rewrite_checkpoint.kind))) {
        throw std::logic_error("planned rewrite checkpoint is unavailable");
    }
    if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::KeepExisting &&
        (!prompt.identity.rewrite_checkpoint || !sequence.rewrite_checkpoint.valid ||
         sequence.rewrite_checkpoint.kind != prompt.identity.rewrite_checkpoint->kind ||
         sequence.rewrite_checkpoint.frontier != prompt.identity.rewrite_checkpoint->frontier ||
         request_plan.reuse == ReusePath::FullReset ||
         !qwen3_6::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                          sequence.rewrite_checkpoint.frontier))) {
        throw std::logic_error("planned rewrite checkpoint retention is unavailable");
    }
    if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::ReclassifyExisting &&
        (!prompt.identity.rewrite_checkpoint || !sequence.rewrite_checkpoint.valid ||
         sequence.rewrite_checkpoint.kind == prompt.identity.rewrite_checkpoint->kind ||
         sequence.rewrite_checkpoint.frontier != prompt.identity.rewrite_checkpoint->frontier ||
         request_plan.reuse == ReusePath::FullReset ||
         !qwen3_6::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                          sequence.rewrite_checkpoint.frontier))) {
        throw std::logic_error("planned rewrite checkpoint reclassification is unavailable");
    }
    if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::CaptureNew &&
        (!request_plan.rewrite_checkpoint_capture || !prompt.identity.rewrite_checkpoint ||
         request_plan.rewrite_checkpoint_capture->kind !=
             prompt.identity.rewrite_checkpoint->kind ||
         request_plan.rewrite_checkpoint_capture->frontier !=
             prompt.identity.rewrite_checkpoint->frontier ||
         request_plan.rewrite_checkpoint_capture->frontier <= request_plan.reuse_base ||
         request_plan.rewrite_checkpoint_capture->frontier > prompt_tokens)) {
        throw std::logic_error("planned rewrite checkpoint capture is invalid");
    }
    if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::Drop &&
        prompt.identity.rewrite_checkpoint) {
        throw std::logic_error("planned rewrite checkpoint drop does not describe the prompt");
    }
    if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::DeferCapture &&
        (!prompt.identity.rewrite_checkpoint || request_plan.reuse == ReusePath::FullReset ||
         prompt.identity.rewrite_checkpoint->frontier > request_plan.reuse_base)) {
        throw std::logic_error("planned rewrite checkpoint deferral is invalid");
    }

    const auto started       = Clock::now();
    const std::uint32_t base = request_plan.reuse_base;
    const std::uint32_t initial_mtp_extent =
        speculative_backend == SpeculativeBackend::Mtp
            ? std::min({draft_window,
                        request_plan.summary.effective_output_tokens > 1
                            ? request_plan.summary.effective_output_tokens - 2
                            : 0U,
                        capacity - prompt_tokens > 0 ? capacity - prompt_tokens - 1 : 0U})
            : 0U;
    request.lifecycle = Lifecycle::Empty;
    sequence.retained = false;
    try {
        if (request_plan.reuse == ReusePath::FullReset) {
            sequence.kv.reset();
            ordered_reset(sequence);
            sequence.ledger.clear();
            sequence.text_kv_valid = 0;
            sequence.mtp_kv_valid  = 0;
            reserve_sequence_kv(sequence, request_plan.text_kv_page_entitlement,
                                request_plan.backend_kv_page_entitlement);
        } else if (request_plan.reuse == ReusePath::AppendAtFrontier) {
            if (!sequence.kv) {
                throw std::logic_error("resident prefix has no KV allocation bundle");
            }
            if (sequence.text_kv_valid < base) {
                throw std::logic_error("resident Text KV is shorter than the append frontier");
            }
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || sequence.mtp_kv_valid < mtp_base) {
                    throw std::logic_error("resident MTP KV is shorter than the bridge frontier");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (speculative_backend == SpeculativeBackend::DFlash &&
                       sequence.dflash_context_frontier != base) {
                throw std::logic_error("resident DFlash context is not at the append frontier");
            }
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            sequence.text_kv_valid = base;
            sequence.ledger.resize(base);
        } else if (is_rewrite_checkpoint_restore(request_plan.reuse)) {
            if (!sequence.kv || sequence.text_kv_valid < base) {
                throw std::logic_error("resident rewrite checkpoint has no complete KV allocation");
            }
            sequence.text_kv_valid = base;
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || sequence.mtp_kv_valid < mtp_base) {
                    throw std::logic_error(
                        "rewrite-checkpoint MTP KV is shorter than the bridge frontier");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (speculative_backend == SpeculativeBackend::DFlash) {
                if (!dflash || dflash->full.has_value() != sequence.kv->backend.has_value() ||
                    sequence.dflash_context_frontier < base) {
                    throw std::logic_error("planned DFlash rewrite checkpoint is unavailable");
                }
                dflash->restore_rewrite_checkpoint(static_cast<std::int32_t>(sequence.lane),
                                                   device.stream);
                sequence.dflash_context_frontier = base;
            }
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            decoder->linear_attention.copy_slot(
                LinearStateSlots::rewrite_checkpoint_state_slot(sequence.lane, max_concurrency),
                LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
                device.stream);
            for (int rank = 1; rank < tp; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                const ScopedDevice scope(peer[slot]->device.device);
                peer[slot]->decoder->linear_attention.copy_slot(
                    LinearStateSlots::rewrite_checkpoint_state_slot(sequence.lane, max_concurrency),
                    LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
                    peer[slot]->device.stream);
            }
            if (base == prompt_tokens) { copy_tail(sequence, sequence.rewrite_checkpoint_hidden); }
            sequence.ledger.resize(base);
        } else if (request_plan.reuse == ReusePath::BlockSnapshot) {
            if (prefix_index == nullptr) {
                throw std::logic_error("hybrid block snapshot has no prefix index");
            }
            const ninfer::prefix_cache::SnapshotId snap_id = request_plan.block_snapshot;
            const ninfer::prefix_cache::Snapshot* snap = prefix_index->snapshot(snap_id);
            if (snap == nullptr || snap->frontier != base || snap->tail_len != 0) {
                throw std::logic_error("planned block snapshot is unavailable");
            }
            // Drop any state inherited from a previous owner on this lane; hybrid reuse is
            // content-addressed and does not depend on a retained sequence.
            sequence.kv.reset();
            ordered_reset(sequence);
            sequence.ledger.clear();
            sequence.text_kv_valid = 0;
            sequence.mtp_kv_valid  = 0;

            // Pin everything this lane will use before any restore work, so slot eviction and
            // subtree erasure cannot touch the planned snapshot or path mid-restore.
            for (ninfer::prefix_cache::NodeId id : request_plan.block_path) {
                prefix_index->pin_node(id);
            }
            prefix_index->pin_snapshot(snap_id);

            std::vector<std::int32_t> text_pages;
            std::vector<std::int32_t> backend_pages;
            // Borrow pages up to the snapshot frontier. Device-resident nodes are borrowed
            // directly; host-resident nodes are restored H2D on every rank's cache stream first.
            const std::uint32_t frontier_blocks = base / ninfer::prefix_cache::kBlockTokens;
            for (std::size_t i = 1; i < request_plan.block_path.size() && (i - 1) < frontier_blocks;
                 ++i) {
                const ninfer::prefix_cache::NodeId node_id = request_plan.block_path[i];
                const ninfer::prefix_cache::BlockNode* node = prefix_index->node(node_id);
                if (node == nullptr) {
                    throw std::logic_error("planned block path node is missing");
                }
                if (node->device_state == ninfer::prefix_cache::Residency::Resident) {
                    text_pages.push_back(node->device_pages.page[0]);
                    if (node->device_pages.page[1] >= 0) {
                        backend_pages.push_back(node->device_pages.page[1]);
                    }
                } else if (node->host_state == ninfer::prefix_cache::Residency::Host &&
                           prefix_host_pool != nullptr) {
                    const ninfer::prefix_cache::PoolPages pages = acquire_cache_bundle();
                    prefix_index->begin_device_fill(node_id, pages);
                    char* slab = static_cast<char*>(prefix_host_pool->data(node->host_page));
                    bool ok = true;
                    for (int rank = 0; ok && rank < tp; ++rank) {
                        ok = copy_cache_block(rank, pages.page[0], pages.page[1],
                                              slab + static_cast<std::size_t>(rank) *
                                                          prefix_block_rank_bytes,
                                              false);
                    }
                    for (int rank = 0; rank < tp; ++rank) {
                        (void)cudaStreamSynchronize(
                            prefix_cache_stream[static_cast<std::size_t>(rank)]);
                    }
                    prefix_index->finish_device_fill(node_id, ok);
                    if (!ok) { throw std::logic_error("host block restore failed"); }
                    text_pages.push_back(pages.page[0]);
                    if (pages.page[1] >= 0) { backend_pages.push_back(pages.page[1]); }
                } else {
                    throw std::logic_error("planned block path node is not resident");
                }
            }

            // Restore the snapshot image to a device slot when it is only host-resident.
            const ninfer::prefix_cache::Snapshot* after = prefix_index->snapshot(snap_id);
            if (after != nullptr && !after->device_resident() &&
                after->host_state == ninfer::prefix_cache::Residency::Host &&
                prefix_host_pool != nullptr) {
                const auto target = prefix_index->begin_snapshot_restore(snap_id);
                if (target.slot == ninfer::prefix_cache::kInvalidSlot) {
                    throw std::logic_error("snapshot restore has no device slot");
                }
                const std::int32_t restore_state_slot = LinearStateSlots::snapshot_state_slot(
                    static_cast<std::uint32_t>(target.slot), max_concurrency,
                    prefix_snapshot_slots);
                char* slab = static_cast<char*>(prefix_host_pool->data(target.page));
                bool ok = true;
                for (int rank = 0; ok && rank < tp; ++rank) {
                    ok = copy_snapshot_slot(rank, restore_state_slot,
                                            slab + static_cast<std::size_t>(rank) *
                                                        prefix_snapshot_rank_bytes,
                                            false);
                }
                for (int rank = 0; rank < tp; ++rank) {
                    (void)cudaStreamSynchronize(
                        prefix_cache_stream[static_cast<std::size_t>(rank)]);
                }
                prefix_index->finish_snapshot_restore(snap_id, ok);
                if (!ok) { throw std::logic_error("snapshot image restore failed"); }
            }
            const auto owned_text    = static_cast<std::uint32_t>(text_pages.size());
            const auto owned_backend = static_cast<std::uint32_t>(backend_pages.size());
            const std::uint32_t text_entitlement =
                request_plan.text_kv_page_entitlement > owned_text
                    ? request_plan.text_kv_page_entitlement - owned_text
                    : 1U;
            const std::uint32_t backend_entitlement =
                backend_kv_cache() == nullptr
                    ? 0U
                    : (request_plan.backend_kv_page_entitlement > owned_backend
                           ? request_plan.backend_kv_page_entitlement - owned_backend
                           : 1U);
            reserve_sequence_kv(sequence, text_entitlement, backend_entitlement);
            if (!text_pages.empty()) {
                sequence.kv->text.borrow_prefix(text_pages);
                if (sequence.kv->backend && !backend_pages.empty()) {
                    sequence.kv->backend->borrow_prefix(backend_pages);
                }
                for (int rank = 1; rank < tp; ++rank) {
                    const auto slot = static_cast<std::size_t>(rank);
                    if (sequence.kv->text_peer[slot]) {
                        sequence.kv->text_peer[slot]->borrow_prefix(text_pages);
                    }
                    if (sequence.kv->backend_peer[slot] && !backend_pages.empty()) {
                        sequence.kv->backend_peer[slot]->borrow_prefix(backend_pages);
                    }
                }
            }
            const ninfer::prefix_cache::Snapshot* restored = prefix_index->snapshot(snap_id);
            if (restored == nullptr || !restored->device_resident()) {
                throw std::logic_error("planned block snapshot is unavailable");
            }
            sequence.prefix_pinned_nodes = request_plan.block_path;
            sequence.prefix_snapshot     = snap_id;
            sequence.prefix_borrowed     = true;
            sequence.text_kv_valid       = base;
            if (speculative_backend == SpeculativeBackend::Mtp) {
                sequence.mtp_kv_valid = base == 0 ? 0 : base - 1;
            }
            const std::int32_t snapshot_slot = LinearStateSlots::snapshot_state_slot(
                static_cast<std::uint32_t>(restored->device_slot), max_concurrency,
                prefix_snapshot_slots);
            decoder->linear_attention.copy_slot(
                snapshot_slot, LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
                device.stream);
            for (int rank = 1; rank < tp; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                const ScopedDevice scope(peer[slot]->device.device);
                peer[slot]->decoder->linear_attention.copy_slot(
                    snapshot_slot,
                    LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
                    peer[slot]->device.stream);
            }
            sequence.ledger.resize(base);
        } else {
            throw std::logic_error("request plan has an invalid prefix reuse path");
        }

        // Hybrid prefix cache: initialize the capture state from the plan. A tap snapshot is
        // published with tail_len == 0 and an anchor on the last committed block, so the state
        // image captured at `cursor` is the state after the block boundary `floor(cursor/64)*64`.
        // Snap every flexible tap down to a block boundary and require its frontier to sit above
        // the reuse base, so the capture that advance_prefill performs at that boundary publishes
        // a state image that is exactly the snapshot's frontier.
        if (prefix_index != nullptr && !request_plan.prefix_taps.empty()) {
            constexpr std::uint32_t kBlock = static_cast<std::uint32_t>(
                ninfer::prefix_cache::kBlockTokens);
            std::vector<std::uint32_t> tap_positions;
            tap_positions.reserve(request_plan.prefix_taps.size());
            for (std::uint32_t position : request_plan.prefix_taps) {
                const std::uint32_t aligned = (position / kBlock) * kBlock;
                if (aligned <= base) { continue; }
                if (!tap_positions.empty() && tap_positions.back() >= aligned) { continue; }
                tap_positions.push_back(aligned);
            }
            sequence.prefix_tap_positions = std::move(tap_positions);
            // A retained lane's state is reused verbatim, so any tap slots left over from its
            // previous life (consumed entries are kInvalidSlot) must be dropped before this
            // request's own slots are reserved, or a stale slot survives at index 0.
            sequence.prefix_tap_slots.clear();
            sequence.prefix_tap_slots.reserve(sequence.prefix_tap_positions.size());
            for (std::size_t tap = 0; tap < sequence.prefix_tap_positions.size(); ++tap) {
                const ninfer::prefix_cache::SlotId slot = prefix_index->reserve_snapshot_slot(0);
                if (slot == ninfer::prefix_cache::kInvalidSlot) { break; }
                sequence.prefix_tap_slots.push_back(slot);
            }
            sequence.prefix_tap_positions.resize(sequence.prefix_tap_slots.size());
            sequence.prefix_tap_next     = 0;
            sequence.prefix_taps_planned = true;
            sequence.prefix_capture      = true;
            sequence.prefix_matched_blocks = request_plan.block_path_blocks;
            sequence.prefix_pinned_nodes = request_plan.block_path;
            if (sequence.prefix_pinned_nodes.empty()) {
                sequence.prefix_pinned_nodes.push_back(prefix_index->root());
            }
            sequence.prefix_parent = sequence.prefix_pinned_nodes.back();
            sequence.prefix_prompt_tokens = prompt.token_ids;
            sequence.prefix_block_hashes  = prompt.block_hashes;
            sequence.prefix_block_extras  = prompt.block_extras;
        }

        trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
        bind_sequence_kv(sequence);
        const std::uint32_t backend_materialized =
            speculative_backend == SpeculativeBackend::Mtp
                ? std::min(capacity,
                           prompt_tokens + (initial_mtp_extent == 0 ? 0U : initial_mtp_extent - 1U))
            : backend_kv_cache() != nullptr ? prompt_tokens : 0U;
        materialize_sequence_kv(sequence, prompt_tokens, backend_materialized);
        install_sampling(sequence, request, request_plan.sampling);
        sequence.rope_delta = prompt.rope_delta;
        set_device_i32(io.rope_delta, sequence.rope_delta);
        for (int rank = 1; rank < tp; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            const ScopedDevice scope(peer[slot]->device.device);
            set_peer_i32(peer[slot]->device, peer[slot]->io.rope_delta, sequence.rope_delta);
        }

        if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::Drop ||
            request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::CaptureNew) {
            sequence.rewrite_checkpoint = {};
        } else if (request_plan.rewrite_checkpoint_action ==
                   RewriteCheckpointAction::ReclassifyExisting) {
            sequence.rewrite_checkpoint.kind = prompt.identity.rewrite_checkpoint->kind;
        }
        request.timings            = {};
        request.pending            = {};
        sequence.mtp_draft_count   = 0;
        sequence.tail_hidden_valid = base == prompt_tokens && sequence.tail_hidden_valid;
        sequence.ledger.assign(prompt.token_ids.begin(), prompt.token_ids.end());
        sequence.prefix_identity.assign(prompt);

        if (speculative_backend == SpeculativeBackend::DFlash) {
            if (!dflash || !io.dflash_decode ||
                dflash->full.has_value() != sequence.kv->backend.has_value()) {
                throw std::logic_error("DFlash prefill state is incomplete");
            }
            *dflash_host_ingress                         = {};
            dflash_host_ingress->lanes[0]                = static_cast<std::int32_t>(sequence.lane);
            dflash_host_ingress->dflash_kv_table_rows[0] =
                sequence.kv->backend ? sequence.kv->backend->bound_row() : 0;
            CUDA_CHECK(cudaMemcpyAsync(io.dflash_decode->ingress.data, dflash_host_ingress,
                                       sizeof(qwen3_6::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                       device.stream));
        }

        if (request_plan.vision) {
            std::vector<bool> used(prompt.media_payloads.size(), false);
            for (const VisionUseSpan& use : request_plan.vision->uses) {
                if (use.item_index >= used.size()) {
                    throw std::logic_error("Vision plan references a missing media payload");
                }
                used[use.item_index] = true;
            }
            for (std::size_t i = 0; i < used.size(); ++i) {
                if (!used[i]) { prompt.media_payloads[i].reset(); }
            }
        }
        if (prompt.has_media() && !request_plan.vision) { prompt.release_all_media_payloads(); }

        RequestControl::Prefill prefill{
            .prompt                     = std::move(prompt),
            .vision_plan                = std::move(request_plan.vision),
            .vision                     = nullptr,
            .transient                  = transient,
            .rewrite_checkpoint_capture = request_plan.rewrite_checkpoint_capture,
            .base                       = base,
            .cursor                     = base,
            .prompt_tokens              = prompt_tokens,
            .initial_mtp_extent         = initial_mtp_extent,
            .elapsed_seconds            = 0.0,
            .prepare_mtp                = request_plan.prepare_mtp,
            .reuse                      = request_plan.reuse,
            .mtp_bridge                 = request_plan.mtp_bridge,
            .reuse_reason               = request_plan.reuse_reason,
            .reuse_divergence           = request_plan.reuse_divergence,
        };
        request.prefill.emplace(std::move(prefill));
        auto& staged = *request.prefill;
        if (staged.vision_plan) {
            if (tp > 1) {
                // Drive the vision tower sharded across the same group as the text tower: every rank
                // owns a private scratch arena and its sharded weights, and the row-parallel
                // projections reduce over the shared collective.
                schedule::VisionTpSetup vtp;
                vtp.ec     = &execution;
                vtp.events = &*collective_comms;
                for (int r = 1; r < tp; ++r) {
                    vtp.peer_models[static_cast<std::size_t>(r)] =
                        &peer[static_cast<std::size_t>(r)]->model;
                }
                staged.vision = std::make_unique<schedule::VisionPrefillSession>(
                    device, model, work, staged.prompt, *staged.vision_plan, staged.transient, &vtp);
            } else {
                staged.vision = std::make_unique<schedule::VisionPrefillSession>(
                    device, model, work, staged.prompt, *staged.vision_plan, staged.transient);
            }
        }
        staged.elapsed_seconds = std::chrono::duration<double>(Clock::now() - started).count();
        request.lifecycle      = Lifecycle::Prefilling;
        return advance_prefill(sequence, request);
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        clear_lane(sequence, request);
        throw;
    }
}

runtime::PrefillStepResult ProgramImplCore::advance_prefill_lane(std::uint32_t lane) {
    tick_prefix_cache();
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    return advance_prefill(sequences[lane], requests[lane]);
}

void ProgramImplCore::resolve_prefill_lane(std::uint32_t lane, bool terminal) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    if (requests[lane].pending.kind != PendingKind::Begin) {
        throw std::logic_error("resolve_prefill_lane requires a pending prefill token");
    }
    resolve_non_speculative_pending(sequences[lane], requests[lane], 1, terminal);
}

void ProgramImplCore::resolve_pending_batch(std::span<const std::uint32_t> lanes,
                                            std::span<const std::uint32_t> accepted_tokens,
                                            std::span<const std::uint8_t> terminal,
                                            std::span<const std::uint8_t> cancelled) {
    if (lanes.empty() || lanes.size() > max_concurrency || accepted_tokens.size() != lanes.size() ||
        terminal.size() != lanes.size() || cancelled.size() != lanes.size()) {
        throw std::invalid_argument("pending batch resolution has inconsistent membership");
    }

    if (speculative_backend == SpeculativeBackend::None) {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending ||
                requests[lane].pending.kind != PendingKind::Ordinary) {
                throw std::logic_error("ordinary pending batch no longer matches Program state");
            }
            if (cancelled[row]) {
                clear_lane(sequences[lane], requests[lane]);
            } else {
                resolve_non_speculative_pending(sequences[lane], requests[lane],
                                                accepted_tokens[row], terminal[row] != 0);
            }
        }
        return;
    }

    if (!replay_records) {
        throw std::logic_error("speculative pending batch has no ReplaySSM records");
    }

    std::array<ops::GdnReplayFoldRow, kMaximumConcurrency> fold_rows{};
    std::array<std::int32_t, kMaximumConcurrency> hidden_selectors{};
    bool needs_hidden_correction = false;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending ||
            requests[lane].pending.kind != PendingKind::Speculative) {
            throw std::logic_error("speculative pending batch no longer matches Program state");
        }
        const PendingCandidate& pending = requests[lane].pending;
        const SequenceState& sequence   = sequences[lane];
        if (sequence.execution_frontier != pending.base_E ||
            sequence.ledger_frontier != pending.base_S ||
            sequence.ledger.size() != pending.base_S ||
            sequence.prefix_identity.size() != pending.base_S ||
            sequence.text_kv_valid != pending.base_E ||
            (speculative_backend == SpeculativeBackend::Mtp &&
             sequence.mtp_kv_valid != pending.base_E) ||
            (speculative_backend == SpeculativeBackend::DFlash &&
             sequence.dflash_context_frontier != pending.base_E)) {
            throw std::logic_error("speculative pending row is not at its recorded base");
        }
        const std::uint32_t committed = cancelled[row] ? 0U : accepted_tokens[row];
        if ((cancelled[row] && accepted_tokens[row] != 0) ||
            (!cancelled[row] && (committed == 0 || committed > pending.produced ||
                                 (!terminal[row] && committed != pending.produced)))) {
            throw std::logic_error("speculative pending row has an invalid committed prefix");
        }
        fold_rows[row] = ops::GdnReplayFoldRow{
            .linear_state_slot = LinearStateSlots::current_state_slot(lane, max_concurrency),
            .commit_columns    = static_cast<std::int32_t>(committed),
        };
        const bool partial_terminal =
            !cancelled[row] && terminal[row] && committed < pending.produced;
        hidden_selectors[row] =
            static_cast<std::int32_t>(partial_terminal ? committed - 1U : pending.produced - 1U);
        needs_hidden_correction = needs_hidden_correction || partial_terminal;
    }

    const auto tail_started = Clock::now();
    try {
        ops::gdn_replay_fold(*replay_records, decoder->linear_attention.all_layers_view(),
                             std::span<const ops::GdnReplayFoldRow>(fold_rows.data(), lanes.size()),
                             device.stream);
        // Rank 1 folds ITS OWN records into ITS OWN halved GDN state, with the identical row set
        // and commit counts: the accepted prefix is a property of the round, not of a device, and
        // the two records differ only in which heads and conv channels they cover. The shard
        // geometry FoldGeometry<48, 8, 24, 5120> this call resolves to is registered explicitly;
        // without that registration the fold would reject the peer's record shape outright.
        for (int rank = 1; rank < tp; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            if (!peer[slot]->replay_records) {
                throw std::logic_error("peer speculative round has no ReplaySSM records");
            }
            const ScopedDevice scope(peer[slot]->device.device);
            ops::gdn_replay_fold(
                *peer[slot]->replay_records, peer[slot]->decoder->linear_attention.all_layers_view(),
                std::span<const ops::GdnReplayFoldRow>(fold_rows.data(), lanes.size()),
                peer[slot]->device.stream);
        }

        if (needs_hidden_correction) {
            const auto batch = static_cast<std::int32_t>(lanes.size());
            Tensor selector_tensor;
            Tensor hidden;
            Tensor selected;
            Tensor destinations;
            if (speculative_backend == SpeculativeBackend::Mtp && io.mtp_decode) {
                qwen3_6::MtpDecodeState& frame = *io.mtp_decode;
                selector_tensor                = frame.current_extents.slice(0, 0, batch);
                hidden                         = frame.target_hidden.slice(2, 0, batch);
                selected     = frame.target_continuation_hidden.slice(1, 0, batch);
                destinations = frame.lanes.slice(0, 0, batch);
            } else if (speculative_backend == SpeculativeBackend::DFlash && io.dflash_decode) {
                qwen3_6::DFlashDecodeState& frame = *io.dflash_decode;
                selector_tensor                   = frame.proposal_extents.slice(0, 0, batch);
                hidden                            = frame.target_hidden.slice(2, 0, batch);
                selected     = frame.target_continuation_hidden.slice(1, 0, batch);
                destinations = frame.lanes.slice(0, 0, batch);
            } else {
                throw std::logic_error("partial speculative commit has no target frame");
            }
            CUDA_CHECK(cudaMemcpyAsync(selector_tensor.data, hidden_selectors.data(),
                                       lanes.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                       device.stream));
            ops::speculative_select_accepted_hidden(hidden, selector_tensor, selected,
                                                    device.stream);
            ops::scatter(selected, destinations, tail_hidden_store, device.stream);
        }

        if (speculative_backend == SpeculativeBackend::DFlash) {
            std::array<std::uint32_t, kMaximumConcurrency> append_lanes{};
            std::array<std::uint32_t, kMaximumConcurrency> append_starts{};
            std::array<std::uint32_t, kMaximumConcurrency> append_counts{};
            std::size_t append_size = 0;
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                if (!cancelled[row] && terminal[row]) {
                    append_lanes[append_size]  = lanes[row];
                    append_starts[append_size] = requests[lanes[row]].pending.base_E;
                    append_counts[append_size] = accepted_tokens[row];
                    ++append_size;
                }
            }
            if (append_size != 0) {
                enqueue_dflash_context_append(
                    std::span<const std::uint32_t>(append_lanes.data(), append_size),
                    std::span<const std::uint32_t>(append_starts.data(), append_size),
                    std::span<const std::uint32_t>(append_counts.data(), append_size));
            }
        }

        // Peer first, then rank 0 -- the same order every other tp2 handler uses, so that
        // `clear_lane` below cannot release KV pages a peer's fold still references.
        synchronize_peers();
        device.synchronize();
        work.reset();
        reset_peer_work();
    } catch (...) {
        try {
            synchronize_peers();
            device.synchronize();
        } catch (...) {}
        work.reset();
        reset_peer_work();
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }

    const double tail_seconds = std::chrono::duration<double>(Clock::now() - tail_started).count();
    const std::uint32_t width = draft_window + 1U;
    try {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence = sequences[lanes[row]];
            RequestControl& request = requests[lanes[row]];
            if (cancelled[row]) {
                clear_lane(sequence, request);
                continue;
            }

            const PendingCandidate pending = request.pending;
            const std::uint32_t committed  = accepted_tokens[row];
            const TokenId* token_base =
                speculative_backend == SpeculativeBackend::Mtp
                    ? mtp_host_egress->licensed_tokens.data() + row * width
                    : dflash_host_egress->licensed_tokens.data() + row * width;
            sequence.ledger.insert(sequence.ledger.end(), token_base, token_base + committed);
            sequence.prefix_identity.append_generated(committed, sequence.rope_delta);
            sequence.execution_frontier = pending.base_E + committed;
            sequence.ledger_frontier    = pending.base_S + committed;
            sequence.text_kv_valid      = sequence.execution_frontier;
            sequence.tail_hidden_valid  = true;

            if (speculative_backend == SpeculativeBackend::Mtp) {
                sequence.mtp_kv_valid = sequence.execution_frontier;
                if (terminal[row]) {
                    sequence.mtp_draft_count = 0;
                } else {
                    const std::int32_t next  = mtp_host_egress->next_extents[row];
                    sequence.mtp_draft_count = static_cast<std::uint32_t>(next);
                    for (std::uint32_t step = 0; step < sequence.mtp_draft_count; ++step) {
                        sequence.mtp_drafts[step] =
                            mtp_host_egress->next_drafts[step * max_concurrency + row];
                    }
                }
            } else {
                sequence.dflash_context_frontier =
                    terminal[row] ? sequence.execution_frontier : pending.base_E;
            }

            trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
            if (terminal[row]) {
                release_sequence_growth_entitlement(sequence);
                unbind_sequence_kv(sequence);
                // Release this request's cache pins now that it is done: its committed blocks stay
                // in the cache but must become evictable, otherwise the next request's admission
                // sees them as permanently unavailable and the tap-slot array accumulates.
                finalize_prefix_cache_state(sequence);
                sequence.retained = true;
                request.lifecycle = Lifecycle::Complete;
            } else {
                request.lifecycle = Lifecycle::Active;
            }
            request.pending = {};
            request.timings.decode_seconds += tail_seconds;
        }
    } catch (...) {
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

void ProgramImplCore::abort_lane(std::uint32_t lane) noexcept {
    if (lane >= max_concurrency) { return; }
    clear_lane(sequences[lane], requests[lane]);
}

bool ProgramImplCore::has_retained_lane(std::uint32_t lane) const noexcept {
    return lane < max_concurrency && sequences[lane].retained;
}

void ProgramImplCore::evict_retained_lane(std::uint32_t lane) noexcept {
    if (!has_retained_lane(lane)) { return; }
    clear_lane(sequences[lane], requests[lane]);
}

GenerationTimings ProgramImplCore::generation_timings_lane(std::uint32_t lane) const noexcept {
    return lane < max_concurrency ? requests[lane].timings : GenerationTimings{};
}

SpeculativeStats ProgramImplCore::speculative_stats_lane(std::uint32_t lane) const noexcept {
    return lane < max_concurrency ? requests[lane].speculative_stats : SpeculativeStats{};
}

void ProgramImplCore::finalize_prefix_cache_state(SequenceState& sequence) noexcept {
    if (prefix_index == nullptr) { return; }
    // M2: persist device-resident blocks that gained no host copy yet. This runs BEFORE the
    // lane's pins drop, and each issue pins its node against eviction for the transfer's
    // duration, so the D2H can never race a page release. Best-effort; never throws.
    if (prefix_host_pool != nullptr) {
        for (ninfer::prefix_cache::NodeId id : sequence.prefix_pinned_nodes) {
            issue_block_write_through(id);
        }
    }
    // Slots consumed by published snapshots were marked invalid at capture time; only the
    // genuinely unconsumed reservations go back to the free list.
    for (ninfer::prefix_cache::SlotId slot : sequence.prefix_tap_slots) {
        if (slot != ninfer::prefix_cache::kInvalidSlot) {
            prefix_index->release_reserved_slot(slot);
        }
    }
    // Drop this request's pins so its committed blocks become evictable again. Without this, a
    // completed (retained) request leaves its cache entries permanently pinned: they can never be
    // evicted, and the next request's admission sees them as unavailable capacity.
    if (sequence.prefix_snapshot != ninfer::prefix_cache::kInvalidSnapshot) {
        prefix_index->unpin_snapshot(sequence.prefix_snapshot);
    }
    for (ninfer::prefix_cache::NodeId id : sequence.prefix_pinned_nodes) {
        prefix_index->unpin_node(id);
    }
    // Reset the per-request capture state so the next request starts clean.
    sequence.prefix_pinned_nodes.clear();
    sequence.prefix_tap_positions.clear();
    sequence.prefix_tap_slots.clear();
    sequence.prefix_tap_next      = 0;
    sequence.prefix_taps_planned  = false;
    sequence.prefix_capture       = false;
    sequence.prefix_matched_blocks = 0;
    sequence.prefix_parent         = ninfer::prefix_cache::kInvalidNode;
    sequence.prefix_prompt_tokens.clear();
    sequence.prefix_block_hashes.clear();
    sequence.prefix_block_extras.clear();
    sequence.prefix_snapshot = ninfer::prefix_cache::kInvalidSnapshot;
    sequence.prefix_borrowed = false;
    // Settle this request's write-throughs now that its pins are dropped: wait for the D2H copies
    // to land and finalize them, so every committed block is host-backed and evictable before any
    // later admission reads the pool counts. Without this, the next request's admission sees the
    // blocks as still in-flight (restore_pins != 0), counts them as non-evictable, and rejects a
    // request that would otherwise fit once the prior blocks can be demoted to host.
    settle_prefix_writes();
}

void ProgramImplCore::clear_lane(SequenceState& sequence, RequestControl& request) noexcept {
    request.prefill.reset();
    finalize_prefix_cache_state(sequence);
    sequence.kv.reset();
    request.lifecycle           = Lifecycle::Empty;
    sequence.execution_frontier = 0;
    sequence.ledger_frontier    = 0;
    sequence.ledger.clear();
    sequence.prefix_identity.clear();
    sequence.text_kv_valid           = 0;
    sequence.mtp_kv_valid            = 0;
    sequence.dflash_context_frontier = 0;
    sequence.mtp_draft_count         = 0;
    sequence.tail_hidden_valid       = false;
    sequence.retained                = false;
    sequence.rewrite_checkpoint      = {};
    request.pending                  = {};
}

qwen3_6::PagedKVCache* ProgramImplCore::backend_kv_cache() noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (speculative_backend == SpeculativeBackend::DFlash && dflash && dflash->full) {
        return &*dflash->full;
    }
    return nullptr;
}

const qwen3_6::PagedKVCache* ProgramImplCore::backend_kv_cache() const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (speculative_backend == SpeculativeBackend::DFlash && dflash && dflash->full) {
        return &*dflash->full;
    }
    return nullptr;
}

std::uint32_t ProgramImplCore::backend_kv_valid(const SequenceState& sequence) const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return sequence.mtp_kv_valid; }
    if (speculative_backend == SpeculativeBackend::DFlash && backend_kv_cache() != nullptr) {
        return sequence.dflash_context_frontier;
    }
    return 0;
}

void ProgramImplCore::reserve_sequence_kv(SequenceState& sequence, std::uint32_t text_pages,
                                          std::uint32_t backend_pages) {
    if (sequence.kv) { throw std::logic_error("sequence already owns a KV allocation bundle"); }
    if (text_pages == 0 || (backend_kv_cache() == nullptr) != (backend_pages == 0)) {
        throw std::invalid_argument("KV allocation entitlement does not match the active backend");
    }

    std::array<PagedKVReservation, 2> reservations{};
    std::size_t count     = 0;
    reservations[count++] = PagedKVReservation{
        .pool             = &decoder->text_kv.pool(),
        .page_entitlement = text_pages,
    };
    if (qwen3_6::PagedKVCache* backend = backend_kv_cache(); backend != nullptr) {
        reservations[count++] = PagedKVReservation{
            .pool             = &backend->pool(),
            .page_entitlement = backend_pages,
        };
    }

    // When the pools are exhausted by cache-resident pages, evict device-resident cache nodes
    // (demoting or erasing them through the sink) until the reservation fits.
    std::vector<PagedKVAllocation> allocations;
    for (;;) {
        try {
            allocations = reserve_paged_kv_bundle(
                std::span<const PagedKVReservation>(reservations.data(), count));
            break;
        } catch (const std::bad_alloc&) {
            if (prefix_index == nullptr ||
                prefix_index->evict_device_lru() == ninfer::prefix_cache::kInvalidNode) {
                throw;
            }
        }
    }
    SequenceKVBundle bundle;
    bundle.text = std::move(allocations[0]);
    if (count == 2) { bundle.backend.emplace(std::move(allocations[1])); }
    for (int rank = 1; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        PeerRuntime& p  = *peer[slot];
        // Same pool geometry, same call order, same entitlement: every rank's reservation takes the
        // same page ids, so the block tables agree without ever being compared. The MTP pool rides
        // the SAME bundle call as the text pool, so its page ids match rank 0's too.
        std::array<PagedKVReservation, 2> peer_reservations{};
        std::size_t peer_count     = 0;
        peer_reservations[peer_count++] =
            PagedKVReservation{.pool = &p.decoder->text_kv.pool(), .page_entitlement = text_pages};
        qwen3_6::PagedKVCache* peer_backend =
            speculative_backend == SpeculativeBackend::Mtp
                ? p.decoder->mtp_cache()
                : (p.dflash && p.dflash->full ? &*p.dflash->full : nullptr);
        if (peer_backend != nullptr) {
            peer_reservations[peer_count++] =
                PagedKVReservation{.pool = &peer_backend->pool(), .page_entitlement = backend_pages};
        }
        if (peer_count != count) {
            throw std::logic_error("peer KV pool set does not match rank 0's");
        }
        std::vector<PagedKVAllocation> peer_allocations = reserve_paged_kv_bundle(
            std::span<const PagedKVReservation>(peer_reservations.data(), peer_count));
        bundle.text_peer[slot].emplace(std::move(peer_allocations[0]));
        if (peer_count == 2) { bundle.backend_peer[slot].emplace(std::move(peer_allocations[1])); }
    }
    sequence.kv.emplace(std::move(bundle));
}

void ProgramImplCore::resize_paged_kv_with_eviction(std::span<PagedKVResize> changes) {
    // `admission` is granted against an estimate that credits reclaimable cache pages
    // (`can_admit_lane`), but `resize_paged_kv_bundle` only accepts room that is free right now.
    // Realize the credited room by evicting device-resident cache nodes, exactly as
    // `reserve_sequence_kv` and `acquire_cache_bundle` do -- otherwise a resize admitted by that
    // estimate throws std::bad_alloc and, because the admission region is engine-scoped, takes the
    // whole Engine down. `resize_paged_kv_bundle` validates every extent before mutating, so a
    // bad_alloc leaves no partial state and the retry is safe.
    for (;;) {
        try {
            resize_paged_kv_bundle(changes);
            return;
        } catch (const std::bad_alloc&) {
            if (prefix_index == nullptr ||
                prefix_index->evict_device_lru() == ninfer::prefix_cache::kInvalidNode) {
                throw;
            }
        }
    }
}

void ProgramImplCore::resize_sequence_kv_entitlement(SequenceState& sequence,
                                                     std::uint32_t text_pages,
                                                     std::uint32_t backend_pages) {
    if (!sequence.kv || text_pages == 0 ||
        (sequence.kv->backend.has_value() != (backend_pages != 0))) {
        throw std::invalid_argument("KV resize entitlement does not match the sequence bundle");
    }
    // `text_pages`/`backend_pages` are the full plan entitlements; borrowed pages (prefix reuse
    // plus already-promoted blocks) are cache-owned, so only `plan - borrowed` belongs to this
    // allocation's entitlement.
    const auto owned_entitlement = [](std::uint32_t plan_pages, std::uint32_t borrowed) {
        return plan_pages > borrowed ? plan_pages - borrowed : 1U;
    };
    const std::uint32_t text_owned =
        owned_entitlement(text_pages, sequence.kv->text.borrowed_page_count());
    const std::uint32_t backend_owned =
        sequence.kv->backend
            ? owned_entitlement(backend_pages, sequence.kv->backend->borrowed_page_count())
            : 0U;

    std::array<PagedKVResize, 2> changes{};
    std::size_t count = 0;
    changes[count++]  = PagedKVResize{
         .allocation       = &sequence.kv->text,
         .mapped_pages     = sequence.kv->text.mapped_page_count(),
         .page_entitlement = text_owned,
    };
    if (sequence.kv->backend) {
        changes[count++] = PagedKVResize{
            .allocation       = &*sequence.kv->backend,
            .mapped_pages     = sequence.kv->backend->mapped_page_count(),
            .page_entitlement = backend_owned,
        };
    }
    resize_paged_kv_with_eviction(std::span<PagedKVResize>(changes.data(), count));
    for (int rank = 1; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (!sequence.kv->text_peer[slot]) { continue; }
        std::array<PagedKVResize, 2> peer_changes{};
        std::size_t peer_count = 0;
        peer_changes[peer_count++] =
            PagedKVResize{.allocation       = &*sequence.kv->text_peer[slot],
                          .mapped_pages     = sequence.kv->text_peer[slot]->mapped_page_count(),
                          .page_entitlement = text_owned};
        if (sequence.kv->backend_peer[slot]) {
            peer_changes[peer_count++] =
                PagedKVResize{.allocation   = &*sequence.kv->backend_peer[slot],
                              .mapped_pages = sequence.kv->backend_peer[slot]->mapped_page_count(),
                              .page_entitlement = backend_owned};
        }
        resize_paged_kv_with_eviction(
            std::span<PagedKVResize>(peer_changes.data(), peer_count));
    }
}

void ProgramImplCore::bind_sequence_kv(SequenceState& sequence) {
    if (!sequence.kv || sequence.kv->text.bound_row() >= 0 ||
        (sequence.kv->backend && sequence.kv->backend->bound_row() >= 0)) {
        throw std::logic_error("KV allocation bundle is unavailable or already bound");
    }
    const std::int32_t row = static_cast<std::int32_t>(sequence.lane);
    sequence.kv->text.bind_row(row, device.stream);
    try {
        if (sequence.kv->backend) { sequence.kv->backend->bind_row(row, device.stream); }
        for (int rank = 1; rank < tp; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            if (!sequence.kv->text_peer[slot]) { continue; }
            const ScopedDevice scope(peer[slot]->device.device);
            sequence.kv->text_peer[slot]->bind_row(row, peer[slot]->device.stream);
            set_peer_i32(peer[slot]->device, peer[slot]->io.text_kv_table_row,
                         sequence.kv->text_peer[slot]->bound_row());
            if (sequence.kv->backend_peer[slot]) {
                sequence.kv->backend_peer[slot]->bind_row(row, peer[slot]->device.stream);
            }
            set_peer_i32(peer[slot]->device, peer[slot]->io.backend_kv_table_row,
                         sequence.kv->backend_peer[slot]
                             ? sequence.kv->backend_peer[slot]->bound_row()
                             : 0);
        }
        set_device_i32(io.text_kv_table_row, sequence.kv->text.bound_row());
        set_device_i32(io.backend_kv_table_row,
                       sequence.kv->backend ? sequence.kv->backend->bound_row() : 0);
    } catch (...) {
        for (int rank = 1; rank < tp; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            if (sequence.kv->backend_peer[slot] &&
                sequence.kv->backend_peer[slot]->bound_row() >= 0) {
                sequence.kv->backend_peer[slot]->unbind_row();
            }
            if (sequence.kv->text_peer[slot] && sequence.kv->text_peer[slot]->bound_row() >= 0) {
                sequence.kv->text_peer[slot]->unbind_row();
            }
        }
        if (sequence.kv->backend && sequence.kv->backend->bound_row() >= 0) {
            sequence.kv->backend->unbind_row();
        }
        sequence.kv->text.unbind_row();
        throw;
    }
}

void ProgramImplCore::unbind_sequence_kv(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    for (int rank = 1; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (sequence.kv->backend_peer[slot]) { sequence.kv->backend_peer[slot]->unbind_row(); }
        if (sequence.kv->text_peer[slot]) { sequence.kv->text_peer[slot]->unbind_row(); }
    }
    if (sequence.kv->backend) { sequence.kv->backend->unbind_row(); }
    sequence.kv->text.unbind_row();
}

void ProgramImplCore::materialize_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                              std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > capacity) {
        throw std::logic_error("KV materialization request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence.kv->backend) {
        throw std::logic_error("backend KV materialization requested without an allocation");
    }
    if (main_tokens > sequence.kv->text.mapped_token_capacity()) {
        sequence.kv->text.materialize_tokens(main_tokens, device.stream);
        for (int rank = 1; rank < tp; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            if (!sequence.kv->text_peer[slot]) { continue; }
            const ScopedDevice scope(peer[slot]->device.device);
            sequence.kv->text_peer[slot]->materialize_tokens(main_tokens, peer[slot]->device.stream);
        }
    }
    if (backend_tokens != 0 && backend_tokens > sequence.kv->backend->mapped_token_capacity()) {
        sequence.kv->backend->materialize_tokens(backend_tokens, device.stream);
        for (int rank = 1; rank < tp; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            if (!sequence.kv->backend_peer[slot]) { continue; }
            const ScopedDevice scope(peer[slot]->device.device);
            sequence.kv->backend_peer[slot]->materialize_tokens(backend_tokens,
                                                                peer[slot]->device.stream);
        }
    }
}

void ProgramImplCore::trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                       std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > main_tokens) {
        throw std::logic_error("KV trim request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence.kv->backend) {
        throw std::logic_error("backend KV trim requested without an allocation");
    }
    sequence.kv->text.trim_tokens(main_tokens);
    if (sequence.kv->backend) { sequence.kv->backend->trim_tokens(backend_tokens); }
    for (int rank = 1; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (sequence.kv->text_peer[slot]) { sequence.kv->text_peer[slot]->trim_tokens(main_tokens); }
        if (sequence.kv->backend_peer[slot]) {
            sequence.kv->backend_peer[slot]->trim_tokens(backend_tokens);
        }
    }
}

void ProgramImplCore::release_sequence_growth_entitlement(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    sequence.kv->text.cancel_unmapped_entitlement();
    if (sequence.kv->backend) { sequence.kv->backend->cancel_unmapped_entitlement(); }
    for (int rank = 1; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (sequence.kv->text_peer[slot]) {
            sequence.kv->text_peer[slot]->cancel_unmapped_entitlement();
        }
        if (sequence.kv->backend_peer[slot]) {
            sequence.kv->backend_peer[slot]->cancel_unmapped_entitlement();
        }
    }
}

qwen3_6::PagedKVCacheView ProgramImplCore::text_kv_view(const SequenceState& sequence) const {
    if (!sequence.kv) { throw std::logic_error("sequence has no KV allocation bundle"); }
    return decoder->text_kv.execution_view(sequence.kv->text);
}

qwen3_6::PagedKVCacheView ProgramImplCore::mtp_kv_view(const SequenceState& sequence) const {
    if (speculative_backend != SpeculativeBackend::Mtp) { return {}; }
    if (decoder->mtp_cache() == nullptr || !sequence.kv || !sequence.kv->backend) {
        throw std::logic_error("sequence has no MTP KV allocation");
    }
    return decoder->mtp_cache()->execution_view(*sequence.kv->backend);
}

std::array<qwen3_6::PagedKVCacheView, kMaximumDevices>
ProgramImplCore::mtp_kv_view_peer(const SequenceState& sequence) const {
    std::array<qwen3_6::PagedKVCacheView, kMaximumDevices> views{};
    if (speculative_backend != SpeculativeBackend::Mtp || tp <= 1) { return views; }
    for (int rank = 1; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (peer[slot]->decoder->mtp_cache() == nullptr || !sequence.kv ||
            !sequence.kv->backend_peer[slot]) {
            throw std::logic_error("sequence has no peer MTP KV allocation");
        }
        views[slot] =
            peer[slot]->decoder->mtp_cache()->execution_view(*sequence.kv->backend_peer[slot]);
    }
    return views;
}

std::array<const schedule::TpPeerCore*, kMaximumDevices>
ProgramImplCore::peer_ptrs() const noexcept {
    std::array<const schedule::TpPeerCore*, kMaximumDevices> out{};
    for (int rank = 1; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (peer_core[slot]) { out[slot] = &*peer_core[slot]; }
    }
    return out;
}

void ProgramImplCore::set_device_i32(Tensor& tensor, std::int32_t value) {
    CUDA_CHECK(
        cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice, device.stream));
}

void ProgramImplCore::set_peer_i32(const DeviceContext& rank_device, Tensor& tensor,
                                   std::int32_t value) {
    CUDA_CHECK(cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice,
                               rank_device.stream));
}

void ProgramImplCore::synchronize_peers() noexcept {
    for (int rank = 1; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (peer[slot] && peer[slot]->device.stream != nullptr) {
            (void)cudaStreamSynchronize(peer[slot]->device.stream);
        }
    }
}

void ProgramImplCore::reset_peer_work() noexcept {
    for (int rank = 1; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (peer[slot]) { peer[slot]->work.reset(); }
    }
}

void ProgramImplCore::ordered_reset(SequenceState& sequence) {
    decoder->linear_attention.zero_slot(
        LinearStateSlots::current_state_slot(sequence.lane, max_concurrency), device.stream);
    for (int rank = 1; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        PeerRuntime& p  = *peer[slot];
        const ScopedDevice scope(p.device.device);
        p.decoder->linear_attention.zero_slot(
            LinearStateSlots::current_state_slot(sequence.lane, max_concurrency), p.device.stream);
        p.work.reset();
        set_peer_i32(p.device, p.io.pos, 0);
        set_peer_i32(p.device, p.io.rope_pos, 0);
        set_peer_i32(p.device, p.io.rope_delta, 0);
    }
    work.reset();
    set_device_i32(io.pos, 0);
    set_device_i32(io.rope_pos, 0);
    set_device_i32(io.rope_delta, 0);
    if (io.mtp) { set_device_i32(io.mtp->position, 0); }
    sequence.text_kv_valid           = 0;
    sequence.mtp_kv_valid            = 0;
    sequence.dflash_context_frontier = 0;
}

void ProgramImplCore::prepare_graphs() {
    if (!use_cuda_graph) { return; }
    SequenceState& sequence = sequences[0];

    // Every helper below has to run the SAME operation on every peer device at the same order,
    // because each peer's runtime is a structural mirror of rank 0's: same page ids, same block
    // tables, same zeroed state. `on_peer` is the one place that establishes a peer device as
    // current; the mirrors below never touch the current device themselves.
    const auto on_peer = [&](auto&& body) {
        for (int rank = 1; rank < tp; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            const ScopedDevice scope(peer[slot]->device.device);
            body(rank, *peer[slot]);
        }
    };
    const auto synchronize_all = [&] {
        for (int rank = 1; rank < tp; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            peer[slot]->device.synchronize();
        }
        device.synchronize();
    };

    std::vector<PagedKVAllocation> text_capture_allocations;
    std::array<std::vector<PagedKVAllocation>, kMaximumDevices> peer_text_capture_allocations;
    std::vector<PagedKVAllocation> mtp_capture_allocations;
    std::array<std::vector<PagedKVAllocation>, kMaximumDevices> peer_mtp_capture_allocations;
    std::vector<PagedKVAllocation> dflash_capture_allocations;
    const auto reserve_rows_in = [&](qwen3_6::PagedKVCache& cache,
                                     std::vector<PagedKVAllocation>& allocations,
                                     cudaStream_t stream, const char* label) {
        PagedKVPool& pool = cache.pool();
        if (pool.page_group_count() < max_concurrency) {
            throw std::invalid_argument(std::string(label) +
                                        " cannot provide one Paged KV page per concurrent request");
        }
        allocations.reserve(max_concurrency);
        for (std::uint32_t row = 0; row < max_concurrency; ++row) {
            allocations.push_back(pool.reserve(1));
            PagedKVAllocation& allocation = allocations.back();
            allocation.bind_row(static_cast<std::int32_t>(row), stream);
            allocation.materialize_pages(1, stream);

            // Capture profiles exercise arbitrary context envelopes. Repeating each row's private
            // page across its temporary table keeps every dummy read/write address valid without
            // reserving C full contexts solely for graph construction.
            const std::int32_t page = allocation.page_ids().front();
            std::vector<std::int32_t> repeated(pool.logical_page_capacity(), page);
            Tensor table = pool.block_table_row(static_cast<std::int32_t>(row));
            CUDA_CHECK(cudaMemcpyAsync(table.data, repeated.data(), table.bytes(),
                                       cudaMemcpyHostToDevice, stream));
        }
    };
    const auto reserve_capture_rows = [&](qwen3_6::PagedKVCache& cache,
                                          std::vector<PagedKVAllocation>& allocations,
                                          const char* label) {
        reserve_rows_in(cache, allocations, device.stream, label);
    };
    reserve_capture_rows(decoder->text_kv, text_capture_allocations, "target KV cache");
    // The peer's pool has identical page geometry and receives the identical reserve sequence, so
    // the two allocations take the same page ids and publish identical block tables -- the same
    // by-construction lockstep the live KV bundle relies on.
    on_peer([&](int rank, PeerRuntime& p) {
        reserve_rows_in(p.decoder->text_kv, peer_text_capture_allocations[rank], p.device.stream,
                        "peer target KV cache");
    });
    if (speculative_backend == SpeculativeBackend::Mtp) {
        reserve_capture_rows(*decoder->mtp_cache(), mtp_capture_allocations, "MTP KV cache");
        on_peer([&](int rank, PeerRuntime& p) {
            reserve_rows_in(*p.decoder->mtp_cache(), peer_mtp_capture_allocations[rank],
                            p.device.stream, "peer MTP KV cache");
        });
    } else if (speculative_backend == SpeculativeBackend::DFlash && dflash->full) {
        reserve_capture_rows(*dflash->full, dflash_capture_allocations, "DFlash Full KV cache");
    }
    synchronize_all();

    // Graph residency is measured on EACH device: the single cross-device graph materializes
    // driver and module state on both, and the plan's allowance is a per-device budget.
    // cudaMemGetInfo insists on a total-bytes out-parameter that nothing here reads, so it is
    // consumed and discarded in one place rather than carried as a dead local.
    const auto free_device_bytes = [] {
        std::size_t free  = 0;
        std::size_t total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        return free;
    };
    std::array<std::size_t, kMaximumDevices> free_before{};
    free_before[0] = free_device_bytes();
    on_peer([&](int rank, PeerRuntime&) { free_before[rank] = free_device_bytes(); });

    const auto clear_stable_controls = [&] {
        std::vector<Tensor> controls{
            io.token,
            io.pos,
            io.rope_pos,
            io.rope_delta,
        };
        if (io.mtp) {
            controls.push_back(io.mtp->position);
            controls.push_back(io.mtp->draft_tokens);
            controls.push_back(io.mtp->target_input_ids);
            controls.push_back(io.mtp->target_positions);
        }
        if (io.dflash_prefill) { controls.push_back(io.dflash_prefill->produced_count); }
        for (const Tensor& tensor : controls) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
        }
        on_peer([&](int, PeerRuntime& p) {
            std::vector<Tensor> peer_controls{p.io.token, p.io.pos, p.io.rope_pos,
                                              p.io.rope_delta};
            if (p.io.mtp) {
                peer_controls.push_back(p.io.mtp->position);
                peer_controls.push_back(p.io.mtp->draft_tokens);
                peer_controls.push_back(p.io.mtp->target_input_ids);
                peer_controls.push_back(p.io.mtp->target_positions);
            }
            for (const Tensor& tensor : peer_controls) {
                CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), p.device.stream));
            }
        });
    };
    const auto zero_capture_pages = [&](qwen3_6::PagedKVCache& cache,
                                        const std::vector<PagedKVAllocation>& allocations,
                                        std::uint32_t batch_size) {
        std::vector<std::int32_t> pages;
        pages.reserve(batch_size);
        for (std::uint32_t row = 0; row < batch_size; ++row) {
            pages.push_back(allocations[row].page_ids().front());
        }
        cache.pool().zero_pages(pages, device.stream);
    };
    const auto zero_peer_capture_pages = [&](std::uint32_t batch_size) {
        on_peer([&](int rank, PeerRuntime& p) {
            const auto zero_pool = [&](qwen3_6::PagedKVCache& cache,
                                       const std::vector<PagedKVAllocation>& allocations) {
                std::vector<std::int32_t> pages;
                pages.reserve(batch_size);
                for (std::uint32_t row = 0; row < batch_size; ++row) {
                    pages.push_back(allocations[row].page_ids().front());
                }
                cache.pool().zero_pages(pages, p.device.stream);
            };
            zero_pool(p.decoder->text_kv, peer_text_capture_allocations[rank]);
            if (p.decoder->mtp_cache() != nullptr) {
                zero_pool(*p.decoder->mtp_cache(), peer_mtp_capture_allocations[rank]);
            }
        });
    };
    const auto zero_cyclic_lane = [&](CyclicKVCache& cache, std::uint32_t lane) {
        for (std::uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = cache.layer_view(layer);
            const Tensor k                    = view.k.slice(3, static_cast<std::int32_t>(lane), 1);
            const Tensor v                    = view.v.slice(3, static_cast<std::int32_t>(lane), 1);
            CUDA_CHECK(cudaMemsetAsync(k.data, 0, k.bytes(), device.stream));
            CUDA_CHECK(cudaMemsetAsync(v.data, 0, v.bytes(), device.stream));
        }
    };

    const auto prepare_representative = [&](std::uint32_t frontier, std::uint32_t batch_size) {
        if (batch_size == 0 || batch_size > max_concurrency) {
            throw std::logic_error("CUDA Graph representative batch is invalid");
        }
        work.reset();
        on_peer([&](int, PeerRuntime& p) { p.work.reset(); });
        clear_stable_controls();
        zero_capture_pages(decoder->text_kv, text_capture_allocations, batch_size);
        zero_peer_capture_pages(batch_size);
        if (decoder->mtp_cache() != nullptr) {
            zero_capture_pages(*decoder->mtp_cache(), mtp_capture_allocations, batch_size);
        }
        if (dflash && dflash->full) {
            zero_capture_pages(*dflash->full, dflash_capture_allocations, batch_size);
        }
        on_peer([&](int, PeerRuntime& p) {
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                p.decoder->linear_attention.zero_slot(
                    LinearStateSlots::current_state_slot(row, max_concurrency), p.device.stream);
            }
        });
        for (std::uint32_t row = 0; row < batch_size; ++row) {
            decoder->linear_attention.zero_slot(
                LinearStateSlots::current_state_slot(row, max_concurrency), device.stream);
            if (dflash) {
                zero_cyclic_lane(dflash->local, row);
                const Tensor pending =
                    dflash->pending_features.slice(2, static_cast<std::int32_t>(row), 1);
                CUDA_CHECK(cudaMemsetAsync(pending.data, 0, pending.bytes(), device.stream));
            }
        }
        set_device_i32(io.pos, checked_i32(frontier, "graph representative position"));
        set_device_i32(io.rope_pos, checked_i32(frontier, "graph representative rope position"));
        on_peer([&](int, PeerRuntime& p) {
            set_peer_i32(p.device, p.io.pos, checked_i32(frontier, "peer graph representative position"));
            set_peer_i32(p.device, p.io.rope_pos,
                         checked_i32(frontier, "peer graph representative rope position"));
        });
        if (io.mtp) {
            set_device_i32(io.mtp->position,
                           checked_i32(frontier, "graph representative MTP position"));
        }
        on_peer([&](int, PeerRuntime& p) {
            if (p.io.mtp) {
                set_peer_i32(p.device, p.io.mtp->position,
                             checked_i32(frontier, "peer graph representative MTP position"));
            }
        });
        if (io.dflash_decode) {
            *dflash_host_ingress       = {};
            *dflash_host_egress        = {};
            const std::uint32_t extent = std::min(draft_window, capacity - frontier - 1U);
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                dflash_host_ingress->anchors[row] = 0;
                dflash_host_ingress->execution_frontiers[row] =
                    checked_i32(frontier, "graph representative DFlash frontier");
                dflash_host_ingress->context_frontiers[row] =
                    checked_i32(frontier, "graph representative DFlash context frontier");
                dflash_host_ingress->proposal_extents[row] = static_cast<std::int32_t>(extent);
                dflash_host_ingress->target_valid_columns[row] =
                    static_cast<std::int32_t>(extent + 1U);
                dflash_host_ingress->text_kv_table_rows[row]   = static_cast<std::int32_t>(row);
                dflash_host_ingress->dflash_kv_table_rows[row] = static_cast<std::int32_t>(row);
                dflash_host_ingress->lanes[row]                = static_cast<std::int32_t>(row);
                dflash_host_ingress->sampling[row]             = {};
            }
            for (int rank = 1; rank < tp; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                if (dflash_peer_host_ingress[slot] != nullptr) {
                    *dflash_peer_host_ingress[slot] = *dflash_host_ingress;
                }
            }
        }
        if (io.mtp_decode) {
            *mtp_host_ingress          = {};
            *mtp_host_egress           = {};
            const std::uint32_t extent = std::min(draft_window, capacity - frontier - 1U);
            const std::uint32_t width  = draft_window + 1U;
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                mtp_host_ingress->anchors[row] = 0;
                mtp_host_ingress->base_frontiers[row] =
                    checked_i32(frontier, "graph representative MTP frontier");
                mtp_host_ingress->remaining_budgets[row] =
                    checked_i32(capacity, "graph representative MTP budget");
                mtp_host_ingress->current_extents[row] = static_cast<std::int32_t>(extent);
                mtp_host_ingress->target_valid_columns[row] =
                    static_cast<std::int32_t>(extent + 1U);
                for (std::uint32_t step = 0; step < draft_window; ++step) {
                    mtp_host_ingress->current_drafts[row * draft_window + step] = 0;
                }
                for (std::uint32_t column = 0; column < width; ++column) {
                    mtp_host_ingress->target_rope_positions[row * width + column] =
                        checked_i32(frontier + std::min(column, extent),
                                    "graph representative MTP RoPE position");
                }
                mtp_host_ingress->text_kv_table_rows[row] = static_cast<std::int32_t>(row);
                mtp_host_ingress->mtp_kv_table_rows[row]  = static_cast<std::int32_t>(row);
                mtp_host_ingress->lanes[row]              = static_cast<std::int32_t>(row);
                mtp_host_ingress->rope_deltas[row]        = 0;
                mtp_host_ingress->sampling[row]           = {};
            }
            // The representative's sampling configs are zeroed, so its counter pointers are null
            // and the peer copy is a plain mirror -- but it still has to exist, because the
            // captured graph bakes in the peer ingress's host ADDRESS and reads it at replay.
            for (int rank = 1; rank < tp; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                if (mtp_peer_host_ingress[slot] != nullptr) {
                    *mtp_peer_host_ingress[slot] = *mtp_host_ingress;
                }
            }
        }
        if (io.ordinary) {
            *ordinary_host_ingress = {};
            *ordinary_host_egress  = {};
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                ordinary_host_ingress->tokens[row] = 0;
                ordinary_host_ingress->cache_positions[row] =
                    checked_i32(frontier, "graph representative ordinary position");
                ordinary_host_ingress->rope_positions[row] =
                    checked_i32(frontier, "graph representative ordinary RoPE position");
                ordinary_host_ingress->text_kv_table_rows[row] = static_cast<std::int32_t>(row);
                ordinary_host_ingress->lanes[row]              = static_cast<std::int32_t>(row);
                ordinary_host_ingress->sampling[row]           = {};
            }
            // Same reason as the MTP mirror above: the representative's counter pointers are
            // already null, but the captured graph bakes in the peer ingress's host ADDRESS and
            // reads it at every replay, so the record has to exist and be current here.
            publish_peer_ordinary_ingress();
        }
    };
    const auto execution_core = [&] {
        return schedule::ExecutionCore{device,
                                       model,
                                       work,
                                       decoder->linear_attention,
                                       replay_records ? &*replay_records : nullptr,
                                       io,
                                       prefill_hidden,
                                       prefill_chunk,
                                       proposal_head,
                                       rope_frequency,
                                       peer_ptrs(),
                                       tp};
    };

    if (speculative_backend == SpeculativeBackend::None) {
        const auto ordinary_profiles = ordinary_graph_profiles(capacity);
        validate_graph_profiles(ordinary_profiles, capacity - 1, "ordinary");
        const std::uint32_t ordinary_batch_limit = max_concurrency;
        schedule::OrdinaryBatchContext ordinary_state{execution_core(),
                                                      decoder->text_kv,
                                                      *io.ordinary,
                                                      *ordinary_host_ingress,
                                                      *ordinary_host_egress,
                                                      tail_hidden_store,
                                                      ordinary_peer_host_ingress};
        // One eager pass first, so every kernel's module is already loaded when capture runs: a
        // lazily loaded module inside a capture region is not capturable, and at tp2 that surface
        // exists on BOTH devices.
        const GraphExecutionProfile code_warm = ordinary_profiles.front();
        prepare_representative(code_warm.min, 1);
        synchronize_all();
        schedule::ordinary_decode_batch(ordinary_state, 1, {code_warm.min + 1, code_warm.max + 1},
                                        nullptr);
        synchronize_all();

        if (tp > 1) {
            // Batch shape selects kernels, and a module first touched INSIDE a capture region
            // cannot be loaded there. At tp1 the code-warm pass above has always been enough in
            // practice (with one observed allowance flake, documented, consistent with a late
            // module load); at
            // tp2 the same surface exists on two devices and both are checked against the same
            // per-device allowance, so every batch size is warmed eagerly before any capture.
            for (std::uint32_t batch_size = 2; batch_size <= ordinary_batch_limit; ++batch_size) {
                prepare_representative(code_warm.min, batch_size);
                synchronize_all();
                schedule::ordinary_decode_batch(ordinary_state,
                                                static_cast<std::int32_t>(batch_size),
                                                {code_warm.min + 1, code_warm.max + 1}, nullptr);
                synchronize_all();
            }
        }

        ordinary_graphs.profiles.reserve(ordinary_profiles.size() * ordinary_batch_limit);
        for (std::uint32_t batch_size = 1; batch_size <= ordinary_batch_limit; ++batch_size) {
            for (const GraphExecutionProfile planned : ordinary_profiles) {
                ordinary_graphs.profiles.emplace_back();
                DecodeGraphProfile& profile    = ordinary_graphs.profiles.back();
                profile.batch_size             = batch_size;
                profile.min_execution_frontier = planned.min;
                profile.max_execution_frontier = planned.max;
                profile.topology_class =
                    planned.topology_class * ordinary_batch_limit + (batch_size - 1U);
                const ops::GqaExecutionEnvelope envelope{planned.min + 1, planned.max + 1};
                schedule::capture_ordinary_decode_batch(ordinary_state,
                                                        static_cast<std::int32_t>(batch_size),
                                                        envelope, profile.definition);
            }
        }
    }

    if (speculative_backend == SpeculativeBackend::Mtp) {
        const auto planned_profiles = mtp_graph_profiles(capacity, draft_window);
        validate_graph_profiles(planned_profiles, capacity - 1, "MTP");
        schedule::MtpBatchContext mtp_state{
            execution_core(),  decoder->text_kv, *decoder->mtp_cache(), *io.mtp_decode,
            *mtp_host_ingress, *mtp_host_egress, tail_hidden_store};
        const GraphExecutionProfile code_warm = planned_profiles.front();
        prepare_representative(code_warm.min, 1);
        synchronize_all();
        schedule::mtp_decode_batch(mtp_state, 1, draft_window,
                                   mtp_gqa_envelopes(code_warm.max, draft_window, capacity),
                                   nullptr);
        synchronize_all();
        if (tp > 1) {
            // Same reason as the ordinary family's tp2 warm loop: batch shape selects kernels and
            // a module first touched inside a capture region cannot be loaded there, and at tp2
            // that surface exists on two devices, both checked against the same per-device
            // graph allowance.
            for (std::uint32_t batch_size = 2; batch_size <= max_concurrency; ++batch_size) {
                prepare_representative(code_warm.min, batch_size);
                synchronize_all();
                schedule::mtp_decode_batch(mtp_state, static_cast<std::int32_t>(batch_size),
                                           draft_window,
                                           mtp_gqa_envelopes(code_warm.max, draft_window,
                                                             capacity),
                                           nullptr);
                synchronize_all();
            }
        }

        mtp_graphs.profiles.reserve(planned_profiles.size() * max_concurrency);
        for (std::uint32_t batch_size = 1; batch_size <= max_concurrency; ++batch_size) {
            for (const GraphExecutionProfile planned : planned_profiles) {
                mtp_graphs.profiles.emplace_back();
                DecodeGraphProfile& profile    = mtp_graphs.profiles.back();
                profile.batch_size             = batch_size;
                profile.min_execution_frontier = planned.min;
                profile.max_execution_frontier = planned.max;
                profile.topology_class =
                    planned.topology_class * max_concurrency + (batch_size - 1U);
                schedule::capture_mtp_decode_batch(
                    mtp_state, static_cast<std::int32_t>(batch_size), draft_window,
                    mtp_gqa_envelopes(planned.max, draft_window, capacity), profile.definition);
            }
        }
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        const auto batch_one_profiles = dflash_graph_profiles(capacity, draft_window, 1);
        validate_graph_profiles(batch_one_profiles, capacity - 1, "DFlash");
        schedule::DFlashBatchContext dflash_state{
            execution_core(),     decoder->text_kv,    *dflash,          *io.dflash_decode,
            *dflash_host_ingress, *dflash_host_egress, tail_hidden_store};
        const GraphExecutionProfile code_warm = batch_one_profiles.front();
        const ops::GqaExecutionEnvelope code_warm_target{
            1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                   capacity, static_cast<std::uint64_t>(code_warm.max) + draft_window + 1ULL))};
        prepare_representative(code_warm.min, 1);
        device.synchronize();
        dflash_state.greedy_target = false;
        schedule::dflash_decode_batch(dflash_state, 1, draft_window,
                                      dflash_envelopes(code_warm.min, code_warm.max, draft_window),
                                      code_warm_target, nullptr);
        device.synchronize();

        // The greedy target-argmax route and the full-logit sampling route are separate CUDA
        // graph topologies. Warm both once so their modules are loaded before capture; runtime
        // request sampling selects the matching profile without changing semantics.
        prepare_representative(code_warm.min, 1);
        device.synchronize();
        dflash_state.greedy_target = true;
        schedule::dflash_decode_batch(dflash_state, 1, draft_window,
                                      dflash_envelopes(code_warm.min, code_warm.max, draft_window),
                                      code_warm_target, nullptr);
        device.synchronize();

        dflash_graphs.profiles.reserve(batch_one_profiles.size() * max_concurrency * 2U);
        for (bool greedy_target : {false, true}) {
            dflash_state.greedy_target = greedy_target;
            for (std::uint32_t batch_size = 1; batch_size <= max_concurrency; ++batch_size) {
                const auto planned_profiles =
                    batch_size == 1 ? batch_one_profiles
                                    : dflash_graph_profiles(capacity, draft_window, batch_size);
                validate_graph_profiles(planned_profiles, capacity - 1, "DFlash");
                for (const GraphExecutionProfile planned : planned_profiles) {
                    dflash_graphs.profiles.emplace_back();
                    DecodeGraphProfile& profile    = dflash_graphs.profiles.back();
                    profile.batch_size             = batch_size;
                    profile.min_execution_frontier = planned.min;
                    profile.max_execution_frontier = planned.max;
                    profile.greedy_target          = greedy_target;
                    profile.topology_class =
                        (planned.topology_class * 2U + (greedy_target ? 1U : 0U)) *
                            max_concurrency +
                        (batch_size - 1U);
                    const ops::GqaExecutionEnvelope target_envelope{
                        1,
                        static_cast<std::uint32_t>(std::min<std::uint64_t>(
                            capacity, static_cast<std::uint64_t>(planned.max) + draft_window + 1ULL))};

                    schedule::capture_dflash_decode_batch(
                        dflash_state, static_cast<std::int32_t>(batch_size), draft_window,
                        dflash_envelopes(planned.min, planned.max, draft_window), target_envelope,
                        profile.definition);
                }
            }
        }
    }

    for (const DecodeGraphFamily* family : {&ordinary_graphs, &mtp_graphs, &dflash_graphs}) {
        if (!family->profiles.empty()) {
            graph_node_count = family->profiles.front().definition.node_count();
            break;
        }
    }

    std::array<int, kMaximumDevices> graph_devices{};
    std::array<cudaStream_t, kMaximumDevices> graph_streams{};
    graph_devices[0] = device.device;
    graph_streams[0] = device.stream;
    for (int rank = 1; rank < tp; ++rank) {
        const DeviceContext& rank_device = *peer_core[static_cast<std::size_t>(rank)]->device;
        graph_devices[rank] = rank_device.device;
        graph_streams[rank] = rank_device.stream;
    }
    const std::span<const int> graph_device_span(graph_devices.data(),
                                                 static_cast<std::size_t>(tp));
    const std::span<const cudaStream_t> graph_stream_span(graph_streams.data(),
                                                          static_cast<std::size_t>(tp));

    if (!ordinary_graphs.profiles.empty()) {
        instantiate_graph_family(ordinary_graphs, "ordinary", graph_device_span, graph_stream_span,
                                 prepare_representative, synchronize_all);
    }
    if (speculative_backend == SpeculativeBackend::Mtp) {
        instantiate_graph_family(mtp_graphs, "MTP", graph_device_span, graph_stream_span,
                                 prepare_representative, synchronize_all);
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        instantiate_graph_family(dflash_graphs, "DFlash", graph_device_span, graph_stream_span,
                                 prepare_representative, synchronize_all);
    }

    ordered_reset(sequence);
    clear_stable_controls();
    for (Tensor& tensor : decoder->linear_attention.conv) {
        CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
    }
    for (Tensor& tensor : decoder->linear_attention.recurrent) {
        CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
    }
    on_peer([&](int, PeerRuntime& p) {
        for (Tensor& tensor : p.decoder->linear_attention.conv) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), p.device.stream));
        }
        for (Tensor& tensor : p.decoder->linear_attention.recurrent) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), p.device.stream));
        }
    });
    if (dflash) {
        const auto zero_cyclic_cache = [&](CyclicKVCache& cache) {
            for (std::uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
                const CyclicKVCacheLayerView view = cache.layer_view(layer);
                CUDA_CHECK(cudaMemsetAsync(view.k.data, 0, view.k.bytes(), device.stream));
                CUDA_CHECK(cudaMemsetAsync(view.v.data, 0, view.v.bytes(), device.stream));
            }
        };
        zero_cyclic_cache(dflash->local);
        zero_cyclic_cache(dflash->rewrite_checkpoint_local);
        CUDA_CHECK(cudaMemsetAsync(dflash->prefill_features.data, 0,
                                   dflash->prefill_features.bytes(), device.stream));
        CUDA_CHECK(cudaMemsetAsync(dflash->prefill_positions.data, 0,
                                   dflash->prefill_positions.bytes(), device.stream));
        CUDA_CHECK(cudaMemsetAsync(dflash->pending_features.data, 0,
                                   dflash->pending_features.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    // Symmetry: capture left each peer's counters as dirty as rank 0's, and the lanes are only
    // meaningful while they agree.
    on_peer([&](int, PeerRuntime& p) {
        CUDA_CHECK(
            cudaMemsetAsync(p.token_counts.data, 0, p.token_counts.bytes(), p.device.stream));
    });
    synchronize_all();

    std::array<std::size_t, kMaximumDevices> free_after{};
    free_after[0] = free_device_bytes();
    on_peer([&](int rank, PeerRuntime&) { free_after[rank] = free_device_bytes(); });
    // The allowance is a per-device budget, so each device is checked against it separately
    // rather than against a doubled or summed figure.
    for (int rank = 0; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        const std::size_t consumed =
            free_before[slot] > free_after[slot] ? free_before[slot] - free_after[slot] : 0;
        graph_observed_bytes[slot] = consumed;
        if (consumed > graph_allowance_bytes) {
            throw std::runtime_error(
                "CUDA Graph preparation consumed " + std::to_string(consumed) +
                " bytes on device " +
                std::to_string(rank == 0 ? device.device : peer[slot]->device.device) +
                ", exceeding the planned per-device allowance of " +
                std::to_string(graph_allowance_bytes) + " bytes");
        }
    }
    for (PagedKVAllocation& allocation : dflash_capture_allocations) { allocation.unbind_row(); }
    dflash_capture_allocations.clear();
    for (PagedKVAllocation& allocation : mtp_capture_allocations) { allocation.unbind_row(); }
    mtp_capture_allocations.clear();
    on_peer([&](int rank, PeerRuntime&) {
        for (PagedKVAllocation& allocation : peer_mtp_capture_allocations[rank]) {
            allocation.unbind_row();
        }
        peer_mtp_capture_allocations[rank].clear();
    });
    for (PagedKVAllocation& allocation : text_capture_allocations) { allocation.unbind_row(); }
    text_capture_allocations.clear();
    on_peer([&](int rank, PeerRuntime&) {
        for (PagedKVAllocation& allocation : peer_text_capture_allocations[rank]) {
            allocation.unbind_row();
        }
        peer_text_capture_allocations[rank].clear();
    });
}

Tensor ProgramImplCore::token_counts_lane(const Tensor& storage, std::uint32_t lane) {
    return storage.slice(1, static_cast<std::int32_t>(lane), 1).view({TextConfig::token_domain});
}

void ProgramImplCore::publish_peer_token_counts(const SequenceState& sequence) {
    if (tp <= 1) { return; }
    // With penalties off the lane is never read on any rank (install_sampling leaves every row's
    // `token_counts` null), so the ~0.95 MiB peer copy per prefill is pure cost. All lanes were
    // zeroed together in install_sampling, so skipping keeps them equal either way.
    if (requests[sequence.lane].sampling_host.token_counts == nullptr) { return; }
    const Tensor source = token_counts_lane(token_counts, sequence.lane);
    // The one increment rank 0 performs and the peers do not: prefill's bonus token is sampled on
    // rank 0 alone (the output head is vocabulary-split and sampling belongs to rank 0). Every
    // later increment is performed by `speculative_accept_greedy_drafts`, which the MTP round runs
    // on EVERY rank over bit-identical inputs, so one copy here per peer is what makes the counter
    // lanes agree at every point either is read.
    //
    // Same cross-device form the collectives use, and for the same reason (src/ops/common/
    // allreduce.cu's `pull_peer`): under unified virtual addressing a device pointer already names
    // its device, so `cudaMemcpyDeviceToDevice` expresses the transfer without
    // `cudaMemcpyPeerAsync`, which CUDA 13.1 rejects inside a stream capture region. This call
    // site is eager prefill, not capture, but keeping one form across every cross-device copy
    // means no future move of this code into a captured region reintroduces that failure.
    for (int rank = 1; rank < tp; ++rank) {
        const auto slot     = static_cast<std::size_t>(rank);
        const Tensor target = token_counts_lane(peer[slot]->token_counts, sequence.lane);
        CUDA_CHECK(cudaMemcpyAsync(target.data, source.data, source.bytes(),
                                   cudaMemcpyDeviceToDevice, device.stream));
    }
}

void ProgramImplCore::publish_peer_mtp_ingress(std::span<const std::uint32_t> lanes) {
    if (mtp_host_ingress == nullptr) { return; }
    for (int rank = 1; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (mtp_peer_host_ingress[slot] == nullptr) { continue; }
        *mtp_peer_host_ingress[slot] = *mtp_host_ingress;
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            ops::SamplingConfig& sampling = mtp_peer_host_ingress[slot]->sampling[row];
            if (sampling.token_counts == nullptr) { continue; }
            sampling.token_counts = static_cast<std::int32_t*>(
                token_counts_lane(peer[slot]->token_counts, lanes[row]).data);
        }
    }
}

void ProgramImplCore::publish_peer_dflash_ingress(std::span<const std::uint32_t> lanes) {
    if (dflash_host_ingress == nullptr) { return; }
    for (int rank = 1; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (dflash_peer_host_ingress[slot] == nullptr) { continue; }
        *dflash_peer_host_ingress[slot] = *dflash_host_ingress;
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            ops::SamplingConfig& sampling = dflash_peer_host_ingress[slot]->sampling[row];
            if (sampling.token_counts == nullptr) { continue; }
            sampling.token_counts = static_cast<std::int32_t*>(
                token_counts_lane(peer[slot]->token_counts, lanes[row]).data);
        }
    }
}

void ProgramImplCore::publish_peer_ordinary_ingress() {
    if (ordinary_host_ingress == nullptr) { return; }
    for (int rank = 1; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (ordinary_peer_host_ingress[slot] == nullptr) { continue; }
        *ordinary_peer_host_ingress[slot] = *ordinary_host_ingress;
        // Each peer mirrors the round for its share of the weights and never samples, so its copy
        // of the sampling configs is inert. The counter pointer is the one field that is a rank-0
        // DEVICE address; it is nulled rather than repointed at the peer's lane (as the MTP ingress
        // does) because nothing on a peer advances an ordinary-round counter -- a repointed lane
        // would be a counter that never moves, which is worse than an absent one. If a future
        // change makes a peer sample in the ordinary round, repoint here exactly as
        // publish_peer_mtp_ingress does.
        for (ops::SamplingConfig& sampling : ordinary_peer_host_ingress[slot]->sampling) {
            sampling.token_counts = nullptr;
        }
    }
}

void ProgramImplCore::enable_peer_egress_check(bool enabled) noexcept {
    peer_egress_check_enabled = enabled;
}

void ProgramImplCore::check_peer_mtp_egress(std::size_t rows) {
    if (!peer_egress_check_enabled || tp <= 1 || mtp_host_egress == nullptr) { return; }
    const std::uint32_t width = draft_window + 1U;
    for (int rank = 1; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (!peer[slot]->io.mtp_decode.has_value()) { continue; }
        qwen3_6::MtpDecodeEgress peer_egress{};
        {
            const ScopedDevice scope(peer[slot]->device.device);
            CUDA_CHECK(cudaMemcpy(&peer_egress, peer[slot]->io.mtp_decode->egress.data,
                                  sizeof(qwen3_6::MtpDecodeEgress), cudaMemcpyDeviceToHost));
        }
        std::uint64_t mismatches = 0;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::int32_t count = mtp_host_egress->licensed_counts[row];
            mismatches += peer_egress.licensed_counts[row] != count ? 1U : 0U;
            mismatches +=
                peer_egress.accepted_drafts[row] != mtp_host_egress->accepted_drafts[row] ? 1U : 0U;
            mismatches +=
                peer_egress.next_extents[row] != mtp_host_egress->next_extents[row] ? 1U : 0U;
            // Only the licensed prefix is defined; the columns past it are round scratch on all
            // ranks.
            const std::size_t licensed =
                count > 0 ? std::min(static_cast<std::size_t>(count), static_cast<std::size_t>(width))
                          : 0U;
            for (std::size_t column = 0; column < licensed; ++column) {
                const std::size_t index = row * width + column;
                mismatches += peer_egress.licensed_tokens[index] !=
                                      mtp_host_egress->licensed_tokens[index]
                                  ? 1U
                                  : 0U;
            }
            // `next_drafts` is deliberately NOT compared: the proposal head is vocabulary-split,
            // so TextContext::mtp_propose_batch's tp2 overload gathers every shard and writes ONE
            // argmax -- rank 0's. A peer's next_drafts region is never written (it reads back as
            // zeros) and is not part of its egress; next round's drafts reach each peer through the
            // pinned MTP ingress record, not through its own egress.
        }
        peer_egress_rounds += 1;
        peer_egress_mismatches += mismatches;
    }
}

void ProgramImplCore::install_sampling(SequenceState& sequence, RequestControl& request,
                                       const ops::SamplingConfig& config) {
    Tensor counts = token_counts_lane(token_counts, sequence.lane);
    CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), device.stream));
    for (int rank = 1; rank < tp; ++rank) {
        const auto slot           = static_cast<std::size_t>(rank);
        PeerRuntime& p            = *peer[slot];
        const ScopedDevice scope(p.device.device);
        const Tensor peer_counts  = token_counts_lane(p.token_counts, sequence.lane);
        CUDA_CHECK(cudaMemsetAsync(peer_counts.data, 0, peer_counts.bytes(), p.device.stream));
    }
    request.sampling_host     = config;
    request.speculative_stats = SpeculativeStats{
        .backend               = speculative_backend,
        .enabled               = speculative_backend != SpeculativeBackend::None,
        .draft_window          = draft_window,
        .accepted_per_position = std::vector<std::uint64_t>(draft_window, 0),
    };
    const bool penalties = request.sampling_host.presence_penalty != 0.0F ||
                           request.sampling_host.frequency_penalty != 0.0F;
    request.sampling_host.token_counts =
        penalties ? static_cast<std::int32_t*>(counts.data) : nullptr;
    Tensor config_lane = sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1);
    CUDA_CHECK(cudaMemcpyAsync(config_lane.data, &request.sampling_host,
                               sizeof(request.sampling_host), cudaMemcpyHostToDevice,
                               device.stream));
}

void ProgramImplCore::copy_tail(SequenceState& sequence, const Tensor& source) {
    if (source.dtype != DType::BF16 || source.ne[0] != TextConfig::hidden || source.ne[1] != 1) {
        throw std::logic_error("target tail hidden has an invalid shape");
    }
    CUDA_CHECK(cudaMemcpyAsync(sequence.tail_hidden.data, source.data, sequence.tail_hidden.bytes(),
                               cudaMemcpyDeviceToDevice, device.stream));
    sequence.tail_hidden_valid = true;
}

std::uint32_t ProgramImplCore::cap_final_chunk_for_taps(const SequenceState& sequence,
                                                        std::uint32_t cursor,
                                                        std::uint32_t prompt_tokens,
                                                        std::uint32_t nominal) const {
    if (prefix_index == nullptr || !sequence.prefix_capture ||
        sequence.prefix_tap_next >= sequence.prefix_tap_positions.size() ||
        cursor + nominal != prompt_tokens) {
        return nominal;
    }
    const std::uint32_t kBlock = static_cast<std::uint32_t>(ninfer::prefix_cache::kBlockTokens);
    const std::uint32_t boundary = (prompt_tokens / kBlock) * kBlock;
    if (boundary > cursor) { nominal = boundary - cursor; }
    return nominal;
}

void ProgramImplCore::capture_prefix_after_chunk(SequenceState& sequence) {
    if (prefix_index == nullptr || !sequence.prefix_capture) { return; }
    const std::uint32_t cursor    = sequence.text_kv_valid;
    const std::uint32_t completed = cursor / ninfer::prefix_cache::kBlockTokens;
    const std::uint32_t already_committed = sequence.prefix_pinned_nodes.size() - 1;
    // Commit newly completed blocks. A block the index adopts transfers its page to the cache
    // (promote_page_at, replicated across the lockstep rank pools); a duplicate commit leaves the
    // page owned by this allocation so release() returns it normally and it is never handed back to
    // the free pool while still live in the block table.
    while (sequence.prefix_pinned_nodes.size() - 1 < completed) {
        const std::uint32_t b      = sequence.prefix_pinned_nodes.size() - 1; // 0-indexed block.
        const std::uint32_t parent = sequence.prefix_pinned_nodes[b];         // node for block b-1.
        const std::int32_t text_page = sequence.kv->text.page_ids()[b];
        const std::int32_t backend_page =
            sequence.kv->backend ? sequence.kv->backend->page_ids()[b] : -1;
        const std::span<TokenId> block_tokens =
            std::span<TokenId>(sequence.prefix_prompt_tokens).subspan(b * 64, 64);
        const ninfer::prefix_cache::PoolPages pages{.page = {text_page, backend_page}};
        bool adopted = false;
        const ninfer::prefix_cache::NodeId node = prefix_index->commit_block(
            ninfer::prefix_cache::BlockCommit{
                .parent      = parent,
                .lookup_hash = sequence.prefix_block_hashes[b],
                .extra       = sequence.prefix_block_extras[b],
                .tokens      = block_tokens,
                .pages       = pages},
            adopted);
        if (node == ninfer::prefix_cache::kInvalidNode) {
            break; // avoid an infinite loop if the commit is rejected
        }
        if (adopted && sequence.kv) {
            // The index took this page: fold it into the cache-owned set on every rank (lockstep
            // pools) so release() never returns it.
            sequence.kv->text.promote_page_at(b);
            if (sequence.kv->backend) { sequence.kv->backend->promote_page_at(b); }
            for (int rank = 1; rank < tp; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                if (sequence.kv->text_peer[slot]) {
                    sequence.kv->text_peer[slot]->promote_page_at(b);
                }
                if (sequence.kv->backend_peer[slot]) {
                    sequence.kv->backend_peer[slot]->promote_page_at(b);
                }
            }
        }
        // adopted == false: the block duplicates an already-resident node, so the page stays owned
        // by this allocation and is returned normally by release().
        sequence.prefix_pinned_nodes.push_back(node);
        prefix_index->pin_node(node);
    }
    (void)already_committed;
    // Capture tap snapshots at chunk boundaries. Only when the boundary is block-aligned: the
    // state image copied below is the state after `cursor`, so a mid-block boundary would publish a
    // snapshot whose frontier is ahead of its state. Defer such taps to the next aligned boundary.
    const bool capture_aligned =
        cursor % static_cast<std::uint32_t>(ninfer::prefix_cache::kBlockTokens) == 0;
    while (capture_aligned && sequence.prefix_tap_next < sequence.prefix_tap_slots.size() &&
           sequence.prefix_tap_positions[sequence.prefix_tap_next] <= cursor) {
        const std::uint32_t position = sequence.prefix_tap_positions[sequence.prefix_tap_next];
        const ninfer::prefix_cache::SlotId slot = sequence.prefix_tap_slots[sequence.prefix_tap_next];
        const std::int32_t state_slot = LinearStateSlots::snapshot_state_slot(
            static_cast<std::uint32_t>(slot), max_concurrency, prefix_snapshot_slots);
        decoder->linear_attention.copy_slot(
            LinearStateSlots::current_state_slot(sequence.lane, max_concurrency), state_slot,
            device.stream);
        for (int rank = 1; rank < tp; ++rank) {
            const auto peer_slot = static_cast<std::size_t>(rank);
            const ScopedDevice scope(peer[peer_slot]->device.device);
            peer[peer_slot]->decoder->linear_attention.copy_slot(
                LinearStateSlots::current_state_slot(sequence.lane, max_concurrency), state_slot,
                peer[peer_slot]->device.stream);
        }
        // cursor is block-aligned, so `completed*64 == cursor`; the state image is the state after
        // exactly `cursor` tokens, so the snapshot frontier is `cursor` with tail_len 0, anchored
        // on block completed-1.
        const ninfer::prefix_cache::NodeId anchor = sequence.prefix_pinned_nodes[completed];
        const std::uint32_t frontier              = cursor;
        const auto snap_id = prefix_index->add_snapshot(ninfer::prefix_cache::SnapshotRequest{
            .anchor        = anchor,
            .frontier      = frontier,
            .kind          = ninfer::prefix_cache::SnapshotKind::Tap,
            .priority      = 0,
            .reserved_slot = slot});
        // Settle the reserved slot: a published snapshot owns it (or a duplicate holds a different
        // one), so clear_lane must never release it back to the free list.
        if (snap_id != ninfer::prefix_cache::kInvalidSnapshot) {
            const ninfer::prefix_cache::Snapshot* published = prefix_index->snapshot(snap_id);
            if (published == nullptr || published->device_slot != slot) {
                prefix_index->release_reserved_slot(slot);
            }
            issue_snapshot_write_through(snap_id);
        } else {
            prefix_index->release_reserved_slot(slot);
        }
        sequence.prefix_tap_slots[sequence.prefix_tap_next] = ninfer::prefix_cache::kInvalidSlot;
        (void)position;
        sequence.prefix_tap_next++;
    }
}

void ProgramImplCore::copy_round_token() {
    CUDA_CHECK(cudaMemcpyAsync(host_tokens, io.token.data, sizeof(TokenId), cudaMemcpyDeviceToHost,
                               device.stream));
}

// See logits_capture's declaration comment. io.logits is [vocab,1] BF16 regardless of tp; this
// is the same rank-0 destination copy_round_token() reads io.token from, so no tp2-specific code
// is needed here. Inert unless the debug capture was explicitly enabled.
void ProgramImplCore::copy_round_logits() {
    if (!logits_capture_enabled) { return; }
    CUDA_CHECK(cudaMemcpyAsync(logits_capture.data(), io.logits.data,
                               logits_capture.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, device.stream));
}

void ProgramImplCore::enable_logits_capture(bool enabled) {
    logits_capture_enabled = enabled;
    if (enabled) {
        logits_capture.assign(static_cast<std::size_t>(io.logits.ne[0]), 0);
    } else {
        logits_capture.clear();
        logits_capture.shrink_to_fit();
    }
}

void ProgramImplCore::mark_workspace_usage(std::size_t phase_bytes) noexcept {
    workspace_logical_peak_bytes = std::max(workspace_logical_peak_bytes, phase_bytes);
}

// M2 host-tier transfer layer.

qwen3_6::PagedKVCache* ProgramImplCore::rank_backend_cache(int rank) noexcept {
    if (rank == 0) { return backend_kv_cache(); }
    PeerRuntime& p = *peer[static_cast<std::size_t>(rank)];
    if (speculative_backend == SpeculativeBackend::Mtp) { return p.decoder->mtp_cache(); }
    if (speculative_backend == SpeculativeBackend::DFlash && p.dflash && p.dflash->full) {
        return &*p.dflash->full;
    }
    return nullptr;
}

ninfer::prefix_cache::PoolPages ProgramImplCore::acquire_cache_bundle() {
    auto acquire_once = [&]() -> ninfer::prefix_cache::PoolPages {
        ninfer::prefix_cache::PoolPages pages;
        std::vector<std::int32_t> text = decoder->text_kv.pool().acquire_cache_pages(1);
        pages.page[0] = text.front();
        if (qwen3_6::PagedKVCache* backend = backend_kv_cache(); backend != nullptr) {
            try {
                std::vector<std::int32_t> be = backend->pool().acquire_cache_pages(1);
                pages.page[1] = be.front();
            } catch (const std::bad_alloc&) {
                const std::int32_t ids[1] = {pages.page[0]};
                decoder->text_kv.pool().release_cache_pages(ids);
                throw;
            }
        }
        // Mirror on every peer rank; lockstep pools must return identical ids.
        for (int rank = 1; rank < tp; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            std::vector<std::int32_t> t =
                peer[slot]->decoder->text_kv.pool().acquire_cache_pages(1);
            if (t.front() != pages.page[0]) {
                throw std::logic_error("cache page lockstep broke across ranks");
            }
            if (pages.page[1] >= 0) {
                qwen3_6::PagedKVCache* be = rank_backend_cache(rank);
                if (be == nullptr) {
                    throw std::logic_error("backend cache missing on peer rank");
                }
                std::vector<std::int32_t> b = be->pool().acquire_cache_pages(1);
                if (b.front() != pages.page[1]) {
                    throw std::logic_error("cache page lockstep broke across ranks");
                }
            }
        }
        return pages;
    };
    try {
        return acquire_once();
    } catch (const std::bad_alloc&) {
        // Evict device-resident cache nodes until the acquire fits. Each eviction returns one
        // page per pool through the sink; stop when nothing is evictable and rethrow.
        for (;;) {
            if (prefix_index == nullptr ||
                prefix_index->evict_device_lru() == ninfer::prefix_cache::kInvalidNode) {
                throw;
            }
            try {
                return acquire_once();
            } catch (const std::bad_alloc&) {
                continue;
            }
        }
    }
}

bool ProgramImplCore::copy_cache_block(int rank, std::int32_t text_page, std::int32_t backend_page,
                                       void* slab_base, bool device_to_host) noexcept {
    const auto slot = static_cast<std::size_t>(rank);
    if (prefix_cache_stream[slot] == nullptr) { return false; }
    const cudaStream_t stream = prefix_cache_stream[slot];
    char* cur = static_cast<char*>(slab_base);
    if (rank != 0) { ScopedDevice scope(peer[slot]->device.device); }
    PagedKVPool& text_pool =
        rank == 0 ? decoder->text_kv.pool() : peer[slot]->decoder->text_kv.pool();
    bool ok = true;
    if (text_page >= 0) {
        ok = device_to_host ? text_pool.copy_page_d2h(cur, text_page, stream)
                            : text_pool.copy_page_h2d(cur, text_page, stream);
        cur += text_pool.per_page_bytes();
    }
    if (ok && backend_page >= 0) {
        qwen3_6::PagedKVCache* backend = rank_backend_cache(rank);
        if (backend == nullptr) { return false; }
        ok = device_to_host ? backend->pool().copy_page_d2h(cur, backend_page, stream)
                            : backend->pool().copy_page_h2d(cur, backend_page, stream);
    }
    return ok;
}

bool ProgramImplCore::copy_snapshot_slot(int rank, std::int32_t slot, void* slab_base,
                                         bool device_to_host) noexcept {
    const auto slot_idx = static_cast<std::size_t>(rank);
    if (prefix_cache_stream[slot_idx] == nullptr) { return false; }
    const cudaStream_t stream = prefix_cache_stream[slot_idx];
    char* cur = static_cast<char*>(slab_base);
    if (rank != 0) { ScopedDevice scope(peer[slot_idx]->device.device); }
    LinearAttentionStatePool& la =
        rank == 0 ? decoder->linear_attention : peer[slot_idx]->decoder->linear_attention;
    const cudaMemcpyKind kind = device_to_host ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
    bool ok = true;
    for (std::uint32_t layer = 0; ok && layer < la.layer_count(); ++layer) {
        // Slot is the last tensor dim in both conv and recurrent storage, so each slot's data is
        // one contiguous run per layer.
        const Tensor conv      = la.conv_slot(layer, slot);
        const Tensor recurrent = la.recurrent_slot(layer, slot);
        if (device_to_host) {
            ok = cudaMemcpyAsync(cur, conv.data, conv.bytes(), kind, stream) == cudaSuccess;
            cur += conv.bytes();
            ok = ok &&
                 cudaMemcpyAsync(cur, recurrent.data, recurrent.bytes(), kind, stream) ==
                     cudaSuccess;
            cur += recurrent.bytes();
        } else {
            ok = cudaMemcpyAsync(conv.data, cur, conv.bytes(), kind, stream) == cudaSuccess;
            cur += conv.bytes();
            ok = ok &&
                 cudaMemcpyAsync(recurrent.data, cur, recurrent.bytes(), kind, stream) ==
                     cudaSuccess;
            cur += recurrent.bytes();
        }
    }
    if (!ok) { (void)cudaGetLastError(); }
    return ok;
}

void ProgramImplCore::order_prefix_cache_after_compute(int rank) noexcept {
    const auto slot = static_cast<std::size_t>(rank);
    if (prefix_cache_src_event[slot] == nullptr || prefix_cache_stream[slot] == nullptr) {
        return;
    }
    cudaStream_t compute =
        rank == 0 ? device.stream : peer[static_cast<std::size_t>(rank)]->device.stream;
    if (rank != 0) {
        const ScopedDevice scope(peer[static_cast<std::size_t>(rank)]->device.device);
        (void)cudaEventRecord(prefix_cache_src_event[slot], compute);
        (void)cudaStreamWaitEvent(prefix_cache_stream[slot], prefix_cache_src_event[slot], 0);
    } else {
        (void)cudaEventRecord(prefix_cache_src_event[slot], compute);
        (void)cudaStreamWaitEvent(prefix_cache_stream[slot], prefix_cache_src_event[slot], 0);
    }
}

void ProgramImplCore::issue_block_write_through(ninfer::prefix_cache::NodeId node) noexcept {
    if (prefix_index == nullptr || prefix_host_pool == nullptr) { return; }
    const ninfer::prefix_cache::BlockNode* n = prefix_index->node(node);
    if (n == nullptr || n->device_state != ninfer::prefix_cache::Residency::Resident ||
        n->host_state != ninfer::prefix_cache::Residency::Absent) {
        return;
    }
    const auto target = prefix_index->begin_host_write(node);
    if (target.page == ninfer::prefix_cache::kInvalidHostPage) { return; }
    char* slab = static_cast<char*>(prefix_host_pool->data(target.page));
    bool ok = true;
    for (int rank = 0; ok && rank < tp; ++rank) {
        order_prefix_cache_after_compute(rank);
        ok = copy_cache_block(rank, n->device_pages.page[0], n->device_pages.page[1],
                              slab + static_cast<std::size_t>(rank) * prefix_block_rank_bytes,
                              true);
    }
    if (ok) {
        for (int rank = 0; rank < tp; ++rank) {
            (void)cudaEventRecord(prefix_cache_event[static_cast<std::size_t>(rank)],
                                  prefix_cache_stream[static_cast<std::size_t>(rank)]);
        }
        prefix_inflight_writes.push_back({false, node, ninfer::prefix_cache::kInvalidSnapshot});
    } else {
        prefix_index->finish_host_write(node, false);
    }
}

void ProgramImplCore::issue_snapshot_write_through(ninfer::prefix_cache::SnapshotId id) noexcept {
    if (prefix_index == nullptr || prefix_host_pool == nullptr) { return; }
    const ninfer::prefix_cache::Snapshot* s = prefix_index->snapshot(id);
    if (s == nullptr) { return; }
    const auto target = prefix_index->begin_snapshot_write(id);
    if (target.page == ninfer::prefix_cache::kInvalidHostPage) { return; }
    char* slab = static_cast<char*>(prefix_host_pool->data(target.page));
    const std::int32_t state_slot = LinearStateSlots::snapshot_state_slot(
        static_cast<std::uint32_t>(s->device_slot), max_concurrency, prefix_snapshot_slots);
    bool ok = true;
    for (int rank = 0; ok && rank < tp; ++rank) {
        order_prefix_cache_after_compute(rank);
        ok = copy_snapshot_slot(rank, state_slot,
                                slab + static_cast<std::size_t>(rank) * prefix_snapshot_rank_bytes,
                                true);
    }
    if (ok) {
        for (int rank = 0; rank < tp; ++rank) {
            (void)cudaEventRecord(prefix_cache_event[static_cast<std::size_t>(rank)],
                                  prefix_cache_stream[static_cast<std::size_t>(rank)]);
        }
        prefix_inflight_writes.push_back({true, ninfer::prefix_cache::kInvalidNode, id});
    } else {
        prefix_index->finish_snapshot_write(id, false);
    }
}

void ProgramImplCore::tick_prefix_cache() noexcept {
    if (prefix_index == nullptr || prefix_inflight_writes.empty()) { return; }
    // The newest event on every rank's cache stream marks the end of the whole pending set: all
    // copies were enqueued before their event, so all events ready implies all writes landed. Each
    // event lives on its own rank's device, so query it with that device current.
    for (int rank = 0; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        DeviceContext& rank_device = rank == 0 ? device : peer[slot]->device;
        const ScopedDevice scope(rank_device.device);
        if (cudaEventQuery(prefix_cache_event[slot]) != cudaSuccess) {
            return;
        }
    }
    for (const InflightCacheWrite& write : prefix_inflight_writes) {
        if (write.is_snapshot) {
            prefix_index->finish_snapshot_write(write.snapshot, true);
        } else {
            prefix_index->finish_host_write(write.node, true);
        }
    }
    prefix_inflight_writes.clear();
}

void ProgramImplCore::settle_prefix_writes() noexcept {
    if (prefix_index == nullptr || prefix_inflight_writes.empty()) { return; }
    // Block until every enqueued D2H copy has landed on each rank's cache stream, then finalize.
    // After this, finish_* has dropped the restore pins and re-linked the nodes into the LRU, so
    // their pages are counted as evictable by a subsequent admission check. Each stream lives on
    // its own rank's device, so synchronize it with that device current.
    for (int rank = 0; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (prefix_cache_stream[slot] != nullptr) {
            DeviceContext& rank_device = rank == 0 ? device : peer[slot]->device;
            const ScopedDevice scope(rank_device.device);
            (void)cudaStreamSynchronize(prefix_cache_stream[slot]);
        }
    }
    tick_prefix_cache();
}

void ProgramImplCore::enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                                    std::span<const std::uint32_t> starts,
                                                    std::span<const std::uint32_t> counts) {
    if (speculative_backend != SpeculativeBackend::DFlash || !dflash || !io.dflash_decode ||
        lanes.empty() || lanes.size() > max_concurrency || starts.size() != lanes.size() ||
        counts.size() != lanes.size()) {
        throw std::logic_error("DFlash context append has invalid membership");
    }

    std::uint32_t minimum_count = draft_window + 1U;
    std::uint32_t maximum_count = 0;
    *dflash_host_ingress        = {};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || counts[row] == 0 || counts[row] > draft_window + 1U ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::logic_error("DFlash context append contains an invalid row");
        }
        SequenceState& sequence   = sequences[lane];
        const std::uint32_t start = starts[row];
        const std::uint64_t end64 = static_cast<std::uint64_t>(start) + counts[row];
        const std::uint32_t end   = static_cast<std::uint32_t>(end64);
        if (!sequence.kv || sequence.kv->text.bound_row() < 0 ||
            (sequence.kv->backend && sequence.kv->backend->bound_row() < 0) ||
            end64 > capacity) {
            throw std::logic_error("DFlash context append is outside retained target storage");
        }
        dflash_host_ingress->context_frontiers[row] =
            checked_i32(start, "DFlash append context frontier");
        dflash_host_ingress->execution_frontiers[row] =
            checked_i32(end, "DFlash append target frontier");
        dflash_host_ingress->dflash_kv_table_rows[row] =
            sequence.kv->backend ? sequence.kv->backend->bound_row() : 0;
        dflash_host_ingress->lanes[row]                = static_cast<std::int32_t>(lane);
        materialize_sequence_kv(sequence, std::max(sequence.text_kv_valid, end),
                                 sequence.kv->backend ? end : 0U);
        minimum_count = std::min(minimum_count, counts[row]);
        maximum_count = std::max(maximum_count, counts[row]);
    }

    qwen3_6::DFlashDecodeState& frame = *io.dflash_decode;
    CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, dflash_host_ingress,
                               sizeof(qwen3_6::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                               device.stream));
    const auto batch     = static_cast<std::int32_t>(lanes.size());
    Tensor lane_tensor   = frame.lanes.slice(0, 0, batch);
    Tensor device_starts = frame.context_frontiers.slice(0, 0, batch);
    Tensor device_ends   = frame.execution_frontiers.slice(0, 0, batch);
    Tensor table_rows    = frame.dflash_kv_table_rows.slice(0, 0, batch);
    Tensor positions     = frame.append_positions.slice(1, 0, batch);
    Tensor device_counts = frame.append_counts.slice(0, 0, batch);

    work.reset();
    Tensor features =
        work.alloc(DType::BF16, {DFlashConfig::feature_rows,
                                 static_cast<std::int32_t>(draft_window + 1U), batch});
    ops::prepare_ragged_prefix(dflash->pending_features, lane_tensor, device_starts, device_ends,
                               features, positions, device_counts, device.stream);

    schedule::DFlashAppendContext state{{device, model, work, decoder->linear_attention,
                                         replay_records ? &*replay_records : nullptr, io,
                                         prefill_hidden, prefill_chunk, proposal_head,
                                         rope_frequency},
                                        *dflash};
    mark_workspace_usage(workspace_plan.dflash_context);
    schedule::dflash_append_context(state, features, positions, device_counts, lane_tensor,
                                    table_rows, {minimum_count, maximum_count});
}

void ProgramImplCore::validate_licensed_tokens(std::span<const TokenId> tokens) const {
    for (const TokenId token : tokens) {
        if (token < 0 || token >= TextConfig::token_domain) {
            throw std::runtime_error("target returned a token outside the 248077-token domain");
        }
    }
}

runtime::PrefillStepResult ProgramImplCore::advance_prefill(SequenceState& sequence,
                                                            RequestControl& request) {
    if (request.lifecycle != Lifecycle::Prefilling || !request.prefill) {
        throw std::logic_error("staged prefill step requires an active concurrent request");
    }

    RequestControl::Prefill& staged = *request.prefill;
    const runtime::BeginSummary summary{.prompt_tokens        = staged.prompt_tokens,
                                        .reused_prompt_tokens = staged.base,
                                        .prefix_reuse_path    = staged.reuse,
                                        .prefix_reuse_reason  = staged.reuse_reason,
                                        .prefix_divergence    = staged.reuse_divergence};
    std::uint32_t processed_prompt_tokens = 0;
    const auto started                    = Clock::now();
    try {
        schedule::PrefillContext schedule_state{
            {device, model, work, decoder->linear_attention,
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, rope_frequency, peer_ptrs(), tp},
            text_kv_view(sequence),
            mtp_kv_view(sequence),
            mtp_kv_view_peer(sequence),
            decoder->text_kv,
            decoder->mtp_cache(),
            dflash ? &*dflash : nullptr,
            staged.cursor,
            static_cast<const ops::SamplingConfig*>(
                sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1).data),
            &sequence.rewrite_checkpoint_hidden,
            LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
            LinearStateSlots::rewrite_checkpoint_state_slot(sequence.lane, max_concurrency),
            staged.initial_mtp_extent,
            dflash_host_ingress};

        if (staged.mtp_bridge == MtpBridgeMode::BeforeSuffix) {
            if (staged.cursor != staged.base || staged.base == 0 ||
                staged.cursor >= staged.prompt_tokens) {
                throw std::logic_error("staged MTP bridge is outside the reusable suffix");
            }
            mark_workspace_usage(workspace_plan.mtp_prefill);
            const Tensor& previous_hidden = is_rewrite_checkpoint_restore(staged.reuse)
                                                ? sequence.rewrite_checkpoint_hidden
                                                : sequence.tail_hidden;
            const schedule::MtpBridgeInput bridge{
                .previous_hidden = &previous_hidden,
                .position        = checked_i32(staged.base - 1, "MTP bridge position"),
                .rope_position   = prompt_rope_position(staged.prompt, staged.base - 1),
            };
            if (staged.vision) {
                schedule::mtp_bridge_multimodal(schedule_state, staged.prompt, *staged.vision,
                                                bridge);
            } else {
                Tensor bridge_token = io.mtp->target_input_ids.slice(0, 0, 1);
                const TokenId token = staged.prompt.token_ids[staged.base];
                CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token),
                                           cudaMemcpyHostToDevice, device.stream));
                schedule::mtp_bridge_and_propose(schedule_state, bridge_token, previous_hidden,
                                                 bridge.position, bridge.rope_position, false);
            }
            sequence.mtp_kv_valid = staged.base;
            staged.mtp_bridge     = MtpBridgeMode::None;
        }

        if (staged.cursor < staged.prompt_tokens) {
            std::uint32_t nominal =
                std::min(prefill_chunk, staged.prompt_tokens - staged.cursor);
            // End the prompt-tail chunk on a block boundary so a pending tail tap can capture at an
            // aligned cursor (see cap_final_chunk_for_taps).
            nominal = cap_final_chunk_for_taps(sequence, staged.cursor, staged.prompt_tokens, nominal);
            const bool final_candidate = staged.cursor + nominal == staged.prompt_tokens;
            mark_workspace_usage(staged.prepare_mtp ? workspace_plan.mtp_prefill
                                                    : workspace_plan.text_prefill);
            if (speculative_backend == SpeculativeBackend::DFlash) {
                mark_workspace_usage(workspace_plan.dflash_context);
            }
            schedule::PrefillChunkResult result;
            const std::optional<std::uint32_t> rewrite_checkpoint_capture_frontier =
                staged.rewrite_checkpoint_capture
                    ? std::optional<std::uint32_t>(staged.rewrite_checkpoint_capture->frontier)
                    : std::nullopt;
            if (staged.vision) {
                mark_workspace_usage(workspace_plan.vision_encode);
                result = schedule::prefill_multimodal_chunk(
                    schedule_state, staged.prompt, *staged.vision, nominal,
                    rewrite_checkpoint_capture_frontier, final_candidate);
            } else {
                result = schedule::prefill_text_chunk(
                    schedule_state, std::span<const TokenId>(staged.prompt.token_ids), nominal,
                    rewrite_checkpoint_capture_frontier, final_candidate);
            }
            if (result.processed_tokens == 0 || result.processed_tokens > nominal) {
                throw std::logic_error("ordinary prefill chunk made invalid progress");
            }
            processed_prompt_tokens = result.processed_tokens;
            if (staged.vision) { staged.vision->release_encoded_media_payloads(); }
            staged.cursor += result.processed_tokens;
            sequence.text_kv_valid = staged.cursor;
            if (staged.prepare_mtp) { sequence.mtp_kv_valid = staged.cursor; }
            if (speculative_backend == SpeculativeBackend::DFlash) {
                sequence.dflash_context_frontier = staged.cursor;
            }

            // Hybrid prefix cache: commit newly completed blocks and capture tap snapshots at
            // chunk boundaries. Shared with the mixed unit so a prefill folded into a mixed round
            // grows the cache identically.
            capture_prefix_after_chunk(sequence);

            if (!result.finalized) {
                if (staged.cursor == staged.prompt_tokens) {
                    throw std::logic_error("staged prefill reached the prompt without sampling");
                }
                staged.elapsed_seconds +=
                    std::chrono::duration<double>(Clock::now() - started).count();
                return runtime::PrefillStepResult{
                    .summary = summary, .processed_prompt_tokens = processed_prompt_tokens};
            }
            if (staged.cursor != staged.prompt_tokens) {
                throw std::logic_error("staged prefill sampled before the prompt frontier");
            }
            copy_tail(sequence, prefill_hidden.slice(
                                    1, static_cast<std::int32_t>(result.processed_tokens) - 1, 1));
        } else {
            mark_workspace_usage(workspace_plan.ordinary_round);
            if (!sequence.tail_hidden_valid) {
                throw std::logic_error("zero-suffix reuse has no target tail hidden");
            }
            schedule::sample_from_hidden(schedule_state, sequence.tail_hidden,
                                         checked_i32(staged.prompt_tokens, "sample position"),
                                         ops::kSamplePurposePrefill);
            set_device_i32(io.rope_pos, checked_i32(staged.prompt_tokens, "rope position") +
                                            sequence.rope_delta);
            if (staged.prepare_mtp) {
                if (staged.mtp_bridge != MtpBridgeMode::AfterExactHit) {
                    throw std::logic_error("zero-suffix MTP reuse has no exact-hit bridge");
                }
                mark_workspace_usage(workspace_plan.mtp_prefill);
                const auto bridge_rope =
                    prompt_rope_position(staged.prompt, staged.prompt_tokens - 1);
                schedule::mtp_bridge_and_propose(
                    schedule_state, io.token, sequence.tail_hidden,
                    checked_i32(staged.prompt_tokens - 1, "MTP full-prefix bridge position"),
                    bridge_rope, staged.initial_mtp_extent != 0);
                sequence.mtp_kv_valid = staged.prompt_tokens;
                staged.mtp_bridge     = MtpBridgeMode::None;
            }
        }

        copy_round_token();
        copy_round_logits();
        // Prefill's bonus token has now been sampled on rank 0, which is the one place rank 0's
        // penalty counters advance without rank 1's doing the same. Bring rank 1's lane level
        // before any decode round reads it; a no-op at tp1 and whenever penalties are off (the
        // counter lane is then never read).
        publish_peer_token_counts(sequence);
        std::array<TokenId, qwen3_6::kMtpDecodeMaximumDrafts> initial_drafts{};
        if (staged.prepare_mtp && staged.initial_mtp_extent != 0) {
            CUDA_CHECK(cudaMemcpyAsync(initial_drafts.data(), io.mtp->draft_tokens.data,
                                       staged.initial_mtp_extent * sizeof(TokenId),
                                       cudaMemcpyDeviceToHost, device.stream));
        }
        device.synchronize();
        staged.elapsed_seconds += std::chrono::duration<double>(Clock::now() - started).count();
        const double vision_seconds = staged.vision ? staged.vision->elapsed_seconds() : 0.0;
        const std::optional<RewriteCheckpointSpec> rewrite_checkpoint_capture =
            staged.rewrite_checkpoint_capture;
        const std::uint32_t prompt_tokens = staged.prompt_tokens;

        validate_licensed_tokens(std::span<const TokenId>(host_tokens, 1));
        if (sequence.ledger.size() != prompt_tokens) {
            throw std::logic_error("candidate token ledger does not match prompt length");
        }
        sequence.ledger.push_back(host_tokens[0]);
        sequence.prefix_identity.append_generated(1, sequence.rope_delta);
        sequence.text_kv_valid = prompt_tokens;
        if (staged.prepare_mtp) {
            if (sequence.mtp_kv_valid != prompt_tokens) {
                throw std::logic_error("staged MTP prefill did not reach the prompt frontier");
            }
            sequence.mtp_draft_count = staged.initial_mtp_extent;
            std::copy_n(initial_drafts.begin(), staged.initial_mtp_extent,
                        sequence.mtp_drafts.begin());
        } else if (speculative_backend == SpeculativeBackend::DFlash &&
                   sequence.dflash_context_frontier != prompt_tokens) {
            throw std::logic_error("staged DFlash prefill did not reach the prompt frontier");
        }
        sequence.tail_hidden_valid      = true;
        request.timings.vision_seconds  = vision_seconds;
        request.timings.prefill_seconds = std::max(0.0, staged.elapsed_seconds - vision_seconds);
        if (rewrite_checkpoint_capture) {
            const std::uint32_t frontier = rewrite_checkpoint_capture->frontier;
            if (frontier == 0 || frontier > prompt_tokens || sequence.text_kv_valid < frontier) {
                throw std::logic_error("rewrite checkpoint was not materialized by Text prefill");
            }
            if (speculative_backend == SpeculativeBackend::Mtp &&
                (!staged.prepare_mtp || sequence.mtp_kv_valid < frontier - 1)) {
                throw std::logic_error("rewrite checkpoint has no complete MTP prefix");
            }
            if (speculative_backend == SpeculativeBackend::DFlash &&
                (!dflash || !sequence.kv ||
                 dflash->full.has_value() != sequence.kv->backend.has_value() ||
                 sequence.dflash_context_frontier < frontier)) {
                throw std::logic_error("rewrite checkpoint has no complete DFlash prefix");
            }
            sequence.rewrite_checkpoint = RewriteCheckpoint{
                .valid = true, .kind = rewrite_checkpoint_capture->kind, .frontier = frontier};
        }

        staged.prompt.release_all_media_payloads();

        request.prefill.reset();
        request.pending   = PendingCandidate{.kind          = PendingKind::Begin,
                                             .base_E        = 0,
                                             .base_S        = 0,
                                             .prompt_tokens = prompt_tokens,
                                             .produced      = 1};
        request.lifecycle = Lifecycle::Pending;
        return runtime::PrefillStepResult{
            .summary = summary,
            .round   = runtime::GeneratedRound{.tokens = std::span<const TokenId>(host_tokens, 1)},
            .processed_prompt_tokens = processed_prompt_tokens,
            .complete                = true,
        };
    } catch (...) {
        try {
            // Retire BOTH devices before tearing the lane down: rank 1 may still have enqueued
            // work referencing the KV pages and GDN slots clear_lane is about to release. Same
            // order as the decode path's handler.
            synchronize_peers();
            device.synchronize();
        } catch (...) {}
        clear_lane(sequence, request);
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                                       std::span<const runtime::RoundBudget> budgets) {
    if (speculative_backend != SpeculativeBackend::None) {
        throw std::logic_error("ordinary batch execution requires the ordinary backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("ordinary batch membership is invalid");
    }

    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("ordinary batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = sequences[lane];
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            sequence.kv->text.bound_row() < 0 || sequence.execution_frontier >= capacity ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier) {
            throw std::logic_error("ordinary batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    const auto start = Clock::now();
    try {
        DecodeGraphExecutable* executable = nullptr;
        ops::GqaExecutionEnvelope envelope{maximum_frontier + 1, maximum_frontier + 1};
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(ordinary_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "ordinary batch");
            executable = &install_graph_profile(ordinary_graphs, profile, "ordinary batch");
            envelope   = {profile.min_execution_frontier + 1, profile.max_execution_frontier + 1};
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence            = sequences[lanes[row]];
            const RequestControl& request      = requests[lanes[row]];
            const std::uint32_t frontier       = sequence.execution_frontier;
            ordinary_host_ingress->tokens[row] = sequence.ledger.back();
            ordinary_host_ingress->cache_positions[row] =
                checked_i32(frontier, "ordinary batch position");
            ordinary_host_ingress->rope_positions[row] =
                checked_i32(frontier, "ordinary batch RoPE position") + sequence.rope_delta;
            ordinary_host_ingress->text_kv_table_rows[row] = sequence.kv->text.bound_row();
            ordinary_host_ingress->lanes[row]    = static_cast<std::int32_t>(sequence.lane);
            ordinary_host_ingress->sampling[row] = request.sampling_host;
            materialize_sequence_kv(sequence, frontier + 1, 0);
        }
        publish_peer_ordinary_ingress();

        schedule::OrdinaryBatchContext schedule_state{
            {device, model, work, decoder->linear_attention,
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, rope_frequency, peer_ptrs(), tp},
            decoder->text_kv,
            *io.ordinary,
            *ordinary_host_ingress,
            *ordinary_host_egress,
            tail_hidden_store,
            ordinary_peer_host_ingress};

        mark_workspace_usage(workspace_plan.ordinary_round);
        schedule::ordinary_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                        envelope, executable);
        synchronize_peers();
        device.synchronize();

        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence    = sequences[lanes[row]];
            RequestControl& request    = requests[lanes[row]];
            const std::uint32_t base_E = sequence.execution_frontier;
            const std::uint32_t base_S = sequence.ledger_frontier;
            const TokenId token        = ordinary_host_egress->sampled_tokens[row];
            validate_licensed_tokens(std::span<const TokenId>(&token, 1));
            sequence.text_kv_valid     = base_E + 1;
            sequence.tail_hidden_valid = true;
            sequence.ledger.push_back(token);
            sequence.prefix_identity.append_generated(1, sequence.rope_delta);
            request.pending   = PendingCandidate{.kind          = PendingKind::Ordinary,
                                                 .base_E        = base_E,
                                                 .base_S        = base_S,
                                                 .prompt_tokens = 0,
                                                 .produced      = 1};
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens = std::span<const TokenId>(ordinary_host_egress->sampled_tokens.data(),
                                               lanes.size())};
    } catch (...) {
        try {
            synchronize_peers();
            device.synchronize();
        } catch (...) {}
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

runtime::MixedBatchResult
ProgramImplCore::decode_mixed_batch(std::span<const std::uint32_t> lanes,
                                    std::span<const runtime::RoundBudget> budgets,
                                    std::uint32_t owner_lane) {
    if (speculative_backend == SpeculativeBackend::Mtp) {
        return decode_mixed_mtp_batch(lanes, budgets, owner_lane);
    }
    if (speculative_backend != SpeculativeBackend::None) {
        throw std::logic_error("mixed unit requires the ordinary or MTP backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size() ||
        owner_lane >= max_concurrency) {
        throw std::invalid_argument("mixed unit membership is invalid");
    }
    if (!io.ordinary || !io.ordinary->mixed) {
        throw std::logic_error("mixed unit requires the mixed frame");
    }
    qwen3_6::OrdinaryDecodeState::MixedState& mixed = *io.ordinary->mixed;

    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || lane == owner_lane ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("mixed unit contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = sequences[lane];
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            sequence.kv->text.bound_row() < 0 || sequence.execution_frontier >= capacity ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier) {
            throw std::logic_error("mixed unit row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    SequenceState& owner_sequence = sequences[owner_lane];
    RequestControl& owner_request = requests[owner_lane];
    if (owner_request.lifecycle != Lifecycle::Prefilling || !owner_request.prefill ||
        !owner_sequence.kv || owner_sequence.kv->text.bound_row() < 0) {
        throw std::logic_error("mixed unit owner is not an active prefill");
    }
    RequestControl::Prefill& staged = *owner_request.prefill;
    if (staged.vision || staged.prepare_mtp || staged.mtp_bridge != MtpBridgeMode::None ||
        staged.cursor > staged.prompt_tokens) {
        throw std::logic_error("mixed unit does not support this prefill mode");
    }
    // M2: the mixed owner participates in the retained-checkpoint and hybrid-prefix machinery
    // exactly as an isolated prefill chunk does. `staged.cursor` is already an absolute prompt/KV
    // position (it starts at `staged.base`), so the owner frontier and the per-token KV positions
    // are the cursor itself; adding `staged.base` again would double-count the reused prefix.
    const std::uint32_t remaining = staged.prompt_tokens - staged.cursor;
    if (remaining == 0) { throw std::logic_error("mixed unit owner has no prompt left"); }
    std::uint32_t nominal = std::min(prefill_chunk, remaining);
    // End the prompt-tail chunk on a block boundary so a pending tail tap captures at an aligned
    // cursor (shared with the standalone prefill path).
    nominal = cap_final_chunk_for_taps(owner_sequence, staged.cursor, staged.prompt_tokens, nominal);
    // The owner's rewrite-checkpoint capture is only materialized when the checkpoint frontier lands
    // exactly on the owner chunk end. A frontier strictly inside the chunk needs the standalone
    // PrefillImpl internal split, which a single flat mixed pass cannot express; that case is left
    // to the standalone prefill step, so the mixed unit never forces a chunk split (which would
    // perturb the decode batch composition).
    const RewriteCheckpointSpec* rewrite_checkpoint_capture =
        staged.rewrite_checkpoint_capture ? &*staged.rewrite_checkpoint_capture : nullptr;
    const bool finalize                = staged.cursor + nominal == staged.prompt_tokens;
    const std::uint32_t owner_frontier = staged.cursor + nominal;
    if (owner_frontier > capacity) {
        throw std::logic_error("mixed unit owner frontier exceeds capacity");
    }
    const std::int32_t B = static_cast<std::int32_t>(lanes.size());
    const std::int32_t c = static_cast<std::int32_t>(nominal);
    const std::int32_t T = B + c;
    if (T > mixed.tokens.ne[0] || B > mixed.decode_kv_table_rows.ne[0]) {
        throw std::logic_error("mixed unit exceeds its configured frame");
    }
    // The owner's rewrite checkpoint is materialized when a capture was planned and the chunk now
    // ends exactly on its frontier: the mixed pass writes the raw hidden of the owner's last token
    // and copies the owner's GDN state into the checkpoint slot.
    const bool captures_checkpoint = rewrite_checkpoint_capture != nullptr &&
                                     rewrite_checkpoint_capture->frontier > staged.cursor &&
                                     rewrite_checkpoint_capture->frontier == owner_frontier;
    const std::int32_t rewrite_checkpoint_column =
        captures_checkpoint ? B + c - 1 : -1;
    const std::int32_t rewrite_checkpoint_state_slot =
        captures_checkpoint
            ? LinearStateSlots::rewrite_checkpoint_state_slot(owner_lane, max_concurrency)
            : -1;
    qwen3_6::MixedDecodeControl& control = mixed_host_control;
    control                              = qwen3_6::MixedDecodeControl{};
    control.decode_rows         = B;
    control.prefill_len         = c;
    control.prefill_linear_slot = LinearStateSlots::current_state_slot(owner_lane, max_concurrency);
    control.prefill_finalize    = finalize ? 1 : 0;
    control.prefill_sampling_valid = sampling_config.data != nullptr ? 1 : 0;
    control.prefill_bonus_position = checked_i32(owner_frontier, "mixed bonus position");
    control.decode_min_visible     = 1;
    control.decode_max_visible     = maximum_frontier + 1;
    control.prefill_min_visible    = owner_frontier;
    control.prefill_max_visible    = owner_frontier;
    control.prefill_sampling       = owner_request.sampling_host;

    const auto started = Clock::now();
    try {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = sequences[lanes[row]];
            const RequestControl& request = requests[lanes[row]];
            const std::uint32_t frontier  = sequence.execution_frontier;
            mixed_host_tokens[row]          = sequence.ledger.back();
            mixed_host_cache_positions[row] = checked_i32(frontier, "mixed position");
            mixed_host_rope_positions[row] =
                checked_i32(frontier, "mixed RoPE position") + sequence.rope_delta;
            mixed_host_decode_kv_rows[row] = sequence.kv->text.bound_row();
            mixed_host_decode_linear_slots[row] =
                LinearStateSlots::current_state_slot(lanes[row], max_concurrency);
            control.lanes[row]    = static_cast<std::int32_t>(lanes[row]);
            control.sampling[row] = request.sampling_host;
            materialize_sequence_kv(sequence, frontier + 1, 0);
        }
        for (std::int32_t t = 0; t < c; ++t) {
            const std::uint32_t prompt_index = staged.cursor + static_cast<std::uint32_t>(t);
            mixed_host_tokens[B + t]         = staged.prompt.token_ids[prompt_index];
            mixed_host_cache_positions[B + t] =
                checked_i32(prompt_index, "mixed owner position");
            mixed_host_rope_positions[B + t] = prompt_rope_position(staged.prompt, prompt_index)[0];
        }
        std::int32_t prefill_kv_row = owner_sequence.kv->text.bound_row();
        materialize_sequence_kv(owner_sequence, owner_frontier, 0);

        schedule::MixedBatchContext schedule_state{
            {device, model, work, decoder->linear_attention,
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, rope_frequency, peer_ptrs(), tp},
            decoder->text_kv,
            *io.ordinary,
            control,
            mixed_host_egress,
            &mixed_host_bonus_token,
            tail_hidden_store,
            std::span<const std::int32_t>(mixed_host_tokens.data(), static_cast<std::size_t>(T)),
            std::span<const std::int32_t>(mixed_host_cache_positions.data(),
                                          static_cast<std::size_t>(T)),
            std::span<const std::int32_t>(mixed_host_rope_positions.data(),
                                          static_cast<std::size_t>(T)),
            std::span<const std::int32_t>(mixed_host_decode_kv_rows.data(),
                                          static_cast<std::size_t>(B)),
            std::span<const std::int32_t>(mixed_host_decode_linear_slots.data(),
                                          static_cast<std::size_t>(B)),
            prefill_kv_row,
            &owner_sequence.rewrite_checkpoint_hidden,
            rewrite_checkpoint_column,
            rewrite_checkpoint_state_slot};
        mark_workspace_usage(workspace_plan.mixed_round);
        schedule::mixed_prefill_decode_batch(schedule_state);
        synchronize_peers();
        device.synchronize();
        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence    = sequences[lanes[row]];
            RequestControl& request    = requests[lanes[row]];
            const std::uint32_t base_E = sequence.execution_frontier;
            const std::uint32_t base_S = sequence.ledger_frontier;
            const TokenId token        = mixed_host_egress.sampled_tokens[row];
            validate_licensed_tokens(std::span<const TokenId>(&token, 1));
            sequence.text_kv_valid     = base_E + 1;
            sequence.tail_hidden_valid = true;
            sequence.ledger.push_back(token);
            sequence.prefix_identity.append_generated(1, sequence.rope_delta);
            request.pending   = PendingCandidate{.kind          = PendingKind::Ordinary,
                                                 .base_E        = base_E,
                                                 .base_S        = base_S,
                                                 .prompt_tokens = 0,
                                                 .produced      = 1};
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }

        const std::uint32_t prompt_tokens = staged.prompt_tokens;
        runtime::PrefillStepResult owner_result;
        owner_result.summary = runtime::BeginSummary{
            .prompt_tokens        = prompt_tokens,
            .reused_prompt_tokens = staged.base,
            .prefix_reuse_path    = staged.reuse,
            .prefix_reuse_reason  = staged.reuse_reason,
            .prefix_divergence    = staged.reuse_divergence};
        owner_result.processed_prompt_tokens = nominal;
        staged.cursor += nominal;
        owner_sequence.text_kv_valid = owner_frontier;
        // Hybrid prefix cache: commit newly completed blocks and capture tap snapshots at chunk
        // boundaries, exactly as the standalone prefill path does after a chunk.
        capture_prefix_after_chunk(owner_sequence);
        if (captures_checkpoint) {
            owner_sequence.rewrite_checkpoint = RewriteCheckpoint{
                .valid    = true,
                .kind     = rewrite_checkpoint_capture->kind,
                .frontier = owner_frontier};
        }
        if (finalize) {
            const TokenId bonus = static_cast<TokenId>(mixed_host_bonus_token);
            validate_licensed_tokens(std::span<const TokenId>(&bonus, 1));
            if (staged.cursor != staged.prompt_tokens) {
                throw std::logic_error("mixed owner sampled before the prompt frontier");
            }
            copy_tail(owner_sequence, mixed.prefill_tail_hidden);
            owner_sequence.ledger.push_back(bonus);
            owner_sequence.prefix_identity.append_generated(1, owner_sequence.rope_delta);
            owner_sequence.tail_hidden_valid = true;
            staged.prompt.release_all_media_payloads();
            owner_request.prefill.reset();
            owner_request.pending =
                PendingCandidate{.kind          = PendingKind::Begin,
                                 .base_E        = 0,
                                 .base_S        = 0,
                                 .prompt_tokens = prompt_tokens,
                                 .produced      = 1};
            owner_request.lifecycle = Lifecycle::Pending;
            owner_result.round = runtime::GeneratedRound{
                .tokens = std::span<const TokenId>(&mixed_host_bonus_token, 1)};
            owner_result.complete = true;
        }
        return runtime::MixedBatchResult{
            .decode = runtime::BatchedGeneratedRound{
                .tokens = std::span<const TokenId>(mixed_host_egress.sampled_tokens.data(),
                                                   lanes.size()),
                .row_counts = {},
                .row_stride = 1},
            .owner = owner_result};
    } catch (...) {
        try {
            synchronize_peers();
            device.synchronize();
        } catch (...) {}
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        clear_lane(owner_sequence, owner_request);
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_mtp_batch(std::span<const std::uint32_t> lanes,
                                  std::span<const runtime::RoundBudget> budgets) {
    if (speculative_backend != SpeculativeBackend::Mtp || !io.mtp_decode ||
        decoder->mtp_cache() == nullptr) {
        throw std::logic_error("MTP batch execution requires the MTP backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("MTP batch membership is invalid");
    }

    const std::uint32_t width      = draft_window + 1;
    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("MTP batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = sequences[lane];
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv || !sequence.kv->backend ||
            sequence.kv->text.bound_row() < 0 || sequence.kv->backend->bound_row() < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.mtp_kv_valid != sequence.execution_frontier ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.mtp_draft_count > draft_window) {
            throw std::logic_error("MTP batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    const auto started = Clock::now();
    try {
        DecodeGraphExecutable* executable = nullptr;
        schedule::MtpGqaEnvelopes envelopes =
            mtp_gqa_envelopes(maximum_frontier, draft_window, capacity);
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(mtp_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "MTP batch");
            executable = &install_graph_profile(mtp_graphs, profile, "MTP batch");
            envelopes  = mtp_gqa_envelopes(profile.max_execution_frontier, draft_window, capacity);
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence           = sequences[lanes[row]];
            const RequestControl& request     = requests[lanes[row]];
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1
                                                    : 0;
            const std::uint32_t extent =
                std::min({sequence.mtp_draft_count, draft_window, max_by_budget,
                          capacity - sequence.execution_frontier - 1});
            mtp_host_ingress->anchors[row]        = sequence.ledger.back();
            mtp_host_ingress->base_frontiers[row] = checked_i32(frontier, "MTP batch frontier");
            mtp_host_ingress->remaining_budgets[row] =
                checked_i32(budgets[row].generated_tokens_remaining, "MTP batch remaining budget");
            mtp_host_ingress->current_extents[row]      = static_cast<std::int32_t>(extent);
            mtp_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1);
            for (std::uint32_t j = 0; j < draft_window; ++j) {
                mtp_host_ingress->current_drafts[row * draft_window + j] =
                    j < extent ? sequence.mtp_drafts[j] : sequence.ledger.back();
            }
            for (std::uint32_t j = 0; j < width; ++j) {
                const std::uint32_t position = frontier + std::min(j, extent);
                mtp_host_ingress->target_rope_positions[row * width + j] =
                    checked_i32(position, "MTP batch RoPE position") + sequence.rope_delta;
            }
            mtp_host_ingress->text_kv_table_rows[row] = sequence.kv->text.bound_row();
            mtp_host_ingress->mtp_kv_table_rows[row]  = sequence.kv->backend->bound_row();
            mtp_host_ingress->lanes[row]              = static_cast<std::int32_t>(sequence.lane);
            mtp_host_ingress->rope_deltas[row]        = sequence.rope_delta;
            mtp_host_ingress->sampling[row]           = request.sampling_host;
            materialize_sequence_kv(sequence, frontier + extent + 1,
                                    std::min(capacity, frontier + extent + draft_window));
        }
        publish_peer_mtp_ingress(lanes);

        schedule::MtpBatchContext schedule_state{{device, model, work, decoder->linear_attention,
                                                  replay_records ? &*replay_records : nullptr, io,
                                                  prefill_hidden, prefill_chunk, proposal_head,
                                                  rope_frequency, peer_ptrs(),
                                                  tp},
                                                 decoder->text_kv,
                                                 *decoder->mtp_cache(),
                                                 *io.mtp_decode,
                                                 *mtp_host_ingress,
                                                 *mtp_host_egress,
                                                 tail_hidden_store};

        mark_workspace_usage(workspace_plan.mtp_round);
        schedule::mtp_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                   draft_window, envelopes, executable);
        // Peer first, then rank 0: rank 1's stream carries the round's own work eagerly and the
        // graph's rank-1 nodes when captured, and the egress read below must not observe a round
        // rank 1 has not finished contributing to.
        synchronize_peers();
        device.synchronize();
        check_peer_mtp_egress(lanes.size());

        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = sequences[lanes[row]];
            RequestControl& request       = requests[lanes[row]];
            const std::uint32_t base_E    = sequence.execution_frontier;
            const std::uint32_t base_S    = sequence.ledger_frontier;
            const std::int32_t count_i    = mtp_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = mtp_host_egress->accepted_drafts[row];
            const std::int32_t next_i     = mtp_host_egress->next_extents[row];
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || next_i < 0 ||
                next_i > static_cast<std::int32_t>(draft_window) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("MTP batch returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(mtp_host_egress->licensed_tokens.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            const std::uint32_t pcur =
                static_cast<std::uint32_t>(mtp_host_ingress->current_extents[row]);
            if (pcur == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += pcur;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
            }
            request.pending = PendingCandidate{
                .kind          = PendingKind::Speculative,
                .base_E        = base_E,
                .base_S        = base_S,
                .prompt_tokens = 0,
                .produced      = static_cast<std::uint32_t>(count_i),
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(mtp_host_egress->licensed_tokens.data(),
                                                   lanes.size() * width),
            .row_counts = std::span<const std::int32_t>(mtp_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .row_stride = width};
    } catch (...) {
        try {
            synchronize_peers();
            device.synchronize();
        } catch (...) {}
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

bool ProgramImplCore::mixed_owner_eligible(std::uint32_t lane) const noexcept {
    if (speculative_backend != SpeculativeBackend::Mtp) { return true; }
    if (lane >= max_concurrency || !requests[lane].prefill) { return false; }
    const RequestControl::Prefill& staged = *requests[lane].prefill;
    if (staged.vision || staged.mtp_bridge != MtpBridgeMode::None) { return false; }
    if (staged.cursor >= staged.prompt_tokens) { return false; }
    // Any non-vision, bridge-free chunk is foldable, including the owner's tail: the mixed MTP unit
    // runs the owner's MTP alignment (draft seeding) after the shared pass. A pending BeforeSuffix
    // bridge (reused suffix's first chunk) still takes the standalone step.
    return true;
}

runtime::MixedBatchResult
ProgramImplCore::decode_mixed_mtp_batch(std::span<const std::uint32_t> lanes,
                                        std::span<const runtime::RoundBudget> budgets,
                                        std::uint32_t owner_lane) {
    if (speculative_backend != SpeculativeBackend::Mtp || !io.mtp_decode ||
        !io.mtp_decode->mixed || decoder->mtp_cache() == nullptr) {
        throw std::logic_error("mixed MTP batch execution requires the MTP backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size() ||
        owner_lane >= max_concurrency) {
        throw std::invalid_argument("mixed MTP membership is invalid");
    }
    const std::uint32_t width      = draft_window + 1;
    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || lane == owner_lane ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("mixed MTP contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = sequences[lane];
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv || !sequence.kv->backend ||
            sequence.kv->text.bound_row() < 0 || sequence.kv->backend->bound_row() < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.mtp_kv_valid != sequence.execution_frontier ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.mtp_draft_count > draft_window) {
            throw std::logic_error("mixed MTP row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    SequenceState& owner_sequence = sequences[owner_lane];
    RequestControl& owner_request = requests[owner_lane];
    if (owner_request.lifecycle != Lifecycle::Prefilling || !owner_request.prefill ||
        !owner_sequence.kv || owner_sequence.kv->text.bound_row() < 0) {
        throw std::logic_error("mixed MTP owner is not an active prefill");
    }
    RequestControl::Prefill& staged = *owner_request.prefill;
    if (staged.vision || staged.mtp_bridge != MtpBridgeMode::None ||
        staged.cursor > staged.prompt_tokens) {
        throw std::logic_error("mixed MTP does not support this prefill mode");
    }
    const std::uint32_t remaining = staged.prompt_tokens - staged.cursor;
    if (remaining == 0) { throw std::logic_error("mixed MTP owner has no prompt left"); }
    std::uint32_t nominal = std::min(prefill_chunk, remaining);
    nominal = cap_final_chunk_for_taps(owner_sequence, staged.cursor, staged.prompt_tokens, nominal);
    const RewriteCheckpointSpec* rewrite_checkpoint_capture =
        staged.rewrite_checkpoint_capture ? &*staged.rewrite_checkpoint_capture : nullptr;
    const bool finalize                = staged.cursor + nominal == staged.prompt_tokens;
    const std::uint32_t owner_frontier = staged.cursor + nominal;
    if (owner_frontier > capacity) {
        throw std::logic_error("mixed MTP owner frontier exceeds capacity");
    }
    const std::int32_t B           = static_cast<std::int32_t>(lanes.size());
    const std::int32_t c           = static_cast<std::int32_t>(nominal);
    const std::int32_t decode_cols = B * static_cast<std::int32_t>(width);
    const std::int32_t T           = decode_cols + c;
    qwen3_6::MtpDecodeState& frame = *io.mtp_decode;
    if (T > frame.mixed->ids.ne[0]) {
        throw std::logic_error("mixed MTP exceeds its configured frame");
    }
    const bool captures_checkpoint = rewrite_checkpoint_capture != nullptr &&
                                     rewrite_checkpoint_capture->frontier > staged.cursor &&
                                     rewrite_checkpoint_capture->frontier == owner_frontier;
    const std::int32_t rewrite_checkpoint_column     = captures_checkpoint ? T - 1 : -1;
    const std::int32_t rewrite_checkpoint_state_slot =
        captures_checkpoint
            ? LinearStateSlots::rewrite_checkpoint_state_slot(owner_lane, max_concurrency)
            : -1;

    qwen3_6::MtpMixedControl& control = mtp_mixed_host_control;
    control                           = qwen3_6::MtpMixedControl{};
    control.decode_rows               = B;
    control.decode_width              = static_cast<std::int32_t>(width);
    control.prefill_len               = c;
    control.prefill_linear_slot =
        LinearStateSlots::current_state_slot(owner_lane, max_concurrency);
    control.prefill_finalize       = finalize ? 1 : 0;
    control.prefill_sampling_valid = sampling_config.data != nullptr ? 1 : 0;
    control.prefill_bonus_position = checked_i32(owner_frontier, "mixed MTP bonus position");
    control.prefill_sampling       = owner_request.sampling_host;

    const auto started = Clock::now();
    try {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = sequences[lanes[row]];
            const RequestControl& request = requests[lanes[row]];
            const std::uint32_t frontier  = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1
                                                    : 0;
            const std::uint32_t extent =
                std::min({sequence.mtp_draft_count, draft_window, max_by_budget,
                          capacity - sequence.execution_frontier - 1});
            mtp_host_ingress->anchors[row]        = sequence.ledger.back();
            mtp_host_ingress->base_frontiers[row] = checked_i32(frontier, "mixed MTP frontier");
            mtp_host_ingress->remaining_budgets[row] =
                checked_i32(budgets[row].generated_tokens_remaining, "mixed MTP remaining budget");
            mtp_host_ingress->current_extents[row]      = static_cast<std::int32_t>(extent);
            mtp_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1);
            for (std::uint32_t j = 0; j < draft_window; ++j) {
                mtp_host_ingress->current_drafts[row * draft_window + j] =
                    j < extent ? sequence.mtp_drafts[j] : sequence.ledger.back();
            }
            for (std::uint32_t j = 0; j < width; ++j) {
                const std::uint32_t position = frontier + std::min(j, extent);
                mtp_host_ingress->target_rope_positions[row * width + j] =
                    checked_i32(position, "mixed MTP RoPE position") + sequence.rope_delta;
            }
            mtp_host_ingress->text_kv_table_rows[row] = sequence.kv->text.bound_row();
            mtp_host_ingress->mtp_kv_table_rows[row]  = sequence.kv->backend->bound_row();
            mtp_host_ingress->lanes[row]              = static_cast<std::int32_t>(sequence.lane);
            mtp_host_ingress->rope_deltas[row]        = sequence.rope_delta;
            mtp_host_ingress->sampling[row]           = request.sampling_host;
            materialize_sequence_kv(sequence, frontier + extent + 1,
                                    std::min(capacity, frontier + extent + draft_window));
        }
        publish_peer_mtp_ingress(lanes);
        for (std::int32_t t = 0; t < c; ++t) {
            const std::uint32_t prompt_index = staged.cursor + static_cast<std::uint32_t>(t);
            mtp_mixed_host_owner_tokens[t]         = staged.prompt.token_ids[prompt_index];
            mtp_mixed_host_owner_positions[t] =
                checked_i32(prompt_index, "mixed MTP owner position");
            mtp_mixed_host_owner_rope_positions[t] =
                prompt_rope_position(staged.prompt, prompt_index)[0];
        }
        const std::int32_t owner_kv_row = owner_sequence.kv->text.bound_row();
        materialize_sequence_kv(owner_sequence, owner_frontier, 0);

        schedule::MtpMixedBatchContext schedule_state{
            {device, model, work, decoder->linear_attention,
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, rope_frequency, peer_ptrs(), tp},
            decoder->text_kv,
            *decoder->mtp_cache(),
            frame,
            *mtp_host_ingress,
            *mtp_host_egress,
            control,
            &mtp_mixed_host_owner_bonus_token,
            tail_hidden_store,
            std::span<const std::int32_t>(mtp_mixed_host_owner_tokens.data(),
                                          static_cast<std::size_t>(c)),
            std::span<const std::int32_t>(mtp_mixed_host_owner_positions.data(),
                                          static_cast<std::size_t>(c)),
            std::span<const std::int32_t>(mtp_mixed_host_owner_rope_positions.data(),
                                          static_cast<std::size_t>(c)),
            owner_kv_row,
            mtp_kv_view(owner_sequence),
            mtp_kv_view_peer(owner_sequence),
            staged.initial_mtp_extent,
            checked_i32(staged.prompt_tokens, "mixed MTP owner position"),
            &owner_sequence.rewrite_checkpoint_hidden,
            rewrite_checkpoint_column,
            rewrite_checkpoint_state_slot};
        mark_workspace_usage(workspace_plan.mtp_mixed_round);
        schedule::mixed_mtp_batch(schedule_state, B, draft_window,
                                  mtp_gqa_envelopes(maximum_frontier, draft_window, capacity));
        synchronize_peers();
        device.synchronize();

        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = sequences[lanes[row]];
            RequestControl& request       = requests[lanes[row]];
            const std::uint32_t base_E    = sequence.execution_frontier;
            const std::uint32_t base_S    = sequence.ledger_frontier;
            const std::int32_t count_i    = mtp_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = mtp_host_egress->accepted_drafts[row];
            const std::int32_t next_i     = mtp_host_egress->next_extents[row];
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || next_i < 0 ||
                next_i > static_cast<std::int32_t>(draft_window) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("mixed MTP returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(
                mtp_host_egress->licensed_tokens.data() + row * width,
                static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            const std::uint32_t pcur =
                static_cast<std::uint32_t>(mtp_host_ingress->current_extents[row]);
            if (pcur == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += pcur;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
            }
            request.pending = PendingCandidate{
                .kind          = PendingKind::Speculative,
                .base_E        = base_E,
                .base_S        = base_S,
                .prompt_tokens = 0,
                .produced      = static_cast<std::uint32_t>(count_i),
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }

        const std::uint32_t prompt_tokens = staged.prompt_tokens;
        runtime::PrefillStepResult owner_result;
        owner_result.summary = runtime::BeginSummary{
            .prompt_tokens        = prompt_tokens,
            .reused_prompt_tokens = staged.base,
            .prefix_reuse_path    = staged.reuse,
            .prefix_reuse_reason  = staged.reuse_reason,
            .prefix_divergence    = staged.reuse_divergence};
        owner_result.processed_prompt_tokens = nominal;
        staged.cursor += nominal;
        owner_sequence.text_kv_valid = owner_frontier;
        // Mirror the standalone prefill chunk's MTP-KV prefix claim (program_impl's staged prefill
        // advances `mtp_kv_valid` to the chunk end for every prepared-MTP chunk).
        owner_sequence.mtp_kv_valid = owner_frontier;
        capture_prefix_after_chunk(owner_sequence);
        if (captures_checkpoint) {
            owner_sequence.rewrite_checkpoint = RewriteCheckpoint{
                .valid    = true,
                .kind     = rewrite_checkpoint_capture->kind,
                .frontier = owner_frontier};
        }
        if (finalize) {
            const TokenId bonus = static_cast<TokenId>(mtp_mixed_host_owner_bonus_token);
            validate_licensed_tokens(std::span<const TokenId>(&bonus, 1));
            if (staged.cursor != staged.prompt_tokens) {
                throw std::logic_error("mixed MTP owner sampled before the prompt frontier");
            }
            copy_tail(owner_sequence, frame.mixed->owner_tail_hidden);
            // The owner's bonus token was sampled on rank 0 alone, so rank 1's penalty counters have
            // not advanced with it; bring them level before any decode round reads them.
            publish_peer_token_counts(owner_sequence);
            // Seed the owner's MTP draft window from the alignment the schedule just ran.
            std::array<TokenId, qwen3_6::kMtpDecodeMaximumDrafts> initial_drafts{};
            if (staged.initial_mtp_extent != 0) {
                if (staged.initial_mtp_extent > qwen3_6::kMtpDecodeMaximumDrafts) {
                    throw std::logic_error("mixed MTP initial extent exceeds the draft window");
                }
                CUDA_CHECK(cudaMemcpyAsync(initial_drafts.data(), io.mtp->draft_tokens.data,
                                           staged.initial_mtp_extent * sizeof(TokenId),
                                           cudaMemcpyDeviceToHost, device.stream));
                device.synchronize();
            }
            owner_sequence.ledger.push_back(bonus);
            owner_sequence.prefix_identity.append_generated(1, owner_sequence.rope_delta);
            owner_sequence.tail_hidden_valid = true;
            owner_sequence.mtp_draft_count  = staged.initial_mtp_extent;
            for (std::uint32_t step = 0; step < staged.initial_mtp_extent; ++step) {
                owner_sequence.mtp_drafts[step] = initial_drafts[step];
            }
            staged.prompt.release_all_media_payloads();
            owner_request.prefill.reset();
            owner_request.pending =
                PendingCandidate{.kind          = PendingKind::Begin,
                                 .base_E        = 0,
                                 .base_S        = 0,
                                 .prompt_tokens = prompt_tokens,
                                 .produced      = 1};
            owner_request.lifecycle = Lifecycle::Pending;
            owner_result.round = runtime::GeneratedRound{
                .tokens = std::span<const TokenId>(&mtp_mixed_host_owner_bonus_token, 1)};
            owner_result.complete = true;
        }
        return runtime::MixedBatchResult{
            .decode = runtime::BatchedGeneratedRound{
                .tokens     = std::span<const TokenId>(mtp_host_egress->licensed_tokens.data(),
                                                       lanes.size() * width),
                .row_counts = std::span<const std::int32_t>(mtp_host_egress->licensed_counts.data(),
                                                            lanes.size()),
                .row_stride = width},
            .owner = owner_result};
    } catch (...) {
        try {
            synchronize_peers();
            device.synchronize();
        } catch (...) {}
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        clear_lane(owner_sequence, owner_request);
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_dflash_batch(std::span<const std::uint32_t> lanes,
                                     std::span<const runtime::RoundBudget> budgets) {
    if (speculative_backend != SpeculativeBackend::DFlash || !io.dflash_decode || !dflash) {
        throw std::logic_error("DFlash batch execution requires the DFlash backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("DFlash batch membership is invalid");
    }

    const std::uint32_t width           = draft_window + 1U;
    std::uint32_t maximum_frontier      = 0;
    std::uint32_t maximum_target_tokens = 1;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("DFlash batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = sequences[lane];
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            dflash->full.has_value() != sequence.kv->backend.has_value() ||
            sequence.kv->text.bound_row() < 0 ||
            (sequence.kv->backend && sequence.kv->backend->bound_row() < 0) ||
            sequence.execution_frontier >= capacity ||
            sequence.text_kv_valid != sequence.execution_frontier ||
            sequence.dflash_context_frontier > sequence.execution_frontier ||
            sequence.execution_frontier - sequence.dflash_context_frontier > width ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier) {
            throw std::logic_error("DFlash batch row is not decode-ready");
        }
        const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                ? budgets[row].generated_tokens_remaining - 1U
                                                : 0U;
        const std::uint32_t extent =
            std::min({draft_window, max_by_budget, capacity - sequence.execution_frontier - 1U});
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
        maximum_target_tokens =
            std::max(maximum_target_tokens, sequence.execution_frontier + extent + 1U);
    }

    const auto started = Clock::now();
    try {
        const bool greedy_target = std::all_of(
            lanes.begin(), lanes.end(), [&](std::uint32_t lane) {
                return !(requests[lane].sampling_host.temperature > 0.0F);
            });
        DecodeGraphExecutable* executable   = nullptr;
        schedule::DFlashEnvelopes envelopes = dflash_envelopes(0, maximum_frontier, draft_window);
        ops::GqaExecutionEnvelope target_envelope{1, maximum_target_tokens};
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(dflash_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "DFlash batch", greedy_target);
            executable      = &install_graph_profile(dflash_graphs, profile, "DFlash batch");
            envelopes       = dflash_envelopes(profile.min_execution_frontier,
                                               profile.max_execution_frontier, draft_window);
            target_envelope = {
                1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                       capacity, static_cast<std::uint64_t>(profile.max_execution_frontier) +
                                     draft_window + 1ULL))};
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence           = sequences[lanes[row]];
            const RequestControl& request     = requests[lanes[row]];
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1U
                                                    : 0U;
            const std::uint32_t extent =
                std::min({draft_window, max_by_budget, capacity - frontier - 1U});
            dflash_host_ingress->anchors[row] = sequence.ledger.back();
            dflash_host_ingress->execution_frontiers[row] =
                checked_i32(frontier, "DFlash batch frontier");
            dflash_host_ingress->context_frontiers[row] =
                checked_i32(sequence.dflash_context_frontier, "DFlash context frontier");
            dflash_host_ingress->proposal_extents[row]     = static_cast<std::int32_t>(extent);
            dflash_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1U);
            dflash_host_ingress->text_kv_table_rows[row]   = sequence.kv->text.bound_row();
            dflash_host_ingress->dflash_kv_table_rows[row] =
                sequence.kv->backend ? sequence.kv->backend->bound_row() : 0;
            dflash_host_ingress->lanes[row]    = static_cast<std::int32_t>(sequence.lane);
            dflash_host_ingress->sampling[row] = request.sampling_host;
            materialize_sequence_kv(sequence, frontier + extent + 1U,
                                     sequence.kv->backend ? frontier : 0U);
        }

        publish_peer_dflash_ingress(lanes);
        schedule::DFlashBatchContext schedule_state{{device, model, work, decoder->linear_attention,
                                                     replay_records ? &*replay_records : nullptr,
                                                     io, prefill_hidden, prefill_chunk,
                                                     proposal_head,                                                      rope_frequency, peer_ptrs(),
                                                     tp},
                                                    decoder->text_kv,
                                                    *dflash,
                                                    *io.dflash_decode,
                                                    *dflash_host_ingress,
                                                    *dflash_host_egress,
                                                    tail_hidden_store,
                                                    greedy_target};

        mark_workspace_usage(workspace_plan.dflash_round);
        schedule::dflash_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                      draft_window, envelopes, target_envelope, executable);
        device.synchronize();

        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = sequences[lanes[row]];
            RequestControl& request       = requests[lanes[row]];
            const std::uint32_t base_E    = sequence.execution_frontier;
            const std::uint32_t base_S    = sequence.ledger_frontier;
            const std::int32_t count_i    = dflash_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = dflash_host_egress->accepted_drafts[row];
            const std::uint32_t extent =
                static_cast<std::uint32_t>(dflash_host_ingress->proposal_extents[row]);
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || accepted_i > static_cast<std::int32_t>(extent) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("DFlash batch returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(dflash_host_egress->licensed_tokens.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            if (extent == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += extent;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
            }
            sequence.dflash_context_frontier = base_E;
            request.pending                  = PendingCandidate{
                                 .kind          = PendingKind::Speculative,
                                 .base_E        = base_E,
                                 .base_S        = base_S,
                                 .prompt_tokens = 0,
                                 .produced      = static_cast<std::uint32_t>(count_i),
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(dflash_host_egress->licensed_tokens.data(),
                                                   lanes.size() * width),
            .row_counts = std::span<const std::int32_t>(dflash_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .row_stride = width};
    } catch (...) {
        try {
            synchronize_peers();
            device.synchronize();
        } catch (...) {}
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_batch(std::span<const std::uint32_t> lanes,
                              std::span<const runtime::RoundBudget> budgets) {
    tick_prefix_cache();
    if (speculative_backend == SpeculativeBackend::None) {
        return decode_ordinary_batch(lanes, budgets);
    }
    if (speculative_backend == SpeculativeBackend::Mtp) { return decode_mtp_batch(lanes, budgets); }
    return decode_dflash_batch(lanes, budgets);
}

void ProgramImplCore::resolve_non_speculative_pending(SequenceState& sequence,
                                                      RequestControl& request,
                                                      std::uint32_t accepted_tokens,
                                                      bool terminal) {
    if (request.lifecycle != Lifecycle::Pending) {
        throw std::logic_error("pending resolution requires a pending generated round");
    }
    if ((request.pending.kind != PendingKind::Begin &&
         request.pending.kind != PendingKind::Ordinary) ||
        request.pending.produced != 1 || accepted_tokens != 1) {
        throw std::logic_error("non-speculative pending round must commit its single token");
    }

    switch (request.pending.kind) {
    case PendingKind::Begin:
        sequence.execution_frontier = request.pending.prompt_tokens;
        sequence.ledger_frontier    = request.pending.prompt_tokens + 1;
        break;
    case PendingKind::Ordinary:
        sequence.execution_frontier = request.pending.base_E + request.pending.produced;
        sequence.ledger_frontier    = request.pending.base_S + request.pending.produced;
        break;
    case PendingKind::Speculative:
    case PendingKind::None:
        throw std::logic_error("non-speculative pending round has an invalid kind");
    }
    if (sequence.ledger_frontier != sequence.execution_frontier + 1 ||
        sequence.ledger.size() != sequence.ledger_frontier ||
        sequence.prefix_identity.size() != sequence.ledger_frontier) {
        throw std::logic_error("resolved round did not establish a valid frontier");
    }
    trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
    if (terminal) {
        sequence.mtp_draft_count = 0;
        release_sequence_growth_entitlement(sequence);
        unbind_sequence_kv(sequence);
        sequence.retained = true;
    }
    request.lifecycle = terminal ? Lifecycle::Complete : Lifecycle::Active;
    request.pending   = {};
}

MemorySummary ProgramImplCore::memory_summary() const noexcept {
    MemorySummary out;
    out.device                = device.device;
    out.max_context           = capacity;
    out.rope_mode             = rope_mode;
    out.effective_max_context = effective_max_context;
    out.yarn_mscale           = yarn_mscale;
    out.kv_capacity           = kv_capacity;
    out.kv_cache = kv_dtype == DType::BF16 ? KvCacheStorage::BFloat16 : KvCacheStorage::Int8Group64;
    DeviceArena& weights = *model.weights_arena;
    out.weights = ArenaMemorySummary{weights.capacity(), weights.used(), weights.peak_used()};
    out.sequence =
        ArenaMemorySummary{persistent.capacity(), persistent.used(), persistent.peak_used()};
    out.workspace = ArenaMemorySummary{workspace_storage.capacity(), work.used(), work.peak_used()};
    out.workspace_logical_peak_bytes   = workspace_logical_peak_bytes;
    out.cuda_graph_allowance_bytes     = graph_allowance_bytes;
    out.cuda_graph_observed_bytes      = graph_observed_bytes[0];
    out.cuda_graph_peer_observed_bytes = graph_observed_bytes[1];
    out.cuda_graph_device_bytes        = graph_observed_bytes;
    out.cuda_graph_node_count          = graph_node_count;
    out.kv_payload_bytes               = kv_payload_bytes;
    out.gdn_state_bytes                = gdn_state_bytes;
    return out;
}

void ProgramImplCore::reset_memory_peaks() noexcept {
    model.weights_arena->reset_peak();
    persistent.reset_peak();
    work.reset_peak();
    workspace_logical_peak_bytes = 0;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
