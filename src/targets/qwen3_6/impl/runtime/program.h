#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/gdn_replay_records.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/sampling.h"
#include "core/decode_graph.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/dflash_context.h"
#include "targets/qwen3_6/impl/runtime/linear_state_slots.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"
#include "runtime/prefix_cache/prefix_index.h"
#include "runtime/prefix_cache/prefix_store.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/vision_prefill.h"

#include <cstdint>
#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using PreparedPromptData    = qwen3_6::PreparedPromptData;
using RewriteCheckpointKind = qwen3_6::RewriteCheckpointKind;
using RewriteCheckpointSpec = qwen3_6::RewriteCheckpointSpec;

using ReusePath = ninfer::PrefixReusePath;

[[nodiscard]] constexpr bool is_rewrite_checkpoint_restore(ReusePath path) noexcept {
    return path == ReusePath::RestoreTurnCheckpoint || path == ReusePath::RestoreResponseCheckpoint;
}

[[nodiscard]] constexpr ReusePath restore_path(RewriteCheckpointKind kind) noexcept {
    return kind == RewriteCheckpointKind::TurnClosure ? ReusePath::RestoreTurnCheckpoint
                                                      : ReusePath::RestoreResponseCheckpoint;
}

enum class RewriteCheckpointAction : std::uint8_t {
    Drop,
    KeepExisting,
    ReclassifyExisting,
    CaptureNew,
    DeferCapture,
};

enum class MtpBridgeMode : std::uint8_t {
    None,
    BeforeSuffix,
    AfterExactHit,
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
struct RequestBasePlanImpl<NINFER_QWEN36_VARIANT> {
    runtime::RequestPlanSummary summary;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    std::shared_ptr<const qwen3_6::VisionControl> vision_control;
    std::size_t vision_transient_bytes = 0;
    std::optional<qwen3_6::RewriteCheckpointSpec> rewrite_checkpoint;
    bool allow_prefix_reuse = false;
};

template <>
struct RequestPlanImpl<NINFER_QWEN36_VARIANT> {
    runtime::RequestPlanSummary summary;
    NINFER_QWEN36_RUNTIME_NS::ReusePath reuse = NINFER_QWEN36_RUNTIME_NS::ReusePath::FullReset;
    std::uint32_t reuse_base                  = 0;
    PrefixReuseReason reuse_reason            = PrefixReuseReason::None;
    PrefixDivergence reuse_divergence;
    PrefixAdmissionDiagnostics reuse_diag;
    NINFER_QWEN36_RUNTIME_NS::MtpBridgeMode mtp_bridge =
        NINFER_QWEN36_RUNTIME_NS::MtpBridgeMode::None;
    bool prepare_mtp = false;
    std::optional<NINFER_QWEN36_RUNTIME_NS::VisionPrefillPlan> vision;
    NINFER_QWEN36_RUNTIME_NS::RewriteCheckpointAction rewrite_checkpoint_action =
        NINFER_QWEN36_RUNTIME_NS::RewriteCheckpointAction::Drop;
    std::optional<qwen3_6::RewriteCheckpointSpec> rewrite_checkpoint_capture;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;

    // Hybrid block-snapshot admission (see PrefixCacheOptions). `block_snapshot` is the chosen
    // resident snapshot, `block_path` the matched block-tree path (index 0 is the root), and
    // `prefix_taps` the chunk-aligned positions to capture during this request's prefill.
    ninfer::prefix_cache::SnapshotId block_snapshot = ninfer::prefix_cache::kInvalidSnapshot;
    std::vector<ninfer::prefix_cache::NodeId> block_path;
    std::uint32_t block_path_blocks = 0;
    std::vector<std::uint32_t> prefix_taps;
};

} // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using RequestPlanImpl     = qwen3_6::detail::RequestPlanImpl<Variant>;
using RequestBasePlanImpl = qwen3_6::detail::RequestBasePlanImpl<Variant>;

enum class PendingKind : std::uint8_t {
    None,
    Begin,
    Ordinary,
    Speculative,
};

struct PendingCandidate {
    PendingKind kind            = PendingKind::None;
    std::uint32_t base_E        = 0;
    std::uint32_t base_S        = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t produced      = 0;
};

enum class Lifecycle : std::uint8_t {
    Empty,
    Prefilling,
    Active,
    Pending,
    Complete,
};

struct RewriteCheckpoint {
    bool valid                 = false;
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::uint32_t frontier     = 0;
};

struct SequenceKVBundle {
    PagedKVAllocation text;
    std::optional<PagedKVAllocation> backend;
    // Each peer rank's allocation in ITS OWN text KV pool at tp > 1, indexed by rank. The pools
    // have identical page geometry (only the per-page byte count differs, because a rank holds
    // only its own shard of the KV heads), and every pool operation below is issued on all ranks
    // in the same order, so all allocations hold the same page ids and publish identical block
    // tables.
    std::array<std::optional<PagedKVAllocation>, kMaximumDevices> text_peer;
    // Each peer rank's allocation in ITS OWN MTP (backend) KV pool at tp > 1. Same lockstep
    // argument as `text_peer`: identical page geometry, every pool operation issued in the same
    // order.
    std::array<std::optional<PagedKVAllocation>, kMaximumDevices> backend_peer;
};

struct DecodeGraphProfile {
    std::uint32_t batch_size             = 1;
    std::uint32_t min_execution_frontier = 0;
    std::uint32_t max_execution_frontier = 0;
    std::uint32_t topology_class         = 0;
    bool greedy_target                   = false;
    DecodeGraphDefinition definition;
};

struct DecodeGraphTopology {
    std::uint32_t topology_class = 0;
    DecodeGraphExecutable executable;
    std::optional<std::size_t> installed_profile;
};

struct DecodeGraphFamily {
    std::vector<DecodeGraphProfile> profiles;
    std::vector<DecodeGraphTopology> topologies;
};

// Target model continuation for one logical sequence. This state remains meaningful after the
// request which produced it has finished, so it is deliberately separate from request lifecycle,
// output, sampling, and round-control state.
struct SequenceState {
    std::optional<SequenceKVBundle> kv;
    Tensor tail_hidden;
    Tensor rewrite_checkpoint_hidden;
    std::uint32_t lane = 0;

    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier    = 0;
    std::vector<TokenId> ledger;
    qwen3_6::detail::ResidentPrefixIdentity prefix_identity;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::array<TokenId, qwen3_6::kMtpDecodeMaximumDrafts> mtp_drafts{};
    std::uint32_t mtp_draft_count = 0;
    bool tail_hidden_valid        = false;
    bool retained                 = false;
    RewriteCheckpoint rewrite_checkpoint;

    // Hybrid content-addressed prefix cache. Nodes/pages borrowed for this lane's life, plus the
    // tap state images captured into the reserved snapshot slots at chunk boundaries.
    std::vector<ninfer::prefix_cache::NodeId> prefix_pinned_nodes;
    std::vector<std::uint32_t> prefix_tap_positions;
    std::vector<ninfer::prefix_cache::SlotId> prefix_tap_slots;
    std::uint32_t prefix_tap_next = 0;
    bool prefix_taps_planned      = false;
    bool prefix_capture           = false;
    std::uint32_t prefix_matched_blocks = 0;
    ninfer::prefix_cache::NodeId prefix_parent = ninfer::prefix_cache::kInvalidNode;
    std::vector<TokenId> prefix_prompt_tokens;
    std::vector<std::uint64_t> prefix_block_hashes;
    std::vector<std::uint64_t> prefix_block_extras;
    ninfer::prefix_cache::SnapshotId prefix_snapshot =
        ninfer::prefix_cache::kInvalidSnapshot;
    bool prefix_borrowed = false;
    // Committed-block node path ([root, node0, ...]) left behind when a lane TERMINATES while it had
    // promoted (cache-owned) pages but its KV allocation survives retention. Reusages that keep that
    // resident KV (AppendAtFrontier / rewrite-checkpoint restore) must seed prefix_pinned_nodes from
    // this so capture resumes at the promotion watermark instead of re-promoting already-cache-owned
    // leading pages. Cleared with the allocation on abort/clear_lane.
    std::vector<ninfer::prefix_cache::NodeId> prefix_retained_commit_nodes;
};

// Request/round control is not retained with a reusable SequenceState. A later concurrent Engine
// gives every occupied request slot its own instance of this state.
struct RequestControl {
    Lifecycle lifecycle = Lifecycle::Empty;
    PendingCandidate pending;
    ops::SamplingConfig sampling_host;
    GenerationTimings timings;
    SpeculativeStats speculative_stats;

    struct Prefill {
        PreparedPromptData prompt;
        std::optional<VisionPrefillPlan> vision_plan;
        std::unique_ptr<schedule::VisionPrefillSession> vision;
        runtime::TransientRegion transient;
        std::optional<RewriteCheckpointSpec> rewrite_checkpoint_capture;
        std::uint32_t base               = 0;
        std::uint32_t cursor             = 0;
        std::uint32_t prompt_tokens      = 0;
        std::uint32_t initial_mtp_extent = 0;
        double elapsed_seconds           = 0.0;
        bool prepare_mtp                 = false;
        ReusePath reuse                  = ReusePath::FullReset;
        MtpBridgeMode mtp_bridge         = MtpBridgeMode::None;
        PrefixReuseReason reuse_reason   = PrefixReuseReason::None;
        PrefixDivergence reuse_divergence;
        PrefixAdmissionDiagnostics reuse_diag;
        std::vector<std::uint32_t> prefix_taps;
    };

    std::optional<Prefill> prefill;
};

// Rank 1's complete runtime mirror: its own arenas, its own shard of the weights, its own halved
// decoder state, and its own RoundState. It owns no bookkeeping -- lanes, page accounting,
// sampling and the pinned host round buffers all live once, on rank 0.
struct PeerRuntime {
    PeerRuntime(DeviceContext& peer_device, const LoadedModelData& peer_model,
                const SequencePlanImpl& plan);

    PeerRuntime(const PeerRuntime&)            = delete;
    PeerRuntime& operator=(const PeerRuntime&) = delete;

    DeviceContext& device;
    const LoadedModelData& model;
    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    std::unique_ptr<qwen3_6::DecoderState> decoder;
    std::optional<DFlashPersistentState> dflash;
    // Rank 1's own GDN replay records: the speculative verify round records this device's own
    // head/channel shard and folds it here, so the two devices commit the same accepted prefix
    // from records neither ever exchanges.
    std::optional<GdnReplayRecords> replay_records;
    qwen3_6::RoundState io;
    Tensor prefill_hidden;
    // Peer rank's private destination for the replicated Vision embeddings a sharded encode
    // produces ([out_hidden, merged_limit] BF16, rank 1..tp-1 only). Owned for the life of the
    // Program -- not per request -- so a native vision request never cudaMallocs, and it outlives
    // every workspace reset (the MTP bridge reads the last encoded item after prefill).
    std::optional<DeviceBuffer> vision_embed;
    // Rank 1's OWN penalty counters. `ops::SamplingConfig::token_counts` is a raw device pointer,
    // and the MTP round's acceptance runs on both devices (see the replicated-accept note in
    // mtp_impl.h), so rank 1 must never be handed rank 0's. It is not a cache: rank 0 remains the
    // source of truth (ProgramImplCore::install_sampling zeroes both, and the one increment that
    // happens on rank 0 alone -- prefill's bonus token -- is copied across before the first decode
    // round), and thereafter both are advanced by the same Op over bit-identical inputs.
    Tensor token_counts;
};

class ProgramImplCore {
public:
    ProgramImplCore(const std::array<const LoadedModelData*, kMaximumDevices>& models,
                    const SequencePlanImpl& plan, ExecutionContext& execution);
    ~ProgramImplCore() noexcept;

    [[nodiscard]] RequestBasePlan
    plan_request_base(const PreparedPromptData& prompt,
                      const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] RequestPlan plan_request_for_lane(std::uint32_t lane,
                                                    const PreparedPromptData& prompt,
                                                    const RequestBasePlan& base);
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const;
    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t lane, const RequestPlan& plan) const;
    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_cache_residue() const;
    [[nodiscard]] runtime::PrefillStepResult start_prefill_lane(std::uint32_t lane,
                                                                PreparedPromptData&& prompt,
                                                                RequestPlan&& plan,
                                                                runtime::TransientRegion transient);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const runtime::RoundBudget> budgets);
    // L2 mixed unit: one flat ragged pass over the decode-ready lanes plus the staged prefill
    // owner's next chunk. Returns the decode rows' tokens (ordinary round shape) and the owner's
    // step. Requires the ordinary backend; M1 rejects vision, MTP bridges and prefix taps.
    [[nodiscard]] runtime::MixedBatchResult
    decode_mixed_batch(std::span<const std::uint32_t> lanes,
                       std::span<const runtime::RoundBudget> budgets, std::uint32_t owner_lane);
    // True iff this Program planned the L2 mixed frame (ordinary backend, concurrency >= 2) or the
    // M3 mixed MTP frame (MTP backend, concurrency >= 2).
    [[nodiscard]] bool supports_mixed_unit() const noexcept {
        if (io.ordinary) { return io.ordinary->mixed.has_value(); }
        if (io.mtp_decode) { return io.mtp_decode->mixed.has_value(); }
        return false;
    }
    [[nodiscard]] bool mixed_owner_eligible(std::uint32_t lane) const noexcept;
    void resolve_prefill_lane(std::uint32_t lane, bool terminal);
    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled);
    void abort_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept;
    void evict_retained_lane(std::uint32_t lane) noexcept;
    // Demand set (node ids) of the currently admitted plan's matched block path, owned by the
    // scheduler thread for the duration of that admission round. Merged with every retained
    // lane's committed node path inside evict_one_prefix_node().
    void set_eviction_demand(std::span<const std::uint32_t> demand_nodes) noexcept;
    // Retained lanes' committed node chains whose ancestor closure an admission reservation on
    // `lane` leaves eviction-protected. Used so can_admit_lane() credits only cache pages eviction
    // is actually willing to reclaim. The plan's own matched/reuse path is deliberately NOT part of
    // the demand: those pages are adopted by the incoming plan, so they remain available capacity.
    // `include_other_lanes` is false for the retained-eviction path (those lanes are evicted first);
    // `include_current_lane` is true only when the plan reuses the lane's resident KV --
    // FullReset/BlockSnapshot clear the lane's retained chain before reserving, making it
    // reclaimable.
    [[nodiscard]] std::vector<ninfer::prefix_cache::NodeId>
    reservation_demand(std::uint32_t lane, bool include_other_lanes,
                       bool include_current_lane) const;
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const noexcept;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    [[nodiscard]] MetricsSnapshot metrics() const noexcept;

    void reset_memory_peaks() noexcept;

    const LoadedModelData& model;
    ExecutionContext& execution;
    DeviceContext& device;
    const int tp;
    const std::uint32_t capacity;
    const std::uint32_t kv_capacity;
    const std::uint32_t max_concurrency;
    const std::uint32_t prefill_chunk;
    const std::uint32_t draft_window;
    const SpeculativeBackend speculative_backend;
    const DType kv_dtype;
    const std::int32_t kv_quant_group;
    const ProposalHead proposal_head;
    const bool vision_enabled;
    const bool use_cuda_graph;
    const std::size_t kv_payload_bytes;
    const std::size_t gdn_state_bytes;
    const std::size_t graph_allowance_bytes;
    // Measured graph residency per device (index = rank). At tp1 only [0] is populated. For tp > 1
    // the single cross-device graph materializes driver state on EVERY rank, and each is checked
    // against the SAME per-device allowance -- graph_allowance_bytes is a per-device budget, like
    // every other field in device_reservation_bytes.
    std::array<std::size_t, kMaximumDevices> graph_observed_bytes{};
    // Node count of ONE captured decode graph (the first profile of the captured family). At tp > 1
    // one graph holds all ranks' nodes, so this is the direct measurement of whether every peer's
    // half of the schedule was captured rather than left out.
    std::size_t graph_node_count = 0;
    const WorkspacePlan workspace_plan;

    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    // YaRN rotary state, resolved once at construction.
    //
    // `rope_frequency_storage[rank]` is a dedicated 32-float device allocation on rank `rank`'s OWN
    // device, made once here and never touched again: CUDA Graph capture bakes the pointer into the
    // replayed rope launch node, so it must outlive every replay, and at tp 2 each rank ropes its
    // own head-local q/k on its own device, where a pointer into the other rank's allocation is not
    // addressable. It is deliberately NOT part of the planned persistent arena: under
    // `RopeMode::Native` nothing is allocated at all, which is what keeps a native plan and its
    // memory summary byte-identical to the pre-YaRN engine.
    //
    // `rope_frequency[rank]` is the descriptor every text rope call site reads (through
    // `ExecutionCore::rope_frequency` -> `TextContext`); a null `inv_frequency` IS the native path.
    std::array<DeviceBuffer, kMaximumDevices> rope_frequency_storage;
    std::array<ops::RopeFrequencyOverride, kMaximumDevices> rope_frequency{};
    const RopeMode rope_mode;
    const std::uint32_t effective_max_context;
    const double yarn_mscale;
    std::array<std::optional<PeerRuntime>, kMaximumDevices> peer;
    std::optional<ops::CollectiveComms> collective_comms;
    std::array<std::optional<schedule::TpPeerCore>, kMaximumDevices> peer_core;
    std::unique_ptr<qwen3_6::DecoderState> decoder;
    std::optional<GdnReplayRecords> replay_records;
    std::optional<DFlashPersistentState> dflash;
    qwen3_6::RoundState io;
    Tensor prefill_hidden;
    Tensor sampling_config;
    Tensor token_counts;
    Tensor tail_hidden_store;
    Tensor rewrite_checkpoint_hidden_store;

    std::array<SequenceState, kMaximumConcurrency> sequences;
    std::array<RequestControl, kMaximumConcurrency> requests;

    DecodeGraphFamily ordinary_graphs;
    DecodeGraphFamily mtp_graphs;
    DecodeGraphFamily dflash_graphs;

    PinnedHostBuffer round_host;
    TokenId* host_tokens = nullptr;

    // Per-rank pinned Vision staging upload blocks (streaming recipes); zero when the tower is
    // device-resident. The plan-time geometry reserved their arena room; the request admission
    // check needs the same per-rank sizes to compare an item's full encode footprint against THIS
    // rank's workspace envelope.
    std::array<std::size_t, kMaximumDevices> vision_staged_bytes{};

    // Hybrid prefix cache (opt-in via SequencePlanImpl::prefix_cache). `prefix_index` owns the
    // content-addressed block tree over this Program's committed text-KV pages and the cached
    // prefix state-image slots. `prefix_sink` returns evicted pages to every rank's pool. Both are
    // null unless the cache is enabled, so the default path is byte-identical to the legacy one.
    struct PrefixCacheSink;
    std::unique_ptr<PrefixCacheSink> prefix_sink;
    std::unique_ptr<ninfer::prefix_cache::PrefixIndex> prefix_index;
    std::unique_ptr<ninfer::prefix_cache::HostSlabPool> prefix_host_pool; // M2
    std::unique_ptr<ninfer::prefix_cache::DiskSlabPool> prefix_disk_pool; // M3a
    // M3a disk tier: container paths bound at construction. The index path holds metadata
    // (topology + per-record disk slot/generation); the slab path holds the KV bytes. Empty
    // disables persistence, leaving the in-memory cache behavior unchanged.
    std::filesystem::path prefix_store_file;
    std::filesystem::path prefix_slabs_file;
    // Bounded flush sweep cursors and the index-write debounce. The sweep spills host-resident
    // records into the disk tier a few per round boundary so the steady-state cost stays bounded;
    // the index is rewritten every `kPrefixIndexDebounceTicks` boundaries while the disk tier is
    // active.
    std::uint64_t prefix_flush_node_cursor = 1;
    std::uint64_t prefix_flush_snap_cursor = 1;
    std::uint64_t prefix_tick_count        = 0;
    std::uint64_t prefix_index_written_tick = 0;
    // Eviction demand for the current admission round (see set_eviction_demand).
    std::vector<ninfer::prefix_cache::NodeId> eviction_demand_;
    std::uint64_t retained_lane_evictions_total_ = 0;
    std::uint32_t prefix_snapshot_slots = 0;
    std::uint32_t prefix_max_new_taps   = 0;
    std::uint32_t prefix_ladder_tokens  = 0;
    std::uint32_t prefix_min_gap        = 0;
    // M2: dedicated cache transfer stream per rank for D2H/H2D copies.
    std::array<cudaStream_t, kMaximumDevices> prefix_cache_stream{};
    std::array<cudaEvent_t, kMaximumDevices> prefix_cache_event{};
    // Source-ordering events, one per rank: each marks "every compute-stream operation submitted
    // so far on this rank is complete". A cache-stream D2H must wait on it before reading device
    // state (a GDN snapshot slot or KV page) that the compute stream just wrote, otherwise the copy
    // can capture stale contents. Recorded immediately before each write-through D2H.
    std::array<cudaEvent_t, kMaximumDevices> prefix_cache_src_event{};
    // Per-rank slice sizes inside a host slab (a slab holds tp consecutive rank slices).
    std::size_t prefix_block_rank_bytes = 0;
    std::size_t prefix_snapshot_rank_bytes = 0;
    // Write-throughs enqueued but not yet polled complete. The newest per-rank cache event marks
    // the end of the whole pending set, so tick_prefix_cache() drains it all-or-nothing.
    struct InflightCacheWrite {
        bool is_snapshot = false;
        ninfer::prefix_cache::NodeId node = ninfer::prefix_cache::kInvalidNode;
        ninfer::prefix_cache::SnapshotId snapshot = ninfer::prefix_cache::kInvalidSnapshot;
    };
    std::vector<InflightCacheWrite> prefix_inflight_writes;
    // Continuation hidden captured with each tap state image; column `slot` matches the
    // LinearStateSlots snapshot slot. Empty unless the cache is enabled.
    Tensor prefix_snapshot_hidden_store;
    // Debug-only capture of the round's full-vocabulary logits, raw BF16 bits. OFF by default and
    // completely inert until enable_logits_capture(true) allocates the host buffer: with it off,
    // copy_round_logits() is a predictable branch and nothing is allocated or copied, so no
    // production path (tp1 or tp2) is changed. With it on, the buffer is overwritten every time a
    // token is sampled from io.logits at prefill finalization, for both tp1 and tp2 (io is rank
    // 0's window; logits_parallel gathers into it the same way the tp1 leaf writes it), so no
    // tp-specific code is needed. Prefill only -- the decode path is CUDA-graph-capturable and an
    // unconditional memcpy inside a captured region is rejected at capture time; a probe that
    // requests exactly one output token completes on prefill's own sampled token and never runs a
    // decode round, which is what the parity harness (tools/tp2/parity.cpp) uses. Valid only
    // immediately after Engine::generate()/wait() returns to the calling thread for a single-lane
    // engine -- concurrent lanes would overwrite it out of order.
    std::vector<std::uint16_t> logits_capture;
    bool logits_capture_enabled = false;
    std::optional<PinnedHostBuffer> ordinary_host;
    qwen3_6::OrdinaryDecodeIngress* ordinary_host_ingress = nullptr;
    qwen3_6::OrdinaryDecodeEgress* ordinary_host_egress   = nullptr;
    // L2 mixed unit host staging. The control block is uploaded whole by the schedule body; the
    // flat [T] token/position/rope arrays and the decode-row selectors are uploaded per unit
    // before the body runs. Sized at construction from the mixed frame's configured capacities.
    qwen3_6::MixedDecodeControl mixed_host_control;
    std::vector<std::int32_t> mixed_host_tokens;
    std::vector<std::int32_t> mixed_host_cache_positions;
    std::vector<std::int32_t> mixed_host_rope_positions;
    std::vector<std::int32_t> mixed_host_decode_kv_rows;
    std::vector<std::int32_t> mixed_host_decode_linear_slots;
    qwen3_6::OrdinaryDecodeEgress mixed_host_egress;
    std::int32_t mixed_host_bonus_token = 0;
    // M3 mixed MTP unit host staging. The speculative slab's in/out records reuse the MTP ingress/
    // egress; only the prefill owner carries extra staging here (its control block, its flat chunk
    // tokens/positions/rope, and the sampled bonus token).
    qwen3_6::MtpMixedControl mtp_mixed_host_control;
    std::vector<std::int32_t> mtp_mixed_host_owner_tokens;
    std::vector<std::int32_t> mtp_mixed_host_owner_positions;
    std::vector<std::int32_t> mtp_mixed_host_owner_rope_positions;
    std::int32_t mtp_mixed_host_owner_bonus_token = 0;
    // Each peer rank's own pinned ordinary ingress: rank 0's record with every row's
    // `sampling[row].token_counts` nulled. A peer mirrors the ordinary round for its shard of the
    // weights and never samples (the output head is vocabulary-split and sampling belongs to rank
    // 0), so the sampling configs in its frame are inert -- but they must not carry a rank-0
    // DEVICE address into the peer's frame: dereferencing one from a peer is a silent cross-device
    // fault. A separate pinned buffer rather than a patched copy at issue time, for the same
    // reason as `mtp_peer_host_ingress`: the upload is inside the captured decode graph, which
    // re-reads this exact host address at every replay.
    std::array<std::optional<PinnedHostBuffer>, kMaximumDevices> ordinary_peer_host;
    std::array<qwen3_6::OrdinaryDecodeIngress*, kMaximumDevices> ordinary_peer_host_ingress{};
    bool peer_egress_check_enabled     = false;
    std::uint64_t peer_egress_rounds     = 0;
    std::uint64_t peer_egress_mismatches = 0;
    std::optional<PinnedHostBuffer> mtp_host;
    qwen3_6::MtpDecodeIngress* mtp_host_ingress = nullptr;
    qwen3_6::MtpDecodeEgress* mtp_host_egress   = nullptr;
    // Each peer rank's own pinned MTP ingress: byte-for-byte rank 0's record except that each
    // row's `sampling[row].token_counts` names that rank's counter lane. It has to be a separate
    // pinned buffer rather than a patched copy made at issue time, because the ingress upload is
    // inside the captured decode graph and the graph re-reads this exact host address at every
    // replay.
    std::array<std::optional<PinnedHostBuffer>, kMaximumDevices> mtp_peer_host;
    std::array<qwen3_6::MtpDecodeIngress*, kMaximumDevices> mtp_peer_host_ingress{};
    std::optional<PinnedHostBuffer> dflash_host;
    qwen3_6::DFlashDecodeIngress* dflash_host_ingress = nullptr;
    qwen3_6::DFlashDecodeEgress* dflash_host_egress   = nullptr;
    std::array<std::optional<PinnedHostBuffer>, kMaximumDevices> dflash_peer_host;
    std::array<qwen3_6::DFlashDecodeIngress*, kMaximumDevices> dflash_peer_host_ingress{};

    std::size_t workspace_logical_peak_bytes = 0;

    // See logits_capture's comment. Read-only; the caller owns thread-safety (single-lane, read
    // after wait() returns). Empty while capture is disabled.
    [[nodiscard]] std::span<const std::uint16_t> last_round_logits_bf16() const noexcept {
        return logits_capture;
    }

    // Turns the debug capture above on or off. Enabling sizes the host buffer to the round's
    // logits row count; disabling releases it. Must not be called while a round is in flight (the
    // executor holds its execution mutex across this call).
    void enable_logits_capture(bool enabled);

    // Debug-only, OFF by default: after each MTP decode round at tp == 2, read rank 1's MTP egress
    // back and compare it field for field with rank 0's. The two ranks run the acceptance Op over
    // bit-identical inputs, so their egress records are argued to agree; enabling this turns that
    // induction into a measurement. Costs one ~1 KiB device-to-host copy plus a host compare per
    // round while enabled, and nothing at all while off. Counters are cumulative over the
    // Program's lifetime; a mismatch is counted (per row, per field) rather than thrown, so a test
    // can read the totals after a clean run.
    void enable_peer_egress_check(bool enabled) noexcept;
    [[nodiscard]] std::uint64_t peer_egress_check_rounds() const noexcept {
        return peer_egress_rounds;
    }
    [[nodiscard]] std::uint64_t peer_egress_check_mismatches() const noexcept {
        return peer_egress_mismatches;
    }

private:
    // Returns cache-owned pages to every rank's text-KV pool (page ids are identical across ranks).
    void release_cached_pages(const ninfer::prefix_cache::PoolPages& pages) noexcept;
    // One pressure unit of the two-tier demand-aware device-tier eviction (demand-protected,
    // dead-branch-first; demote preferred over delete; trim onto the demand set only as a last
    // resort). Returns the topmost removed node, or kInvalidNode when nothing is evictable.
    [[nodiscard]] ninfer::prefix_cache::NodeId evict_one_prefix_node();
    // Release a request's prefix-cache state: persist blocks to the host tier (M2), return any
    // unconsumed tap-slot reservations, and drop the request's node/snapshot pins so its committed
    // blocks become evictable. Called when a request completes or its lane is cleared; idempotent.
    void finalize_prefix_cache_state(SequenceState& sequence) noexcept;
    void clear_lane(SequenceState& sequence, RequestControl& request) noexcept;
    void ordered_reset(SequenceState& sequence);
    void prepare_graphs();
    // Copies rank 0's counter lane onto rank 1. Called once, after prefill's bonus token is
    // sampled -- the only counter increment that happens on rank 0 and not on rank 1.
    [[nodiscard]] static Tensor token_counts_lane(const Tensor& storage, std::uint32_t lane);
    void publish_peer_token_counts(const SequenceState& sequence);
    // Mirrors `mtp_host_ingress` into `mtp_peer_host_ingress`, swapping every row's counter
    // pointer for rank 1's. No-op at tp1 or without MTP.
    void publish_peer_mtp_ingress(std::span<const std::uint32_t> lanes);
    void publish_peer_dflash_ingress(std::span<const std::uint32_t> lanes);
    // Mirrors `ordinary_host_ingress` into `ordinary_peer_host_ingress` with every row's counter
    // pointer nulled. No-op at tp1 or without an ordinary frame.
    void publish_peer_ordinary_ingress();
    // Debug-only: reads rank 1's MTP egress back and compares it, field for field, with rank 0's.
    // No-op unless the check is enabled and a peer exists.
    void check_peer_mtp_egress(std::size_t rows);
    void install_sampling(SequenceState& sequence, RequestControl& request,
                          const ops::SamplingConfig& config);
    void set_device_i32(Tensor& tensor, std::int32_t value);
    void set_peer_i32(const DeviceContext& rank_device, Tensor& tensor, std::int32_t value);
    // Synchronizes every peer rank's stream (rank 0 is synchronized separately by the caller).
    void synchronize_peers() noexcept;
    void reset_peer_work() noexcept;
    void copy_tail(SequenceState& sequence, const Tensor& source);
    // Hybrid prefix cache: commit every newly completed block and capture the tap snapshots whose
    // frontier the sequence's text KV has now reached, using `sequence.text_kv_valid` as the
    // completed cursor. Reached from both the standalone prefill step (advance_prefill) and the
    // mixed unit (decode_mixed_batch) so a prefill folded into a mixed round grows the cache
    // exactly as an isolated chunk does. No-op unless the sequence is capturing.
    void capture_prefix_after_chunk(SequenceState& sequence);
    // Block-aligned final-chunk cut: when a capturing sequence is about to finish its prompt and a
    // pending tap still lies at the tail, cap `nominal` so the chunk ends on a kBlockTokens
    // boundary (a tap's state image must be captured at an aligned cursor). Mirrors the cut in
    // advance_prefill. Returns the possibly-reduced chunk length.
    [[nodiscard]] std::uint32_t cap_final_chunk_for_taps(const SequenceState& sequence,
                                                         std::uint32_t cursor,
                                                         std::uint32_t prompt_tokens,
                                                         std::uint32_t nominal) const;
    void copy_round_token();
    void copy_round_logits();
    void resolve_non_speculative_pending(SequenceState& sequence, RequestControl& request,
                                         std::uint32_t accepted_tokens, bool terminal);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill(SequenceState& sequence,
                                                             RequestControl& request);
    void enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                       std::span<const std::uint32_t> starts,
                                       std::span<const std::uint32_t> counts);
    void validate_licensed_tokens(std::span<const TokenId> tokens) const;
    void mark_workspace_usage(std::size_t phase_bytes) noexcept;
    // M2 host-tier transfer layer. All copies run on the per-rank cache streams; slabs are
    // rank-sliced (rank r owns bytes [r * rank_bytes, (r+1) * rank_bytes) of a slab).
    // Returns false on the first failed CUDA copy (never throws).
    [[nodiscard]] qwen3_6::PagedKVCache* rank_backend_cache(int rank) noexcept;
    // Acquire one cache-owned page bundle (text pool + backend pool when present) on every rank.
    // The pools are lockstep, so every rank hands out the same ids; a mismatch is a logic error.
    // When a pool is exhausted, evict one device-resident cache node through the index (demoting
    // or erasing it via its sink) and retry until the acquire fits or nothing is evictable, in
    // which case std::bad_alloc propagates.
    [[nodiscard]] ninfer::prefix_cache::PoolPages acquire_cache_bundle();
    bool copy_cache_block(int rank, std::int32_t text_page, std::int32_t backend_page,
                          void* slab_base, bool device_to_host) noexcept;
    bool copy_snapshot_slot(int rank, std::int32_t slot, void* slab_base,
                            bool device_to_host) noexcept;
    // Issue an async D2H write-through for one cache entry (block node or snapshot image).
    // Best-effort and non-throwing: allocation or enqueue failure rolls the reservation back.
    void issue_block_write_through(ninfer::prefix_cache::NodeId node) noexcept;
    void issue_snapshot_write_through(ninfer::prefix_cache::SnapshotId id) noexcept;
    // Order the rank's cache stream after all compute-stream work submitted so far, so a D2H
    // write-through cannot read a GDN snapshot slot or KV page before the compute stream that wrote
    // it has finished. Called immediately before each per-rank D2H.
    void order_prefix_cache_after_compute(int rank) noexcept;
    // Poll in-flight write-throughs; complete (and unpin) every transfer whose per-rank events
    // have all fired. Called at round boundaries.
    void tick_prefix_cache() noexcept;
    // Blocking drain: wait for every pending cache write to land on host, then finalize it. A
    // completed request settles its writes this way so its committed blocks are host-backed and
    // evictable before any later admission reads the pool counts.
    void settle_prefix_writes() noexcept;
    // M3a disk tier: the geometry the persisted index is bound to (mirrors the disk pool).
    [[nodiscard]] ninfer::prefix_cache::PrefixStoreGeometry prefix_store_geometry() const noexcept;
    // M3a: spill up to a bounded number of host-resident, not-yet-spilled records into the disk
    // tier, then rewrite the index if the debounce has elapsed. Called at every round boundary;
    // bounded so a round boundary never stalls on a full sweep. No-op without a disk tier.
    void sweep_prefix_disk_tier() noexcept;
    // M3a: spill every host-resident record into the disk tier, then serialize the index to
    // `prefix_store_file`. No-op unless the disk tier exists. Best case only: a record that cannot
    // reach the disk tier is dropped by the serializer with its subtree.
    void persist_prefix_store() noexcept;
    // M3a: write the index to `prefix_store_file` atomically. Best-effort; logs on failure.
    void write_prefix_index() noexcept;
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                          std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_mtp_batch(std::span<const std::uint32_t> lanes,
                     std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_dflash_batch(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets);
    // M3: the speculative decode lanes run as one uniform verify slab in the same flat pass as the
    // staged prefill owner's next chunk.
    [[nodiscard]] runtime::MixedBatchResult
    decode_mixed_mtp_batch(std::span<const std::uint32_t> lanes,
                           std::span<const runtime::RoundBudget> budgets,
                           std::uint32_t owner_lane);
    void reserve_sequence_kv(SequenceState& sequence, std::uint32_t text_pages,
                             std::uint32_t backend_pages);
    void resize_sequence_kv_entitlement(SequenceState& sequence, std::uint32_t text_pages,
                                        std::uint32_t backend_pages);
    void resize_paged_kv_with_eviction(std::span<PagedKVResize> changes);
    void bind_sequence_kv(SequenceState& sequence);
    void unbind_sequence_kv(SequenceState& sequence) noexcept;
    void materialize_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                 std::uint32_t backend_tokens = 0);
    void trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                          std::uint32_t backend_tokens = 0);
    void release_sequence_growth_entitlement(SequenceState& sequence) noexcept;
    [[nodiscard]] qwen3_6::PagedKVCache* backend_kv_cache() noexcept;
    [[nodiscard]] const qwen3_6::PagedKVCache* backend_kv_cache() const noexcept;
    [[nodiscard]] std::uint32_t backend_kv_valid(const SequenceState& sequence) const noexcept;
    [[nodiscard]] qwen3_6::PagedKVCacheView text_kv_view(const SequenceState& sequence) const;
    [[nodiscard]] qwen3_6::PagedKVCacheView mtp_kv_view(const SequenceState& sequence) const;
    [[nodiscard]] std::array<qwen3_6::PagedKVCacheView, kMaximumDevices>
    mtp_kv_view_peer(const SequenceState& sequence) const;
    [[nodiscard]] std::array<const schedule::TpPeerCore*, kMaximumDevices>
    peer_ptrs() const noexcept;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
class ProgramImpl<NINFER_QWEN36_VARIANT> final : public NINFER_QWEN36_RUNTIME_NS::ProgramImplCore {
public:
    using NINFER_QWEN36_RUNTIME_NS::ProgramImplCore::ProgramImplCore;
};

} // namespace ninfer::targets::qwen3_6::detail
