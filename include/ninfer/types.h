#pragma once

#include <chrono>
#include <cstddef>
#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer {

using TokenId = std::int32_t;

inline constexpr std::uint32_t kMaximumConcurrency = 8;
// Aggregate encoded image/video payload retained by one prompt, independent of item count.
inline constexpr std::size_t kMaximumPromptMediaBytes = 256ULL << 20;
inline constexpr std::size_t kDefaultMediaCacheBytes  = 1ULL << 30;
inline constexpr std::size_t kDefaultMediaLiveBytes   = 2ULL << 30;

enum class KvCacheStorage : std::uint8_t {
    BFloat16,
    Int8Group64,
    // IQ4_NL-G32: 4-bit codes indexing a 16-entry asymmetric LUT (kvalues_iq4nl), one FP16 scale
    // per contiguous 32-value group along the head dim. Same fused-codec model as Int8Group64;
    // logical decode is LUT[code] * FP16(scale) (docs/maintainer/iq4nl-kv-plan.md).
    Int4NLGroup32,
};

enum class KvCapacityMode : std::uint8_t {
    Explicit,
    Automatic,
};

inline constexpr std::size_t kDefaultKvCapacityHeadroomBytes = 1024ULL * 1024ULL * 1024ULL;

struct KvCapacityPolicy {
    KvCapacityMode mode                  = KvCapacityMode::Explicit;
    std::uint32_t explicit_tokens        = 2048;
    std::size_t automatic_headroom_bytes = 0;

    [[nodiscard]] static constexpr KvCapacityPolicy
    explicit_capacity(std::uint32_t tokens) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Explicit, tokens, 0};
    }

    [[nodiscard]] static constexpr KvCapacityPolicy
    automatic(std::size_t headroom_bytes = kDefaultKvCapacityHeadroomBytes) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Automatic, 0, headroom_bytes};
    }
};

enum class ProposalHead : std::uint8_t {
    Full,
    Optimized,
};

// Rotary position-embedding regime. `Native` is the checkpoint's own RoPE: the registered native
// context capacity is the ceiling and nothing about the rope path changes. `Yarn` applies YaRN
// frequency correction (`--yarn-factor` x `--yarn-origin`), which raises the addressable context
// ceiling to `yarn_origin * yarn_factor` and substitutes a corrected inverse-frequency table plus
// an `mscale` multiplier inside the rope kernels. Native is bit-for-bit the pre-YaRN path.
enum class RopeMode : std::uint8_t {
    Native,
    Yarn,
};

// Product ceiling on any YaRN-extended context, independent of factor/origin.
inline constexpr std::uint32_t kMaximumYarnContext = 1048576;

enum class SpeculativeBackend : std::uint8_t {
    None,
    Mtp,
    DFlash,
};

struct SpeculativeOptions {
    SpeculativeBackend backend = SpeculativeBackend::None;
    std::uint32_t draft_tokens = 0;
    ProposalHead proposal_head = ProposalHead::Full;
};

// Content-addressed prefix-cache configuration. The cache stores reusable sequence states along
// the prompt's block path so any request sharing a prefix can resume from the deepest surviving
// boundary, independent of which session created it. This is device-only in the current milestone;
// when `enabled` is false the engine uses only the legacy owner/checkpoint reuse paths.
struct PrefixCacheOptions {
    bool enabled                 = false;
    // Extra per-lane state-image slots reserved for cache snapshots. Zero derives
    // `max_concurrency + 1` at startup.
    std::uint32_t snapshot_slots = 0;
    // Cap on new snapshots captured per request. Zero derives `2` without a host tier.
    std::uint32_t max_new_taps   = 0;
    // Geometric ladder spacing in tokens. Zero derives `max(4096, 2 * prefill_chunk)`.
    std::uint32_t ladder_tokens  = 0;
    // Minimum token gap between ladder taps. Zero derives `max(1024, prefill_chunk)`.
    std::uint32_t min_gap        = 0;
    // Pinned host RAM for the prefix pool in MiB. Zero = device-only (M1).
    std::uint32_t host_cache_mib = 0;
    // M3a disk tier: bounded disk budget in MiB for the lower cache tier. Requires the host tier
    // (`host_cache_mib > 0`) and a `file` path. Zero disables the disk tier. Spilled host slabs are
    // written to a fixed-slot slab file alongside the index at runtime and restored into the host
    // tier on admission; the budget bounds the slab file and is evicted per dead -> superseded ->
    // GDSF policy.
    std::uint32_t disk_cache_mib = 0;
    // M3a disk tier: base path for persistence. Empty = disabled. A path that names an existing
    // directory resolves the index to `<dir>/prefix-cache.bin` and the slab file to
    // `<dir>/prefix-cache.slabs`; any other path is used verbatim for the index with its extension
    // replaced by `.slabs` for the slab file.
    std::filesystem::path file;
};

struct LoadProgress {
    std::function<void(std::string_view phase, std::uint64_t done, std::uint64_t total)> callback;
};

struct EngineOptions {
    std::filesystem::path artifact_path;
    int device = 0;
    // Tensor-parallel degree: 1 (default, single device), 2, or 4. `tp > 1` splits the resident
    // model across that many CUDA devices and requires `devices` to name exactly `tp` distinct
    // ids of the same compute capability. It is supported by the 27B execution package
    // (`qwen3.6-27b`, `qwen3.8-27b`) with `SpeculativeBackend::None` or `Mtp`; `qwen3.6-35b-a3b`,
    // `SpeculativeBackend::DFlash`, and `enable_vision` are rejected at construction. `tp == 1`
    // is bit-identical to the single-device path.
    int tp = 1;
    // Explicit device ids, one per tp rank. Empty means "derive from `device`" (i.e. {device});
    // this lets callers that construct EngineOptions directly (tests, embedders) omit it. When
    // non-empty its size must equal tp.
    std::vector<int> devices;
    std::uint32_t max_context          = 2048; // Exact logical ceiling of each request.
    // Rotary regime. `RopeMode::Yarn` raises the ceiling `max_context` is validated against from
    // the variant's registered native capacity to `yarn_origin * yarn_factor` (capped at
    // `kMaximumYarnContext`); `yarn_origin` must equal that registered native capacity. Native
    // leaves every rope call site on the checkpoint's own constant frequency table.
    RopeMode rope_mode                 = RopeMode::Native;
    double yarn_factor                 = 4.0;
    std::uint32_t yarn_origin          = 262144;
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = 30000;
    std::uint32_t prefill_chunk        = 1024;
    KvCacheStorage kv_cache            = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    PrefixCacheOptions prefix_cache;
    std::size_t media_cache_bytes = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes  = kDefaultMediaLiveBytes;
    // Zero selects a bounded worker count from the detected host concurrency.
    std::uint32_t media_preprocess_threads = 0;
    bool enable_vision                     = false;
    // CUDA Graph capture for decode. The CLI and server default this off when `tp > 1` because the
    // multi-device graph replays with GPU idle bubbles and is slower than eager on the V100
    // target, and on when `tp == 1`; both modes produce the same tokens. Set explicitly to
    // override either default.
    bool use_cuda_graph                    = true;
    LoadProgress load_progress;
};

enum class SamplingMode : std::uint8_t {
    Thinking,
    NonThinking,
};

// Immutable model-owned values used when a request does not override a sampling field. Seed is
// deliberately excluded: it is an execution choice rather than a model recommendation.
struct SamplingPreset {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
};

struct ModelSamplingDefaults {
    SamplingPreset thinking;
    SamplingPreset non_thinking;

    [[nodiscard]] constexpr const SamplingPreset& for_mode(SamplingMode mode) const noexcept {
        return mode == SamplingMode::Thinking ? thinking : non_thinking;
    }
};

// Public request-side overrides. std::nullopt means "use the registered model/mode default";
// explicit zero remains a real override (including temperature=0 for exact argmax).
struct SamplingOverrides {
    std::optional<float> temperature;
    std::optional<std::int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

// Complete parameters after Engine resolution. Target runtimes consume only this type.
struct ResolvedSamplingParameters {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
    std::uint64_t seed      = 0;
};

enum class OutputChannel : std::uint8_t {
    Content,
    Reasoning,
};

struct StopString {
    std::string text;
    OutputChannel channel  = OutputChannel::Content;
    bool include_in_output = false;
};

struct StopPolicy {
    std::vector<TokenId> token_ids;
    std::vector<StopString> strings;
    bool include_model_defaults = true;
    bool publish_stop_token     = false;
};

struct ExecutionOptions {
    SamplingOverrides sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
};

struct OutputOptions {
    bool raw                     = false;
    bool preserve_special_tokens = false;
};

struct RequestOptions {
    ExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
};

enum class MediaKind : std::uint8_t {
    Image,
    Video,
};

struct OwnedMedia {
    MediaKind kind = MediaKind::Image;
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

// Wire-independent conversation authority. Protocol adapters preserve these roles and their
// ordering; a target frontend owns any model-specific role lowering.
enum class ChatRole : std::uint8_t {
    System,
    Developer,
    User,
    Assistant,
    Tool,
};

enum class MessagePartKind : std::uint8_t {
    Text,
    Media,
};

struct MessagePart {
    MessagePartKind kind = MessagePartKind::Text;
    std::string text;
    OwnedMedia media;
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::vector<MessagePart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
};

enum class ReasoningEffort : std::uint8_t {
    Low,
    Medium,
    XHigh,
};

struct ReasoningEffortCapabilities {
    bool low    = false;
    bool medium = false;
    bool xhigh  = false;
    std::optional<ReasoningEffort> default_effort;

    [[nodiscard]] constexpr bool supports(ReasoningEffort effort) const noexcept {
        switch (effort) {
        case ReasoningEffort::Low:
            return low;
        case ReasoningEffort::Medium:
            return medium;
        case ReasoningEffort::XHigh:
            return xhigh;
        }
        return false;
    }
};

struct PromptCapabilities {
    bool enable_thinking = false;
    ReasoningEffortCapabilities reasoning_effort;
};

struct PromptOptions {
    bool add_generation_prompt = true;
    bool enable_thinking       = true;
    std::optional<ReasoningEffort> reasoning_effort;
    bool preserve_thinking = false;
    bool add_vision_id     = false;
    std::vector<std::string> tool_jsons;
};

struct PromptInput {
    std::vector<ChatMessage> messages;
    PromptOptions options;
};

enum class RequestErrorKind : std::uint8_t {
    ContextLengthExceeded,
    MediaBudgetExceeded,
    Overloaded,
    QueueTimeout,
    Cancelled,
    Unavailable,
};

class RequestError final : public std::invalid_argument {
public:
    RequestError(RequestErrorKind kind, std::string message)
        : std::invalid_argument(std::move(message)), kind_(kind) {}

    [[nodiscard]] RequestErrorKind kind() const noexcept { return kind_; }

private:
    RequestErrorKind kind_;
};

struct PromptSummary {
    std::uint32_t prompt_tokens = 0;
    bool has_media              = false;
};

struct PromptPreparationStats {
    double seconds                       = 0.0;
    double media_preprocess_seconds      = 0.0;
    double media_preprocess_work_seconds = 0.0;
    double tokenize_seconds              = 0.0;
    std::size_t media_items              = 0;
    std::size_t media_bytes              = 0;
    std::uint64_t raw_patches            = 0;
    std::uint64_t vision_tokens          = 0;
    std::size_t patch_bytes              = 0;
    std::size_t media_cache_hits         = 0;
    std::size_t media_cache_misses       = 0;
    std::size_t media_singleflight_waits = 0;
    std::size_t built_patch_bytes        = 0;
    std::size_t reused_patch_bytes       = 0;
};

struct MediaCacheSummary {
    std::size_t capacity_bytes       = 0;
    std::size_t live_capacity_bytes  = 0;
    std::size_t retained_bytes       = 0;
    std::size_t live_bytes           = 0;
    std::size_t entries              = 0;
    std::size_t inflight             = 0;
    std::size_t queued_tasks         = 0;
    std::size_t active_tasks         = 0;
    std::uint32_t preprocess_threads = 0;
    std::uint64_t hits               = 0;
    std::uint64_t misses             = 0;
    std::uint64_t singleflight_waits = 0;
    std::uint64_t evictions          = 0;
    std::uint64_t oversize_bypasses  = 0;
};

enum class FinishReason : std::uint8_t {
    None,
    OutputLimit,
    ContextCapacity,
    StopToken,
    StopString,
    Cancelled,
};

struct OutputDelta {
    OutputChannel channel = OutputChannel::Content;
    std::string text;
};

class OutputSink {
public:
    virtual ~OutputSink()                   = default;
    virtual void publish(OutputDelta delta) = 0;
};

class CancellationView {
public:
    CancellationView() = default;
    explicit CancellationView(std::function<bool()> requested);

    [[nodiscard]] bool requested() const;

private:
    std::function<bool()> requested_;
};

// Deadline and cancellation apply to all host-side prompt preparation work. Empty values mean
// unbounded preparation.
struct PreparationControl {
    std::chrono::steady_clock::time_point deadline;
    CancellationView cancellation;
};

struct GenerationTimings {
    double prepare_seconds     = 0.0;
    double first_token_seconds = 0.0;
    double vision_seconds      = 0.0;
    double prefill_seconds     = 0.0;
    double decode_seconds      = 0.0;
    double total_seconds       = 0.0;
};

struct SpeculativeStats {
    SpeculativeBackend backend    = SpeculativeBackend::None;
    bool enabled                  = false;
    std::uint32_t draft_window    = 0;
    std::uint64_t rounds          = 0;
    std::uint64_t drafted_tokens  = 0;
    std::uint64_t accepted_tokens = 0;
    std::uint64_t fallback_steps  = 0;
    std::vector<std::uint64_t> accepted_per_position;
};

enum class PrefixReusePath : std::uint8_t {
    FullReset,
    AppendAtFrontier,
    RestoreTurnCheckpoint,
    RestoreResponseCheckpoint,
    BlockSnapshot,
};

// Diagnostic explanation of `PrefixReusePath`. A successful reuse reports which continuation state
// matched; a `FullReset` reports the dominant gate that discarded the retained sequence, so the
// request log can explain why a turn did not resume from the previous KV state.
enum class PrefixReuseReason : std::uint8_t {
    None = 0,
    ReusedFrontier,
    ReusedTurnCheckpoint,
    ReusedResponseCheckpoint,
    ReusedBlockSnapshot,
    ReuseDisabled,
    PromptNotReusable,
    NoRetainedSequence,
    FrontierEmpty,
    DflashFrontierMismatch,
    FrontierPrefixMismatch,
    CheckpointUnavailable,
    MtpTailHiddenInvalid,
    MtpCacheMissing,
    MtpKvTooShort,
    DflashCheckpointInvalid,
};

// First point where an incoming prompt stops matching a retained prefix. Captured only when a
// frontier/checkpoint match fails, so the request log can localize the divergence. The arrays hold
// a window of token ids starting at `window_begin` (centered on the divergence point); out-of-range
// entries are -1.
inline constexpr std::size_t kPrefixDivergenceWindow = 16;

struct PrefixDivergence {
    std::uint32_t frontier_index   = 0;
    std::uint32_t divergence_index = 0;
    bool tokens_matched            = true;
    std::uint32_t window_begin     = 0;
    std::array<std::int32_t, kPrefixDivergenceWindow> prompt_tokens{};
    std::array<std::int32_t, kPrefixDivergenceWindow> resident_tokens{};
};

// Why the content-addressed block walk stopped short of the prompt's eligible blocks.
enum class PrefixStallCause : std::uint8_t {
    NotRun = 0, // match not attempted (hybrid cache off, no block hashes, or prompt < 2 blocks)
    Completed,  // every eligible block matched; the miss is at the snapshot layer, not the walk
    NoChild,    // the parent has no live child with this block's chained hash
    HashCollision, // a same-hash child existed but its stored tokens or media extra differ
};

// Why a found best candidate was rejected when it made it past PrefixIndex::match.
enum class PrefixSelectionVeto : std::uint8_t {
    None = 0,       // no candidate survived (or hybrid layer produced nothing)
    SnapshotGone,   // the chosen snapshot vanished or changed between match and lookup
    FrontierInvalid,// frontier regressed to 0 or beyond prompt-1
    TailNonZero,    // only block-aligned (tail_len == 0) snapshots are consumable
};

// One snapshot seen on the matched block path and the gate that decided its fate (0 = eligible).
struct PrefixDiagCandidate {
    std::uint32_t frontier        = 0;
    std::uint32_t anchor_depth    = 0; // block index of the anchor node
    std::uint8_t  kind            = 0; // SnapshotKind code; opaque to consumers
    bool          device_resident = false;
    bool          host_restorable = false;
    // Restorable using the device, host, or disk tier (a superset of host_restorable); false only
    // when some node on the path has no copy in any tier.
    bool          disk_restorable = false;
    std::uint8_t  reject          = 0; // 1 superseded/gone, 2 frontier past prompt, 3 tail mismatch,
                                       // 4 no usable storage on the path, 5 outranked by the best
};

inline constexpr std::size_t kPrefixDiagCandidates = 8;

// Per-request explanation of the prefix-reuse decision. Filled once at plan time; carried to the
// done log alongside PrefixReusePath/PrefixReuseReason, which state WHAT happened, while this
// carries the input facts and the hybrid-cache internals that explain WHY.
struct PrefixAdmissionDiagnostics {
    // Input facts, always filled.
    std::uint32_t prompt_tokens          = 0;
    std::uint32_t eligible_blocks        = 0; // floor((n-1)/64) full blocks matchable from the root
    bool allow_prefix_reuse              = true;
    bool identity_reusable               = true;
    bool hybrid_enabled                  = false;
    bool retained_sequence               = false;
    std::uint32_t retained_frontier      = 0;
    bool mtp_backend                     = false;
    std::uint32_t mtp_kv_valid           = 0;
    bool mtp_cache_present               = false;
    bool tail_hidden_valid               = false;
    std::uint32_t snapshot_slots_total   = 0;
    std::uint32_t snapshot_slots_used    = 0;
    std::uint32_t max_new_taps           = 0;

    // M3a bounded disk tier. Occupancy is the tier's used/budget bytes when the request was planned;
    // `served_from_disk` is set when the consumed snapshot had to be materialized out of the disk
    // tier (device/host had no copy of some path node), i.e. the disk tier actually served this turn.
    bool          disk_tier_enabled      = false;
    std::uint64_t disk_tier_used_bytes   = 0;
    std::uint64_t disk_tier_budget_bytes = 0;
    bool          served_from_disk       = false;

    // Hybrid match section; zero when PrefixStallCause::NotRun.
    PrefixStallCause walk_cause          = PrefixStallCause::NotRun;
    std::uint32_t matched_blocks         = 0;
    std::uint32_t stall_sibling_count    = 0; // live children under the stall parent (0 => pristine)
    PrefixSelectionVeto selection_veto   = PrefixSelectionVeto::None;
    std::uint32_t candidates_total       = 0;
    std::array<PrefixDiagCandidate, kPrefixDiagCandidates> candidates{}; // deepest frontier first
    std::uint32_t candidates_shown   = 0; // filled entries in candidates[]

    // Outcome; reuse_base is the resumed-without-recompute token count.
    std::uint32_t reuse_base             = 0;
};

struct GenerationResult {
    PromptSummary prompt;
    std::vector<TokenId> generated_token_ids;
    std::string content;
    std::string reasoning;
    std::uint32_t reasoning_tokens     = 0;
    FinishReason finish_reason         = FinishReason::None;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::FullReset;
    PrefixReuseReason prefix_reuse_reason = PrefixReuseReason::None;
    PrefixDivergence prefix_divergence;
    PrefixAdmissionDiagnostics prefix_admission;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

struct ArenaMemorySummary {
    std::size_t capacity_bytes  = 0;
    std::size_t used_bytes      = 0;
    std::size_t peak_used_bytes = 0;
};

struct MemorySummary {
    int device                                = 0;
    std::uint32_t max_context                 = 0;
    // Rotary regime as the target runtime resolved it. `effective_max_context` is the ceiling
    // `max_context` was admitted against; `yarn_mscale` is the rope-path cos/sin multiplier the
    // corrected frequency table carries (exactly 1 under `RopeMode::Native`).
    RopeMode rope_mode                        = RopeMode::Native;
    std::uint32_t effective_max_context       = 0;
    double yarn_mscale                        = 1.0;
    KvCapacityMode kv_capacity_mode           = KvCapacityMode::Explicit;
    std::uint32_t kv_capacity                 = 0; // Resolved page-aligned Main KV capacity.
    std::uint32_t kv_capacity_page_groups     = 0;
    std::uint32_t kv_capacity_max_page_groups = 0;
    KvCacheStorage kv_cache                   = KvCacheStorage::BFloat16;
    ArenaMemorySummary weights;
    ArenaMemorySummary sequence;
    ArenaMemorySummary workspace;
    ArenaMemorySummary request_transient;
    std::size_t minimum_runtime_reservation_bytes = 0;
    std::size_t kv_capacity_increment_bytes       = 0;
    std::size_t runtime_reservation_bytes         = 0;
    std::size_t available_after_weights_bytes     = 0;
    std::size_t available_after_startup_bytes     = 0;
    std::size_t kv_capacity_headroom_bytes        = 0;
    std::size_t planned_slack_bytes               = 0;
    std::size_t workspace_logical_peak_bytes      = 0;
    std::size_t cuda_graph_allowance_bytes        = 0;
    // Graph residency measured on the primary rank, and per rank in `cuda_graph_device_bytes`. The
    // allowance above is a per-device budget, so every figure is per-device and none is a total to
    // be divided across ranks.
    std::size_t cuda_graph_observed_bytes         = 0;
    // Graph residency measured on the first peer rank (rank 1), and 0 at tp == 1. Retained for the
    // serving schema; the per-rank figures are in `cuda_graph_device_bytes`. Each rank captures its
    // OWN graph (the NCCL collectives are recorded as nodes in that rank's graph), and every rank
    // materializes the same per-class driver and NCCL module state against the same allowance.
    std::size_t cuda_graph_peer_observed_bytes    = 0;
    // Per-rank graph residency, indexed by rank; entries at or beyond `tp` are 0. This is the
    // authoritative per-device figure the load-summary table reports.
    std::array<std::size_t, 4> cuda_graph_device_bytes{};
    // Node count of one captured decode graph, 0 when graphs are disabled. Each rank captures its
    // own graph holding that rank's schedule plus its NCCL collective nodes, so this is the tp 1
    // node count plus the collective nodes, not a sum over ranks.
    std::size_t cuda_graph_node_count             = 0;
    std::size_t kv_payload_bytes                  = 0;
    // GDN (linear-attention) recurrent + convolution state for THIS device. At tp > 1 the head split
    // divides it by the tensor-parallel width, so it is reported separately from the rest of the
    // sequence arena.
    std::size_t gdn_state_bytes = 0;
};

// One row of the per-device memory table the load summary prints. At tp == 1 only `devices[0]` is
// meaningful. Every byte count is measured or planned for that device specifically -- nothing here
// is a model total divided by the tensor-parallel width.
struct DeviceMemoryReport {
    int device                              = -1;
    std::uint64_t weights_bytes             = 0; // this device's weight arena (its own shard)
    std::uint64_t kv_pool_bytes             = 0; // paged KV payload
    std::uint64_t gdn_state_bytes           = 0; // GDN recurrent + conv state
    std::uint64_t sequence_bytes            = 0; // whole persistent arena (KV + GDN + round state)
    std::uint64_t workspace_bytes           = 0; // workspace arena capacity
    std::uint64_t cuda_graph_bytes          = 0; // measured graph residency on THIS device
    std::uint64_t reserved_bytes            = 0; // weights + runtime reservation
    std::uint64_t free_after_startup_bytes  = 0;
    std::uint64_t total_bytes               = 0;
};

// Monotonic execution counters plus one boundary-consistent scheduler snapshot. Consumers derive
// interval throughput by subtracting two snapshots and dividing by their own monotonic wall time.
struct RuntimeStats {
    // Actual prompt tokens evaluated by prefill; resident prefix hits are excluded.
    std::uint64_t computed_prefill_tokens = 0;
    // Tokens committed by decode rounds; the first token emitted by prefill is excluded.
    std::uint64_t committed_decode_tokens = 0;
    // Decode batch executions and the sum of their batch sizes.
    std::uint64_t decode_rounds         = 0;
    std::uint64_t decode_row_rounds     = 0;
    std::uint32_t running_requests      = 0;
    std::uint32_t prefilling_requests   = 0;
    std::uint32_t decode_ready_requests = 0;
    std::uint32_t waiting_requests      = 0;
};

// Current memory on one tensor-parallel rank's device, from cudaMemGetInfo. `used_bytes` is
// `total_bytes - free_bytes`.
struct DeviceMemoryMetrics {
    int device                = -1;
    std::uint64_t total_bytes = 0;
    std::uint64_t free_bytes  = 0;
    std::uint64_t used_bytes  = 0;
};

// Occupancy of one paged KV pool. `capacity_pages` is the physical page-group count; `mapped_pages`
// belong to live sequences, `cached_pages` to the prefix cache, `free_pages` to neither. `page_bytes`
// is the byte size of one physical page across all storage planes.
struct KvPoolMetrics {
    std::uint32_t capacity_pages = 0;
    std::uint32_t mapped_pages   = 0;
    std::uint32_t cached_pages   = 0;
    std::uint32_t free_pages     = 0;
    std::uint64_t page_bytes     = 0;
};

// Pinned host-RAM KV tier. `pinned_bytes` is the allocation the tier took; `used_bytes` of it is
// occupied by block and snapshot slab data. `enabled` is false when the tier is off or its
// allocation failed.
struct HostCacheMetrics {
    bool enabled                      = false;
    std::uint64_t pinned_bytes        = 0;
    std::uint64_t used_bytes          = 0;
    std::uint32_t block_slab_count    = 0;
    std::uint32_t snapshot_slab_count = 0;
};

// Bounded disk KV tier (M3a). `budget_bytes` is the configured slab-file size (`--disk-cache-mib`);
// `used_bytes` of it holds live block and snapshot slots. `enabled` is false when the tier is off or
// its slab file could not be opened.
struct DiskCacheMetrics {
    bool enabled                      = false;
    std::uint64_t budget_bytes        = 0;
    std::uint64_t used_bytes          = 0;
    std::uint32_t block_slab_count    = 0;
    std::uint32_t snapshot_slab_count = 0;
};

// Hybrid prefix cache residency. Device snapshot slots are the on-device state images; the host
// tier carries demoted blocks and state images. The cumulative counters record lifetime eviction
// behavior on rank 0 so "the pool had room" stops being the only explanation available.
struct PrefixCacheMetrics {
    bool enabled                             = false;
    std::uint32_t device_snapshot_slots      = 0;
    std::uint32_t device_snapshot_slots_used = 0;
    HostCacheMetrics host;
    DiskCacheMetrics disk;
    // Cumulative, process-lifetime, rank 0.
    std::uint64_t device_blocks_demoted_total   = 0; // lossless device -> host demotions
    std::uint64_t subtree_erased_total          = 0; // lossy subtree erasures (devices w/o host copy)
    std::uint64_t branch_gc_removed_blocks_total = 0; // dead-branch nodes removed by demand-aware GC
    std::uint64_t snapshots_deleted_by_eviction_total = 0;
    std::uint64_t retained_lane_evictions_total = 0;
};

// Live resource metrics for the serving /metrics endpoint: a boundary-consistent snapshot. This is
// not a performance ledger; throughput is derived by consumers from RuntimeStats as usual.
struct MetricsSnapshot {
    RuntimeStats scheduler;
    std::vector<DeviceMemoryMetrics> devices;
    KvPoolMetrics main_kv;    // text KV pool (always present)
    KvPoolMetrics backend_kv; // MTP/DFlash KV pool; zero when the backend has none
    PrefixCacheMetrics prefix_cache;
};

struct LoadSummary {
    int tp = 1;
    std::array<DeviceMemoryReport, 4> devices{};
    // Rotary regime resolved at construction. `effective_max_context` is the ceiling `max_context`
    // was validated against (the variant's native capacity under `RopeMode::Native`,
    // `yarn_origin * yarn_factor` under `RopeMode::Yarn`); `yarn_mscale` is the rotary cos/sin
    // multiplier the YaRN table carries and is exactly 1 under Native.
    RopeMode rope_mode                    = RopeMode::Native;
    double yarn_factor                    = 0.0;
    std::uint32_t yarn_origin             = 0;
    std::uint32_t effective_max_context   = 0;
    double yarn_mscale                    = 1.0;
    std::string target;
    std::string model_id;
    std::string weights_id;
    double load_seconds                = 0.0;
    double upload_seconds              = 0.0;
    std::uint64_t artifact_bytes_read  = 0;
    std::uint64_t host_to_device_bytes = 0;
    std::uint64_t peak_staging_bytes   = 0;
    std::size_t tensor_count           = 0;
    std::size_t resource_count         = 0;
};

} // namespace ninfer
