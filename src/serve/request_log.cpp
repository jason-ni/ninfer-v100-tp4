#include "serve/request_log.h"
#include "product/speculative_options.h"
#include "serve/console_log.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <unistd.h>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

// Must match prefix_cache's kBlockTokens (the content-addressed block size).
inline constexpr std::uint32_t kDiagBlockTokens = 64U;

std::uint64_t unix_time_ms() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

std::string new_server_instance_id() {
    const auto now    = std::chrono::system_clock::now().time_since_epoch();
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
    return "serve-" + std::to_string(static_cast<long long>(::getpid())) + '-' +
           std::to_string(micros);
}

std::filesystem::path normalized_absolute_path(const std::string& value) {
    std::error_code error;
    std::filesystem::path path = std::filesystem::weakly_canonical(value, error);
    if (!error) { return path; }
    error.clear();
    path = std::filesystem::absolute(value, error);
    return error ? std::filesystem::path(value).lexically_normal() : path.lexically_normal();
}

std::string cuda_version_string(int version) {
    if (version <= 0) { return {}; }
    return std::to_string(version / 1000) + '.' + std::to_string((version % 1000) / 10);
}

std::string cuda_uuid_string(const cudaUUID_t& uuid) {
    std::ostringstream out;
    out << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) { out << '-'; }
        out << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(uuid.bytes[i]));
    }
    return out.str();
}

const char* finish_reason_name(ninfer::FinishReason reason) {
    switch (reason) {
    case ninfer::FinishReason::None:
        return "none";
    case ninfer::FinishReason::OutputLimit:
        return "output_limit";
    case ninfer::FinishReason::ContextCapacity:
        return "context_capacity";
    case ninfer::FinishReason::StopToken:
        return "stop_token";
    case ninfer::FinishReason::StopString:
        return "stop_string";
    case ninfer::FinishReason::Cancelled:
        return "cancelled";
    }
    return "unknown";
}

std::string tool_choice_name(const ToolChoice& choice) {
    switch (choice.mode) {
    case ToolChoiceMode::Auto:
        return "auto";
    case ToolChoiceMode::None:
        return "none";
    case ToolChoiceMode::Required:
        return "required";
    case ToolChoiceMode::Named:
        return choice.name.empty() ? "named" : choice.name;
    }
    return "unknown";
}

const char* kv_cache_name(ninfer::KvCacheStorage storage) {
    switch (storage) {
    case ninfer::KvCacheStorage::BFloat16: return "bf16";
    case ninfer::KvCacheStorage::Int8Group64: return "int8-group64";
    case ninfer::KvCacheStorage::Int4NLGroup32: return "iq4nl-group32";
    }
    return "unknown";
}

const char* kv_capacity_mode_name(ninfer::KvCapacityMode mode) {
    return mode == ninfer::KvCapacityMode::Automatic ? "auto" : "explicit";
}

const char* proposal_head_name(ninfer::ProposalHead proposal) {
    return proposal == ninfer::ProposalHead::Optimized ? "optimized" : "full";
}

const char* prefix_reuse_path_name(ninfer::PrefixReusePath path) {
    switch (path) {
    case ninfer::PrefixReusePath::FullReset:
        return "full_reset";
    case ninfer::PrefixReusePath::AppendAtFrontier:
        return "append_frontier";
    case ninfer::PrefixReusePath::RestoreTurnCheckpoint:
        return "restore_turn_checkpoint";
    case ninfer::PrefixReusePath::RestoreResponseCheckpoint:
        return "restore_response_checkpoint";
    case ninfer::PrefixReusePath::BlockSnapshot:
        return "block_snapshot";
    }
    return "unknown";
}

const char* prefix_reuse_reason_name(ninfer::PrefixReuseReason reason) {
    switch (reason) {
    case ninfer::PrefixReuseReason::None:
        return "none";
    case ninfer::PrefixReuseReason::ReusedFrontier:
        return "reused_frontier";
    case ninfer::PrefixReuseReason::ReusedTurnCheckpoint:
        return "reused_turn_checkpoint";
    case ninfer::PrefixReuseReason::ReusedResponseCheckpoint:
        return "reused_response_checkpoint";
    case ninfer::PrefixReuseReason::ReusedBlockSnapshot:
        return "reused_block_snapshot";
    case ninfer::PrefixReuseReason::ReuseDisabled:
        return "reuse_disabled";
    case ninfer::PrefixReuseReason::PromptNotReusable:
        return "prompt_not_reusable";
    case ninfer::PrefixReuseReason::NoRetainedSequence:
        return "no_retained_sequence";
    case ninfer::PrefixReuseReason::FrontierEmpty:
        return "frontier_empty";
    case ninfer::PrefixReuseReason::DflashFrontierMismatch:
        return "dflash_frontier_mismatch";
    case ninfer::PrefixReuseReason::FrontierPrefixMismatch:
        return "frontier_prefix_mismatch";
    case ninfer::PrefixReuseReason::CheckpointUnavailable:
        return "checkpoint_unavailable";
    case ninfer::PrefixReuseReason::MtpTailHiddenInvalid:
        return "mtp_tail_hidden_invalid";
    case ninfer::PrefixReuseReason::MtpCacheMissing:
        return "mtp_cache_missing";
    case ninfer::PrefixReuseReason::MtpKvTooShort:
        return "mtp_kv_too_short";
    case ninfer::PrefixReuseReason::DflashCheckpointInvalid:
        return "dflash_checkpoint_invalid";
    }
    return "unknown";
}

std::string prefix_divergence_str(const ninfer::PrefixDivergence& divergence) {
    if (divergence.frontier_index == 0) { return {}; }
    std::ostringstream out;
    out << " divergence=@" << divergence.divergence_index << "/" << divergence.frontier_index
        << (divergence.tokens_matched ? "(identity)" : "(token)") << " window_from="
        << divergence.window_begin << " prompt=[";
    for (std::size_t k = 0; k < ninfer::kPrefixDivergenceWindow; ++k) {
        out << (k == 0 ? "" : ",") << divergence.prompt_tokens[k];
    }
    out << "] resident=[";
    for (std::size_t k = 0; k < ninfer::kPrefixDivergenceWindow; ++k) {
        out << (k == 0 ? "" : ",") << divergence.resident_tokens[k];
    }
    out << "]";
    return out.str();
}

const char* stall_cause_name(ninfer::PrefixStallCause cause) {
    switch (cause) {
        case ninfer::PrefixStallCause::NotRun: return "not_run";
        case ninfer::PrefixStallCause::Completed: return "completed";
        case ninfer::PrefixStallCause::NoChild: return "no_child";
        case ninfer::PrefixStallCause::HashCollision: return "hash_collision";
    }
    return "unknown";
}

const char* selection_veto_name(ninfer::PrefixSelectionVeto veto) {
    switch (veto) {
        case ninfer::PrefixSelectionVeto::None: return "none";
        case ninfer::PrefixSelectionVeto::SnapshotGone: return "snapshot_gone";
        case ninfer::PrefixSelectionVeto::FrontierInvalid: return "frontier_invalid";
        case ninfer::PrefixSelectionVeto::TailNonZero: return "tail_nonzero";
    }
    return "unknown";
}

const char* candidate_reject_code_name(std::uint8_t reject) {
    switch (reject) {
        case 0: return "eligible";
        case 1: return "superseded_or_gone";
        case 2: return "frontier_past_prompt";
        case 3: return "tail_mismatch";
        case 4: return "no_usable_storage";
        case 5: return "outranked";
    }
    return "unknown";
}

const char* snapshot_kind_name(std::uint8_t kind) {
    switch (kind) {
        case 0: return "tap";
        case 1: return "endpoint";
        case 2: return "output_boundary";
        case 3: return "boundary";
    }
    return "unknown";
}



Json event_base(const std::string& server_instance_id, std::uint64_t timestamp, const char* event) {
    return Json{{"artifact_type", kRequestLogArtifactType},
                {"schema_version", kRequestLogSchemaVersion},
                {"event", event},
                {"timestamp_unix_ms", timestamp},
                {"server_instance_id", server_instance_id}};
}

Json sampler_json(const ninfer::ResolvedSamplingParameters& sampling) {
    return Json{{"temperature", sampling.temperature},
                {"top_p", sampling.top_p},
                {"top_k", sampling.top_k},
                {"min_p", sampling.min_p},
                {"presence_penalty", sampling.presence_penalty},
                {"frequency_penalty", sampling.frequency_penalty},
                {"seed", sampling.seed}};
}

Json preset_json(const ninfer::SamplingPreset& preset) {
    return Json{{"temperature", preset.temperature},
                {"top_p", preset.top_p},
                {"top_k", preset.top_k},
                {"min_p", preset.min_p},
                {"presence_penalty", preset.presence_penalty},
                {"frequency_penalty", preset.frequency_penalty}};
}

Json overrides_json(const ninfer::SamplingOverrides& overrides) {
    Json result{{"temperature", nullptr},
                {"top_p", nullptr},
                {"top_k", nullptr},
                {"min_p", nullptr},
                {"presence_penalty", nullptr},
                {"frequency_penalty", nullptr},
                {"seed", nullptr}};
    if (overrides.temperature) { result["temperature"] = *overrides.temperature; }
    if (overrides.top_p) { result["top_p"] = *overrides.top_p; }
    if (overrides.top_k) { result["top_k"] = *overrides.top_k; }
    if (overrides.min_p) { result["min_p"] = *overrides.min_p; }
    if (overrides.presence_penalty) { result["presence_penalty"] = *overrides.presence_penalty; }
    if (overrides.frequency_penalty) { result["frequency_penalty"] = *overrides.frequency_penalty; }
    if (overrides.seed) { result["seed"] = *overrides.seed; }
    return result;
}

Json request_json(const RequestLogContext& context) {
    return Json{{"request_id", context.id},
                {"protocol", context.protocol},
                {"model", context.model},
                {"stream", context.stream},
                {"message_count", context.message_count},
                {"media_item_count", context.media_item_count},
                {"requested_output_tokens", context.requested_output_tokens},
                {"requested_output_tokens_source",
                 context.requested_output_tokens_client_set ? "client" : "server_default"},
                {"tool_count", context.tool_count},
                {"tool_choice", tool_choice_name(context.tool_choice)},
                {"has_tool_history", context.has_tool_history},
                {"enable_thinking", context.enable_thinking},
                {"preserve_thinking", context.preserve_thinking},
                {"preserve_thinking_semantic_change", context.preserve_thinking_semantic_change},
                {"sampling", sampler_json(context.sampling)}};
}

Json preparation_json(const RequestLogContext& context) {
    const PromptPreparationStats& stats = context.preparation;
    return Json{{"total", stats.seconds},
                {"acquisition", context.acquisition_seconds},
                {"media_preprocess", stats.media_preprocess_seconds},
                {"media_preprocess_work", stats.media_preprocess_work_seconds},
                {"tokenize", stats.tokenize_seconds},
                {"media_items", stats.media_items},
                {"media_bytes", stats.media_bytes},
                {"raw_patches", stats.raw_patches},
                {"vision_tokens", stats.vision_tokens},
                {"patch_bytes", stats.patch_bytes},
                {"cache_hits", stats.media_cache_hits},
                {"cache_misses", stats.media_cache_misses},
                {"singleflight_waits", stats.media_singleflight_waits},
                {"built_patch_bytes", stats.built_patch_bytes},
                {"reused_patch_bytes", stats.reused_patch_bytes}};
}

Json rejected_request_json(const RequestRejectionLogContext& context) {
    return Json{{"request_id", context.id},
                {"protocol", context.protocol},
                {"model", context.model},
                {"stream", context.stream},
                {"message_count", context.message_count},
                {"media_item_count", context.media_item_count},
                {"requested_output_tokens", context.requested_output_tokens},
                {"requested_output_tokens_source",
                 context.requested_output_tokens_client_set ? "client" : "server_default"},
                {"tool_count", context.tool_count},
                {"tool_choice", tool_choice_name(context.tool_choice)},
                {"has_tool_history", context.has_tool_history}};
}

Json error_json(const ApiError& error) {
    Json code  = error.code.empty() ? Json(nullptr) : Json(error.code);
    Json param = error.param.empty() ? Json(nullptr) : Json(error.param);
    return Json{{"status", error.status},
                {"type", error.type},
                {"code", std::move(code)},
                {"param", std::move(param)},
                {"message", error.message}};
}

Json arena_json(const ninfer::ArenaMemorySummary& arena) {
    return Json{{"capacity_bytes", arena.capacity_bytes},
                {"used_bytes", arena.used_bytes},
                {"peak_used_bytes", arena.peak_used_bytes}};
}

Json speculative_json(const GenerationMetrics& metrics) {
    return Json{{"backend", product::speculative_backend_name(metrics.speculative_backend)},
                {"draft_window", metrics.speculative_draft_window},
                {"rounds", metrics.speculative_rounds},
                {"drafted_tokens", metrics.speculative_draft_tokens},
                {"accepted_tokens", metrics.speculative_accepted_tokens},
                {"fallback_steps", metrics.speculative_fallback_steps},
                {"accepted_per_position", metrics.speculative_accepted_per_position}};
}

// Tokens/second with fixed precision, or "n/a" when the interval is degenerate.
std::string rate(double tokens, double seconds) {
    std::ostringstream out;
    if (seconds > 0.0 && tokens > 0.0) {
        out << std::fixed << std::setprecision(1) << (tokens / seconds) << "tok/s";
    } else {
        out << "n/a";
    }
    return out.str();
}

std::string seconds_str(double seconds) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << seconds << 's';
    return out.str();
}

// Compact resolved-sampler summary. temperature <= 0 is the exact-argmax path.
std::string sampler_str(const ninfer::ResolvedSamplingParameters& sampling) {
    if (sampling.temperature <= 0.0f) { return "greedy"; }
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << "temp=" << sampling.temperature
        << " top_p=" << sampling.top_p << " top_k=" << sampling.top_k;
    if (sampling.min_p > 0.0f) { out << " min_p=" << sampling.min_p; }
    if (sampling.presence_penalty != 0.0f) { out << " pres=" << sampling.presence_penalty; }
    if (sampling.frequency_penalty != 0.0f) { out << " freq=" << sampling.frequency_penalty; }
    out << " seed=" << sampling.seed;
    return out.str();
}

std::string speculative_str(const GenerationMetrics& metrics) {
    if (metrics.speculative_backend == SpeculativeBackend::None) { return "off"; }
    std::ostringstream out;
    out << product::speculative_backend_name(metrics.speculative_backend) << ' ' << std::fixed
        << std::setprecision(2);
    if (metrics.speculative_rounds > 0) {
        const double per_round = 1.0 + static_cast<double>(metrics.speculative_accepted_tokens) /
                                           static_cast<double>(metrics.speculative_rounds);
        out << per_round << "tok/round";
    } else {
        out << "n/a";
    }
    if (metrics.speculative_draft_tokens > 0) {
        const double accept_pct = 100.0 * static_cast<double>(metrics.speculative_accepted_tokens) /
                                  static_cast<double>(metrics.speculative_draft_tokens);
        out << " (" << std::setprecision(1) << accept_pct << "%)";
    }
    return out.str();
}
} // namespace

std::string format_prefix_admission_compact(const ninfer::PrefixAdmissionDiagnostics& diag,
                                            ninfer::PrefixReusePath verdict_path,
                                            ninfer::PrefixReuseReason verdict_reason) {
    // Silent when nothing was recomputed; the explanation exists to explain recompute.
    if (diag.reuse_base >= diag.prompt_tokens) { return {}; }
    std::ostringstream out;
    out << " prefix_admission=";
    if (!diag.allow_prefix_reuse || !diag.identity_reusable) {
        out << "gates(allow=" << (diag.allow_prefix_reuse ? 1 : 0) << ",reusable="
            << (diag.identity_reusable ? 1 : 0) << ")";
    } else if (!diag.hybrid_enabled) {
        out << "hybrid_off path=" << prefix_reuse_path_name(verdict_path);
        if (diag.retained_sequence) {
            out << " retained_frontier=F" << diag.retained_frontier;
        } else {
            out << " reason=" << prefix_reuse_reason_name(verdict_reason);
        }
    } else if (diag.walk_cause != ninfer::PrefixStallCause::Completed) {
        out << "walk_stalled@B" << diag.matched_blocks << "/B" << diag.eligible_blocks
            << " cause=" << stall_cause_name(diag.walk_cause);
        if (diag.stall_sibling_count > 0U) {
            out << " siblings=" << diag.stall_sibling_count;
        }
        if (diag.walk_cause == ninfer::PrefixStallCause::HashCollision) {
            out << " (tokens or media extra differ)";
        }
    } else if (diag.selection_veto != ninfer::PrefixSelectionVeto::None) {
        out << "walk_completed veto=" << selection_veto_name(diag.selection_veto);
    } else if (verdict_path == ninfer::PrefixReusePath::BlockSnapshot) {
        out << "block_reused B" << diag.reuse_base / kDiagBlockTokens << "/B"
            << diag.eligible_blocks << " slots=" << diag.snapshot_slots_used << '/'
            << diag.snapshot_slots_total;
    } else {
        out << "miss path=" << prefix_reuse_path_name(verdict_path)
            << " reason=" << prefix_reuse_reason_name(verdict_reason);
        if (diag.mtp_backend) { out << " mtp(downgraded)"; }
    }
    return out.str();
}

std::string format_prefix_admission_verbose(std::uint64_t request_id,
                                            const ninfer::PrefixAdmissionDiagnostics& diag,
                                            ninfer::PrefixReusePath verdict_path,
                                            ninfer::PrefixReuseReason verdict_reason) {
    if (!diag.hybrid_enabled) { return {}; }
    const auto gi = [](std::uint32_t n) {
        // Rough estimate; assumes an INT8 group-64 KV entry of ~16 bytes per token.
        std::ostringstream o;
        o << n << " tok [~" << std::fixed << std::setprecision(2)
          << static_cast<double>(n) / 16.0 / 1048576.0 << " GiB]";
        return o.str();
    };
    std::ostringstream v;
    v << "[req " << request_id << "] prefix_recompute="
      << (diag.prompt_tokens - diag.reuse_base) << " tok explanation:";
v << "\n  [inputs] prompt=" << diag.prompt_tokens << " eligible_blocks=" << diag.eligible_blocks;
    if (diag.eligible_blocks > 0) {
        v << " (block=" << kDiagBlockTokens << " toks, trailing partial block excluded)";
    }
    v << " allow_reuse=" << (diag.allow_prefix_reuse ? "yes" : "NO")
      << " identity_reusable=" << (diag.identity_reusable ? "yes" : "NO");
    if (diag.retained_sequence) {
        v << " retained_seq=YES frontier(F)=" << diag.retained_frontier;
        if (diag.mtp_backend) {
            const std::uint32_t mtp_valid = diag.mtp_kv_valid >= 1U ? diag.mtp_kv_valid - 1U : 0U;
            v << " mtp_cache(F-1)valid=" << mtp_valid;
        } else {
            v << " mtp_cache=no";
        }
    }
    if (diag.mtp_backend) {
        v << " mtp_downgrade_gate(kv_valid=" << diag.mtp_kv_valid << (diag.mtp_cache_present ? "" : "(no cache)")
          << ",tail_hidden_valid=" << (diag.tail_hidden_valid ? "yes" : "no") << ")";
    }
    v << "\n  [hybrid walk] eligible=" << diag.eligible_blocks
      << " matched_blocks(B)=" << diag.matched_blocks << "/" << diag.eligible_blocks
      << " => " << gi(diag.matched_blocks * kDiagBlockTokens);
    switch (diag.walk_cause) {
        case ninfer::PrefixStallCause::NotRun:
            v << " [skipped] "
              << (diag.eligible_blocks < 2
                     ? "prompt < 2 blocks"
                     : diag.matched_blocks == 0U ? "hash-chain start failed"
                                                 : "");
            break;
        case ninfer::PrefixStallCause::Completed: v << " [OK] every eligible block matched the index."; break;
        case ninfer::PrefixStallCause::NoChild:
            v << " [STALLED at block " << diag.matched_blocks + 1U
              << "] parent of block #" << diag.matched_blocks << ' ';
            if (diag.stall_sibling_count == 0U) {
                v << "has no live children: this exact block hash was never committed here "
                  "(first turn in this thread, or a prior history was evicted).";
            } else {
                v << "has "
                  << (diag.stall_sibling_count + 1U)
                  << " other child blocks: an earlier request diverged at exactly this position, "
                    "so the cache continues along that chain instead.";
            }
            break;
        case ninfer::PrefixStallCause::HashCollision:
            v << " [STALLED at block " << diag.matched_blocks + 1U << "] a same-hash child exists "
              "but its stored 64 tokens or media extra differ from this prompt's block: an "
                                   "earlier turn used different content at this position.";
            break;
    }
    v << "\n  [snapshots on matched path] candidates_seen=" << diag.candidates_total
      << ", showing deepest " << diag.candidates_shown << ":";
    for (std::uint32_t i = 0; i < diag.candidates_shown; ++i) {
        const auto& c = diag.candidates[i];
        const std::uint32_t begin = c.anchor_depth * kDiagBlockTokens;
        v << "\n    B" << begin / kDiagBlockTokens << " (" << begin << ".."
          << c.frontier + 1U << ") kind=" << snapshot_kind_name(c.kind);
        v << (c.device_resident ? " device=yes" : " device=no");
        v << (c.host_restorable ? " host=yes" : " host=no");
        v << " -> " << candidate_reject_code_name(c.reject);
    }
    if (diag.candidates_total > diag.candidates_shown) {
        v << "\n    ... and " << diag.candidates_total - diag.candidates_shown
          << " shallower ones not shown";
    }
    v << (diag.selection_veto != ninfer::PrefixSelectionVeto::None
             ? " selection_veto=" : " selection_veto=none")
      << selection_veto_name(diag.selection_veto);
    v << "\n  [slots] device_snapshot_slots_used=" << diag.snapshot_slots_used << '/'
      << diag.snapshot_slots_total << " max_new_taps_per_req=" << diag.max_new_taps;
    v << "\n  [verdict] path=" << prefix_reuse_path_name(verdict_path)
      << " reason=" << prefix_reuse_reason_name(verdict_reason)
      << " reuse_base=" << diag.reuse_base << " => recomputed "
      << (diag.prompt_tokens - diag.reuse_base) << ' ';
    if (verdict_path == ninfer::PrefixReusePath::FullReset) {
        if (!diag.allow_prefix_reuse || !diag.identity_reusable) {
            v << "tokens because the prompt is not reusable for the engine (gate above).";
        } else if (diag.walk_cause != ninfer::PrefixStallCause::Completed) {
            v << "tokens because the content-addressed walk stalled: no matching chain past "
               "the stall point exists in the cache.";
        } else if (diag.selection_veto != ninfer::PrefixSelectionVeto::None) {
            v << "tokens because the best candidate on the chain was rejected: "
               << selection_veto_name(diag.selection_veto) << '.';
        } else {
            v << "tokens because no consumable snapshot survived on the matched chain.";
        }
    }
    return v.str();
}

RequestLogContext make_request_log_context(std::uint64_t id, std::string protocol,
                                           const GenerationRequest& request,
                                           const PreparedRequest& prepared) {
    RequestLogContext context;
    context.id                                 = id;
    context.protocol                           = std::move(protocol);
    context.model                              = request.model;
    context.stream                             = request.stream;
    context.message_count                      = request.messages.size();
    context.media_item_count                   = request.media_item_count();
    context.requested_output_tokens            = request.max_tokens;
    context.requested_output_tokens_client_set = request.max_tokens_set;
    context.tool_count                         = request.tools.size();
    context.tool_choice                        = request.tool_choice;
    context.has_tool_history                   = request.has_tool_history();
    context.enable_thinking                    = prepared.enable_thinking;
    context.preserve_thinking                  = prepared.preserve_thinking;
    context.preserve_thinking_semantic_change  = prepared.preserve_thinking_semantic_change;
    context.sampling                           = prepared.sampling;
    context.acquisition_seconds                = prepared.acquisition_seconds;
    context.preparation                        = prepared.preparation;
    return context;
}

RequestRejectionLogContext make_request_rejection_log_context(std::uint64_t id,
                                                              std::string protocol,
                                                              const GenerationRequest& request,
                                                              ApiError error) {
    RequestRejectionLogContext context;
    context.id                                 = id;
    context.protocol                           = std::move(protocol);
    context.model                              = request.model;
    context.stream                             = request.stream;
    context.message_count                      = request.messages.size();
    context.media_item_count                   = request.media_item_count();
    context.requested_output_tokens            = request.max_tokens;
    context.requested_output_tokens_client_set = request.max_tokens_set;
    context.tool_count                         = request.tools.size();
    context.tool_choice                        = request.tool_choice;
    context.has_tool_history                   = request.has_tool_history();
    context.error                              = std::move(error);
    return context;
}

std::string format_request_start(const RequestLogContext& context) {
    std::ostringstream out;
    out << "[req " << context.id << "] " << context.protocol << ' '
        << (context.stream ? "stream" : "non-stream") << " msgs=" << context.message_count
        << " max_tokens=" << context.requested_output_tokens << ' '
        << (context.requested_output_tokens_client_set ? "(client)" : "(server default)")
        << " tools=" << context.tool_count
        << " tool_choice=" << tool_choice_name(context.tool_choice)
        << " tool_history=" << (context.has_tool_history ? "yes" : "no")
        << " thinking=" << (context.enable_thinking ? "on" : "off")
        << " preserve_thinking=" << (context.preserve_thinking ? "on" : "off")
        << " preserve_change=" << (context.preserve_thinking_semantic_change ? "yes" : "no")
        << " sampler=[" << sampler_str(context.sampling) << ']';
    if (context.media_item_count != 0) {
        out << " prepare=" << seconds_str(context.preparation.seconds)
            << " acquire=" << seconds_str(context.acquisition_seconds)
            << " media=" << seconds_str(context.preparation.media_preprocess_seconds) << '/'
            << seconds_str(context.preparation.media_preprocess_work_seconds)
            << " tokenize=" << seconds_str(context.preparation.tokenize_seconds)
            << " media_cache=" << context.preparation.media_cache_hits << '/'
            << context.preparation.media_cache_misses << '/'
            << context.preparation.media_singleflight_waits;
    }
    out << " \xE2\x86\x92 submitted";
    return out.str();
}

std::string format_request_rejected(const RequestRejectionLogContext& context) {
    std::ostringstream out;
    out << "[req " << context.id << "] rejected phase=prepare protocol=" << context.protocol << ' '
        << (context.stream ? "stream" : "non-stream") << " msgs=" << context.message_count
        << " media=" << context.media_item_count << " tools=" << context.tool_count
        << " status=" << context.error.status;
    if (!context.error.code.empty()) { out << " code=" << context.error.code; }
    out << " message=" << context.error.message;
    return out.str();
}

std::string format_request_done(const RequestLogContext& context,
                                const GenerationOutcome& outcome) {
    const GenerationMetrics& metrics = outcome.metrics;
    const double ttft_ms             = metrics.ttft_seconds * 1000.0;
    // Prefill emits the first token; the remaining (gen - 1) come from decode.
    const double decode_tokens =
        outcome.completion_tokens > 0 ? static_cast<double>(outcome.completion_tokens - 1) : 0.0;
    const double computed_prefill_tokens = static_cast<double>(
        std::max(0, outcome.prompt_tokens - static_cast<int>(metrics.prefix_cache_hit_tokens)));

    std::ostringstream out;
    out << "[req " << context.id << "] done finish="
        << (outcome.tool_calls.empty() ? finish_reason_name(outcome.finish_reason) : "tool_calls");
    if (!outcome.tool_calls.empty()) { out << " tool_calls=" << outcome.tool_calls.size(); }
    out << " prompt=" << outcome.prompt_tokens << " gen=" << outcome.completion_tokens
        << " cache=" << metrics.prefix_cache_hit_tokens
        << " reuse=" << prefix_reuse_path_name(metrics.prefix_reuse_path)
        << " reuse_reason=" << prefix_reuse_reason_name(metrics.prefix_reuse_reason)
        << prefix_divergence_str(metrics.prefix_divergence)
        << format_prefix_admission_compact(metrics.prefix_admission, metrics.prefix_reuse_path,
                                           metrics.prefix_reuse_reason)
        << " ttft=" << std::fixed
        << std::setprecision(0) << ttft_ms << "ms"
        << " prefill=" << rate(computed_prefill_tokens, metrics.prefill_seconds)
        << " decode=" << rate(decode_tokens, metrics.decode_seconds)
        << " wall=" << seconds_str(metrics.total_seconds)
        << " speculative=" << speculative_str(metrics);
    return out.str();
}

std::string format_request_error(const RequestLogContext& context, const std::string& message) {
    std::ostringstream out;
    out << "[req " << context.id << "] error " << message;
    return out.str();
}

std::string format_throughput(const ThroughputReport& report) {
    const double prefill_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.computed_prefill_tokens) / report.interval_seconds
            : 0.0;
    const double decode_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.committed_decode_tokens) / report.interval_seconds
            : 0.0;
    std::ostringstream out;
    out << "throughput interval=" << std::fixed << std::setprecision(3) << report.interval_seconds
        << "s prefill=" << std::setprecision(1) << prefill_rate << "tok/s decode=" << decode_rate
        << "tok/s running=" << report.scheduler.running_requests
        << " prefilling=" << report.scheduler.prefilling_requests
        << " decode_ready=" << report.scheduler.decode_ready_requests
        << " waiting=" << report.scheduler.waiting_requests << " avg_decode_batch=";
    if (report.decode_rounds == 0) {
        out << "n/a";
    } else {
        out << std::setprecision(2)
            << static_cast<double>(report.decode_row_rounds) /
                   static_cast<double>(report.decode_rounds);
    }
    return out.str();
}

std::string format_server_start_json(
    const std::string& server_instance_id, std::uint64_t timestamp, const ServeOptions& options,
    const ninfer::ModelSamplingDefaults& sampling_defaults, const std::string& public_model_id,
    const ninfer::LoadSummary& load, const ninfer::MemorySummary& memory,
    const ServerLogEnvironment& environment, std::optional<std::uint64_t> artifact_size_bytes) {
    Json record = event_base(server_instance_id, timestamp, "server_start");

    Json artifact_size = nullptr;
    if (artifact_size_bytes.has_value()) { artifact_size = *artifact_size_bytes; }

    record["server"]   = Json{{"host", options.host},
                              {"port", options.port},
                              {"public_model_id", public_model_id},
                              {"api_key_configured", !options.api_key.empty()},
                              {"cors_enabled", options.enable_cors},
                              {"max_request_bytes", options.max_request_bytes},
                              {"media_cache_bytes", options.media_cache_bytes},
                              {"media_live_bytes", options.media_live_bytes},
                              {"media_preprocess_threads", options.media_preprocess_threads},
                              {"request_log_jsonl", options.request_log_jsonl},
                              {"default_output_tokens", options.default_max_tokens},
                              {"default_thinking", options.enable_thinking},
                              {"default_preserve_thinking", options.preserve_thinking}};
    record["artifact"] = Json{{"path", options.artifact_path},
                              {"size_bytes", std::move(artifact_size)},
                              {"target", load.target},
                              {"weights_id", load.weights_id},
                              {"bytes_read", load.artifact_bytes_read},
                              {"host_to_device_bytes", load.host_to_device_bytes},
                              {"peak_staging_bytes", load.peak_staging_bytes},
                              {"tensor_count", load.tensor_count},
                              {"resource_count", load.resource_count},
                              {"load_seconds", load.load_seconds},
                              {"upload_seconds", load.upload_seconds}};
    record["engine"]   = Json{
          {"device", options.device},
          // Tensor-parallel width and the resolved per-rank device list. At `tp` 2 every other
          // engine and memory field in this record is rank 0's; `devices` is what names the
          // second GPU, so a downstream reader can tell a dual-device run from a single-device
          // one without parsing argv. `devices` always has exactly `tp` entries.
          {"tp", options.tp},
          {"devices", options.devices},
          {"max_context", options.max_context},
          {"rope_mode", options.rope_mode == RopeMode::Yarn ? "yarn" : "native"},
          {"yarn_factor", options.rope_mode == RopeMode::Yarn ? options.yarn_factor : 0.0},
          {"yarn_origin", options.rope_mode == RopeMode::Yarn ? options.yarn_origin : 0U},
          {"effective_max_context", memory.effective_max_context},
          {"yarn_mscale", memory.yarn_mscale},
          {"kv_capacity_mode", kv_capacity_mode_name(memory.kv_capacity_mode)},
          {"kv_capacity", memory.kv_capacity},
          {"kv_capacity_page_groups", memory.kv_capacity_page_groups},
          {"kv_capacity_max_page_groups", memory.kv_capacity_max_page_groups},
          {"max_concurrency", options.max_concurrency},
          {"max_pending_requests", options.max_pending_requests},
          {"pending_timeout_ms", options.pending_timeout_ms},
          {"prefill_chunk", options.prefill_chunk},
          {"log_stats_interval_ms", options.log_stats_interval_ms},
          {"kv_cache", kv_cache_name(options.kv_cache)},
          {"vision", options.enable_vision},
          {"cuda_graph", options.use_cuda_graph},
          {"prefix_reuse", options.allow_prefix_reuse},
          {"speculative_backend", product::speculative_backend_name(options.speculative.backend)},
          {"speculative_draft_window", options.speculative.draft_tokens},
          {"proposal_head", proposal_head_name(options.speculative.proposal_head)}};
    record["sampling_defaults"] =
        Json{{"thinking", preset_json(sampling_defaults.thinking)},
             {"non_thinking", preset_json(sampling_defaults.non_thinking)},
             {"server_overrides", overrides_json(options.sampling_overrides)},
             {"omitted_seed", "random"},
             {"greedy", options.greedy}};
    record["memory"] =
        Json{{"weights", arena_json(memory.weights)},
             {"sequence", arena_json(memory.sequence)},
             {"workspace", arena_json(memory.workspace)},
             {"request_transient", arena_json(memory.request_transient)},
             {"minimum_runtime_reservation_bytes", memory.minimum_runtime_reservation_bytes},
             {"kv_capacity_increment_bytes", memory.kv_capacity_increment_bytes},
             {"runtime_reservation_bytes", memory.runtime_reservation_bytes},
             {"available_after_weights_bytes", memory.available_after_weights_bytes},
             {"available_after_startup_bytes", memory.available_after_startup_bytes},
             {"kv_capacity_headroom_bytes", memory.kv_capacity_headroom_bytes},
             {"planned_slack_bytes", memory.planned_slack_bytes},
             {"cuda_graph_allowance_bytes", memory.cuda_graph_allowance_bytes},
             // Both observed figures are per device and are checked against the same
             // per-device allowance; the peer field is 0 at tp 1. `cuda_graph_node_count`
             // is one captured decode graph's node count, roughly 2x at tp 2 because one
             // graph holds both devices' nodes.
             {"cuda_graph_observed_bytes", memory.cuda_graph_observed_bytes},
             {"cuda_graph_peer_observed_bytes", memory.cuda_graph_peer_observed_bytes},
             {"cuda_graph_node_count", memory.cuda_graph_node_count},
             {"kv_payload_bytes", memory.kv_payload_bytes}};
    record["environment"] =
        Json{{"device", environment.device},
             {"gpu_name", environment.gpu_name},
             {"gpu_uuid", environment.gpu_uuid},
             {"total_device_memory_bytes", environment.total_device_memory_bytes},
             {"compute_capability_major", environment.compute_capability_major},
             {"compute_capability_minor", environment.compute_capability_minor},
             {"cuda_compile_version", environment.cuda_compile_version},
             {"cuda_runtime_version", environment.cuda_runtime_version},
             {"cuda_driver_version", environment.cuda_driver_version}};
    record["argv"] = options.startup_argv;
    return record.dump();
}

std::string format_request_start_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp, const RequestLogContext& context) {
    Json record                   = event_base(server_instance_id, timestamp, "request_start");
    record["request"]             = request_json(context);
    record["preparation_seconds"] = preparation_json(context);
    return record.dump();
}

std::string format_request_rejected_json(const std::string& server_instance_id,
                                         std::uint64_t timestamp,
                                         const RequestRejectionLogContext& context) {
    Json record       = event_base(server_instance_id, timestamp, "request_rejected");
    record["phase"]   = "prepare";
    record["request"] = rejected_request_json(context);
    record["error"]   = error_json(context.error);
    return record.dump();
}

std::string format_request_done_json(const std::string& server_instance_id, std::uint64_t timestamp,
                                     const RequestLogContext& context,
                                     const GenerationOutcome& outcome) {
    Json record       = event_base(server_instance_id, timestamp, "request_done");
    record["request"] = request_json(context);
    record["result"] =
        Json{{"finish_reason", finish_reason_name(outcome.finish_reason)},
             {"prompt_tokens", outcome.prompt_tokens},
             {"completion_tokens", outcome.completion_tokens},
             {"computed_prefill_tokens",
              std::max(0, outcome.prompt_tokens -
                              static_cast<int>(outcome.metrics.prefix_cache_hit_tokens))},
             {"prefix_cache_hit_tokens", outcome.metrics.prefix_cache_hit_tokens},
             {"prefix_reuse_path", prefix_reuse_path_name(outcome.metrics.prefix_reuse_path)},
             {"prefix_reuse_reason",
              prefix_reuse_reason_name(outcome.metrics.prefix_reuse_reason)},
             {"tool_call_count", outcome.tool_calls.size()}};
    const ninfer::PrefixDivergence& divergence = outcome.metrics.prefix_divergence;
    if (divergence.frontier_index != 0) {
        Json prompt_window = Json::array();
        Json resident_window = Json::array();
        for (std::size_t k = 0; k < ninfer::kPrefixDivergenceWindow; ++k) {
            prompt_window.push_back(divergence.prompt_tokens[k]);
            resident_window.push_back(divergence.resident_tokens[k]);
        }
record["result"]["prefix_divergence"] =
             Json{{"divergence_index", divergence.divergence_index},
                  {"frontier_index", divergence.frontier_index},
                  {"tokens_matched", divergence.tokens_matched},
                  {"window_begin", divergence.window_begin},
                  {"prompt_tokens", std::move(prompt_window)},
                  {"resident_tokens", std::move(resident_window)}};
    }
    const ninfer::PrefixAdmissionDiagnostics& admission = outcome.metrics.prefix_admission;
    if (admission.hybrid_enabled) {
        Json candidates = Json::array();
        for (std::uint32_t i = 0; i < admission.candidates_shown; ++i) {
            const auto& c = admission.candidates[i];
            candidates.push_back(Json{
                {"frontier", c.frontier},
                {"anchor_depth", c.anchor_depth},
                {"kind", snapshot_kind_name(c.kind)},
                {"device_resident", c.device_resident},
                {"host_restorable", c.host_restorable},
                {"reject", candidate_reject_code_name(c.reject)},
            });
        }
        record["result"]["prefix_admission"] = Json{
{"reuse_path", prefix_reuse_path_name(outcome.metrics.prefix_reuse_path)},
            {"prompt_tokens", admission.prompt_tokens},
            {"eligible_blocks", admission.eligible_blocks},
            {"allow_prefix_reuse", admission.allow_prefix_reuse},
            {"identity_reusable", admission.identity_reusable},
            {"retained_sequence", admission.retained_sequence},
            {"retained_frontier", admission.retained_frontier},
            {"mtp_backend", admission.mtp_backend},
            {"mtp_kv_valid", admission.mtp_kv_valid},
            {"mtp_cache_present", admission.mtp_cache_present},
            {"tail_hidden_valid", admission.tail_hidden_valid},
            {"snapshot_slots", Json{{"used", admission.snapshot_slots_used},
                                    {"total", admission.snapshot_slots_total},
                                    {"max_new_taps_per_request", admission.max_new_taps}}},
            {"walk",
             Json{{"cause", stall_cause_name(admission.walk_cause)},
                  {"matched_blocks", admission.matched_blocks},
                  {"stall_sibling_count", admission.stall_sibling_count}}},
            {"selection_veto", selection_veto_name(admission.selection_veto)},
            {"candidates_total", admission.candidates_total},
            {"candidates", std::move(candidates)},
            {"reuse_base", admission.reuse_base},
        };
    }
    record["timings_seconds"] = Json{
        {"prepare", outcome.metrics.prepare_seconds}, {"ttft", outcome.metrics.ttft_seconds},
        {"vision", outcome.metrics.vision_seconds},   {"prefill", outcome.metrics.prefill_seconds},
        {"decode", outcome.metrics.decode_seconds},   {"total", outcome.metrics.total_seconds}};
    record["speculative"] = speculative_json(outcome.metrics);
    return record.dump();
}

std::string format_request_error_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp, const RequestLogContext& context,
                                      const std::string& message) {
    Json record       = event_base(server_instance_id, timestamp, "request_error");
    record["request"] = request_json(context);
    record["error"]   = Json{{"message", message}};
    return record.dump();
}

std::string format_throughput_json(const std::string& server_instance_id, std::uint64_t timestamp,
                                   const ThroughputReport& report) {
    Json record = event_base(server_instance_id, timestamp, "throughput");
    const double prefill_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.computed_prefill_tokens) / report.interval_seconds
            : 0.0;
    const double decode_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.committed_decode_tokens) / report.interval_seconds
            : 0.0;
    Json average_batch = nullptr;
    if (report.decode_rounds != 0) {
        average_batch = static_cast<double>(report.decode_row_rounds) /
                        static_cast<double>(report.decode_rounds);
    }
    record["interval_seconds"] = report.interval_seconds;
    record["tokens"]           = Json{{"computed_prefill", report.computed_prefill_tokens},
                                      {"committed_decode", report.committed_decode_tokens}};
    record["throughput_tokens_per_second"] =
        Json{{"prefill", prefill_rate}, {"decode", decode_rate}};
    record["scheduler"]    = Json{{"running", report.scheduler.running_requests},
                                  {"prefilling", report.scheduler.prefilling_requests},
                                  {"decode_ready", report.scheduler.decode_ready_requests},
                                  {"waiting", report.scheduler.waiting_requests}};
    record["decode_batch"] = Json{{"rounds", report.decode_rounds},
                                  {"row_rounds", report.decode_row_rounds},
                                  {"average_size", std::move(average_batch)}};
    return record.dump();
}

ServerLogEnvironment query_server_log_environment(int device) {
    ServerLogEnvironment environment;
    environment.device               = device;
    environment.cuda_compile_version = cuda_version_string(CUDART_VERSION);

    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        environment.cuda_runtime_version = cuda_version_string(runtime_version);
    }
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
        environment.cuda_driver_version = cuda_version_string(driver_version);
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
        environment.gpu_name                  = properties.name;
        environment.gpu_uuid                  = cuda_uuid_string(properties.uuid);
        environment.total_device_memory_bytes = properties.totalGlobalMem;
        environment.compute_capability_major  = properties.major;
        environment.compute_capability_minor  = properties.minor;
    }
    return environment;
}

JsonlRequestLog::JsonlRequestLog(const std::string& path,
                                 const std::string& protected_artifact_path)
    : path_(path) {
    if (path_.empty()) { return; }
    if (!protected_artifact_path.empty() &&
        normalized_absolute_path(path_) == normalized_absolute_path(protected_artifact_path)) {
        throw std::invalid_argument("request JSONL log must not overwrite the model artifact");
    }
    server_instance_id_ = new_server_instance_id();
    output_.open(path_, std::ios::out | std::ios::app);
    if (!output_) {
        throw std::runtime_error("failed to open request JSONL log for append: " + path_);
    }
}

void JsonlRequestLog::write_server_start(const ServeOptions& options,
                                         const ninfer::ModelSamplingDefaults& sampling_defaults,
                                         const std::string& public_model_id,
                                         const ninfer::LoadSummary& load,
                                         const ninfer::MemorySummary& memory) {
    if (!enabled()) { return; }
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(options.artifact_path, error);
    const std::optional<std::uint64_t> artifact_size =
        error ? std::nullopt : std::optional<std::uint64_t>(size);
    append(format_server_start_json(server_instance_id_, unix_time_ms(), options, sampling_defaults,
                                    public_model_id, load, memory,
                                    query_server_log_environment(options.device), artifact_size));
}

void JsonlRequestLog::write_request_start(const RequestLogContext& context) {
    if (!enabled()) { return; }
    append(format_request_start_json(server_instance_id_, unix_time_ms(), context));
}

void JsonlRequestLog::write_request_rejected(const RequestRejectionLogContext& context) {
    if (!enabled()) { return; }
    append(format_request_rejected_json(server_instance_id_, unix_time_ms(), context));
}

void JsonlRequestLog::write_request_done(const RequestLogContext& context,
                                         const GenerationOutcome& outcome) {
    if (!enabled()) { return; }
    append(format_request_done_json(server_instance_id_, unix_time_ms(), context, outcome));
}

void JsonlRequestLog::write_request_error(const RequestLogContext& context,
                                          const std::string& message) {
    if (!enabled()) { return; }
    append(format_request_error_json(server_instance_id_, unix_time_ms(), context, message));
}

void JsonlRequestLog::write_throughput(const ThroughputReport& report) {
    if (!enabled()) { return; }
    append(format_throughput_json(server_instance_id_, unix_time_ms(), report));
}

void JsonlRequestLog::append(std::string record) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (failed_) { return; }
    output_ << record << '\n';
    output_.flush();
    if (!output_) {
        failed_ = true;
        write_console_log(ConsoleLogLevel::Error, "request JSONL logging failed for " + path_);
    }
}

} // namespace ninfer::serve
