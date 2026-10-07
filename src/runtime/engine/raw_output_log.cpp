#include "runtime/engine/raw_output_log.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>

namespace ninfer::runtime {
namespace {

using Json = nlohmann::json;

const char* finish_reason_name(FinishReason reason) {
    switch (reason) {
    case FinishReason::None:
        return "none";
    case FinishReason::OutputLimit:
        return "output_limit";
    case FinishReason::ContextCapacity:
        return "context_capacity";
    case FinishReason::StopToken:
        return "stop_token";
    case FinishReason::StopString:
        return "stop_string";
    case FinishReason::Cancelled:
        return "cancelled";
    }
    return "unknown";
}

const char* prefix_reuse_path_name(PrefixReusePath path) {
    switch (path) {
    case PrefixReusePath::FullReset:
        return "full_reset";
    case PrefixReusePath::AppendAtFrontier:
        return "append_frontier";
    case PrefixReusePath::RestoreTurnCheckpoint:
        return "restore_turn_checkpoint";
    case PrefixReusePath::RestoreResponseCheckpoint:
        return "restore_response_checkpoint";
    case PrefixReusePath::BlockSnapshot:
        return "block_snapshot";
    }
    return "unknown";
}

const char* prefix_reuse_reason_name(PrefixReuseReason reason) {
    switch (reason) {
    case PrefixReuseReason::None:
        return "none";
    case PrefixReuseReason::ReusedFrontier:
        return "reused_frontier";
    case PrefixReuseReason::ReusedTurnCheckpoint:
        return "reused_turn_checkpoint";
    case PrefixReuseReason::ReusedResponseCheckpoint:
        return "reused_response_checkpoint";
    case PrefixReuseReason::ReusedBlockSnapshot:
        return "reused_block_snapshot";
    case PrefixReuseReason::ReuseDisabled:
        return "reuse_disabled";
    case PrefixReuseReason::PromptNotReusable:
        return "prompt_not_reusable";
    case PrefixReuseReason::NoRetainedSequence:
        return "no_retained_sequence";
    case PrefixReuseReason::FrontierEmpty:
        return "frontier_empty";
    case PrefixReuseReason::DflashFrontierMismatch:
        return "dflash_frontier_mismatch";
    case PrefixReuseReason::FrontierPrefixMismatch:
        return "frontier_prefix_mismatch";
    case PrefixReuseReason::CheckpointUnavailable:
        return "checkpoint_unavailable";
    case PrefixReuseReason::MtpTailHiddenInvalid:
        return "mtp_tail_hidden_invalid";
    case PrefixReuseReason::MtpCacheMissing:
        return "mtp_cache_missing";
    case PrefixReuseReason::MtpKvTooShort:
        return "mtp_kv_too_short";
    case PrefixReuseReason::DflashCheckpointInvalid:
        return "dflash_checkpoint_invalid";
    }
    return "unknown";
}

} // namespace

RawOutputLog& RawOutputLog::instance() {
    static RawOutputLog log;
    return log;
}

RawOutputLog::RawOutputLog() {
    const char* path = std::getenv("NINFER_RAW_OUTPUT_LOG");
    if (path == nullptr || *path == '\0') { return; }
    path_ = path;
    output_.open(path_, std::ios::out | std::ios::app | std::ios::binary);
    enabled_ = output_.is_open();
    if (!enabled_) {
        std::fprintf(stderr,
                     "ninfer: NINFER_RAW_OUTPUT_LOG=%s could not be opened; raw output capture "
                     "disabled\n",
                     path_.c_str());
    }
}

RawOutputLog::~RawOutputLog() = default;

void RawOutputLog::write(const RawOutputRecord& record) {
    Json value;
    value["schema"]           = "ninfer_raw_output_debug";
    value["schema_version"]   = 1;
    value["ts_unix_ms"]       = record.timestamp_unix_ms;
    value["request_id"]       = record.request_id;
    value["prompt_tokens"]    = record.prompt_tokens;
    value["generated_tokens"] = record.token_ids.size();
    value["reasoning_tokens"] = record.reasoning_tokens;
    value["finish_reason"]    = finish_reason_name(record.finish_reason);
    value["prefix"]           = Json{{"reused_prompt_tokens", record.reused_prompt_tokens},
                                     {"path", prefix_reuse_path_name(record.prefix_reuse_path)},
                                     {"reason", prefix_reuse_reason_name(record.prefix_reuse_reason)}};
    value["output"]           = Json{{"raw", record.raw},
                                     {"preserve_special", record.preserve_special}};
    value["prompt_tail"]      = record.prompt_tail;
    value["raw"]              = record.raw_text;
    value["reasoning"]        = record.reasoning;
    value["content"]          = record.content;
    value["token_ids"]        = record.token_ids;

    std::lock_guard lock(mutex_);
    output_ << value.dump() << '\n';
    output_.flush();
}

} // namespace ninfer::runtime
