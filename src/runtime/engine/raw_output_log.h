#pragma once

// TEMPORARY debug capture. When the environment variable NINFER_RAW_OUTPUT_LOG names a writable
// file this appends one JSON object per completed request describing the exact model output that
// reached the reasoning/content parser and how the parser split it. It exists to diagnose the
// tool-call/reasoning routing leak, whose failing raw output is otherwise lost once the deltas have
// been routed. It is inert (one cached bool check per completion) when the variable is unset.
//
// File layout (JSONL, one record per line; append mode so a run can collect many requests):
//
//   {"schema":"ninfer_raw_output_debug","schema_version":1,"ts_unix_ms":<int>,
//    "request_id":<int>,"prompt_tokens":<int>,"generated_tokens":<int>,
//    "reasoning_tokens":<int>,"finish_reason":"stop_token",
//    "prefix":{"reused_prompt_tokens":<int>,"path":"append_frontier","reason":"reused_frontier"},
//    "output":{"raw":false,"preserve_special":false},
//    "prompt_tail":"<decoded tail of the prompt fed to the model>",
//    "raw":"<the exact decoded model output, markers intact>",
//    "reasoning":"<text routed to the reasoning channel>",
//    "content":"<text routed to the content channel>",
//    "token_ids":[<int>,...]}
//
// `raw` is authoritative: `reasoning`/`content` are its parser split. A structural `</think>`
// close is consumed by the parser (present in neither channel) while a quoted copy stays on the
// reasoning channel; comparing the three strings localizes a misrouted marker. `token_ids`
// re-decodes the exact stream if the text is ambiguous. `prompt_tail` shows whether the model was
// fed a continuation (a prompt ending mid-tool-call leaves no `</think>` to close on); the
// `prefix` block reports which retained-sequence path resumed this request. Read a field with
// e.g. `jq -r .raw file.jsonl`.
//
// Not part of any product contract; delete this file, its .cpp, the CMake source entry, the
// ConcurrentExecutor call, and OutputSession::debug_raw_text when the investigation concludes.

#include "ninfer/types.h"

#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace ninfer::runtime {

struct RawOutputRecord {
    std::uint64_t request_id        = 0;
    std::uint64_t timestamp_unix_ms = 0;
    std::uint32_t prompt_tokens     = 0;
    std::uint32_t reasoning_tokens  = 0;
    FinishReason finish_reason      = FinishReason::None;
    bool raw                        = false;
    bool preserve_special           = false;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::FullReset;
    PrefixReuseReason prefix_reuse_reason = PrefixReuseReason::None;
    std::vector<TokenId> token_ids;
    std::string prompt_tail;
    std::string raw_text;
    std::string reasoning;
    std::string content;
};

class RawOutputLog {
public:
    static RawOutputLog& instance();

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }

    void write(const RawOutputRecord& record);

private:
    RawOutputLog();
    ~RawOutputLog();

    bool enabled_ = false;
    std::string path_;
    std::mutex mutex_;
    std::ofstream output_;
};

} // namespace ninfer::runtime
