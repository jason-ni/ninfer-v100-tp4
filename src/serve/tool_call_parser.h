#pragma once

#include "serve/request.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

struct ParsedToolCallOutput {
    bool is_tool_call_response = false;
    // Set when the text opens a <tool_call> that is never closed and the caller asked to drop
    // an unterminated call (a generation cut off by the output/context limit). `content` then
    // holds only the text before <tool_call>; the dangling call syntax is discarded instead of
    // being fed back to the client as ordinary text.
    bool unterminated_tool_call = false;
    std::string content;
    std::vector<ToolCall> tool_calls;
};

// Parses a Qwen XML tool-call response. When `drop_unterminated` is true (the generation ended
// on an output/context limit) an opening <tool_call> with no closing tag is reported through
// `unterminated_tool_call` and withheld from `content` rather than falling back to the whole
// text. When false the historical byte-preserving fallback is retained.
ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length,
                                                 bool drop_unterminated = false);

// Terminal disposition of the region withheld behind a possible Qwen <tool_call>.
enum class ToolCallTerminal {
    Flush, // no structured call: emit the withheld region as ordinary content
    Drop,  // structured call, or a truncated dangling one: discard the withheld region
};

// Incrementally publishes text that is provably outside a possible Qwen
// <tool_call> suffix. At terminal time the withheld region is either flushed
// verbatim (ordinary/malformed text) or discarded (a parsed or truncated call).
class ToolCallStreamFilter {
public:
    std::string feed(std::string_view text);
    std::string finish(ToolCallTerminal terminal);

    [[nodiscard]] std::size_t emitted_bytes() const noexcept { return emitted_bytes_; }

private:
    std::string pending_;
    std::string tool_region_;
    std::size_t emitted_bytes_ = 0;
    bool saw_tool_marker_      = false;
    bool finished_             = false;
};

} // namespace ninfer::serve
