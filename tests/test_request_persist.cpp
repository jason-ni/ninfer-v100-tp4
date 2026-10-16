#include "serve/request_persist.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::vector<char> from_hex(const std::string& hex) {
    std::vector<char> out;
    out.reserve(hex.size() / 2);
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') { return c - '0'; }
        if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
        if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
        return -1;
    };
    for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
        const int hi = nibble(hex[i]);
        const int lo = nibble(hex[i + 1]);
        if (hi < 0 || lo < 0) { break; }
        out.push_back(static_cast<char>((hi << 4) | lo));
    }
    return out;
}

std::string to_hex(const std::string& bytes) {
    static const char* digits = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (unsigned char c : bytes) {
        out.push_back(digits[c >> 4]);
        out.push_back(digits[c & 0x0f]);
    }
    return out;
}

} // namespace

int main() {
    int failures = 0;

    // Golden vector pins the exact cross-language wire format (see
    // docs/maintainer/request-persistence.md). This byte
    // stream is asserted identically by the Python receiver's test.
    const std::string golden_hex =
        "0100000000000000002a00000192cc091400000000176f70656e61695f636861745f636f6d706c6574696"
        "f6e73000000117177656e332e382d3237622f6e76667034000000397b226d6f64656c223a226d222c226d"
        "65737361676573223a5b7b22726f6c65223a2275736572222c22636f6e74656e74223a226869227d5d7d";

    const std::string encoded = encode_persist_frame(
        kPersistProtocolVersion, 42, 1730000000000, "openai_chat_completions",
        "qwen3.8-27b/nvfp4", R"({"model":"m","messages":[{"role":"user","content":"hi"}]})");
    failures += check(to_hex(encoded) == golden_hex, "encoded envelope does not match golden bytes");

    const std::vector<char> golden = from_hex(golden_hex);
    PersistFrame decoded;
    failures += check(decode_persist_frame(golden.data(), golden.size(), decoded),
                      "golden envelope failed to decode");
    failures += check(decoded.version == kPersistProtocolVersion, "decoded version mismatch");
    failures += check(decoded.request_id == 42, "decoded request_id mismatch");
    failures += check(decoded.created_unix_ms == 1730000000000, "decoded created_unix_ms mismatch");
    failures += check(decoded.protocol == "openai_chat_completions", "decoded protocol mismatch");
    failures += check(decoded.model == "qwen3.8-27b/nvfp4", "decoded model mismatch");
    failures += check(decoded.body == R"({"model":"m","messages":[{"role":"user","content":"hi"}]})",
                      "decoded body mismatch");

    // Round-trip with edge values: max u64 id, negative timestamp, multi-byte UTF-8 body.
    const std::uint64_t max_id   = ~static_cast<std::uint64_t>(0);
    const std::int64_t neg_time  = -5;
    const std::string unicode_body =
        R"({"messages":[{"role":"user","content":"héllo 世界"}]})";
    const std::string rt = encode_persist_frame(kPersistProtocolVersion, max_id, neg_time,
                                                "openai_chat_completions", "m", unicode_body);
    PersistFrame rt_decoded;
    failures += check(decode_persist_frame(rt.data(), rt.size(), rt_decoded),
                      "edge-case envelope failed to decode");
    failures += check(rt_decoded.request_id == max_id, "round-trip request_id mismatch");
    failures += check(rt_decoded.created_unix_ms == neg_time, "round-trip created_unix_ms mismatch");
    failures += check(rt_decoded.body == unicode_body, "round-trip body mismatch");

    // Unknown version is rejected.
    std::string bad_version = encoded;
    bad_version[0]          = static_cast<char>(kPersistProtocolVersion + 1);
    PersistFrame v_out;
    failures += check(!decode_persist_frame(bad_version.data(), bad_version.size(), v_out),
                      "unknown version was accepted");

    // Truncated input is rejected.
    failures += check(!decode_persist_frame(encoded.data(), encoded.size() - 1, v_out),
                      "truncated envelope was accepted");
    failures += check(!decode_persist_frame(nullptr, 0, v_out), "null envelope was accepted");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}