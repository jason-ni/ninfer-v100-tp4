#pragma once

// Request persistence transport: a bounded in-memory queue fed by the HTTP handler plus a
// dedicated sender thread that ships framed records to a remote persistence server over a
// length-prefixed TCP binary protocol. Enqueue is non-blocking (drops when full) so the serving
// path never waits on the network. The wire codec is pure and shared with the Python receiver
// (tools/request_persist); the frame layout is documented in
// docs/maintainer/request-persistence.md.

#include "serve/serve_options.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

namespace ninfer::serve {

// Wire format version understood by the Python receiver. Bump on any change to the envelope
// layout; the receiver rejects unknown versions.
inline constexpr std::uint8_t kPersistProtocolVersion = 1;

// One persisted request: the raw OpenAI JSON body plus serve-side provenance.
struct PersistRecord {
    std::uint64_t request_id    = 0;
    std::int64_t created_unix_ms = 0;
    std::string protocol;
    std::string model;
    std::string body; // raw JSON body exactly as received
};

// A parsed envelope (transport-agnostic). Mirrors the Python receiver's ParsedRecord.
struct PersistFrame {
    std::uint8_t version         = 0;
    std::uint64_t request_id     = 0;
    std::int64_t created_unix_ms = 0;
    std::string protocol;
    std::string model;
    std::string body;
};

// Encodes an envelope WITHOUT the u32be length prefix; the transport prepends it. Big-endian.
std::string encode_persist_frame(std::uint8_t version, std::uint64_t request_id,
                                 std::int64_t created_unix_ms, std::string_view protocol,
                                 std::string_view model, std::string_view body);

// Parses an envelope (the bytes after the u32be length prefix). Returns false on a malformed or
// unknown-version envelope, leaving out unchanged.
bool decode_persist_frame(const char* data, std::size_t len, PersistFrame& out);

class RequestPersistor {
public:
    explicit RequestPersistor(const ServeOptions& options);
    RequestPersistor(const RequestPersistor&)            = delete;
    RequestPersistor& operator=(const RequestPersistor&) = delete;
    ~RequestPersistor();

    [[nodiscard]] bool enabled() const noexcept { return config_enabled_; }

    // Non-blocking. No-op when disabled; drops (and counts) when the queue is full.
    void enqueue(PersistRecord record);

    // Signals the worker, drains best-effort, and joins. Called from the destructor only: a
    // signal handler must not join a thread, so HttpServer::stop() must not invoke this.
    void stop();

    [[nodiscard]] std::uint64_t sent_count() const noexcept {
        return sent_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t dropped_count() const noexcept {
        return dropped_.load(std::memory_order_relaxed);
    }

private:
    void run();
    bool ensure_connected();
    bool send_frame(const PersistRecord& record);
    void close_socket();

    struct Config {
        std::string host;
        int port               = 0;
        std::uint32_t queue_capacity = kDefaultPersistQueueCapacity;
        int reconnect_ms       = kDefaultPersistReconnectMs;
    };

    Config config_;
    bool config_enabled_ = false;

    std::deque<PersistRecord> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    bool stopping_ = false;

    int socket_ = -1; // owned by the worker thread only

    std::atomic<std::uint64_t> sent_{0};
    std::atomic<std::uint64_t> dropped_{0};
};

} // namespace ninfer::serve