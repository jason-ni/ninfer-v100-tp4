#include "serve/request_persist.h"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <utility>

namespace ninfer::serve {
namespace {

constexpr int kConnectTimeoutMs = 3000;
constexpr int kIdlePollMs      = 20;

void put_u8(std::string& out, std::uint8_t v) { out.push_back(static_cast<char>(v)); }

void put_u32_be(std::string& out, std::uint32_t v) {
    out.push_back(static_cast<char>((v >> 24) & 0xff));
    out.push_back(static_cast<char>((v >> 16) & 0xff));
    out.push_back(static_cast<char>((v >> 8) & 0xff));
    out.push_back(static_cast<char>(v & 0xff));
}

void put_u64_be(std::string& out, std::uint64_t v) {
    for (int shift = 56; shift >= 0; shift -= 8) {
        out.push_back(static_cast<char>((v >> shift) & 0xff));
    }
}

std::uint32_t get_u32_be(const char* p) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) { v = (v << 8) | static_cast<unsigned char>(p[i]); }
    return v;
}

std::uint64_t get_u64_be(const char* p) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) { v = (v << 8) | static_cast<unsigned char>(p[i]); }
    return v;
}

// Writes exactly len bytes, retrying on EINTR and partial sends. MSG_NOSIGNAL keeps a closing
// peer from raising SIGPIPE in the process.
bool write_all(int fd, const char* data, std::size_t len) {
    std::size_t off = 0;
    while (off < len) {
        const ssize_t n = ::send(fd, data + off, len - off, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) { continue; }
            return false;
        }
        if (n == 0) { return false; }
        off += static_cast<std::size_t>(n);
    }
    return true;
}

} // namespace

std::string encode_persist_frame(std::uint8_t version, std::uint64_t request_id,
                                 std::int64_t created_unix_ms, std::string_view protocol,
                                 std::string_view model, std::string_view body) {
    std::string out;
    out.reserve(2 + 8 + 8 + 4 + protocol.size() + 4 + model.size() + 4 + body.size());
    put_u8(out, version);
    put_u8(out, 0); // flags (reserved)
    put_u64_be(out, request_id);
    put_u64_be(out, static_cast<std::uint64_t>(created_unix_ms));
    put_u32_be(out, static_cast<std::uint32_t>(protocol.size()));
    out.append(protocol.data(), protocol.size());
    put_u32_be(out, static_cast<std::uint32_t>(model.size()));
    out.append(model.data(), model.size());
    put_u32_be(out, static_cast<std::uint32_t>(body.size()));
    out.append(body.data(), body.size());
    return out;
}

bool decode_persist_frame(const char* data, std::size_t len, PersistFrame& out) {
    constexpr std::size_t kHeaderMin = 2 + 8 + 8; // version, flags, request_id, created
    if (data == nullptr || len < kHeaderMin) { return false; }

    std::size_t off = 0;
    const std::uint8_t version = static_cast<unsigned char>(data[off++]);
    if (version != kPersistProtocolVersion) { return false; }
    off++; // flags

    const std::uint64_t request_id    = get_u64_be(data + off);
    off += 8;
    const std::int64_t created_unix_ms = static_cast<std::int64_t>(get_u64_be(data + off));
    off += 8;

    auto read_str = [&](std::string& dst) -> bool {
        if (off + 4 > len) { return false; }
        const std::uint32_t n = get_u32_be(data + off);
        off += 4;
        if (off + n > len) { return false; }
        dst.assign(data + off, n);
        off += n;
        return true;
    };

    PersistFrame parsed;
    parsed.version         = version;
    parsed.request_id      = request_id;
    parsed.created_unix_ms = created_unix_ms;
    if (!read_str(parsed.protocol) || !read_str(parsed.model) || !read_str(parsed.body)) {
        return false;
    }
    out = parsed;
    return true;
}

RequestPersistor::RequestPersistor(const ServeOptions& options) {
    config_.queue_capacity = options.persist_queue_capacity;
    config_.reconnect_ms   = options.persist_reconnect_ms;
    if (options.persist_endpoint.empty()) {
        return; // persistence disabled
    }
    const std::size_t colon = options.persist_endpoint.rfind(':');
    if (colon == std::string::npos || colon == 0) {
        return; // malformed; leave disabled rather than fail startup (validated at parse time)
    }
    config_.host = options.persist_endpoint.substr(0, colon);
    const std::string port_text = options.persist_endpoint.substr(colon + 1);
    char* end                   = nullptr;
    const long port             = std::strtol(port_text.c_str(), &end, 10);
    if (end == port_text.c_str() || *end != '\0' || port < 1 || port > 65535) {
        return;
    }
    config_.port         = static_cast<int>(port);
    config_enabled_      = true;
    worker_              = std::thread([this] { run(); });
}

RequestPersistor::~RequestPersistor() { stop(); }

void RequestPersistor::enqueue(PersistRecord record) {
    if (!config_enabled_) { return; }
    {
        std::lock_guard lock(mutex_);
        if (queue_.size() >= config_.queue_capacity) {
            dropped_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        queue_.push_back(std::move(record));
    }
    cv_.notify_one();
}

void RequestPersistor::stop() {
    if (!config_enabled_) { return; }
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) { worker_.join(); }
}

void RequestPersistor::run() {
    while (true) {
        PersistRecord record;
        bool have = false;
        {
            std::unique_lock lock(mutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(kIdlePollMs), [this] {
                return !queue_.empty() || stopping_;
            });
            if (!queue_.empty()) {
                record = std::move(queue_.front());
                queue_.pop_front();
                have = true;
            } else if (stopping_) {
                break;
            }
        }
        if (!have) { continue; }

        if (!ensure_connected() || !send_frame(record)) {
            close_socket();
            if (stopping_) {
                std::lock_guard lock(mutex_);
                dropped_.fetch_add(queue_.size() + 1, std::memory_order_relaxed);
                queue_.clear();
                break;
            }
            std::lock_guard lock(mutex_);
            queue_.push_front(std::move(record));
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.reconnect_ms));
            continue;
        }
        sent_.fetch_add(1, std::memory_order_relaxed);
    }
}

bool RequestPersistor::ensure_connected() {
    if (socket_ != -1) { return true; }

    std::string addr = config_.host;
    if (addr.size() >= 2 && addr.front() == '[' && addr.back() == ']') {
        addr = addr.substr(1, addr.size() - 2); // bracketed IPv6 literal
    }
    const std::string service = std::to_string(config_.port);

    struct addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* results = nullptr;
    const int rc             = ::getaddrinfo(addr.c_str(), service.c_str(), &hints, &results);
    if (rc != 0 || results == nullptr) {
        if (results != nullptr) { ::freeaddrinfo(results); }
        return false;
    }

    int fd = -1;
    for (struct addrinfo* it = results; it != nullptr; it = it->ai_next) {
        fd = ::socket(it->ai_family, it->ai_socktype, it->ai_protocol);
        if (fd < 0) { continue; }

        const int flags = ::fcntl(fd, F_GETFL, 0);
        if (flags >= 0 && ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0) {
            const int crc = ::connect(fd, it->ai_addr, it->ai_addrlen);
            if (crc == 0) {
                // Connected immediately.
            } else if (errno == EINPROGRESS) {
                struct pollfd pfd{};
                pfd.fd     = fd;
                pfd.events = POLLOUT;
                const int prc = ::poll(&pfd, 1, kConnectTimeoutMs);
                bool ok       = false;
                if (prc > 0 && (pfd.revents & POLLOUT)) {
                    int so_error = 0;
                    socklen_t elen = sizeof(so_error);
                    ok             = ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &elen) == 0 &&
                             so_error == 0;
                }
                if (!ok) {
                    ::close(fd);
                    fd = -1;
                    continue;
                }
            } else {
                ::close(fd);
                fd = -1;
                continue;
            }
            if (flags >= 0) { ::fcntl(fd, F_SETFL, flags); } // back to blocking
        } else {
            if (::connect(fd, it->ai_addr, it->ai_addrlen) != 0) {
                ::close(fd);
                fd = -1;
                continue;
            }
        }
        int nodelay = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
        break;
    }
    ::freeaddrinfo(results);
    if (fd < 0) { return false; }
    socket_ = fd;
    return true;
}

bool RequestPersistor::send_frame(const PersistRecord& record) {
    const std::string envelope = encode_persist_frame(
        kPersistProtocolVersion, record.request_id, record.created_unix_ms, record.protocol,
        record.model, record.body);
    std::string frame;
    frame.reserve(4 + envelope.size());
    put_u32_be(frame, static_cast<std::uint32_t>(envelope.size()));
    frame.append(envelope);
    return write_all(socket_, frame.data(), frame.size());
}

void RequestPersistor::close_socket() {
    if (socket_ != -1) {
        ::close(socket_);
        socket_ = -1;
    }
}

} // namespace ninfer::serve