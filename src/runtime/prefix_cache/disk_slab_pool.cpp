#include "runtime/prefix_cache/disk_slab_pool.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>

namespace ninfer::prefix_cache {
namespace {

constexpr std::uint64_t kFileMagic   = 0x4E46'444B'534C'4231ULL; // "NFDKSB1"
constexpr std::uint64_t kSlotMagic   = 0x4E46'534C'4F54'3031ULL; // "NFSLOT01"
constexpr std::uint32_t kFileVersion = 1;

constexpr std::size_t kFileHeaderBytes = 4096;
constexpr std::size_t kSlotHeaderBytes = 64;

// File header written once per creation. The payload regions follow it.
struct FileHeader {
    std::uint64_t magic              = kFileMagic;
    std::uint32_t version            = kFileVersion;
    std::uint32_t block_slab_count   = 0;
    std::uint32_t snapshot_slab_count = 0;
    std::uint32_t reserved           = 0;
    std::uint64_t block_slab_bytes   = 0;
    std::uint64_t snapshot_slab_bytes = 0;
    std::uint64_t file_uid           = 0;
};

// Per-slot header, committed after the payload. `kind` is 0 for a block slot, 1 for a snapshot.
struct SlotHeader {
    std::uint64_t magic         = kSlotMagic;
    std::uint32_t kind          = 0;
    std::uint32_t generation    = 0;
    std::uint64_t payload_bytes = 0;
    std::uint64_t checksum      = 0;
    std::uint64_t reserved[4]   = {0, 0, 0, 0};
};
static_assert(sizeof(FileHeader) <= kFileHeaderBytes, "file header must fit");
static_assert(sizeof(SlotHeader) == kSlotHeaderBytes, "slot header must be 64 bytes");

[[nodiscard]] std::uint64_t mix(std::uint64_t value) noexcept {
    value += 0x9E37'79B9'7F4A'7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58'476D'1CE4'E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D0'49BB'1331'11EBULL;
    return value ^ (value >> 31U);
}

// Word-wise integrity tag over a slot payload. Detects torn writes and gross corruption; it is
// not a security boundary.
[[nodiscard]] std::uint64_t checksum_bytes(const void* data, std::size_t bytes) noexcept {
    const auto* p = static_cast<const std::uint8_t*>(data);
    std::uint64_t hash = 0x9E37'79B9'7F4A'7C15ULL;
    std::size_t i = 0;
    for (; i + 8U <= bytes; i += 8U) {
        std::uint64_t word = 0;
        std::memcpy(&word, p + i, sizeof(word));
        hash = mix(hash ^ word);
    }
    if (i < bytes) {
        std::uint64_t tail = 0;
        std::memcpy(&tail, p + i, bytes - i);
        hash = mix(hash ^ tail);
    }
    return mix(hash ^ static_cast<std::uint64_t>(bytes));
}

[[nodiscard]] bool pwrite_all(int fd, const void* data, std::size_t bytes, std::size_t offset) {
    const auto* p = static_cast<const char*>(data);
    std::size_t done = 0;
    while (done < bytes) {
        const ssize_t w = ::pwrite(fd, p + done, bytes - done, static_cast<off_t>(offset + done));
        if (w <= 0) { return false; }
        done += static_cast<std::size_t>(w);
    }
    return true;
}

[[nodiscard]] bool pread_all(int fd, void* data, std::size_t bytes, std::size_t offset) {
    auto* p = static_cast<char*>(data);
    std::size_t done = 0;
    while (done < bytes) {
        const ssize_t r = ::pread(fd, p + done, bytes - done, static_cast<off_t>(offset + done));
        if (r <= 0) { return false; }
        done += static_cast<std::size_t>(r);
    }
    return true;
}

[[nodiscard]] std::uint64_t fresh_uid() noexcept {
    std::random_device rd;
    std::uint64_t uid = (static_cast<std::uint64_t>(rd()) << 32U) | static_cast<std::uint64_t>(rd());
    if (uid == 0) { uid = 1; }
    return uid;
}

} // namespace

DiskSlabPool::DiskSlabPool(const DiskSlabPoolOptions& options) : options_(options) {
    if (options_.total_bytes == 0 || options_.block_slab_count == 0 ||
        options_.snapshot_slab_count == 0 || options_.block_slab_bytes == 0 ||
        options_.snapshot_slab_bytes == 0 || options_.file.empty()) {
        return; // disabled by configuration
    }

    layout_.file_header_bytes = kFileHeaderBytes;
    layout_.slot_header_bytes = kSlotHeaderBytes;
    layout_.block_stride      = kSlotHeaderBytes + options_.block_slab_bytes;
    layout_.snapshot_stride   = kSlotHeaderBytes + options_.snapshot_slab_bytes;
    layout_.block_offset      = kFileHeaderBytes;
    layout_.snapshot_offset   = layout_.block_offset + options_.block_slab_count * layout_.block_stride;
    layout_.file_size         = layout_.snapshot_offset +
                                options_.snapshot_slab_count * layout_.snapshot_stride;

    if (!open_file()) {
        return;
    }
    block_slabs_.resize(options_.block_slab_count);
    snapshot_slabs_.resize(options_.snapshot_slab_count);
    pinned_bytes_ = options_.block_slab_count * options_.block_slab_bytes +
                    options_.snapshot_slab_count * options_.snapshot_slab_bytes;
}

DiskSlabPool::~DiskSlabPool() {
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

bool DiskSlabPool::open_file() {
    std::error_code ec;
    const std::filesystem::path parent = options_.file.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            std::fprintf(stderr, "[disk-cache] cannot create %s: %s; disk tier disabled\n",
                         parent.string().c_str(), ec.message().c_str());
            return false;
        }
    }
    const std::string path = options_.file.string();
    fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd_ < 0) {
        std::fprintf(stderr, "[disk-cache] cannot open %s; disk tier disabled\n", path.c_str());
        return false;
    }

    FileHeader existing;
    bool reuse = false;
    if (pread_all(fd_, &existing, sizeof(existing), 0) &&
        existing.magic == kFileMagic && existing.version == kFileVersion &&
        existing.block_slab_count == options_.block_slab_count &&
        existing.snapshot_slab_count == options_.snapshot_slab_count &&
        existing.block_slab_bytes == options_.block_slab_bytes &&
        existing.snapshot_slab_bytes == options_.snapshot_slab_bytes &&
        existing.file_uid != 0) {
        reuse     = true;
        file_uid_ = existing.file_uid;
    }

    struct stat st{};
    if (::fstat(fd_, &st) != 0) {
        ::close(fd_);
        fd_ = -1;
        return false;
    }
    const bool wrong_size = static_cast<std::uint64_t>(st.st_size) != layout_.file_size;
    if (!reuse || wrong_size) {
        // Initialize (or reinitialize) the file: a foreign geometry or size never validates.
        FileHeader fresh;
        fresh.block_slab_count    = static_cast<std::uint32_t>(options_.block_slab_count);
        fresh.snapshot_slab_count = static_cast<std::uint32_t>(options_.snapshot_slab_count);
        fresh.block_slab_bytes    = options_.block_slab_bytes;
        fresh.snapshot_slab_bytes = options_.snapshot_slab_bytes;
        fresh.file_uid            = fresh_uid();
        std::vector<std::uint8_t> header_bytes(kFileHeaderBytes, 0);
        std::memcpy(header_bytes.data(), &fresh, sizeof(fresh));
        if (::ftruncate(fd_, static_cast<off_t>(layout_.file_size)) != 0 ||
            !pwrite_all(fd_, header_bytes.data(), header_bytes.size(), 0)) {
            std::fprintf(stderr, "[disk-cache] cannot initialize %s; disk tier disabled\n",
                         path.c_str());
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        file_uid_ = fresh.file_uid;
    }
    return true;
}

std::size_t DiskSlabPool::used_bytes() const noexcept {
    std::size_t used = 0;
    for (const DiskSlab& slab : block_slabs_) {
        if (slab.occupied) { used += options_.block_slab_bytes; }
    }
    for (const DiskSlab& slab : snapshot_slabs_) {
        if (slab.occupied) { used += options_.snapshot_slab_bytes; }
    }
    return used;
}

std::size_t DiskSlabPool::free_bytes() const noexcept { return pinned_bytes_ - used_bytes(); }

void DiskSlabPool::reset_slab(DiskSlab& slab) noexcept {
    slab.occupied       = false;
    slab.pinned         = false;
    slab.dead           = false;
    slab.dead_at        = 0;
    slab.superseded     = false;
    slab.superseded_at  = 0;
    slab.freq           = 0;
    slab.last_hit_tick  = 0;
    // `generation` is monotonic for the pool's lifetime (not reset) so a slot reallocated after a
    // crash-before-index-write still fails the loader's generation check.
}

void DiskSlabPool::occupy(DiskPageId page) noexcept {
    DiskSlab* slab = slab_mut(page);
    if (slab == nullptr) {
        return;
    }
    slab->occupied      = true;
    slab->pinned        = false;
    slab->dead          = false;
    slab->dead_at       = 0;
    slab->superseded    = false;
    slab->superseded_at = 0;
    slab->freq          = 1;
    slab->last_hit_tick = ++tick_;
    slab->generation    = slab->generation + 1U;
}

bool DiskSlabPool::valid(DiskPageId page) const noexcept {
    if (page < 0) {
        return false;
    }
    const auto idx = static_cast<std::size_t>(page);
    return idx < block_slabs_.size() || idx - block_slabs_.size() < snapshot_slabs_.size();
}

const DiskSlab* DiskSlabPool::slab(DiskPageId page) const noexcept {
    if (page < 0) {
        return nullptr;
    }
    const auto idx = static_cast<std::size_t>(page);
    if (idx < block_slabs_.size()) {
        return &block_slabs_[idx];
    }
    const std::size_t snap = idx - block_slabs_.size();
    if (snap < snapshot_slabs_.size()) {
        return &snapshot_slabs_[snap];
    }
    return nullptr;
}

DiskSlab* DiskSlabPool::slab_mut(DiskPageId page) noexcept {
    return const_cast<DiskSlab*>(slab(page));
}

DiskPageId DiskSlabPool::find_free(std::size_t offset, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t idx = offset + i;
        const DiskSlab* s = slab(static_cast<DiskPageId>(idx));
        if (s != nullptr && !s->occupied) {
            return static_cast<DiskPageId>(idx);
        }
    }
    return kInvalidDiskPage;
}

DiskPageId DiskSlabPool::pick_victim(std::size_t offset, std::size_t count) noexcept {
    std::uint64_t oldest = UINT64_MAX;
    DiskPageId best = kInvalidDiskPage;
    for (std::size_t i = 0; i < count; ++i) {
        const DiskSlab* s = slab(static_cast<DiskPageId>(offset + i));
        if (s == nullptr || !s->occupied || s->pinned || !s->dead) { continue; }
        if (s->dead_at < oldest) { oldest = s->dead_at; best = static_cast<DiskPageId>(offset + i); }
    }
    if (best != kInvalidDiskPage) { return best; }

    oldest = UINT64_MAX;
    for (std::size_t i = 0; i < count; ++i) {
        const DiskSlab* s = slab(static_cast<DiskPageId>(offset + i));
        if (s == nullptr || !s->occupied || s->pinned || !s->superseded) { continue; }
        if (s->superseded_at < oldest) {
            oldest = s->superseded_at;
            best   = static_cast<DiskPageId>(offset + i);
        }
    }
    if (best != kInvalidDiskPage) { return best; }

    std::uint64_t min_freq = UINT64_MAX;
    std::uint64_t min_tick = UINT64_MAX;
    best = kInvalidDiskPage;
    for (std::size_t i = 0; i < count; ++i) {
        const DiskSlab* s = slab(static_cast<DiskPageId>(offset + i));
        if (s == nullptr || !s->occupied || s->pinned) { continue; }
        if (s->freq < min_freq || (s->freq == min_freq && s->last_hit_tick < min_tick)) {
            min_freq = s->freq;
            min_tick = s->last_hit_tick;
            best     = static_cast<DiskPageId>(offset + i);
        }
    }
    return best;
}

DiskSlotAllocation DiskSlabPool::allocate_block() {
    DiskSlotAllocation result;
    if (!enabled()) { return result; }
    DiskPageId page = find_free(0, block_slabs_.size());
    if (page == kInvalidDiskPage) {
        page = pick_victim(0, block_slabs_.size());
        if (page == kInvalidDiskPage) { return result; }
        result.evicted = page;
    }
    occupy(page);
    result.page = page;
    return result;
}

DiskSlotAllocation DiskSlabPool::allocate_snapshot() {
    DiskSlotAllocation result;
    if (!enabled()) { return result; }
    const std::size_t offset = block_slabs_.size();
    DiskPageId page = find_free(offset, snapshot_slabs_.size());
    if (page == kInvalidDiskPage) {
        page = pick_victim(offset, snapshot_slabs_.size());
        if (page == kInvalidDiskPage) { return result; }
        result.evicted = page;
    }
    occupy(page);
    result.page = page;
    return result;
}

void DiskSlabPool::release(DiskPageId page) noexcept {
    DiskSlab* slab_ptr = slab_mut(page);
    if (slab_ptr != nullptr) {
        reset_slab(*slab_ptr);
    }
}

void DiskSlabPool::occupy_loaded(DiskPageId page, bool dead, bool superseded, std::uint64_t freq,
                                 std::uint64_t last_hit_tick, std::uint32_t generation) noexcept {
    DiskSlab* slab_ptr = slab_mut(page);
    if (slab_ptr == nullptr) { return; }
    slab_ptr->occupied      = true;
    slab_ptr->pinned        = false;
    slab_ptr->dead          = dead;
    slab_ptr->dead_at       = dead ? ++tick_ : 0;
    slab_ptr->superseded    = superseded;
    slab_ptr->superseded_at = superseded ? ++tick_ : 0;
    slab_ptr->freq          = freq;
    slab_ptr->last_hit_tick = last_hit_tick;
    slab_ptr->generation    = generation;
    if (last_hit_tick > tick_) { tick_ = last_hit_tick; }
}

void DiskSlabPool::clear() noexcept {
    for (DiskSlab& s : block_slabs_) { reset_slab(s); }
    for (DiskSlab& s : snapshot_slabs_) { reset_slab(s); }
    tick_ = 0;
}

std::size_t DiskSlabPool::slot_offset(DiskPageId page) const noexcept {
    const auto idx = static_cast<std::size_t>(page);
    if (idx < block_slabs_.size()) {
        return layout_.block_offset + idx * layout_.block_stride;
    }
    return layout_.snapshot_offset + (idx - block_slabs_.size()) * layout_.snapshot_stride;
}

std::size_t DiskSlabPool::slot_payload_bytes(DiskPageId page) const noexcept {
    if (page < 0) { return 0; }
    const auto idx = static_cast<std::size_t>(page);
    return idx < block_slabs_.size() ? options_.block_slab_bytes : options_.snapshot_slab_bytes;
}

std::uint32_t DiskSlabPool::generation(DiskPageId page) const noexcept {
    const DiskSlab* s = slab(page);
    return s != nullptr ? s->generation : 0U;
}

bool DiskSlabPool::slot_valid(DiskPageId page) const noexcept { return valid(page); }

bool DiskSlabPool::write_slot(DiskPageId page, const void* src) noexcept {
    if (!enabled() || !valid(page) || src == nullptr) { return false; }
    const std::size_t bytes = slot_payload_bytes(page);
    const std::size_t off   = slot_offset(page);
    const std::uint64_t checksum = checksum_bytes(src, bytes);

    SlotHeader header;
    header.kind          = static_cast<std::size_t>(page) < block_slabs_.size() ? 0U : 1U;
    header.generation    = generation(page);
    header.payload_bytes = bytes;
    header.checksum      = checksum;

    // Payload first, header last: a torn write leaves the previous header, which the loader then
    // rejects against the record it names.
    if (!pwrite_all(fd_, src, bytes, off + kSlotHeaderBytes)) { return false; }
    if (!pwrite_all(fd_, &header, sizeof(header), off)) { return false; }
    return true;
}

bool DiskSlabPool::read_slot(DiskPageId page, std::uint32_t expected_generation, void* dst) noexcept {
    if (!enabled() || !valid(page) || dst == nullptr) { return false; }
    const std::size_t bytes = slot_payload_bytes(page);
    const std::size_t off   = slot_offset(page);
    SlotHeader header;
    if (!pread_all(fd_, &header, sizeof(header), off)) { return false; }
    const std::uint32_t kind = static_cast<std::size_t>(page) < block_slabs_.size() ? 0U : 1U;
    if (header.magic != kSlotMagic || header.kind != kind ||
        header.generation != expected_generation || header.payload_bytes != bytes) {
        return false;
    }
    if (!pread_all(fd_, dst, bytes, off + kSlotHeaderBytes)) { return false; }
    return checksum_bytes(dst, bytes) == header.checksum;
}

bool DiskSlabPool::validate_slot(DiskPageId page, std::uint32_t expected_generation) const noexcept {
    if (!enabled() || !valid(page)) { return false; }
    const std::size_t bytes = slot_payload_bytes(page);
    const std::size_t off   = slot_offset(page);
    SlotHeader header;
    if (!pread_all(fd_, &header, sizeof(header), off)) { return false; }
    const std::uint32_t kind = static_cast<std::size_t>(page) < block_slabs_.size() ? 0U : 1U;
    return header.magic == kSlotMagic && header.kind == kind &&
           header.generation == expected_generation && header.payload_bytes == bytes;
}

void DiskSlabPool::pin(DiskPageId page) noexcept {
    DiskSlab* s = slab_mut(page);
    if (s != nullptr && s->occupied) { s->pinned = true; }
}

void DiskSlabPool::unpin(DiskPageId page) noexcept {
    DiskSlab* s = slab_mut(page);
    if (s != nullptr && s->occupied) { s->pinned = false; }
}

void DiskSlabPool::mark_dead(DiskPageId page) noexcept {
    DiskSlab* s = slab_mut(page);
    if (s != nullptr && s->occupied && !s->dead) {
        s->dead    = true;
        s->dead_at = ++tick_;
    }
}

void DiskSlabPool::mark_alive(DiskPageId page) noexcept {
    DiskSlab* s = slab_mut(page);
    if (s != nullptr && s->occupied) {
        s->dead    = false;
        s->dead_at = 0;
    }
}

void DiskSlabPool::supersede(DiskPageId page) noexcept {
    DiskSlab* s = slab_mut(page);
    if (s != nullptr && s->occupied && !s->superseded) {
        s->superseded    = true;
        s->superseded_at = ++tick_;
    }
}

void DiskSlabPool::clear_superseded(DiskPageId page) noexcept {
    DiskSlab* s = slab_mut(page);
    if (s != nullptr && s->occupied) {
        s->superseded    = false;
        s->superseded_at = 0;
    }
}

void DiskSlabPool::hit(DiskPageId page, std::uint64_t tick) noexcept {
    DiskSlab* s = slab_mut(page);
    if (s == nullptr || !s->occupied) { return; }
    ++s->freq;
    s->last_hit_tick = tick;
}

} // namespace ninfer::prefix_cache
