#pragma once

// Disk tier for the hybrid prefix cache (M3a): a bounded, crash-durable lower tier below the
// pinned host tier. See docs/maintainer/hybrid-prefix-cache-plan.md section "M3a".
//
// Storage is a single preallocated, fixed-slot file. A block region and a snapshot region mirror
// the host tier's slab geometry (each slot is `tp * per_rank_bytes`), so spilling a host slab and
// restoring one are straight whole-slot copies and a slot's file offset is O(1):
//
//     [ file header ][ block slot 0 | block slot 1 | ... ][ snapshot slot 0 | ... ]
//
// Each slot is `[slot header | payload]`. The payload is written first and the slot header last,
// so a torn write (process killed mid-payload) leaves a header whose generation/checksum no longer
// match the record the index names; the loader drops that record. Eviction metadata (dead,
// superseded, GDSF frequency/recency) mirrors HostSlabPool so the two tiers share one policy.
//
// The file header carries a per-creation uid. The container stores that uid, so a slab file that
// was reinitialized (different geometry, or replaced) never validates against a stale index.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace ninfer::prefix_cache {

using DiskPageId = std::int32_t;

inline constexpr DiskPageId kInvalidDiskPage = -1;

// One slot in the disk pool. Data lives in the file; this is metadata only.
struct DiskSlab {
    bool occupied = false;
    // In-flight transfer; never evicted while pinned.
    bool pinned = false;
    // Dead-KV: unreachable from any live snapshot; evicted before live slots.
    bool dead = false;
    std::uint64_t dead_at = 0;
    // Superseded by a newer version of the same state; evicted before live slabs.
    bool superseded = false;
    std::uint64_t superseded_at = 0;
    // GDSF recency/frequency metadata.
    std::uint64_t freq = 0;
    std::uint64_t last_hit_tick = 0;
    // Bumped on every reuse; identifies the record currently occupying the slot.
    std::uint32_t generation = 0;
};

struct DiskSlabPoolOptions {
    // 0 = disabled.
    std::size_t total_bytes = 0;
    std::size_t block_slab_bytes = 0;
    std::size_t snapshot_slab_bytes = 0;
    std::size_t block_slab_count = 0;
    std::size_t snapshot_slab_count = 0;
    // Slab data file. Parent directories are created if needed.
    std::filesystem::path file;
};

// Result of allocating a slot: the page obtained and the page evicted to make room
// (kInvalidDiskPage when none). The caller reconciles the evicted page's owner before the new
// payload is written.
struct DiskSlotAllocation {
    DiskPageId page = kInvalidDiskPage;
    DiskPageId evicted = kInvalidDiskPage;
};

class DiskSlabPool {
public:
    explicit DiskSlabPool(const DiskSlabPoolOptions& options);
    ~DiskSlabPool();

    DiskSlabPool(const DiskSlabPool&) = delete;
    DiskSlabPool& operator=(const DiskSlabPool&) = delete;

    [[nodiscard]] bool enabled() const noexcept { return fd_ >= 0; }
    [[nodiscard]] std::size_t total_bytes() const noexcept { return options_.total_bytes; }
    [[nodiscard]] std::size_t pinned_bytes() const noexcept { return pinned_bytes_; }
    [[nodiscard]] std::size_t used_bytes() const noexcept;
    [[nodiscard]] std::size_t free_bytes() const noexcept;
    [[nodiscard]] std::size_t block_slab_count() const noexcept { return block_slabs_.size(); }
    [[nodiscard]] std::size_t snapshot_slab_count() const noexcept { return snapshot_slabs_.size(); }
    [[nodiscard]] std::size_t block_slab_bytes() const noexcept { return options_.block_slab_bytes; }
    [[nodiscard]] std::size_t snapshot_slab_bytes() const noexcept {
        return options_.snapshot_slab_bytes;
    }
    // Identity of the current slab file, or 0 when disabled. Stored in the index so a replaced or
    // reinitialized file never validates against a stale index.
    [[nodiscard]] std::uint64_t file_uid() const noexcept { return file_uid_; }

    // Allocate a slot, evicting another slot of the same kind when the region is full. Returns
    // kInvalidDiskPage when no slot can be obtained (all pinned).
    [[nodiscard]] DiskSlotAllocation allocate_block();
    [[nodiscard]] DiskSlotAllocation allocate_snapshot();

    // Return a slot to the free list. Resets all eviction metadata.
    void release(DiskPageId page) noexcept;

    // Load-time: mark a specific free slot occupied, adopt its eviction metadata and generation.
    void occupy_loaded(DiskPageId page, bool dead, bool superseded, std::uint64_t freq,
                       std::uint64_t last_hit_tick, std::uint32_t generation) noexcept;

    // Load-time: return every slot to the free state.
    void clear() noexcept;

    // --- payload I/O ---

    // Write `bytes(page)` bytes from `src` into the slot payload, then commit the slot header
    // (magic, kind, current generation, checksum) last. Returns false on any I/O error.
    [[nodiscard]] bool write_slot(DiskPageId page, const void* src) noexcept;
    // Read the slot payload into `dst`, validating that the on-disk header matches the expected
    // generation and payload checksum. Returns false when the header is stale or corrupt.
    [[nodiscard]] bool read_slot(DiskPageId page, std::uint32_t expected_generation,
                                 void* dst) noexcept;
    // Load-time: check only the on-disk slot header (magic/kind/generation/payload size) without
    // reading the payload. The header is committed last, so a matching header proves the payload
    // write for that generation completed; a stale header marks the slot as not part of the record.
    [[nodiscard]] bool validate_slot(DiskPageId page, std::uint32_t expected_generation) const noexcept;

    [[nodiscard]] std::uint32_t generation(DiskPageId page) const noexcept;
    [[nodiscard]] bool slot_valid(DiskPageId page) const noexcept;
    [[nodiscard]] std::size_t slot_payload_bytes(DiskPageId page) const noexcept;

    // --- eviction metadata (driven by PrefixIndex policy) ---

    void pin(DiskPageId page) noexcept;
    void unpin(DiskPageId page) noexcept;
    void mark_dead(DiskPageId page) noexcept;
    void mark_alive(DiskPageId page) noexcept;
    void supersede(DiskPageId page) noexcept;
    void clear_superseded(DiskPageId page) noexcept;
    void hit(DiskPageId page, std::uint64_t tick) noexcept;

    [[nodiscard]] const DiskSlab* slab(DiskPageId page) const noexcept;

private:
    struct Layout {
        std::size_t file_header_bytes = 0;
        std::size_t slot_header_bytes = 0;
        std::size_t block_offset = 0;
        std::size_t snapshot_offset = 0;
        std::size_t block_stride = 0;
        std::size_t snapshot_stride = 0;
        std::size_t file_size = 0;
    };

    [[nodiscard]] bool open_file();
    [[nodiscard]] DiskPageId find_free(std::size_t offset, std::size_t count) noexcept;
    [[nodiscard]] DiskPageId pick_victim(std::size_t offset, std::size_t count) noexcept;
    void occupy(DiskPageId page) noexcept;
    void reset_slab(DiskSlab& slab) noexcept;
    [[nodiscard]] bool valid(DiskPageId page) const noexcept;
    [[nodiscard]] DiskSlab* slab_mut(DiskPageId page) noexcept;
    [[nodiscard]] std::size_t slot_offset(DiskPageId page) const noexcept;

    DiskSlabPoolOptions options_;
    Layout layout_;
    std::vector<DiskSlab> block_slabs_;
    std::vector<DiskSlab> snapshot_slabs_;
    int fd_ = -1;
    std::uint64_t file_uid_ = 0;
    std::size_t pinned_bytes_ = 0;
    std::uint64_t tick_ = 0;
};

} // namespace ninfer::prefix_cache
