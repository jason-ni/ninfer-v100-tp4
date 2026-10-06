#pragma once

// Pinned host memory slab pool for the hybrid prefix cache host tier (M2).
//
// Holds KV blocks and snapshot images in pinned host RAM. Allocations are fixed-size slabs:
//   - Block slabs: one per KV page group (64 tokens); sized from the real pool layout per rank.
//   - Snapshot slabs: one per state image (GDN recurrent + conv history); sized from the real
//     linear-attention layout per rank.
//
// The pool is storage only: it carves a pinned allocation into slabs and tracks occupancy plus
// GDSF eviction metadata. All policy decisions (which node/snapshot owns a slab, when a slab is
// dead or superseded) live in PrefixIndex, which calls mark_dead/mark_alive/supersede/hit/pin.
// Eviction order inside the pool: pinned slabs are never touched; among the rest, dead slabs
// first (oldest), then superseded slabs (oldest), then by (frequency, recency) — GreedyDual-
// Size-Frequency.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::prefix_cache {

using HostPageId = std::int32_t;

inline constexpr HostPageId kInvalidHostPage = -1;

// One slab in the host pool. Each slab is a fixed-size pinned buffer.
struct HostSlab {
    void* data = nullptr;
    std::size_t bytes = 0;
    bool occupied = false;
    // In-flight transfer destination/source; never evicted while pinned.
    bool pinned = false;
    // Dead-KV: unreachable from any live snapshot; evicted before live slabs.
    bool dead = false;
    std::uint64_t dead_at = 0;
    // Superseded by a newer version of the same state; evicted before live slabs. Sticky until
    // the slab is freed and re-allocated.
    bool superseded = false;
    std::uint64_t superseded_at = 0;
    // GDSF recency/frequency metadata.
    std::uint64_t freq = 0;
    std::uint64_t last_hit_tick = 0;
};

struct HostSlabPoolOptions {
    // Total pinned memory in bytes. 0 = disabled (device-only mode).
    std::size_t total_bytes = 0;
    // Size of one KV block slab in bytes (per rank).
    std::size_t block_slab_bytes = 0;
    // Size of one snapshot image slab in bytes (per rank).
    std::size_t snapshot_slab_bytes = 0;
    // Number of block slabs to carve.
    std::size_t block_slab_count = 0;
    // Number of snapshot slabs to carve.
    std::size_t snapshot_slab_count = 0;
};

// Result of a slab allocation: the page obtained, and the page that was evicted to make room
// (kInvalidHostPage when none). The caller must reconcile the evicted page's owner before the
// new slab's data is written.
struct SlabAllocation {
    HostPageId page = kInvalidHostPage;
    HostPageId evicted = kInvalidHostPage;
};

class HostSlabPool {
public:
    explicit HostSlabPool(HostSlabPoolOptions options);
    ~HostSlabPool();

    HostSlabPool(const HostSlabPool&) = delete;
    HostSlabPool& operator=(const HostSlabPool&) = delete;

    [[nodiscard]] bool enabled() const noexcept { return block_slabs_ != nullptr; }
    [[nodiscard]] std::size_t total_bytes() const noexcept { return options_.total_bytes; }
    [[nodiscard]] std::size_t used_bytes() const noexcept;
    [[nodiscard]] std::size_t free_bytes() const noexcept;
    [[nodiscard]] std::size_t block_slab_count() const noexcept { return block_slab_count_; }
    [[nodiscard]] std::size_t snapshot_slab_count() const noexcept { return snapshot_slab_count_; }

    // Allocate a block slab, evicting another BLOCK slab if the pool is full. Returns
    // kInvalidHostPage when no block slab can be obtained (all pinned).
    [[nodiscard]] SlabAllocation allocate_block();
    // Allocate a snapshot slab, evicting another SNAPSHOT slab if the pool is full.
    [[nodiscard]] SlabAllocation allocate_snapshot();

    // Release a slab back to the free list. Resets all eviction metadata.
    void release(HostPageId page) noexcept;

    // Raw pointer and size of a slab's data (for D2H/H2D transfers).
    [[nodiscard]] void* data(HostPageId page) const noexcept;
    [[nodiscard]] std::size_t bytes(HostPageId page) const noexcept;

    // --- eviction metadata (driven by PrefixIndex policy) ---

    // Protect a slab from eviction while a transfer is in flight.
    void pin(HostPageId page) noexcept;
    void unpin(HostPageId page) noexcept;

    // Mark a slab dead (no live snapshot below its node) or alive again.
    void mark_dead(HostPageId page) noexcept;
    void mark_alive(HostPageId page) noexcept;

    // Mark a slab superseded by a newer version (sticky until re-allocated).
    void supersede(HostPageId page) noexcept;

    // Clear the superseded mark after the owner is hit again.
    void clear_superseded(HostPageId page) noexcept;

    // Record a hit: bumps frequency and recency. Does not clear dead/superseded.
    void hit(HostPageId page, std::uint64_t tick) noexcept;

    // Slab metadata access.
    [[nodiscard]] const HostSlab* slab(HostPageId page) const noexcept;
    [[nodiscard]] HostSlab* slab_mut(HostPageId page) noexcept;

private:
    HostPageId find_free(std::size_t offset, std::size_t count) noexcept;
    // Pick a victim among slabs [offset, offset+count): dead first (oldest), then superseded
    // (oldest), then (min freq, oldest tick). Pinned and unoccupied slabs are skipped.
    HostPageId pick_victim(std::size_t offset, std::size_t count) noexcept;
    void occupy(HostPageId page) noexcept;
    void reset_slab(HostSlab& slab) noexcept;

    HostSlabPoolOptions options_;
    HostSlab* block_slabs_ = nullptr;
    HostSlab* snapshot_slabs_ = nullptr;
    std::size_t block_slab_count_ = 0;
    std::size_t snapshot_slab_count_ = 0;
    std::uint64_t tick_ = 0;
};

} // namespace ninfer::prefix_cache
