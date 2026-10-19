#pragma once

// Model-agnostic content-addressed prefix index for the hybrid prefix cache.
//
// This is the host-side half of the design in docs/maintainer/hybrid-prefix-cache-plan.md
// (HPC sections 5-7, 9). It owns no device memory and performs no transfers; it tracks the
// block tree, snapshot table, device KV LRU and device snapshot slots. The consuming Program
// supplies physical pages when a block or tail is committed and receives them back through a
// PrefixPageSink when the index drops its last copy.

#include <ninfer/types.h>
#include "runtime/prefix_cache/disk_slab_pool.h"
#include "runtime/prefix_cache/host_slab_pool.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace ninfer::prefix_cache {

// M3 disk-tier index container (defined in prefix_store.h). Forward-declared so the index can
// expose its state to the serializer without pulling the file-format header into every TU.
struct PrefixStoreGeometry;
struct PrefixStoreStats;
struct PrefixStore;

// One tree node / snapshot tail covers exactly one PagedKV page.
inline constexpr std::int32_t kBlockTokens = 64;
// Enabled KV pools per rank: the text pool always, the speculative backend pool when the
// artifact owns one. The plan's `kMaxPools`.
inline constexpr std::size_t kMaxPools = 2;

using NodeId     = std::uint32_t;
using SnapshotId = std::uint32_t;
using SlotId     = std::int32_t;

inline constexpr NodeId kInvalidNode         = 0;
inline constexpr SnapshotId kInvalidSnapshot = 0;
inline constexpr SlotId kInvalidSlot         = -1;

// Physical page ids, one per enabled pool; -1 marks the pool absent for this record.
struct PoolPages {
    std::array<std::int32_t, kMaxPools> page{-1, -1};
};

// Chain a parent's lookup hash with one block's exact tokens and media key. The hash only
// selects a child; equality is always decided by an exact token/media compare.
[[nodiscard]] std::uint64_t block_hash(std::uint64_t parent_hash, std::span<const TokenId> tokens,
                                       std::uint64_t extra) noexcept;

// Root seed; see HPC section 5.1 (`lookup_hash(-1) = seed`).
inline constexpr std::uint64_t kBlockHashSeed = 0x243F'6A88'85A3'08D3ULL;

// Parent-chained lookup hash for every full block of `tokens`. The result has
// `tokens.size() / kBlockTokens` entries; the chain starts from `kBlockHashSeed`. `extras`, when
// non-empty, must supply one media key per full block, else every block uses key 0.
[[nodiscard]] std::vector<std::uint64_t>
chain_block_hashes(std::span<const TokenId> tokens, std::span<const std::uint64_t> extras = {});

enum class SnapshotKind : std::uint8_t {
    Tap,            // page-aligned snapshot taken by the tap planner (tail_len == 0)
    Endpoint,       // request finish/cancel at an exact frontier; owns the partial last page
    OutputBoundary, // exact frontier child of the generation opener
    Boundary,       // explicit client breakpoint or structural boundary, shareable
};

// A node may own a device copy, a host copy, a disk copy, no copy, or a copy being filled in.
enum class Residency : std::uint8_t {
    Absent,   // no copy exists anywhere
    Resident, // device-resident
    Host,     // host-resident (data in pinned host memory)
    Disk,     // disk-resident (data in the disk slab file; not directly device-addressable)
    Filling,  // in-flight transfer (D2H or H2D)
};

struct BlockNode {
    NodeId parent             = kInvalidNode;
    std::int32_t depth        = -1; // block index b (node covers [64b, 64b+64)); root is -1
    std::uint64_t lookup_hash = 0;
    std::uint64_t extra       = 0;
    std::array<TokenId, kBlockTokens> tokens{};
    std::vector<NodeId> children;

    PoolPages device_pages{};
    Residency device_state = Residency::Absent;

    // Host tier (M2): pinned host slab holding this block's KV data.
    HostPageId host_page = kInvalidHostPage;
    Residency host_state = Residency::Absent;

    // Disk tier (M3a): slot in the disk slab file holding this block's KV data. Populated when the
    // host copy is spilled (evicted) and on load; dropped when the record is erased. Only the
    // generation stored in the pool identifies the record currently occupying the slot.
    DiskPageId disk_page = kInvalidDiskPage;
    Residency disk_state = Residency::Absent;

    std::uint32_t active_refs          = 0; // active sequences whose table maps this node
    std::uint32_t restore_pins         = 0; // in-flight transfers reading/writing this node
    std::uint32_t live_snapshots_below = 0; // snapshots whose path includes this node
    std::uint64_t last_access_tick     = 0;

    // Intrusive device LRU. Linked iff active_refs == 0 && device_state == Resident.
    NodeId lru_prev = kInvalidNode;
    NodeId lru_next = kInvalidNode;
    bool lru_linked = false;

    std::vector<SnapshotId> snapshots; // snapshots anchored here

    std::uint32_t generation = 1;
    bool live                = true;

    // True when a host or disk copy exists and can back the device copy (safe to demote on
    // eviction). A disk-backed node can be materialized back into a host slab on demand.
    [[nodiscard]] bool is_backed() const noexcept {
        return host_state == Residency::Host || disk_state == Residency::Disk;
    }
};

struct Snapshot {
    NodeId anchor      = kInvalidNode;
    std::uint32_t frontier = 0; // F: state is after tokens [0,F)
    std::uint8_t tail_len  = 0; // F - 64*(anchor.depth+1)
    std::array<TokenId, kBlockTokens - 1> tail_tokens{};

    PoolPages tail_pages{};
    Residency tail_state = Residency::Absent;

    SlotId device_slot = kInvalidSlot; // index into the D device snapshot slots
    Residency device_state = Residency::Absent;

    // Host tier (M2): pinned host slab holding this snapshot's state image.
    HostPageId host_page = kInvalidHostPage;
    Residency host_state = Residency::Absent;

    // Disk tier (M3a): slot in the disk slab file holding this snapshot's state image.
    DiskPageId disk_page = kInvalidDiskPage;
    Residency disk_state = Residency::Absent;

    SnapshotKind kind = SnapshotKind::Tap;
    bool superseded   = false;
    std::uint64_t superseded_at = 0; // tick when superseded; orders slot reclamation
    std::uint32_t restore_pins  = 0; // requests currently using this state image

    // Host eviction (M2); kept so snapshot ordering is stable across milestones.
    std::uint64_t gdsf_h      = 0;
    std::uint32_t freq        = 1;
    std::uint64_t last_hit_tick = 0;

    std::uint32_t generation = 1;
    bool live                = true;

    [[nodiscard]] bool device_resident() const noexcept {
        return device_state == Residency::Resident;
    }
    [[nodiscard]] bool is_backed() const noexcept {
        return host_state == Residency::Host || disk_state == Residency::Disk;
    }
};

// Returned when the index drops the last reference to physical storage. Device eviction never
// copies: a backed node demotes (host copy survives), an unbacked node erases its subtree.
class PrefixPageSink {
public:
    virtual ~PrefixPageSink() = default;
    virtual void release_block_pages(const PoolPages& pages) noexcept = 0;
    virtual void release_tail_pages(const PoolPages& pages) noexcept  = 0;
    virtual void release_snapshot_slot(SlotId slot) noexcept          = 0;
};

struct PrefixIndexOptions {
    // D in the plan: device state-image slots, C + 1 with a host tier, C + 2 without.
    std::uint32_t device_snapshot_slots = 0;
    std::uint64_t now_tick              = 0;
};

// A request to publish a block or a snapshot. Page ownership transfers to the index on success.
struct BlockCommit {
    NodeId parent               = kInvalidNode;
    std::uint64_t lookup_hash   = 0;
    std::uint64_t extra         = 0;
    std::span<const TokenId> tokens; // exactly kBlockTokens
    PoolPages pages{};
};

struct SnapshotRequest {
    NodeId anchor         = kInvalidNode;
    std::uint32_t frontier = 0;
    std::span<const TokenId> tail_tokens{}; // exactly frontier - 64*(anchor.depth+1) if non-empty
    PoolPages tail_pages{};
    SnapshotKind kind = SnapshotKind::Tap;
    // Device-only snapshots may be evicted by a newer tap whose estimated value is at least this.
    std::uint64_t priority = 0;
    // The snapshot this request's lineage previously resumed from; superseded before a new slot is
    // taken. kInvalidSnapshot when none.
    SnapshotId supersede = kInvalidSnapshot;
    // A device slot previously obtained from reserve_snapshot_slot() whose state image has already
    // been written. When set, add_snapshot uses this slot instead of taking a fresh one.
    SlotId reserved_slot = kInvalidSlot;
};

// The unique child under `parent` matching `lookup_hash` with equal tokens and extra.
struct ChildMatch {
    NodeId node       = kInvalidNode;
    bool hash_collided = false; // a child had the hash but differed in tokens/extra
};

struct SnapshotCandidate {
    SnapshotId id      = kInvalidSnapshot;
    std::uint32_t frontier = 0;
    bool device_resident   = false;
    bool host_resident     = false;
    bool disk_resident     = false;
};

struct PrefixMatch {
    std::size_t matched_blocks = 0;
    std::vector<NodeId> path;              // size matched_blocks + 1; path[0] == root
    std::vector<SnapshotCandidate> candidates;
    // Deepest candidate with F <= n-1 whose storage is fully device resident, or invalid.
    SnapshotId best_device = kInvalidSnapshot;
    std::uint32_t best_device_frontier = 0;
    // Deepest candidate with F <= n-1 whose image is host resident and whose path is restorable
    // (every node has a device or host copy), or invalid.
    SnapshotId best_host = kInvalidSnapshot;
    std::uint32_t best_host_frontier = 0;
    // Deepest candidate with F <= n-1 whose image and path are restorable from a device, host, or
    // disk copy (a disk-only path must first be materialized into the host tier), or invalid.
    SnapshotId best_disk = kInvalidSnapshot;
    std::uint32_t best_disk_frontier = 0;

    // --- D1b: miss explanation, filled in the same walk (cheap: bounded per-snapshot work) ---
    // What happened to the block walk. Completed means every eligible block matched; the miss
    // (if any) is then at the snapshot layer. NotRun means no walk executed.
    ninfer::PrefixStallCause walk_cause = ninfer::PrefixStallCause::NotRun;
    // Live children under the stall parent: 0 => nothing was ever committed below (never cached
    // or fully evicted); >0 with HashCollision => history genuinely branched here.
    std::uint32_t stall_sibling_count = 0;
    // Full count of snapshots seen on the path; `diagnostics` holds the deepest 8 of them.
    std::uint32_t diagnostics_total = 0;
    // Every snapshot seen on the matched path and the gate that decided it, sorted by descending
    // frontier. reject codes: 1 superseded/gone, 2 frontier past prompt, 3 tail mismatch,
    // 4 no usable storage on the path, 5 outranked by the best.
    std::vector<ninfer::PrefixDiagCandidate> diagnostics;
};

// Cumulative lifetime victim statistics of the device-tier eviction policy (see evict_node).
struct EvictionCounters {
    std::uint64_t demoted                  = 0; // lossless device -> host demotions
    std::uint64_t subtree_erased           = 0; // lossy erasures outside the dead-branch GC tier
    std::uint64_t branch_gc_removed_blocks = 0; // dead-branch nodes removed by demand-aware GC
    std::uint64_t snapshots_deleted        = 0; // live snapshots lost by an eviction
};

// Result of begin_host_write: the node and the host slab that will receive its KV data. The
// caller enqueues the D2H copy and calls finish_host_write when it completes.
struct HostWriteTarget {
    NodeId node = kInvalidNode;
    HostPageId page = kInvalidHostPage;
};

// Result of begin_snapshot_write: the snapshot and the host slab that will receive its image.
struct SnapshotWriteTarget {
    SnapshotId id = kInvalidSnapshot;
    HostPageId page = kInvalidHostPage;
};

// Result of begin_snapshot_restore: the snapshot, the device slot to fill, and the host slab to
// read from. The caller enqueues the H2D copy and calls finish_snapshot_restore when done.
struct SnapshotRestoreTarget {
    SnapshotId id = kInvalidSnapshot;
    SlotId slot = kInvalidSlot;
    HostPageId page = kInvalidHostPage;
};

class PrefixIndex {
public:
    explicit PrefixIndex(PrefixIndexOptions options, PrefixPageSink* sink = nullptr);
    ~PrefixIndex();

    PrefixIndex(const PrefixIndex&)            = delete;
    PrefixIndex& operator=(const PrefixIndex&) = delete;

    [[nodiscard]] NodeId root() const noexcept { return root_; }
    [[nodiscard]] const BlockNode* node(NodeId id) const noexcept;
    [[nodiscard]] SnapshotId snapshot_id_at(std::size_t index) const noexcept;
    [[nodiscard]] const Snapshot* snapshot(SnapshotId id) const noexcept;
    [[nodiscard]] std::size_t live_node_count() const noexcept { return live_nodes_; }
    [[nodiscard]] std::size_t live_snapshot_count() const noexcept { return live_snapshots_; }
    [[nodiscard]] std::uint32_t device_snapshot_slots() const noexcept {
        return static_cast<std::uint32_t>(slot_owner_.size());
    }
    [[nodiscard]] std::uint32_t device_snapshot_slots_free() const noexcept {
        return static_cast<std::uint32_t>(free_slots_.size());
    }
    [[nodiscard]] std::uint64_t tick() const noexcept { return tick_; }

    // Raw slot capacities, for the disk-tier serializer's iteration (ids are 1-based; a slot is
    // live iff node(id)/snapshot(id) returns non-null).
    [[nodiscard]] std::size_t node_capacity() const noexcept { return nodes_.size(); }
    [[nodiscard]] std::size_t snapshot_capacity() const noexcept { return snapshots_.size(); }

    // Reset to the empty index (a fresh root). Used by the disk-tier loader when a container fails
    // validation after a partial read. Host slabs are cleared by the caller through HostSlabPool.
    void clear();

    // --- lookup ---

    // Walk full blocks from the root; only blocks b < floor((n-1)/64) are matched so at least one
    // prompt token is always prefilled. `block_hashes[b]` is the frontend-chained hash for block
    // b; `block_extras[b]` its media key (0 without Vision). Collects snapshots anchored on the
    // matched path whose tail matches the prompt and whose frontier is <= n-1.
    [[nodiscard]] PrefixMatch match(std::span<const TokenId> prompt_tokens,
                                    std::span<const std::uint64_t> block_hashes,
                                    std::span<const std::uint64_t> block_extras) const;

    // Exact child under `parent`, or invalid. `hash_collided` reports a same-hash mismatch.
    [[nodiscard]] ChildMatch find_child(NodeId parent, std::uint64_t lookup_hash,
                                        std::span<const TokenId> tokens,
                                        std::uint64_t extra) const;

    // --- mutation ---

    // Insert or adopt a committed full block. When an equal child already owns a device copy the
    // sequence page stays private: the index does not take `commit.pages` and returns it unchanged
    // (`adopted == false`). Otherwise the index owns the pages (`adopted == true`).
    NodeId commit_block(const BlockCommit& commit, bool& adopted);

    // Publish a snapshot. A duplicate (same anchor and tail tokens) is a no-op that refreshes
    // recency. Returns the snapshot id, or kInvalidSnapshot when no device slot could be taken.
    SnapshotId add_snapshot(const SnapshotRequest& request);

    // Reserve a device snapshot slot before its state image is captured, so a request can stage
    // tap images during prefill and publish them at completion. Returns kInvalidSlot if none is
    // available.
    [[nodiscard]] SlotId reserve_snapshot_slot(std::uint64_t priority);
    void release_reserved_slot(SlotId slot) noexcept;

    // Refresh a hit: clears `superseded`, bumps frequency and recency.
    void hit(SnapshotId id) noexcept;

    // Remove a snapshot and release its tail pages and device slot.
    void remove_snapshot(SnapshotId id);

    // --- pinning and eviction ---

    void pin_node(NodeId id) noexcept;   // active_refs++ / unlink from LRU
    void unpin_node(NodeId id) noexcept; // active_refs-- / relink at LRU tail
    void pin_restore(NodeId id) noexcept;
    void unpin_restore(NodeId id) noexcept;

    // Protect a snapshot from slot eviction while a request is using the state it names.
    void pin_snapshot(SnapshotId id) noexcept;
    void unpin_snapshot(SnapshotId id) noexcept;

    // Evict one unpinned device-resident node from the LRU set using the demand-aware policy:
    // tier 1 demotes the deepest non-demand backed node (lossless; keeps warm shared prefixes
    // device-resident by shedding the tails), tier 2 erases a whole dead branch (the shallowest
    // non-demand branch root that anchors no live non-superseded snapshot and hides no demand or
    // pinned data below it), tier 3 only then trims onto the demand set or the unbacked remainder,
    // deepest leaf first, demote-preferred. `protected_nodes` carries the incoming work's demand
    // (matched block paths + retained commit paths); their ANCESTORS are protected implicitly.
    // Returns the topmost evicted node, or kInvalidNode when nothing is evictable.
    [[nodiscard]] NodeId evict_node(std::span<const NodeId> protected_nodes);

    [[nodiscard]] EvictionCounters eviction_counters() const noexcept {
        return {demoted_total_, subtree_erased_total_, branch_gc_removed_blocks_total_,
                snapshots_deleted_by_eviction_total_};
    }

    // Delete `node` and its subtree, releasing pages and deleting anchored snapshots. Pinned nodes
    // are left untouched. With non-null outputs, the deleted snapshot count and removed node count
    // are reported back (eviction accounting).
    void erase_subtree(NodeId node, std::uint32_t* deleted_snapshots = nullptr,
                       std::uint32_t* removed_nodes = nullptr);

    // Set the host slab pool for M2 write-through/restore. Must be called before any
    // commit_block or match that involves host residency.
    void set_host_pool(HostSlabPool* pool) noexcept { host_pool_ = pool; }

    // Set the disk slab pool for M3a disk-resident spill/restore. Optional; when null or disabled
    // the disk tier is inactive and host eviction simply drops the record as before.
    void set_disk_pool(DiskSlabPool* pool) noexcept { disk_pool_ = pool; }
    [[nodiscard]] DiskSlabPool* disk_pool() const noexcept { return disk_pool_; }

    // --- M3a disk tier ---

    // Ensure a disk copy of a node's host-resident KV data. Allocates a disk slot (evicting another
    // per disk policy), writes the host slab's bytes, and marks disk_state=Disk. Returns true when
    // the node is disk-resident on return (already was, or the spill succeeded). No-op (false) when
    // the node has no host copy or the disk tier is inactive.
    bool spill_node(NodeId id);
    bool spill_snapshot(SnapshotId id);

    // Ensure a host copy for a record that is currently disk-only (materialize disk -> host slab so
    // the existing H2D path can restore it). Returns the host page, or kInvalidHostPage when the
    // record has no disk copy, the disk read failed, or the host tier is unavailable. Records that
    // are already host- or device-resident return their host page / kInvalidHostPage respectively.
    [[nodiscard]] HostPageId materialize_node_host(NodeId id);
    [[nodiscard]] HostPageId materialize_snapshot_host(SnapshotId id);

    // --- M2 async fill APIs ---
    //
    // Each begin_* reserves its destination, marks the record Filling, and pins it (and, for a
    // snapshot write, its anchor node) against eviction while the transfer is in flight. The
    // caller enqueues the copies on its own streams and calls the matching finish_* exactly once;
    // finish(ok=false) rolls the reservation back.

    // Begin D2H write-through of a device-resident node that has no host copy yet. Allocates a
    // host slab, marks host_state=Filling, pins the node and the slab against eviction. Returns
    // the host page for the caller to enqueue the D2H copy, or kInvalidHostPage if not possible.
    [[nodiscard]] HostWriteTarget begin_host_write(NodeId id);

    // Complete (or abort) a D2H write-through. ok=true marks host_state=Host; ok=false releases
    // the slab again. Unpins the node either way.
    void finish_host_write(NodeId id, bool ok) noexcept;

    // Begin H2D restore of a host-resident node to freshly allocated device pages. Caller supplies
    // the pages (from pool.acquire_cache_pages). Marks device_state=Filling, pins the node and the
    // source slab. No-op when the node is not device-absent.
    void begin_device_fill(NodeId id, PoolPages pages) noexcept;

    // Complete (or abort) an H2D restore. ok=true marks device_state=Resident and re-links the
    // node into the eviction LRU; ok=false releases the pages through the sink.
    void finish_device_fill(NodeId id, bool ok) noexcept;

    // Begin D2H write-through of a snapshot image that has no host copy yet. Allocates a host
    // slab, marks host_state=Filling, pins the snapshot and its anchor node against eviction.
    // Returns the host page for the caller to enqueue the D2H copy, or kInvalidHostPage if not
    // possible.
    [[nodiscard]] SnapshotWriteTarget begin_snapshot_write(SnapshotId id);

    // Complete (or abort) a snapshot image D2H write-through. ok=true marks host_state=Host;
    // ok=false releases the slab again. Unpins the anchor node either way.
    void finish_snapshot_write(SnapshotId id, bool ok) noexcept;

    // Begin H2D restore of a snapshot image to a device slot. Acquires a slot (evicting per GDSF
    // if needed), marks device_state=Filling, pins the snapshot and its source slab. Returns the
    // slot and host page for the caller to enqueue the H2D copy, or kInvalidSlot if not possible.
    [[nodiscard]] SnapshotRestoreTarget begin_snapshot_restore(SnapshotId id);

    // Complete (or abort) a snapshot image H2D restore. ok=true marks device_state=Resident;
    // ok=false returns the slot to the free list.
    void finish_snapshot_restore(SnapshotId id, bool ok) noexcept;

    // Pages the admission check may reclaim while preserving the demand path: one entry per pool,
    // counting LRU-linked nodes that own a page in that pool and are not on the protected demand
    // path. `protected_nodes` must be the SAME set later passed to `evict_node()`. The count must
    // never exceed what eviction can release WITHOUT breaking that path: an unbacked protected
    // node (no host/disk copy) cannot be demoted, and erasing it would kill a demanded prefix
    // another request still addresses, so crediting it would admit a reservation the pool cannot
    // satisfy (-> std::bad_alloc). Excluding the whole protected closure can under-credit a
    // demanded tail that eviction would only demote as a last resort; that is safe (it can only
    // refuse an admission), whereas over-crediting is not. Passing an empty span credits every
    // LRU-linked page.
    [[nodiscard]] std::array<std::uint32_t, kMaxPools>
    evictable_page_counts(std::span<const NodeId> protected_nodes);

    [[nodiscard]] std::uint32_t evictable_pages() const noexcept { return evictable_pages_; }

private:
    friend struct PrefixStore;
    BlockNode& node_mut(NodeId id) noexcept { return nodes_[id - 1U]; }
    [[nodiscard]] bool valid(NodeId id) const noexcept {
        return id != kInvalidNode && id <= nodes_.size() && nodes_[id - 1U].live;
    }
    NodeId allocate_node();
    void free_node(NodeId id) noexcept;

    void link_lru_tail(NodeId id) noexcept;
    void unlink_lru(NodeId id) noexcept;

    // Mark a snapshot superseded, making it the next slot victim.
    void supersede(SnapshotId id) noexcept;

    // Acquire a device snapshot slot for a new snapshot, evicting per HPC section 9.2.
    SlotId acquire_slot(std::uint64_t priority);

    // Remove a snapshot's bookkeeping and release its tail pages, returning its device slot
    // without releasing it (used when the slot is immediately reused by a newer snapshot).
    SlotId detach_snapshot(SnapshotId id);
    // Release a snapshot's device slot and tail pages but keep it alive as host-resident. Used when
    // the victim has a host copy that should survive the device-slot eviction. Returns the freed
    // slot (caller clears slot_owner_ and returns it to the free list).
    SlotId demote_snapshot_device(SnapshotId id);
    void delete_snapshot(SnapshotId id);
    void dec_live_below(NodeId node, std::uint32_t count) noexcept;
    void inc_live_below(NodeId node) noexcept;

    // The slab pool evicted `page` to make room. Clear the owner's host reference (the slab is
    // already freed by the pool, so it must NOT be released again) and, if the owner now holds no
    // data anywhere and is unpinned, drop it (erase subtree / delete snapshot).
    void reconcile_host_eviction(HostPageId page) noexcept;

    // The disk pool evicted `page` to make room. Clear the owner's disk reference (the slot is
    // already freed by the pool, so it must NOT be released again) and, if the owner now holds no
    // copy anywhere and is unpinned, drop it.
    void reconcile_disk_eviction(DiskPageId page) noexcept;

    // Spill a record's host slab into a disk slot before the host slab is cleared. Called from
    // reconcile_host_eviction (while the host page still holds the data) and from the flush sweep.
    void spill_node_from_host(NodeId id, HostPageId host_page) noexcept;
    void spill_snapshot_from_host(SnapshotId id, HostPageId host_page) noexcept;

    PrefixIndexOptions options_;
    PrefixPageSink* sink_ = nullptr;
    HostSlabPool* host_pool_ = nullptr; // M2: pinned host memory for write-through/restore
    DiskSlabPool* disk_pool_ = nullptr; // M3a: disk slab file for the lower tier

    std::vector<BlockNode> nodes_;
    std::vector<Snapshot> snapshots_; // index + 1 == SnapshotId
    std::vector<NodeId> free_nodes_;
    std::vector<SnapshotId> free_snapshots_;
    std::size_t live_nodes_     = 0;
    std::size_t live_snapshots_ = 0;

    NodeId root_ = kInvalidNode;

    NodeId lru_head_ = kInvalidNode; // eviction end (oldest)
    NodeId lru_tail_ = kInvalidNode; // recency end (newest)

    // Device snapshot slots: slot_owner[s] is the snapshot holding slot s, or invalid.
    std::vector<SnapshotId> slot_owner_;
    std::vector<SlotId> free_slots_;

    std::uint64_t tick_            = 0;
    std::uint32_t evictable_pages_ = 0;

    // Demand-aware eviction (evict_node) bookkeeping. Scratch vectors are resized on demand.
    std::uint64_t demoted_total_                 = 0;
    std::uint64_t subtree_erased_total_          = 0;
    std::uint64_t branch_gc_removed_blocks_total_ = 0;
    std::uint64_t snapshots_deleted_by_eviction_total_ = 0;
    std::vector<std::uint8_t> evict_protected_;  // protected + ancestors closure
    std::vector<std::uint8_t> evict_demand_under_;// demand anywhere in the subtree
};

} // namespace ninfer::prefix_cache
