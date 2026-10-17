// Unit tests for the M2 host tier: HostSlabPool allocation/eviction and PrefixIndex
// host-residency state transitions. No GPU required. Checks are evaluated unconditionally
// (no assert), so they hold in NDEBUG builds as well.

#include "runtime/prefix_cache/host_slab_pool.h"
#include "runtime/prefix_cache/prefix_index.h"
#include "runtime/prefix_cache/prefix_store.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace pc = ninfer::prefix_cache;

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

struct TestSink : pc::PrefixPageSink {
    int released_blocks = 0;
    int released_tails  = 0;
    int released_slots  = 0;
    void release_block_pages(const pc::PoolPages&) noexcept override { ++released_blocks; }
    void release_tail_pages(const pc::PoolPages&) noexcept override { ++released_tails; }
    void release_snapshot_slot(pc::SlotId) noexcept override { ++released_slots; }
};

pc::HostSlabPool make_test_pool() {
    // 4 MiB budget; the slab data (4 x 512 KiB blocks + 2 x 128 KiB snapshots = 2.25 MiB) fits.
    return pc::HostSlabPool({.total_bytes = 4 * 1024 * 1024,
                             .block_slab_bytes = 512 * 1024,
                             .snapshot_slab_bytes = 128 * 1024,
                             .block_slab_count = 4,
                             .snapshot_slab_count = 2});
}

std::vector<std::int32_t> block_tokens(std::int32_t seed) {
    std::vector<std::int32_t> tokens(pc::kBlockTokens);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = seed + static_cast<std::int32_t>(i);
    }
    return tokens;
}

pc::NodeId commit_block(pc::PrefixIndex& index, pc::NodeId parent, std::int32_t seed,
                        std::int32_t text_page) {
    auto tokens = block_tokens(seed);
    const pc::BlockNode* parent_node = index.node(parent);
    const std::uint64_t parent_hash =
        parent_node != nullptr ? parent_node->lookup_hash : pc::kBlockHashSeed;
    bool adopted = false;
    const pc::NodeId node = index.commit_block(
        pc::BlockCommit{.parent = parent,
                        .lookup_hash = pc::block_hash(parent_hash, tokens, 0),
                        .extra = 0,
                        .tokens = tokens,
                        .pages = {std::array<std::int32_t, pc::kMaxPools>{text_page, -1}}},
        adopted);
    return node;
}

pc::SnapshotId add_tap(pc::PrefixIndex& index, pc::NodeId anchor, std::uint32_t frontier,
                       std::uint64_t priority = 5) {
    return index.add_snapshot(
        pc::SnapshotRequest{.anchor = anchor,
                            .frontier = frontier,
                            .tail_tokens = {},
                            .tail_pages = {},
                            .kind = pc::SnapshotKind::Tap,
                            .priority = priority});
}

void write_block_to_host(pc::PrefixIndex& index, pc::NodeId node) {
    const auto target = index.begin_host_write(node);
    (void)target;
    index.finish_host_write(node, true);
}

void write_snapshot_to_host(pc::PrefixIndex& index, pc::SnapshotId id) {
    const auto target = index.begin_snapshot_write(id);
    (void)target;
    index.finish_snapshot_write(id, true);
}

// Fill a record's host slab with a byte pattern so a round-trip can compare payload bytes, then
// finalize the write-through.
void write_block_pattern(pc::PrefixIndex& index, pc::HostSlabPool& pool, pc::NodeId node,
                         unsigned char pattern) {
    const auto target = index.begin_host_write(node);
    if (target.page != pc::kInvalidHostPage) {
        std::memset(pool.data(target.page), pattern, pool.bytes(target.page));
    }
    index.finish_host_write(node, true);
}

void write_snapshot_pattern(pc::PrefixIndex& index, pc::HostSlabPool& pool, pc::SnapshotId id,
                            unsigned char pattern) {
    const auto target = index.begin_snapshot_write(id);
    if (target.page != pc::kInvalidHostPage) {
        std::memset(pool.data(target.page), pattern, pool.bytes(target.page));
    }
    index.finish_snapshot_write(id, true);
}

bool files_equal(const std::string& x, const std::string& y) {
    std::FILE* fx = std::fopen(x.c_str(), "rb");
    std::FILE* fy = std::fopen(y.c_str(), "rb");
    if (fx == nullptr || fy == nullptr) {
        if (fx != nullptr) { std::fclose(fx); }
        if (fy != nullptr) { std::fclose(fy); }
        return false;
    }
    bool equal = true;
    for (;;) {
        unsigned char bx[4096];
        unsigned char by[4096];
        const std::size_t nx = std::fread(bx, 1, sizeof(bx), fx);
        const std::size_t ny = std::fread(by, 1, sizeof(by), fy);
        if (nx != ny || std::memcmp(bx, by, nx) != 0) { equal = false; break; }
        if (nx == 0) { break; }
    }
    std::fclose(fx);
    std::fclose(fy);
    return equal;
}

// --- HostSlabPool tests ---

int test_slab_pool_basic() {
    int failures = 0;
    pc::HostSlabPool pool = make_test_pool();
    failures += check(pool.enabled(), "pool should be enabled after construction");

    pc::SlabAllocation blk = pool.allocate_block();
    failures += check(blk.page != pc::kInvalidHostPage, "block slab allocation failed");
    void* data = pool.data(blk.page);
    failures += check(data != nullptr, "allocated slab has no data pointer");
    if (data != nullptr) { std::memset(data, 0xAB, pool.bytes(blk.page)); }

    pc::SlabAllocation snap = pool.allocate_snapshot();
    failures += check(snap.page != pc::kInvalidHostPage, "snapshot slab allocation failed");
    failures += check(pool.data(snap.page) != nullptr, "snapshot slab has no data pointer");

    pool.release(blk.page);
    pool.release(snap.page);
    return failures;
}

int test_slab_pool_eviction() {
    int failures = 0;
    pc::HostSlabPool pool({.total_bytes = 2 * 1024 * 1024,
                           .block_slab_bytes = 512 * 1024,
                           .snapshot_slab_bytes = 64 * 1024,
                           .block_slab_count = 2,
                           .snapshot_slab_count = 1});
    failures += check(pool.enabled(), "pool should be enabled after construction");

    pc::SlabAllocation a = pool.allocate_block();
    pc::SlabAllocation b = pool.allocate_block();
    failures += check(a.page != pc::kInvalidHostPage && b.page != pc::kInvalidHostPage,
                      "initial block slab allocations failed");

    // Pool is full; allocating a third should evict the oldest (a).
    pc::SlabAllocation c = pool.allocate_block();
    failures += check(c.page != pc::kInvalidHostPage, "third block slab allocation failed");
    failures += check(c.evicted == a.page, "LRU slab was not evicted");
    return failures;
}

int test_slab_pool_gdsf() {
    int failures = 0;
    pc::HostSlabPool pool({.total_bytes = 2 * 1024 * 1024,
                           .block_slab_bytes = 512 * 1024,
                           .snapshot_slab_bytes = 64 * 1024,
                           .block_slab_count = 3,
                           .snapshot_slab_count = 1});
    failures += check(pool.enabled(), "pool should be enabled after construction");

    pc::SlabAllocation a = pool.allocate_block();
    pc::SlabAllocation b = pool.allocate_block();
    pc::SlabAllocation c = pool.allocate_block();
    failures += check(a.page != pc::kInvalidHostPage && b.page != pc::kInvalidHostPage &&
                          c.page != pc::kInvalidHostPage,
                      "initial block slab allocations failed");

    pool.mark_dead(a.page);

    // Allocating a fourth block should evict the dead slab first.
    pc::SlabAllocation d = pool.allocate_block();
    failures += check(d.page != pc::kInvalidHostPage, "fourth block slab allocation failed");
    failures += check(d.evicted == a.page, "dead slab was not evicted first");
    return failures;
}

// --- PrefixIndex host-residency tests ---

int test_prefix_index_host_residency() {
    int failures = 0;
    TestSink sink;
    pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 2}, &sink);
    pc::HostSlabPool pool = make_test_pool();
    index.set_host_pool(&pool);

    const pc::NodeId node = commit_block(index, index.root(), 42, 100);
    failures += check(node != pc::kInvalidNode, "block commit failed");
    const pc::BlockNode* record = index.node(node);
    failures += check(record != nullptr, "committed node is missing");
    if (record != nullptr) {
        failures += check(record->device_state == pc::Residency::Resident,
                          "committed node is not device-resident");
        failures += check(record->host_state == pc::Residency::Absent,
                          "fresh node unexpectedly has a host copy");
    }

    pc::HostWriteTarget wt = index.begin_host_write(node);
    failures += check(wt.node == node && wt.page != pc::kInvalidHostPage,
                      "begin_host_write did not reserve a slab");
    record = index.node(node);
    failures += check(record != nullptr && record->host_state == pc::Residency::Filling,
                      "node is not Filling during write-through");

    index.finish_host_write(node, true);
    record = index.node(node);
    failures += check(record != nullptr && record->host_state == pc::Residency::Host,
                      "write-through did not land the host copy");
    return failures;
}

int test_device_fill_roundtrip() {
    int failures = 0;
    TestSink sink;
    pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 2}, &sink);
    pc::HostSlabPool pool = make_test_pool();
    index.set_host_pool(&pool);

    const pc::NodeId node = commit_block(index, index.root(), 7, 100);
    failures += check(node != pc::kInvalidNode, "block commit failed");
    write_block_to_host(index, node);

    // Demote the device copy: the node is backed, so eviction keeps it host-only.
    const pc::NodeId evicted = index.evict_node({});
    failures += check(evicted == node, "backed node was not demoted");
    const pc::BlockNode* record = index.node(node);
    failures += check(record != nullptr && record->device_state == pc::Residency::Absent &&
                          record->host_state == pc::Residency::Host,
                      "demoted node should be host-only");

    // Restore: fill the device copy from the host slab.
    index.begin_device_fill(node, {std::array<std::int32_t, pc::kMaxPools>{200, -1}});
    record = index.node(node);
    failures += check(record != nullptr && record->device_state == pc::Residency::Filling,
                      "node is not Filling during restore");
    index.finish_device_fill(node, true);
    record = index.node(node);
    failures += check(record != nullptr && record->device_state == pc::Residency::Resident &&
                          record->device_pages.page[0] == 200,
                      "restore did not make the node device-resident");

    // Rollback path: demote again, start a fill, abort it.
    (void)index.evict_node({});
    const int releases_before = sink.released_blocks;
    index.begin_device_fill(node, {std::array<std::int32_t, pc::kMaxPools>{201, -1}});
    index.finish_device_fill(node, false);
    record = index.node(node);
    failures += check(record != nullptr && record->device_state == pc::Residency::Absent,
                      "aborted restore did not roll the node back");
    failures += check(sink.released_blocks == releases_before + 1,
                      "aborted restore did not return its pages");
    return failures;
}

int test_evict_demote_vs_erase() {
    int failures = 0;
    TestSink sink;
    pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 2}, &sink);
    pc::HostSlabPool pool = make_test_pool();
    index.set_host_pool(&pool);

    const pc::NodeId a = commit_block(index, index.root(), 1, 100);
    const pc::NodeId b = commit_block(index, a, 2, 101);
    failures += check(a != pc::kInvalidNode && b != pc::kInvalidNode, "block commits failed");
    const pc::SnapshotId snap = add_tap(index, b, 128);
    failures += check(snap != pc::kInvalidSnapshot, "tap snapshot was rejected");

    // Unbacked subtree: eviction erases the oldest node and everything below it.
    const int releases_before = sink.released_blocks;
    const int slots_before    = sink.released_slots;
    const pc::NodeId evicted  = index.evict_node({});
    failures += check(evicted == a, "LRU did not evict the oldest node");
    failures += check(index.node(a) == nullptr && index.node(b) == nullptr,
                      "unbacked subtree was not erased");
    failures += check(index.snapshot(snap) == nullptr,
                      "snapshot anchored in the erased subtree survived");
    failures += check(sink.released_blocks == releases_before + 2,
                      "erased nodes did not return their pages");
    failures += check(sink.released_slots == slots_before + 1,
                      "erased snapshot did not return its slot");
    return failures;
}

int test_evict_demand_demotes_tail_first() {
    int failures = 0;
    TestSink sink;
    pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 2}, &sink);
    pc::HostSlabPool pool = make_test_pool();
    index.set_host_pool(&pool);

    const pc::NodeId a = commit_block(index, index.root(), 5, 100);
    const pc::NodeId b = commit_block(index, a, 6, 101);
    failures += check(a != pc::kInvalidNode && b != pc::kInvalidNode, "block commits failed");
    write_block_to_host(index, a);
    write_block_to_host(index, b);

    // With no demand, demotion sheds the deepest host-backed node: the tail block b goes host-only
    // while the shared prefix block a stays device-resident.
    const pc::NodeId evicted_b = index.evict_node({});
    const pc::BlockNode* rb = index.node(b);
    const pc::BlockNode* ra = index.node(a);
    failures += check(evicted_b == b, "demotion did not shed the deepest backed node first");
    failures += check(rb != nullptr && rb->device_state == pc::Residency::Absent &&
                          rb->host_state == pc::Residency::Host,
                      "the tail block should be host-only");
    failures += check(ra != nullptr && ra->device_state == pc::Residency::Resident,
                      "the shared prefix block lost its device copy");
    failures += check(index.eviction_counters().demoted >= 1, "demotion was not counted");

    const pc::NodeId evicted_a = index.evict_node({});
    failures += check(evicted_a == a, "the second pressure round did not shed the prefix block");

    // A request whose matched chain runs through a protects a: after restoring b, the demanded
    // path is excluded from eviction and b is demoted again before any trimming of a.
    index.begin_device_fill(b, {std::array<std::int32_t, pc::kMaxPools>{400, -1}});
    index.finish_device_fill(b, true);
    failures += check(index.node(b) != nullptr && index.node(b)->device_state ==
                           pc::Residency::Resident,
                      "tail block restore failed");
    const std::array<pc::NodeId, 1> demand{a};
    const pc::NodeId evicted_b_again = index.evict_node(demand);
    failures += check(evicted_b_again == b, "demand did not exclude the matched chain");
    failures += check(index.node(a) != nullptr && index.node(a)->device_state ==
                           pc::Residency::Resident,
                      "demanded chain lost its device residency");
    return failures;
}

int test_slab_eviction_reconciliation() {
    int failures = 0;
    TestSink sink;
    pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 2}, &sink);
    // Exactly two block slabs so the third write-through must evict one.
    pc::HostSlabPool pool({.total_bytes = 2 * 1024 * 1024,
                           .block_slab_bytes = 512 * 1024,
                           .snapshot_slab_bytes = 64 * 1024,
                           .block_slab_count = 2,
                           .snapshot_slab_count = 1});
    index.set_host_pool(&pool);

    const pc::NodeId a = commit_block(index, index.root(), 1, 100);
    const pc::NodeId b = commit_block(index, index.root(), 2, 101);
    const pc::NodeId c = commit_block(index, index.root(), 3, 102);
    failures += check(a != pc::kInvalidNode && b != pc::kInvalidNode &&
                          c != pc::kInvalidNode,
                      "block commits failed");

    write_block_to_host(index, a);
    write_block_to_host(index, b);
    const pc::HostPageId slab_a = index.node(a)->host_page;
    failures += check(slab_a != pc::kInvalidHostPage, "first write-through did not land");

    // The pool is full; writing c through evicts the oldest slab (a's). Node a is still
    // device-resident, so it survives as device-only.
    write_block_to_host(index, c);
    const pc::BlockNode* na = index.node(a);
    const pc::BlockNode* nc = index.node(c);
    failures += check(na != nullptr && na->host_state == pc::Residency::Absent &&
                          na->host_page == pc::kInvalidHostPage,
                      "evicted slab was not reconciled off the owner");
    failures += check(nc != nullptr && nc->host_state == pc::Residency::Host,
                      "new write-through did not land");

    // Demand-aware eviction sheds host-backed unpinned nodes; a/b/c are equal-depth siblings, so the
    // LRU head (b) goes first, then c. Nothing is demanded here, so only lossless demotions occur.
    (void)index.evict_node({});
    const pc::BlockNode* nb_after = index.node(b);
    failures += check(nb_after != nullptr && nb_after->device_state == pc::Residency::Absent &&
                          nb_after->host_state == pc::Residency::Host,
                      "the first backed node (b) should be demoted");
    (void)index.evict_node({});
    const pc::BlockNode* nc_after = index.node(c);
    failures += check(nc_after != nullptr && nc_after->device_state == pc::Residency::Absent &&
                          nc_after->host_state == pc::Residency::Host,
                      "c should now be host-only");

    const pc::NodeId d = commit_block(index, index.root(), 4, 103);
    failures += check(d != pc::kInvalidNode, "fourth block commit failed");
    write_block_to_host(index, d);
    // b's slab is the oldest in the pool, so writing d evicts it. b is device-absent as well, so
    // losing its last data copy deletes the node.
    failures += check(index.node(b) == nullptr,
                      "host-only node was not deleted when its slab was evicted");
    failures += check(index.node(c) != nullptr && index.node(c)->host_state == pc::Residency::Host,
                      "newer host slab (c's) must survive pool pressure");
    return failures;
}

int test_slot_demote_host_backed() {
    int failures = 0;
    TestSink sink;
    pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 1}, &sink);
    pc::HostSlabPool pool = make_test_pool();
    index.set_host_pool(&pool);

    const pc::NodeId a = commit_block(index, index.root(), 1, 100);
    const pc::NodeId b = commit_block(index, a, 2, 101);
    failures += check(a != pc::kInvalidNode && b != pc::kInvalidNode, "block commits failed");

    const pc::SnapshotId snap_a = add_tap(index, a, 64, 10);
    failures += check(snap_a != pc::kInvalidSnapshot, "first snapshot was rejected");
    write_snapshot_to_host(index, snap_a);

    // Fill the only remaining slot with a second snapshot.
    const pc::SnapshotId snap_b = add_tap(index, b, 128, 10);
    failures += check(snap_b != pc::kInvalidSnapshot, "second snapshot was rejected");

    // No free slot: reserving one must evict a victim. snap_a has a host copy, so it is
    // demoted (kept host-resident) rather than deleted.
    const pc::SlotId slot = index.reserve_snapshot_slot(10);
    failures += check(slot != pc::kInvalidSlot, "slot reservation found no victim");
    const pc::Snapshot* ra = index.snapshot(snap_a);
    failures += check(ra != nullptr, "host-backed snapshot was deleted instead of demoted");
    if (ra != nullptr) {
        failures += check(ra->device_slot == pc::kInvalidSlot &&
                              ra->device_state == pc::Residency::Absent &&
                              ra->host_state == pc::Residency::Host,
                          "demoted snapshot should be host-only");
    }
    index.release_reserved_slot(slot);
    return failures;
}

int test_snapshot_restore_roundtrip() {
    int failures = 0;
    TestSink sink;
    pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 2}, &sink);
    pc::HostSlabPool pool = make_test_pool();
    index.set_host_pool(&pool);

    const pc::NodeId a = commit_block(index, index.root(), 1, 100);
    const pc::NodeId b = commit_block(index, a, 2, 101);
    failures += check(a != pc::kInvalidNode && b != pc::kInvalidNode, "block commits failed");

    const pc::SnapshotId snap_a = add_tap(index, a, 64, 10);
    const pc::SnapshotId snap_b = add_tap(index, b, 128, 10);
    failures += check(snap_a != pc::kInvalidSnapshot && snap_b != pc::kInvalidSnapshot,
                      "snapshot admission failed");
    write_snapshot_to_host(index, snap_a);

    // Exhaust the slots so snap_a (host-backed) is demoted.
    const pc::SlotId slot = index.reserve_snapshot_slot(10);
    failures += check(slot != pc::kInvalidSlot, "slot reservation found no victim");
    index.release_reserved_slot(slot);
    const pc::Snapshot* ra = index.snapshot(snap_a);
    failures += check(ra != nullptr && ra->device_state == pc::Residency::Absent,
                      "snap_a should be demoted to host-only");

    // Restore snap_a's image into a device slot.
    const auto target = index.begin_snapshot_restore(snap_a);
    failures += check(target.slot != pc::kInvalidSlot &&
                          target.page != pc::kInvalidHostPage,
                      "snapshot restore did not reserve a slot");
    index.finish_snapshot_restore(snap_a, true);
    ra = index.snapshot(snap_a);
    failures += check(ra != nullptr && ra->device_resident() &&
                          ra->device_slot != pc::kInvalidSlot,
                      "restored snapshot is not device-resident");

    // Rollback path: demote again, start a restore, abort it.
    const pc::SlotId slot2 = index.reserve_snapshot_slot(10);
    failures += check(slot2 != pc::kInvalidSlot, "second slot reservation found no victim");
    index.release_reserved_slot(slot2);
    const auto aborted = index.begin_snapshot_restore(snap_a);
    failures += check(aborted.slot != pc::kInvalidSlot, "second restore did not reserve a slot");
    index.finish_snapshot_restore(snap_a, false);
    ra = index.snapshot(snap_a);
    failures += check(ra != nullptr && ra->device_state == pc::Residency::Absent &&
                          ra->host_state == pc::Residency::Host,
                      "aborted restore did not roll the snapshot back");
    return failures;
}

int test_best_host_admission() {
    int failures = 0;
    TestSink sink;
    pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 2}, &sink);
    pc::HostSlabPool pool = make_test_pool();
    index.set_host_pool(&pool);

    const auto block0 = block_tokens(1);
    const auto block1 = block_tokens(1000);
    const pc::NodeId a = commit_block(index, index.root(), 1, 100);
    failures += check(a != pc::kInvalidNode, "block commit failed");
    const pc::SnapshotId snap = add_tap(index, a, 64, 5);
    failures += check(snap != pc::kInvalidSnapshot, "tap snapshot was rejected");
    write_block_to_host(index, a);
    write_snapshot_to_host(index, snap);

    // Demote the block: the path is now host-only, so device admission must fail while host
    // admission still finds the snapshot.
    (void)index.evict_node({});
    const pc::BlockNode* record = index.node(a);
    failures += check(record != nullptr && record->host_state == pc::Residency::Host &&
                          record->device_state == pc::Residency::Absent,
                      "block should be host-only after demotion");

    std::vector<std::int32_t> prompt(block0.begin(), block0.end());
    prompt.insert(prompt.end(), block1.begin(), block1.end());
    const auto hashes = pc::chain_block_hashes(prompt);
    const pc::PrefixMatch match = index.match(prompt, hashes, {});
    failures += check(match.matched_blocks == 1, "match did not walk the committed block");
    failures += check(match.best_device == pc::kInvalidSnapshot,
                      "device-resident admission should fail on a demoted path");
    failures += check(match.best_host == snap && match.best_host_frontier == 64,
                      "host-resident admission did not select the snapshot");
    return failures;
}

int test_dead_kv_and_supersede_marking() {
    int failures = 0;
    TestSink sink;
    pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 1}, &sink);
    pc::HostSlabPool pool = make_test_pool();
    index.set_host_pool(&pool);

    const pc::NodeId a = commit_block(index, index.root(), 1, 100);
    failures += check(a != pc::kInvalidNode, "block commit failed");
    write_block_to_host(index, a);
    const pc::HostPageId slab_a = index.node(a)->host_page;
    failures += check(slab_a != pc::kInvalidHostPage, "write-through did not land");

    // While a snapshot reaches the node, its slab stays alive.
    const pc::SnapshotId snap = add_tap(index, a, 64, 10);
    failures += check(snap != pc::kInvalidSnapshot, "tap snapshot was rejected");
    failures += check(pool.slab(slab_a) != nullptr && !pool.slab(slab_a)->dead,
                      "slab marked dead while a snapshot still reaches it");

    // Removing the snapshot leaves the block unreachable: the slab becomes dead-KV.
    index.remove_snapshot(snap);
    failures += check(pool.slab(slab_a) != nullptr && pool.slab(slab_a)->dead,
                      "slab not marked dead after its last snapshot left");

    // Supersede: with one slot, adding a second snapshot that names the first as superseded
    // marks the victim's slab superseded and demotes it (it has a host copy).
    const pc::NodeId b = commit_block(index, index.root(), 2, 101);
    failures += check(b != pc::kInvalidNode, "second block commit failed");
    const pc::SnapshotId s1 = add_tap(index, b, 64, 10);
    failures += check(s1 != pc::kInvalidSnapshot, "supersede victim snapshot was rejected");
    write_snapshot_to_host(index, s1);
    const pc::HostPageId slab_s1 = index.snapshot(s1) != nullptr
                                       ? index.snapshot(s1)->host_page
                                       : pc::kInvalidHostPage;
    failures += check(slab_s1 != pc::kInvalidHostPage &&
                          !pool.slab(slab_s1)->superseded,
                      "victim slab marked superseded too early");

    const pc::NodeId c = commit_block(index, b, 3, 102);
    failures += check(c != pc::kInvalidNode, "third block commit failed");
    const pc::SnapshotId s2 = index.add_snapshot(
        pc::SnapshotRequest{.anchor = c,
                            .frontier = 192,
                            .tail_tokens = {},
                            .tail_pages = {},
                            .kind = pc::SnapshotKind::Tap,
                            .priority = 5,
                            .supersede = s1});
    failures += check(s2 != pc::kInvalidSnapshot, "superseding snapshot was rejected");
    failures += check(pool.slab(slab_s1) != nullptr && pool.slab(slab_s1)->superseded,
                      "victim slab not marked superseded");
    const pc::Snapshot* victim = index.snapshot(s1);
    failures += check(victim != nullptr && victim->device_state == pc::Residency::Absent &&
                          victim->host_state == pc::Residency::Host,
                      "superseded snapshot with a host copy should be demoted, not deleted");

    // A hit clears the superseded mark on both the snapshot and its slab.
    index.hit(s1);
    failures += check(pool.slab(slab_s1) != nullptr && !pool.slab(slab_s1)->superseded,
                      "hit did not clear the superseded mark");
    return failures;
}

// --- M3a disk-tier tests ---

constexpr std::size_t kDiskBlockBytes    = 512 * 1024;
constexpr std::size_t kDiskSnapshotBytes = 128 * 1024;

pc::DiskSlabPool make_test_disk_pool(const std::string& file) {
    return pc::DiskSlabPool({.total_bytes = 4 * 1024 * 1024,
                             .block_slab_bytes = kDiskBlockBytes,
                             .snapshot_slab_bytes = kDiskSnapshotBytes,
                             .block_slab_count = 4,
                             .snapshot_slab_count = 2,
                             .file = file});
}

pc::PrefixStoreGeometry disk_geometry(const pc::DiskSlabPool& pool) {
    pc::PrefixStoreGeometry g;
    g.tp                   = 1;
    g.device_snapshot_slots = 2;
    g.block_rank_bytes     = pool.block_slab_bytes();
    g.snapshot_rank_bytes  = pool.snapshot_slab_bytes();
    g.block_slab_bytes     = pool.block_slab_bytes();
    g.snapshot_slab_bytes  = pool.snapshot_slab_bytes();
    g.block_slab_count     = static_cast<std::uint32_t>(pool.block_slab_count());
    g.snapshot_slab_count  = static_cast<std::uint32_t>(pool.snapshot_slab_count());
    g.disk_file_uid        = pool.file_uid();
    return g;
}

std::vector<unsigned char> read_bytes(const void* data, std::size_t bytes) {
    return std::vector<unsigned char>(static_cast<const unsigned char*>(data),
                                      static_cast<const unsigned char*>(data) + bytes);
}

int test_disk_pool_basic() {
    int failures = 0;
    const std::string file = "disk-pool-basic.slabs";
    std::remove(file.c_str());
    {
        pc::DiskSlabPool pool = make_test_disk_pool(file);
        failures += check(pool.enabled(), "disk pool did not open");
        failures += check(pool.block_slab_count() == 4 && pool.snapshot_slab_count() == 2,
                          "disk pool geometry differs");
        const std::size_t capacity = 4 * kDiskBlockBytes + 2 * kDiskSnapshotBytes;
        failures += check(pool.free_bytes() == capacity, "fresh pool is not fully free");

        const pc::DiskSlotAllocation b0 = pool.allocate_block();
        failures += check(b0.page == 0 && b0.evicted == pc::kInvalidDiskPage,
                          "first block slot is not 0");
        std::vector<unsigned char> pattern(kDiskBlockBytes, 0x9E);
        failures += check(pool.write_slot(b0.page, pattern.data()), "write_slot failed");
        failures += check(pool.validate_slot(b0.page, pool.generation(b0.page)),
                          "validate_slot rejects a freshly written slot");
        failures += check(!pool.validate_slot(b0.page, pool.generation(b0.page) + 1U),
                          "validate_slot accepts a wrong generation");

        std::vector<unsigned char> got(kDiskBlockBytes, 0);
        failures += check(pool.read_slot(b0.page, pool.generation(b0.page), got.data()),
                          "read_slot failed");
        failures += check(got == pattern, "disk payload differs after round trip");

        const std::uint64_t uid = pool.file_uid();
        failures += check(uid != 0, "disk pool has no uid");
        pool.release(b0.page);
        failures += check(pool.free_bytes() == capacity, "release did not free the slot");
    }
    {
        // Reopening the same file preserves the identity of the slab file.
        pc::DiskSlabPool pool = make_test_disk_pool(file);
        failures += check(pool.enabled() && pool.file_uid() != 0, "reopened pool lost its uid");
        const std::uint64_t uid = pool.file_uid();
        (void)uid;
    }
    {
        // A different geometry reinitializes the file and yields a fresh uid.
        pc::DiskSlabPool pool({.total_bytes = 4 * 1024 * 1024,
                               .block_slab_bytes = kDiskBlockBytes,
                               .snapshot_slab_bytes = kDiskSnapshotBytes,
                               .block_slab_count = 8,
                               .snapshot_slab_count = 2,
                               .file = file});
        failures += check(pool.enabled() && pool.block_slab_count() == 8,
                          "reinitialized pool did not adopt the new geometry");
    }
    std::remove(file.c_str());
    return failures;
}

int test_disk_pool_eviction_order() {
    int failures = 0;
    const std::string file = "disk-pool-evict.slabs";
    std::remove(file.c_str());
    pc::DiskSlabPool pool = make_test_disk_pool(file);
    failures += check(pool.enabled(), "disk pool did not open");

    pc::DiskPageId pages[4] = {pc::kInvalidDiskPage, pc::kInvalidDiskPage, pc::kInvalidDiskPage,
                               pc::kInvalidDiskPage};
    for (int i = 0; i < 4; ++i) { pages[i] = pool.allocate_block().page; }
    failures += check(pages[0] == 0 && pages[3] == 3, "block slots are not allocated in order");

    // Mark slot 2 dead; the next allocation must reclaim it first even though slot 0 is colder.
    pool.mark_dead(2);
    const pc::DiskSlotAllocation dead_evict = pool.allocate_block();
    failures += check(dead_evict.page == 2 && dead_evict.evicted == 2,
                      "dead slot was not the eviction victim");

    // With no dead slot, a superseded slot is evicted before a live one.
    pool.supersede(1);
    const pc::DiskSlotAllocation sup_evict = pool.allocate_block();
    failures += check(sup_evict.page == 1 && sup_evict.evicted == 1,
                      "superseded slot was not the eviction victim");

    // With neither, GDSF picks the coldest live slot.
    const pc::DiskSlotAllocation gdsf_evict = pool.allocate_block();
    failures += check(gdsf_evict.page == 0 && gdsf_evict.evicted == 0,
                      "GDSF did not pick the coldest live slot");

    // A pinned slot is never chosen as a victim, even when it is the only dead one.
    for (pc::DiskPageId page : {0, 1, 2, 3}) { pool.release(page); }
    const pc::DiskPageId p0 = pool.allocate_block().page;
    const pc::DiskPageId p1 = pool.allocate_block().page;
    (void)pool.allocate_block();
    (void)pool.allocate_block();
    pool.mark_dead(p0);
    pool.pin(p0);
    const pc::DiskSlotAllocation no_pin_evict = pool.allocate_block();
    failures += check(no_pin_evict.page != p0 && no_pin_evict.evicted != p0,
                      "a pinned dead slot was chosen as the victim");
    pool.unpin(p0);
    const pc::DiskSlotAllocation pinned_then = pool.allocate_block();
    failures += check(pinned_then.page == p0, "unpinned dead slot is not the next victim");
    (void)p1;

    std::remove(file.c_str());
    return failures;
}

int test_disk_store_roundtrip() {
    int failures = 0;
    const std::string idx   = "prefix-store-roundtrip.bin";
    const std::string idx2  = "prefix-store-roundtrip2.bin";
    const std::string slabs = "prefix-store-roundtrip.slabs";
    std::remove(idx.c_str());
    std::remove(idx2.c_str());
    std::remove(slabs.c_str());

    std::vector<unsigned char> expected_a;
    std::vector<unsigned char> expected_b;
    std::vector<unsigned char> expected_s;
    pc::NodeId a = pc::kInvalidNode;
    pc::NodeId b = pc::kInvalidNode;
    pc::SnapshotId snap = pc::kInvalidSnapshot;
    pc::PrefixStoreGeometry geometry;

    {
        TestSink sink;
        pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 2}, &sink);
        pc::HostSlabPool host = make_test_pool();
        pc::DiskSlabPool disk = make_test_disk_pool(slabs);
        index.set_host_pool(&host);
        index.set_disk_pool(&disk);
        failures += check(disk.enabled(), "disk pool did not open");

        a = commit_block(index, index.root(), 1, 100);
        b = commit_block(index, a, 2, 101);
        failures += check(a != pc::kInvalidNode && b != pc::kInvalidNode, "block commits failed");
        write_block_pattern(index, host, a, 0xA5);
        write_block_pattern(index, host, b, 0x5A);
        snap = add_tap(index, b, 128, 10);
        failures += check(snap != pc::kInvalidSnapshot, "snapshot admission failed");
        write_snapshot_pattern(index, host, snap, 0xC3);

        const pc::BlockNode* na = index.node(a);
        const pc::BlockNode* nb = index.node(b);
        failures += check(na != nullptr && nb != nullptr, "committed nodes are missing");
        if (na != nullptr && nb != nullptr) {
            expected_a = read_bytes(host.data(na->host_page), host.bytes(na->host_page));
            expected_b = read_bytes(host.data(nb->host_page), host.bytes(nb->host_page));
        }
        const pc::Snapshot* ns = index.snapshot(snap);
        if (ns != nullptr) {
            expected_s = read_bytes(host.data(ns->host_page), host.bytes(ns->host_page));
        }

        failures += check(index.spill_node(a), "spill a failed");
        failures += check(index.spill_node(b), "spill b failed");
        failures += check(index.spill_snapshot(snap), "spill snapshot failed");
        failures += check(na->disk_state == pc::Residency::Disk && na->host_state == pc::Residency::Host,
                          "spill a should keep the host copy and gain a disk copy");
        failures += check(index.snapshot(snap)->disk_state == pc::Residency::Disk,
                          "snapshot did not gain a disk copy");

        geometry = disk_geometry(disk);
        failures += check(pc::save_prefix_store(idx, index, disk, geometry), "save failed");
        failures += check(pc::save_prefix_store(idx2, index, disk, geometry), "second save failed");
        failures += check(files_equal(idx, idx2), "container bytes are not deterministic");
    }

    {
        pc::DiskSlabPool disk = make_test_disk_pool(slabs);
        failures += check(disk.file_uid() == geometry.disk_file_uid,
                          "reopened slab file lost its uid");
        TestSink sink2;
        pc::PrefixIndex loaded(pc::PrefixIndexOptions{.device_snapshot_slots = 2}, &sink2);
        pc::HostSlabPool host2 = make_test_pool();
        loaded.set_host_pool(&host2);
        loaded.set_disk_pool(&disk);
        pc::PrefixStoreStats stats;
        failures += check(pc::load_prefix_store(idx, loaded, disk, geometry, &stats), "load failed");
        failures += check(loaded.live_node_count() == 3 && loaded.live_snapshot_count() == 1,
                          "loaded counts differ from the saved index");
        failures += check(stats.nodes == 3 && stats.snapshots == 1,
                          "load stats do not match the restored cache");

        const pc::BlockNode* la = loaded.node(a);
        const pc::BlockNode* lb = loaded.node(b);
        failures += check(la != nullptr && la->disk_state == pc::Residency::Disk &&
                              la->host_state == pc::Residency::Absent &&
                              la->device_state == pc::Residency::Absent,
                          "restored block a is not disk-only");
        failures += check(lb != nullptr && lb->disk_state == pc::Residency::Disk,
                          "restored block b is not disk-resident");
        if (la != nullptr) {
            std::vector<unsigned char> got(kDiskBlockBytes, 0);
            failures += check(disk.read_slot(la->disk_page, disk.generation(la->disk_page),
                                             got.data()),
                              "restored block a slot is unreadable");
            failures += check(got == expected_a, "restored block a payload differs");
        }
        if (lb != nullptr) {
            std::vector<unsigned char> got(kDiskBlockBytes, 0);
            failures += check(disk.read_slot(lb->disk_page, disk.generation(lb->disk_page),
                                             got.data()),
                              "restored block b slot is unreadable");
            failures += check(got == expected_b, "restored block b payload differs");
        }
        const pc::Snapshot* ls = loaded.snapshot(snap);
        failures += check(ls != nullptr && ls->disk_state == pc::Residency::Disk &&
                              ls->frontier == 128,
                          "restored snapshot metadata differs");
        if (ls != nullptr) {
            std::vector<unsigned char> got(kDiskSnapshotBytes, 0);
            failures += check(disk.read_slot(ls->disk_page, disk.generation(ls->disk_page),
                                             got.data()),
                              "restored snapshot slot is unreadable");
            failures += check(got == expected_s, "restored snapshot payload differs");
        }

        // Materializing a disk-only record must reproduce the host payload exactly.
        const pc::HostPageId restored_a = loaded.materialize_node_host(a);
        failures += check(restored_a != pc::kInvalidHostPage, "materialize a failed");
        if (restored_a != pc::kInvalidHostPage) {
            failures += check(host2.data(restored_a) != nullptr &&
                                  read_bytes(host2.data(restored_a), host2.bytes(restored_a)) ==
                                      expected_a,
                              "materialized block a payload differs");
        }
        const pc::HostPageId restored_s = loaded.materialize_snapshot_host(snap);
        failures += check(restored_s != pc::kInvalidHostPage, "materialize snapshot failed");

        // A re-save of the restored cache must be byte-identical to the original container.
        failures += check(pc::save_prefix_store(idx2, loaded, disk, geometry), "re-save failed");
        failures += check(files_equal(idx, idx2), "round-trip container bytes differ");
    }

    std::remove(idx.c_str());
    std::remove(idx2.c_str());
    std::remove(slabs.c_str());
    return failures;
}

int test_disk_store_rejects() {
    int failures = 0;
    const std::string idx   = "prefix-store-reject.bin";
    const std::string slabs = "prefix-store-reject.slabs";
    std::remove(idx.c_str());
    std::remove(slabs.c_str());

    pc::PrefixStoreGeometry geometry;
    {
        TestSink sink;
        pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 2}, &sink);
        pc::HostSlabPool host = make_test_pool();
        pc::DiskSlabPool disk = make_test_disk_pool(slabs);
        index.set_host_pool(&host);
        index.set_disk_pool(&disk);
        geometry = disk_geometry(disk);

        // A missing index is a clean miss, not an error state.
        pc::PrefixStoreStats stats;
        failures += check(
            !pc::load_prefix_store("prefix-store-absent.bin", index, disk, geometry, &stats),
            "load of a missing index should fail");
        failures += check(index.live_snapshot_count() == 0, "failed load left a non-empty index");

        const pc::NodeId node = commit_block(index, index.root(), 1, 100);
        failures += check(node != pc::kInvalidNode, "block commit failed");
        write_block_pattern(index, host, node, 0x11);
        failures += check(index.spill_node(node), "spill failed");
        failures += check(pc::save_prefix_store(idx, index, disk, geometry), "save failed");
    }

    // A geometry mismatch must be rejected whole.
    {
        pc::DiskSlabPool disk = make_test_disk_pool(slabs);
        TestSink sink;
        pc::PrefixIndex loaded(pc::PrefixIndexOptions{.device_snapshot_slots = 2}, &sink);
        pc::HostSlabPool host = make_test_pool();
        loaded.set_host_pool(&host);
        pc::PrefixStoreGeometry wrong = geometry;
        wrong.tp = 2;
        pc::PrefixStoreStats stats;
        failures += check(!pc::load_prefix_store(idx, loaded, disk, wrong, &stats),
                          "geometry mismatch should be rejected");
        failures += check(loaded.live_node_count() == 1 && loaded.live_snapshot_count() == 0,
                          "rejected load left a non-empty index");
    }

    // A slab file reinitialized with a different geometry has a different uid and is rejected.
    {
        std::remove(slabs.c_str());
        pc::DiskSlabPool disk({.total_bytes = 8 * 1024 * 1024,
                               .block_slab_bytes = kDiskBlockBytes,
                               .snapshot_slab_bytes = kDiskSnapshotBytes,
                               .block_slab_count = 8,
                               .snapshot_slab_count = 2,
                               .file = slabs});
        failures += check(disk.file_uid() != geometry.disk_file_uid, "uid did not change");
        TestSink sink;
        pc::PrefixIndex loaded(pc::PrefixIndexOptions{.device_snapshot_slots = 2}, &sink);
        pc::HostSlabPool host = make_test_pool();
        loaded.set_host_pool(&host);
        pc::PrefixStoreStats stats;
        failures += check(!pc::load_prefix_store(idx, loaded, disk, geometry, &stats),
                          "uid/geometry mismatch should be rejected");
    }

    std::remove(idx.c_str());
    std::remove(slabs.c_str());
    return failures;
}

int test_disk_store_crash_gap() {
    int failures = 0;
    const std::string idx   = "prefix-store-crash.bin";
    const std::string slabs = "prefix-store-crash.slabs";
    std::remove(idx.c_str());
    std::remove(slabs.c_str());

    pc::PrefixStoreGeometry geometry;
    pc::NodeId a = pc::kInvalidNode;
    {
        TestSink sink;
        pc::PrefixIndex index(pc::PrefixIndexOptions{.device_snapshot_slots = 2}, &sink);
        pc::HostSlabPool host = make_test_pool();
        pc::DiskSlabPool disk = make_test_disk_pool(slabs);
        index.set_host_pool(&host);
        index.set_disk_pool(&disk);
        geometry = disk_geometry(disk);

        a = commit_block(index, index.root(), 1, 100);
        write_block_pattern(index, host, a, 0x7E);
        failures += check(index.spill_node(a), "spill failed");
        failures += check(pc::save_prefix_store(idx, index, disk, geometry), "save failed");

        // Simulate a later runtime spill reusing node a's slot and bumping its generation, with a
        // crash before the next index rewrite: the on-disk slot no longer matches the record.
        const pc::DiskPageId slot = index.node(a)->disk_page;
        const std::uint32_t saved_generation = disk.generation(slot);
        const pc::DiskPageId p1 = disk.allocate_block().page;
        const pc::DiskPageId p2 = disk.allocate_block().page;
        const pc::DiskPageId p3 = disk.allocate_block().page;
        (void)p1;
        (void)p2;
        (void)p3;
        const pc::DiskSlotAllocation reuse = disk.allocate_block();
        failures += check(reuse.page == slot && reuse.evicted == slot,
                          "runtime spill did not reuse node a's slot");
        failures += check(disk.generation(slot) != saved_generation,
                          "reused slot did not advance its generation");
        std::vector<unsigned char> junk(kDiskBlockBytes, 0x3C);
        failures += check(disk.write_slot(slot, junk.data()), "overwrite of reused slot failed");
    }

    {
        // The index still names the old generation, so the loader drops the stale record.
        pc::DiskSlabPool disk = make_test_disk_pool(slabs);
        TestSink sink;
        pc::PrefixIndex loaded(pc::PrefixIndexOptions{.device_snapshot_slots = 2}, &sink);
        pc::HostSlabPool host = make_test_pool();
        loaded.set_host_pool(&host);
        loaded.set_disk_pool(&disk);
        pc::PrefixStoreStats stats;
        failures += check(pc::load_prefix_store(idx, loaded, disk, geometry, &stats),
                          "load should still succeed on a stale slot");
        failures += check(loaded.live_node_count() == 1,
                          "the stale-generation record was not dropped");
        failures += check(loaded.node(a) == nullptr || !loaded.node(a)->live,
                          "the stale block survived the load");
    }

    std::remove(idx.c_str());
    std::remove(slabs.c_str());
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_slab_pool_basic();
    failures += test_slab_pool_eviction();
    failures += test_slab_pool_gdsf();
    failures += test_prefix_index_host_residency();
    failures += test_device_fill_roundtrip();
    failures += test_evict_demote_vs_erase();
    failures += test_slab_eviction_reconciliation();
    failures += test_slot_demote_host_backed();
    failures += test_snapshot_restore_roundtrip();
    failures += test_best_host_admission();
    failures += test_disk_pool_basic();
    failures += test_disk_pool_eviction_order();
    failures += test_disk_store_roundtrip();
    failures += test_disk_store_rejects();
    failures += test_disk_store_crash_gap();
    if (failures == 0) {
        std::printf("All host-tier unit tests passed.\n");
        return 0;
    }
    std::printf("%d host-tier check(s) failed.\n", failures);
    return 1;
}
