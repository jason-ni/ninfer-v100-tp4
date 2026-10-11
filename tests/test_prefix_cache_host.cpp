// Unit tests for the M2 host tier: HostSlabPool allocation/eviction and PrefixIndex
// host-residency state transitions. No GPU required. Checks are evaluated unconditionally
// (no assert), so they hold in NDEBUG builds as well.

#include "runtime/prefix_cache/host_slab_pool.h"
#include "runtime/prefix_cache/prefix_index.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
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
    const pc::NodeId evicted = index.evict_device_lru();
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
    (void)index.evict_device_lru();
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
    const pc::NodeId evicted  = index.evict_device_lru();
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

    // Now demote a (device-only) and force its... a has no host copy; instead demote c's
    // sibling b: b still owns a slab? No - b's slab was never evicted. Demote b to host-only,
    // then evict b's slab by writing through a fourth node.
    (void)index.evict_device_lru(); // demotes the oldest backed node (b or a; a has no host copy)
    const pc::BlockNode* nb = index.node(b);
    failures += check(nb != nullptr && nb->host_state == pc::Residency::Host,
                      "b should be host-backed before slab eviction");
    if (nb != nullptr && nb->device_state == pc::Residency::Resident) {
        (void)index.evict_device_lru();
    }
    const pc::HostPageId slab_b = index.node(b) != nullptr ? index.node(b)->host_page
                                                           : pc::kInvalidHostPage;
    const pc::NodeId d = commit_block(index, index.root(), 4, 103);
    failures += check(d != pc::kInvalidNode, "fourth block commit failed");
    write_block_to_host(index, d);
    // b was host-only; losing its slab leaves it with no data anywhere, so it is deleted.
    failures += check(index.node(b) == nullptr,
                      "host-only node was not deleted when its slab was evicted");
    (void)slab_b;
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
    (void)index.evict_device_lru();
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
    if (failures == 0) {
        std::printf("All host-tier unit tests passed.\n");
        return 0;
    }
    std::printf("%d host-tier check(s) failed.\n", failures);
    return 1;
}
