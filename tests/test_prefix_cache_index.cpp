#include "runtime/prefix_cache/prefix_index.h"
#include "runtime/prefix_cache/tap_planner.h"

#include <array>
#include <algorithm>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using ninfer::TokenId;
using namespace ninfer::prefix_cache;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::vector<TokenId> tokens(std::size_t count, TokenId seed) {
    std::vector<TokenId> out(count);
    for (std::size_t i = 0; i < count; ++i) { out[i] = seed + static_cast<TokenId>(i); }
    return out;
}

// Frontend-style chained hashes for every full 64-token block of `prompt`.
std::vector<std::uint64_t> chain(std::span<const TokenId> prompt, std::uint64_t extra = 0) {
    std::vector<std::uint64_t> hashes;
    const std::size_t full = prompt.size() / kBlockTokens;
    std::uint64_t hash     = kBlockHashSeed;
    for (std::size_t block = 0; block < full; ++block) {
        hash = block_hash(hash, prompt.subspan(block * kBlockTokens, kBlockTokens), extra);
        hashes.push_back(hash);
    }
    return hashes;
}

PoolPages pages(std::int32_t first, std::int32_t second = -1) {
    return PoolPages{std::array<std::int32_t, kMaxPools>{first, second}};
}

struct CountingSink final : PrefixPageSink {
    int block_releases = 0;
    int tail_releases  = 0;
    int slot_releases  = 0;
    void release_block_pages(const PoolPages&) noexcept override { ++block_releases; }
    void release_tail_pages(const PoolPages&) noexcept override { ++tail_releases; }
    void release_snapshot_slot(SlotId) noexcept override { ++slot_releases; }
};

NodeId commit(PrefixIndex& index, NodeId parent, std::span<const TokenId> block,
              std::uint64_t hash, PoolPages block_pages, bool* adopted = nullptr) {
    BlockCommit request;
    request.parent      = parent;
    request.lookup_hash = hash;
    request.extra       = 0;
    request.tokens      = block;
    request.pages       = block_pages;
    bool local          = false;
    const NodeId id     = index.commit_block(request, local);
    if (adopted != nullptr) { *adopted = local; }
    return id;
}

SnapshotId add(PrefixIndex& index, NodeId anchor, std::uint32_t frontier,
               std::span<const TokenId> tail, PoolPages tail_pages = {},
               SnapshotKind kind = SnapshotKind::Tap, std::uint64_t priority = 5,
               SnapshotId supersede = kInvalidSnapshot) {
    SnapshotRequest request;
    request.anchor     = anchor;
    request.frontier   = frontier;
    request.tail_tokens = tail;
    request.tail_pages = tail_pages;
    request.kind       = kind;
    request.priority   = priority;
    request.supersede  = supersede;
    return index.add_snapshot(request);
}

int test_block_hash() {
    int failures         = 0;
    const auto a         = tokens(kBlockTokens, 10);
    const auto b         = tokens(kBlockTokens, 11);
    const auto hash_a    = block_hash(kBlockHashSeed, a, 0);
    const auto hash_again = block_hash(kBlockHashSeed, a, 0);
    failures += check(hash_a == hash_again, "block hash is not deterministic");
    failures += check(hash_a != block_hash(kBlockHashSeed, b, 0), "block hash ignores tokens");
    failures += check(hash_a != block_hash(kBlockHashSeed, a, 1), "block hash ignores media key");
    failures += check(hash_a != block_hash(0x1234, a, 0), "block hash ignores the parent hash");
    return failures;
}

int test_exact_recompare_under_collision() {
    int failures = 0;
    CountingSink sink;
    PrefixIndex index(PrefixIndexOptions{2, 0}, &sink);

    const auto a = tokens(kBlockTokens, 100);
    const auto b = tokens(kBlockTokens, 200);
    const auto c = tokens(kBlockTokens, 300);
    const std::uint64_t forced = 0xDEADBEEF;

    bool adopted = false;
    const NodeId node_a = commit(index, index.root(), a, forced, pages(1), &adopted);
    failures += check(node_a != kInvalidNode && adopted, "first colliding block was not inserted");
    const NodeId node_b = commit(index, index.root(), b, forced, pages(2), &adopted);
    failures += check(node_b != kInvalidNode && adopted && node_b != node_a,
                      "same-hash different tokens did not create a distinct node");

    const ChildMatch find_a = index.find_child(index.root(), forced, a, 0);
    const ChildMatch find_b = index.find_child(index.root(), forced, b, 0);
    const ChildMatch find_c = index.find_child(index.root(), forced, c, 0);
    failures += check(find_a.node == node_a && !find_a.hash_collided,
                      "exact re-compare did not resolve the first colliding child");
    failures += check(find_b.node == node_b, "exact re-compare did not resolve the second child");
    failures += check(find_c.node == kInvalidNode && find_c.hash_collided,
                      "same-hash miss did not report a collision");
    return failures;
}

int test_commit_adopt_duplicate_and_match() {
    int failures = 0;
    CountingSink sink;
    PrefixIndex index(PrefixIndexOptions{2, 0}, &sink);

    const auto prompt = tokens(200, 1000);
    const auto hashes = chain(prompt);
    const auto block0 = std::span<const TokenId>(prompt.data(), kBlockTokens);

    bool adopted = false;
    const NodeId node0 = commit(index, index.root(), block0, hashes[0], pages(7), &adopted);
    failures += check(node0 != kInvalidNode && adopted, "block0 was not committed");

    // Re-committing the same block is a duplicate; the existing device copy is kept.
    const NodeId again = commit(index, index.root(), block0, hashes[0], pages(8), &adopted);
    failures += check(again == node0 && !adopted, "duplicate block was not deduplicated");

    // A tap snapshot sits exactly on the block boundary (tail_len == 0).
    const SnapshotId tap = add(index, node0, 64, {}, {}, SnapshotKind::Tap, 5);
    failures += check(tap != kInvalidSnapshot, "block-boundary tap snapshot was rejected");
    failures += check(index.snapshot(tap)->tail_len == 0, "tap snapshot carries a tail");

    // A tail snapshot at frontier 100 with tokens [64,100).
    const auto tail = std::span<const TokenId>(prompt.data() + 64, 36);
    const SnapshotId tail_snap = add(index, node0, 100, tail, pages(9), SnapshotKind::Endpoint, 5);
    failures += check(tail_snap != kInvalidSnapshot, "tail snapshot was rejected");

    PrefixMatch match = index.match(prompt, hashes, {});
    failures += check(match.matched_blocks == 1, "matched blocks did not stop at the first miss");
    failures += check(match.candidates.size() == 2, "not every anchored snapshot was collected");
    failures += check(match.best_device == tail_snap && match.best_device_frontier == 100,
                      "deepest device snapshot was not selected");

    // F must be <= n-1: at n == 100 the frontier-100 snapshot is excluded, the tap survives.
    PrefixMatch short_match = index.match(std::span<const TokenId>(prompt.data(), 100),
                                          std::span<const std::uint64_t>(hashes.data(),
                                                                         hashes.size()),
                                          {});
    failures += check(short_match.best_device == tap && short_match.best_device_frontier == 64,
                      "frontier == n was incorrectly admitted");

    const std::uint64_t missing_hash = block_hash(block_hash(kBlockHashSeed, block0, 0),
                                                  std::span<const TokenId>(prompt.data() + 64,
                                                                           kBlockTokens),
                                                  0);
    const std::array<std::uint64_t, 1> unmatched{missing_hash};
    PrefixMatch miss = index.match(prompt, unmatched, {});
    failures += check(miss.matched_blocks == 0 && miss.best_device == kInvalidSnapshot,
                      "uncommitted block was treated as a match");
    return failures;
}

int test_live_counts_and_removal() {
    int failures = 0;
    CountingSink sink;
    PrefixIndex index(PrefixIndexOptions{2, 0}, &sink);

    const auto block0 = tokens(kBlockTokens, 1);
    const std::uint64_t hash = block_hash(kBlockHashSeed, block0, 0);
    const NodeId node0 = commit(index, index.root(), block0, hash, pages(1));
    const SnapshotId snap = add(index, node0, 64, {});
    failures += check(index.node(node0)->live_snapshots_below == 1 &&
                          index.node(index.root())->live_snapshots_below == 1,
                      "live_snapshots_below was not propagated to the root");

    index.remove_snapshot(snap);
    failures += check(index.live_snapshot_count() == 0, "removing a snapshot did not decrease count");
    failures += check(index.node(node0)->live_snapshots_below == 0 &&
                          index.node(index.root())->live_snapshots_below == 0,
                      "live_snapshots_below was not decremented on removal");
    failures += check(sink.slot_releases == 1, "removing a snapshot did not release its device slot");
    return failures;
}

int test_supersede_and_slot_reuse() {
    int failures = 0;
    CountingSink sink;
    PrefixIndex index(PrefixIndexOptions{1, 0}, &sink);

    const auto prompt = tokens(200, 500);
    const auto block0 = std::span<const TokenId>(prompt.data(), kBlockTokens);
    const NodeId node0 = commit(index, index.root(), block0, chain(prompt)[0], pages(1));

    const SnapshotId first = add(index, node0, 64, {}, {}, SnapshotKind::Tap, 5);
    const auto tail = std::span<const TokenId>(prompt.data() + 64, 36);
    const SnapshotId second =
        add(index, node0, 100, tail, pages(2), SnapshotKind::Endpoint, 5, first);

    failures += check(first != kInvalidSnapshot && second != kInvalidSnapshot,
                      "snapshot admission failed");
    failures += check(index.snapshot(first) == nullptr, "superseded snapshot was not reclaimed");
    failures += check(index.live_snapshot_count() == 1, "superseded snapshot still counted as live");
    failures += check(index.snapshot(second)->device_slot == 0,
                      "reused device slot was not transferred to the new snapshot");
    failures += check(sink.slot_releases == 0,
                      "a slot reused in place should not be released to the integration");
    return failures;
}

int test_lru_eviction_and_pinning() {
    int failures = 0;
    CountingSink sink;
    PrefixIndex index(PrefixIndexOptions{2, 0}, &sink);

    const auto block = tokens(kBlockTokens, 42);
    const std::uint64_t hash = block_hash(kBlockHashSeed, block, 0);
    const NodeId a = commit(index, index.root(), block, hash, pages(1));
    const NodeId b = commit(index, a, block, hash, pages(2));
    const SnapshotId snap = add(index, b, 128, {}, {}, SnapshotKind::Tap, 5);

    index.pin_node(a);
    const NodeId evicted = index.evict_node({});
    failures += check(evicted == b, "LRU did not evict the oldest unpinned node");
    failures += check(index.node(a) != nullptr, "pinned node was evicted");
    failures += check(index.snapshot(snap) == nullptr && index.live_snapshot_count() == 0,
                      "evicting a node did not delete its anchored snapshot");
    failures += check(index.node(a)->live_snapshots_below == 0,
                      "subtree eviction did not decrement live_snapshots_below");
    failures += check(sink.block_releases == 1 && sink.slot_releases == 1,
                      "eviction did not return the block page and snapshot slot");

    index.unpin_node(a);
    failures += check(index.evict_node({}) == a, "unpinned node was not evictable");
    failures += check(index.live_node_count() == 1, "root plus no children expected after eviction");
    return failures;
}

int test_match_miss_diagnostics() {
    int failures = 0;
    CountingSink sink;
    PrefixIndex index(PrefixIndexOptions{2, 0}, &sink);

    const auto p1 = tokens(kBlockTokens, 1);
    const auto p2 = tokens(kBlockTokens, 2);
    const auto p3 = tokens(kBlockTokens, 3);
    const std::uint64_t h1 = block_hash(kBlockHashSeed, p1, 0);
    const std::uint64_t h2 = block_hash(h1, p2, 0);
    const std::uint64_t h3 = block_hash(h2, p3, 0);
    const NodeId n1 = commit(index, index.root(), p1, h1, pages(1));
    const NodeId n2 = commit(index, n1, p2, h2, pages(2));
    const NodeId n3 = commit(index, n2, p3, h3, pages(3));
    failures += check(n1 != kInvalidNode && n2 != kInvalidNode && n3 != kInvalidNode,
                      "diagnostic chain commits failed");

    // The incoming prompt branches at block 1 (the second block differs).
    const std::vector<TokenId> branch_prompt = [&] {
        std::vector<TokenId> prompt;
        for (TokenId value : p1) { prompt.push_back(value); }
        for (TokenId value : tokens(kBlockTokens, 40)) { prompt.push_back(value); }
        prompt.push_back(0xACE); // a partial token keeps the second block eligible
        return prompt;
    }();
    const std::array<std::uint64_t, 2> branch_hashes{h1, 0xC0FFEE};
    const PrefixMatch branch = index.match(branch_prompt, branch_hashes, {});
    failures += check(branch.matched_blocks == 1, "branch did not walk to the first match");
    failures += check(branch.walk_cause == ninfer::PrefixStallCause::NoChild,
                      "branching prompt was not reported as NoChild");
    failures += check(branch.stall_sibling_count == 1,
                      "no-child stall should count the sibling child block");

    // Same lookup hash on two committed children and a third token set colliding with both.
    CountingSink sink2;
    PrefixIndex collided(PrefixIndexOptions{2, 0}, &sink2);
    const std::uint64_t forced = 0xDEAD0BAF;
    const auto a = tokens(kBlockTokens, 10);
    const auto b = tokens(kBlockTokens, 20);
    std::vector<TokenId> colliding_prompt;
    for (TokenId value : tokens(kBlockTokens, 30)) { colliding_prompt.push_back(value); }
    colliding_prompt.push_back(0xBEEF); // a partial 65th token makes block 0 eligible
    (void)commit(collided, collided.root(), a, forced, pages(1));
    (void)commit(collided, collided.root(), b, forced, pages(2));
    const std::array<std::uint64_t, 1> forced_hashes{forced};
    const PrefixMatch hit = collided.match(colliding_prompt, forced_hashes, {});
    failures += check(hit.walk_cause == ninfer::PrefixStallCause::HashCollision,
                      "same-hash different-token child was not reported as a collision");
    failures += check(hit.stall_sibling_count == 2, "collision should count both sibling blocks");

    // Completed walk with two eligible taps: the deeper one wins, the shallower is outranked.
    const SnapshotId tap_shallow = add(index, n2, 2U * kBlockTokens, {});
    const SnapshotId tap_deep    = add(index, n3, 3U * kBlockTokens, {});
    std::vector<TokenId> full_prompt;
    for (TokenId value : p1) { full_prompt.push_back(value); }
    for (TokenId value : p2) { full_prompt.push_back(value); }
    for (TokenId value : p3) { full_prompt.push_back(value); }
    full_prompt.push_back(0xFEED); // one extra token so all three blocks stay eligible
    const std::array<std::uint64_t, 3> full_hashes{h1, h2, h3};
    const PrefixMatch complete = index.match(full_prompt, full_hashes, {});
    failures += check(complete.walk_cause == ninfer::PrefixStallCause::Completed,
                      "full-chain walk should report completed");
    failures += check(complete.best_device == tap_deep, "deeper tap was not selected");
    failures += check(complete.diagnostics_total == 2 && complete.diagnostics.size() == 2,
                      "both path snapshots should be collected");
    bool shallow_outranked = false;
    bool deep_eligible     = false;
    for (const auto& entry : complete.diagnostics) {
        if (entry.frontier == 2U * kBlockTokens && entry.reject == 5) { shallow_outranked = true; }
        if (entry.frontier == 3U * kBlockTokens && entry.reject == 0) { deep_eligible = true; }
    }
    failures += check(shallow_outranked && deep_eligible,
                      "candidate fates were not recorded (shallow outranked / deep eligible)");
    return failures;
}

int test_eviction_policy_tiers() {
    int failures = 0;
    CountingSink sink;
    PrefixIndex index(PrefixIndexOptions{2, 0}, &sink);

    const auto xa = tokens(kBlockTokens, 21);
    const auto ya = tokens(kBlockTokens, 22);
    const auto za = tokens(kBlockTokens, 23);
    const std::uint64_t hx = block_hash(kBlockHashSeed, xa, 0);
    const std::uint64_t hy = block_hash(kBlockHashSeed, ya, 0);
    const std::uint64_t hz = block_hash(hy, za, 0);
    const NodeId x = commit(index, index.root(), xa, hx, pages(1));
    const NodeId y = commit(index, index.root(), ya, hy, pages(2));
    const SnapshotId snap_y = add(index, y, kBlockTokens, {});
    failures += check(x != kInvalidNode && y != kInvalidNode && snap_y != kInvalidSnapshot,
                      "tier-test setup failed");

    // Tier 2: with no lossless move possible, dead-branch GC removes the snapshot-free leaf x and
    // leaves y (still anchoring a live snapshot) alone.
    const NodeId evicted_x = index.evict_node({});
    failures += check(evicted_x == x && index.node(x) == nullptr,
                      "dead-branch GC did not remove the snapshot-free leaf");
    failures += check(index.node(y) != nullptr && index.snapshot(snap_y) != nullptr,
                      "snapshot-anchored branch was removed by GC");
    failures += check(index.eviction_counters().branch_gc_removed_blocks == 1,
                      "GC removal was not counted");

    // A newly committed leaf under y is trimmable even though y carries demand: the demand closure
    // protects y and its ancestors, everything else stays trimmable.
    const NodeId z = commit(index, y, za, hz, pages(3));
    failures += check(z != kInvalidNode, "demand-scoped leaf commit failed");
    const std::array<NodeId, 1> demand{y};
    const NodeId evicted_z = index.evict_node(demand);
    failures += check(evicted_z == z && index.node(z) == nullptr,
                      "demanded branch's un-demanded leaf was not trimmed");
    failures += check(index.node(y) != nullptr, "demanded node was erased");

    // Final pressure: only y remains under the root with no usable host tier, so tier 3 trims it
    // lossily and the anchored snapshot dies with it.
    const NodeId evicted_y = index.evict_node({});
    failures += check(evicted_y == y && index.node(y) == nullptr, "final tier lost the last candidate");
    failures += check(index.eviction_counters().snapshots_deleted >= 1,
                      "eviction-lane snapshot deletion was not counted");
    return failures;
}

int test_tap_planner_priority_and_clustering() {
    int failures = 0;
    TapPlanOptions options;
    options.prompt_tokens  = 8000;
    options.base_frontier  = 0;
    options.ladder_tokens  = 8192; // no ladder below n
    options.min_gap_tokens = 1024;
    options.max_new_taps   = 8;

    TapHint hints;
    hints.explicit_markers      = {1000};
    hints.generation_openers    = {2000};
    hints.structural_boundaries = {2001, 3000};

    const auto taps = plan_taps(options, hints);
    failures += check(taps.size() == 4, "unexpected number of taps");
    if (taps.size() == 4) {
        failures += check(taps[0].position == 1000 && taps[0].kind == SnapshotKind::Boundary &&
                              taps[0].exact,
                          "explicit marker was not planned as an exact boundary");
        failures += check(taps[1].position == 2000 && taps[1].kind == SnapshotKind::OutputBoundary,
                          "generation opener was not planned");
        failures += check(taps[2].position == 3000, "structural boundary was not planned");
        failures += check(taps[3].position == 7999 && !taps[3].exact, "prompt tail was not planned");
    }

    options.max_new_taps = 2;
    const auto budgeted = plan_taps(options, hints);
    failures += check(budgeted.size() == 2 && budgeted[0].position == 1000 &&
                          budgeted[1].position == 2000,
                      "budget did not keep the highest-priority taps");
    return failures;
}

int test_tap_planner_vision_and_ladder() {
    int failures = 0;
    TapPlanOptions options;
    options.prompt_tokens  = 100000;
    options.base_frontier  = 0;
    options.ladder_tokens  = 8192;
    options.min_gap_tokens = 1024;
    options.max_new_taps   = 8;

    TapHint hints;
    hints.explicit_markers = {1050};
    hints.vision_spans     = {{1000, 1100}};
    hints.message_boundaries = {91000, 83000};

    const auto taps = plan_taps(options, hints);
    // 1050 -> 1000 (Vision span start); ladder 91808->91000, 83616->83000, 67232, 34464; tail 99999.
    failures += check(!taps.empty() && taps.front().position == 1000,
                      "vision-interior frontier did not move to the span start");
    failures += check(std::find_if(taps.begin(), taps.end(), [](const PlannedTap& tap) {
                          return tap.position == 91000;
                      }) != taps.end(),
                      "ladder tap was not snapped to a message boundary");
    failures += check(taps.back().position == 99999, "prompt tail was not the last tap");

    const auto again = plan_taps(options, hints);
    failures += check(taps.size() == again.size() &&
                          std::equal(taps.begin(), taps.end(), again.begin(),
                                     [](const PlannedTap& a, const PlannedTap& b) {
                                         return a.position == b.position && a.exact == b.exact &&
                                                a.kind == b.kind;
                                     }),
                      "tap planning is not deterministic");

    options.ladder_tokens = 0;
    const auto no_ladder = plan_taps(options, hints);
    failures += check(no_ladder.size() == 2, "disabling the ladder left extra taps");
    return failures;
}

int test_chain_block_hashes() {
    int failures = 0;
    const auto prompt = tokens(200, 5); // 3 full blocks plus an 8-token tail
    failures += check(chain_block_hashes(prompt) == chain(prompt, 0),
                      "chain_block_hashes matches the reference chain");

    std::vector<std::uint64_t> extras{0x11, 0x22, 0x33};
    std::vector<std::uint64_t> expected(3);
    std::uint64_t parent = kBlockHashSeed;
    const std::span<const TokenId> view(prompt);
    for (std::size_t block = 0; block < expected.size(); ++block) {
        parent = block_hash(parent, view.subspan(block * kBlockTokens, kBlockTokens), extras[block]);
        expected[block] = parent;
    }
    failures += check(chain_block_hashes(prompt, extras) == expected,
                      "per-block media keys chain exactly");

    auto edited = prompt;
    edited[128] ^= 0x5A5A;
    const auto edited_hashes = chain_block_hashes(edited, extras);
    failures += check(edited_hashes[0] == expected[0] && edited_hashes[1] == expected[1] &&
                          edited_hashes[2] != expected[2],
                      "an edited block changes only its own suffix hash");

    bool threw = false;
    try {
        (void)chain_block_hashes(prompt, std::span<const std::uint64_t>{extras.data(), 2});
    } catch (const std::invalid_argument&) { threw = true; }
    failures += check(threw, "a mismatched media-key count is rejected");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_block_hash();
    failures += test_chain_block_hashes();
    failures += test_exact_recompare_under_collision();
    failures += test_commit_adopt_duplicate_and_match();
    failures += test_live_counts_and_removal();
    failures += test_supersede_and_slot_reuse();
    failures += test_lru_eviction_and_pinning();
    failures += test_match_miss_diagnostics();
    failures += test_eviction_policy_tiers();
    failures += test_tap_planner_priority_and_clustering();
    failures += test_tap_planner_vision_and_ladder();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
