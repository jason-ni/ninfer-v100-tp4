#include "runtime/prefix_cache/prefix_index.h"

#include "core/paged_kv_cache.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::prefix_cache {
namespace {

static_assert(kBlockTokens == kPagedKVPageSize,
              "prefix-cache block must equal the paged KV page size");

[[nodiscard]] std::uint64_t mix(std::uint64_t value) noexcept {
    value += 0x9E37'79B9'7F4A'7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58'476D'1CE4'E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D0'49BB'1331'11EBULL;
    return value ^ (value >> 31U);
}

} // namespace

std::uint64_t block_hash(std::uint64_t parent_hash, std::span<const TokenId> tokens,
                         std::uint64_t extra) noexcept {
    std::uint64_t hash = mix(parent_hash ^ 0xA076'1D64'CF15'1E25ULL);
    for (TokenId token : tokens) {
        hash = mix(hash ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(token)) *
                           0xE703'7B9D'2C1F'1235ULL));
    }
    return mix(hash ^ extra);
}

std::vector<std::uint64_t> chain_block_hashes(std::span<const TokenId> tokens,
                                              std::span<const std::uint64_t> extras) {
    const std::size_t blocks = tokens.size() / static_cast<std::size_t>(kBlockTokens);
    if (!extras.empty() && extras.size() != blocks) {
        throw std::invalid_argument("block extras must supply one key per full block");
    }
    std::vector<std::uint64_t> hashes(blocks);
    std::uint64_t parent = kBlockHashSeed;
    for (std::size_t b = 0; b < blocks; ++b) {
        const std::uint64_t extra = extras.empty() ? 0U : extras[b];
        parent = block_hash(parent,
                            tokens.subspan(b * static_cast<std::size_t>(kBlockTokens),
                                           static_cast<std::size_t>(kBlockTokens)),
                            extra);
        hashes[b] = parent;
    }
    return hashes;
}

PrefixIndex::PrefixIndex(PrefixIndexOptions options, PrefixPageSink* sink)
    : options_(options), sink_(sink), tick_(options.now_tick) {
    nodes_.reserve(256);
    snapshots_.reserve(64);
    slot_owner_.assign(options_.device_snapshot_slots, kInvalidSnapshot);
    free_slots_.reserve(options_.device_snapshot_slots);
    for (std::int32_t slot = static_cast<std::int32_t>(options_.device_snapshot_slots) - 1;
         slot >= 0; --slot) {
        free_slots_.push_back(slot);
    }

    root_ = allocate_node();
    BlockNode& root = nodes_[root_ - 1U];
    root.parent       = kInvalidNode;
    root.depth        = -1;
    root.lookup_hash  = kBlockHashSeed;
    root.device_state = Residency::Absent;
}

PrefixIndex::~PrefixIndex() = default;

const BlockNode* PrefixIndex::node(NodeId id) const noexcept {
    if (!valid(id)) { return nullptr; }
    return &nodes_[id - 1U];
}

const Snapshot* PrefixIndex::snapshot(SnapshotId id) const noexcept {
    if (id == kInvalidSnapshot || id > snapshots_.size()) { return nullptr; }
    const Snapshot& record = snapshots_[id - 1U];
    return record.live ? &record : nullptr;
}

SnapshotId PrefixIndex::snapshot_id_at(std::size_t index) const noexcept {
    if (index >= snapshots_.size()) { return kInvalidSnapshot; }
    const Snapshot& record = snapshots_[index];
    return record.live ? static_cast<SnapshotId>(index + 1U) : kInvalidSnapshot;
}

NodeId PrefixIndex::allocate_node() {
    NodeId id;
    if (!free_nodes_.empty()) {
        id = free_nodes_.back();
        free_nodes_.pop_back();
        BlockNode& reused = nodes_[id - 1U];
        const std::uint32_t generation = reused.generation + 1U;
        reused = BlockNode{};
        reused.generation = generation;
    } else {
        nodes_.emplace_back();
        id = static_cast<NodeId>(nodes_.size());
    }
    ++live_nodes_;
    return id;
}

void PrefixIndex::free_node(NodeId id) noexcept {
    BlockNode& record = nodes_[id - 1U];
    record.live       = false;
    record.children.clear();
    record.snapshots.clear();
    record.device_state = Residency::Absent;
    free_nodes_.push_back(id);
    --live_nodes_;
}

void PrefixIndex::link_lru_tail(NodeId id) noexcept {
    BlockNode& record = nodes_[id - 1U];
    record.lru_prev   = lru_tail_;
    record.lru_next   = kInvalidNode;
    if (lru_tail_ != kInvalidNode) {
        nodes_[lru_tail_ - 1U].lru_next = id;
    } else {
        lru_head_ = id;
    }
    lru_tail_          = id;
    record.lru_linked  = true;
    ++evictable_pages_;
}

void PrefixIndex::unlink_lru(NodeId id) noexcept {
    BlockNode& record = nodes_[id - 1U];
    if (!record.lru_linked) { return; }
    if (record.lru_prev != kInvalidNode) {
        nodes_[record.lru_prev - 1U].lru_next = record.lru_next;
    } else {
        lru_head_ = record.lru_next;
    }
    if (record.lru_next != kInvalidNode) {
        nodes_[record.lru_next - 1U].lru_prev = record.lru_prev;
    } else {
        lru_tail_ = record.lru_prev;
    }
    record.lru_prev   = kInvalidNode;
    record.lru_next   = kInvalidNode;
    record.lru_linked = false;
    if (evictable_pages_ > 0U) { --evictable_pages_; }
}

void PrefixIndex::pin_node(NodeId id) noexcept {
    if (!valid(id)) { return; }
    BlockNode& record = nodes_[id - 1U];
    if (record.active_refs == 0U) { unlink_lru(id); }
    ++record.active_refs;
}

void PrefixIndex::unpin_node(NodeId id) noexcept {
    if (!valid(id)) { return; }
    BlockNode& record = nodes_[id - 1U];
    if (record.active_refs == 0U) { return; }
    --record.active_refs;
    if (record.active_refs == 0U && record.device_state == Residency::Resident &&
        !record.lru_linked) {
        link_lru_tail(id);
    }
}

void PrefixIndex::pin_restore(NodeId id) noexcept {
    if (!valid(id)) { return; }
    BlockNode& record = nodes_[id - 1U];
    if (record.restore_pins == 0U && record.active_refs == 0U &&
        record.device_state == Residency::Resident) {
        unlink_lru(id);
    }
    ++record.restore_pins;
}

void PrefixIndex::unpin_restore(NodeId id) noexcept {
    if (!valid(id)) { return; }
    BlockNode& record = nodes_[id - 1U];
    if (record.restore_pins == 0U) { return; }
    --record.restore_pins;
    if (record.restore_pins == 0U && record.active_refs == 0U &&
        record.device_state == Residency::Resident && !record.lru_linked) {
        link_lru_tail(id);
    }
}

void PrefixIndex::pin_snapshot(SnapshotId id) noexcept {
    if (id == kInvalidSnapshot || id > snapshots_.size()) { return; }
    Snapshot& record = snapshots_[id - 1U];
    if (record.live) { ++record.restore_pins; }
}

void PrefixIndex::unpin_snapshot(SnapshotId id) noexcept {
    if (id == kInvalidSnapshot || id > snapshots_.size()) { return; }
    Snapshot& record = snapshots_[id - 1U];
    if (record.live && record.restore_pins != 0U) { --record.restore_pins; }
}

void PrefixIndex::inc_live_below(NodeId node) noexcept {
    for (NodeId cur = node; cur != kInvalidNode; cur = nodes_[cur - 1U].parent) {
        ++nodes_[cur - 1U].live_snapshots_below;
    }
}

void PrefixIndex::dec_live_below(NodeId node, std::uint32_t count) noexcept {
    for (NodeId cur = node; cur != kInvalidNode; cur = nodes_[cur - 1U].parent) {
        BlockNode& record = nodes_[cur - 1U];
        record.live_snapshots_below =
            record.live_snapshots_below >= count ? record.live_snapshots_below - count : 0U;
    }
}

ChildMatch PrefixIndex::find_child(NodeId parent, std::uint64_t lookup_hash,
                                   std::span<const TokenId> tokens, std::uint64_t extra) const {
    const BlockNode* parent_node = node(parent);
    if (parent_node == nullptr || tokens.size() != kBlockTokens) { return {}; }
    ChildMatch match;
    for (NodeId child_id : parent_node->children) {
        const BlockNode& child = nodes_[child_id - 1U];
        if (!child.live) { continue; }
        if (child.lookup_hash != lookup_hash) { continue; }
        if (child.extra == extra &&
            std::equal(tokens.begin(), tokens.end(), child.tokens.begin())) {
            match.node = child_id;
            return match;
        }
        match.hash_collided = true;
    }
    return match;
}

PrefixMatch PrefixIndex::match(std::span<const TokenId> prompt_tokens,
                               std::span<const std::uint64_t> block_hashes,
                               std::span<const std::uint64_t> block_extras) const {
    PrefixMatch result;
    result.path.push_back(root_);

    const std::size_t token_count = prompt_tokens.size();
    std::size_t eligible = 0;
    if (token_count >= 1U) { eligible = (token_count - 1U) / kBlockTokens; }

    NodeId current = root_;
    std::size_t block = 0;
    for (; block < eligible && block < block_hashes.size(); ++block) {
        const std::span<const TokenId> tokens =
            prompt_tokens.subspan(block * kBlockTokens, kBlockTokens);
        const std::uint64_t extra = block < block_extras.size() ? block_extras[block] : 0U;
        const ChildMatch child = find_child(current, block_hashes[block], tokens, extra);
        if (child.node == kInvalidNode) { break; }
        current = child.node;
        result.path.push_back(current);
    }
    result.matched_blocks = result.path.size() - 1U;

    for (NodeId node_id : result.path) {
        const BlockNode& record = nodes_[node_id - 1U];
        for (SnapshotId snapshot_id : record.snapshots) {
            const Snapshot& candidate = snapshots_[snapshot_id - 1U];
            if (!candidate.live || candidate.superseded) { continue; }
            if (candidate.frontier > token_count - 1U) { continue; }

            const BlockNode& anchor = nodes_[candidate.anchor - 1U];
            const std::uint32_t tail_begin =
                static_cast<std::uint32_t>(kBlockTokens) *
                static_cast<std::uint32_t>(anchor.depth + 1);
            if (candidate.frontier != tail_begin + candidate.tail_len) { continue; }
            if (candidate.tail_len != 0U &&
                !std::equal(candidate.tail_tokens.begin(),
                            candidate.tail_tokens.begin() + candidate.tail_len,
                            prompt_tokens.begin() + tail_begin)) {
                continue;
            }

            // A snapshot is valid iff every node on its anchor path, its tail bundle and its image
            // are resident. Path residency is checked against the matched path prefix.
            bool resident = candidate.device_state == Residency::Resident &&
                            (candidate.tail_len == 0U ||
                             candidate.tail_state == Residency::Resident);
            if (resident) {
                const std::size_t anchor_index = static_cast<std::size_t>(anchor.depth + 1);
                // path[0] is the root, which owns no KV pages; residency of a prefix is carried by
                // the deepest matching node.
                for (std::size_t i = 1; i <= anchor_index && i < result.path.size(); ++i) {
                    if (nodes_[result.path[i] - 1U].device_state != Residency::Resident) {
                        resident = false;
                        break;
                    }
                }
            }

            result.candidates.push_back(SnapshotCandidate{snapshot_id, candidate.frontier,
                                                          resident});
            if (resident && candidate.frontier >= result.best_device_frontier) {
                result.best_device          = snapshot_id;
                result.best_device_frontier = candidate.frontier;
            }
        }
    }
    return result;
}

NodeId PrefixIndex::commit_block(const BlockCommit& commit, bool& adopted) {
    adopted = false;
    if (!valid(commit.parent) || commit.tokens.size() != kBlockTokens) { return kInvalidNode; }

    const ChildMatch existing =
        find_child(commit.parent, commit.lookup_hash, commit.tokens, commit.extra);
    if (existing.node != kInvalidNode) {
        BlockNode& record = nodes_[existing.node - 1U];
        if (record.device_state != Residency::Resident) {
            record.device_pages  = commit.pages;
            record.device_state  = Residency::Resident;
            record.lru_backed    = false;
            link_lru_tail(existing.node);
            adopted = true;
        }
        record.last_access_tick = ++tick_;
        return existing.node;
    }

    const NodeId id = allocate_node();
    BlockNode& record = nodes_[id - 1U];
    record.parent      = commit.parent;
    record.depth       = nodes_[commit.parent - 1U].depth + 1;
    record.lookup_hash = commit.lookup_hash;
    record.extra       = commit.extra;
    std::copy(commit.tokens.begin(), commit.tokens.end(), record.tokens.begin());
    record.device_pages     = commit.pages;
    record.device_state     = Residency::Resident;
    record.lru_backed       = false;
    record.last_access_tick = ++tick_;
    nodes_[commit.parent - 1U].children.push_back(id);
    link_lru_tail(id);
    adopted = true;
    return id;
}

SlotId PrefixIndex::acquire_slot(std::uint64_t priority) {
    if (!free_slots_.empty()) {
        const SlotId slot = free_slots_.back();
        free_slots_.pop_back();
        slot_owner_[static_cast<std::size_t>(slot)] = kInvalidSnapshot;
        return slot;
    }

    // Oldest superseded snapshot's slot first.
    SnapshotId superseded_victim = kInvalidSnapshot;
    std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t index = 0; index < snapshots_.size(); ++index) {
        const Snapshot& record = snapshots_[index];
        if (!record.live || !record.superseded || record.restore_pins != 0U ||
            record.device_slot == kInvalidSlot) {
            continue;
        }
        if (record.superseded_at < oldest) {
            oldest           = record.superseded_at;
            superseded_victim = static_cast<SnapshotId>(index + 1U);
        }
    }
    if (superseded_victim != kInvalidSnapshot) {
        const SlotId slot = detach_snapshot(superseded_victim);
        slot_owner_[static_cast<std::size_t>(slot)] = kInvalidSnapshot;
        return slot;
    }

    // Device-only snapshot with the lowest GDSF priority, if the new one is at least as valuable.
    SnapshotId lowest_victim = kInvalidSnapshot;
    std::uint64_t lowest     = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t index = 0; index < snapshots_.size(); ++index) {
        const Snapshot& record = snapshots_[index];
        if (!record.live || record.restore_pins != 0U || record.device_slot == kInvalidSlot) {
            continue;
        }
        if (record.gdsf_h < lowest) {
            lowest        = record.gdsf_h;
            lowest_victim = static_cast<SnapshotId>(index + 1U);
        }
    }
    if (lowest_victim != kInvalidSnapshot && lowest <= priority) {
        const SlotId slot = detach_snapshot(lowest_victim);
        slot_owner_[static_cast<std::size_t>(slot)] = kInvalidSnapshot;
        return slot;
    }
    return kInvalidSlot;
}

SnapshotId PrefixIndex::add_snapshot(const SnapshotRequest& request) {
    if (!valid(request.anchor)) {
        return kInvalidSnapshot;
    }
    const BlockNode& anchor = nodes_[request.anchor - 1U];
    const std::uint32_t tail_begin =
        static_cast<std::uint32_t>(kBlockTokens) * static_cast<std::uint32_t>(anchor.depth + 1);
    if (request.frontier < tail_begin) {
        return kInvalidSnapshot;
    }
    const std::uint32_t tail_len = request.frontier - tail_begin;
    if (tail_len >= kBlockTokens || request.tail_tokens.size() != tail_len) {
        // tail_len == 0 is a frontier exactly on a block boundary (a tap snapshot).
        return kInvalidSnapshot;
    }

    // At most one snapshot per (anchor, exact tail tokens).
    for (SnapshotId existing : anchor.snapshots) {
        const Snapshot& record = snapshots_[existing - 1U];
        if (!record.live) { continue; }
        if (record.frontier == request.frontier &&
            record.tail_len == tail_len &&
            std::equal(record.tail_tokens.begin(), record.tail_tokens.begin() + tail_len,
                       request.tail_tokens.begin())) {
            hit(existing);
            return existing;
        }
    }

    // Take a slot. A reserved slot was already removed from free_slots_ by reserve_snapshot_slot,
    // so use it directly. Otherwise acquire a new one from the free/GDSF paths.
    SlotId slot = request.reserved_slot;
    if (slot == kInvalidSlot) {
        slot = acquire_slot(request.priority);
        if (slot == kInvalidSlot && request.supersede != kInvalidSnapshot) {
            supersede(request.supersede);
            slot = acquire_slot(request.priority);
        }
    }
    if (slot == kInvalidSlot) { return kInvalidSnapshot; }

    // Never recycle the id of the snapshot just superseded in this call: a caller may still hold it
    // to supersede on finish.
    SnapshotId id = kInvalidSnapshot;
    for (std::size_t index = free_snapshots_.size(); index-- > 0U;) {
        if (free_snapshots_[index] == request.supersede) { continue; }
        id                  = free_snapshots_[index];
        free_snapshots_[index] = free_snapshots_.back();
        free_snapshots_.pop_back();
        break;
    }
    if (id != kInvalidSnapshot) {
        Snapshot& reused               = snapshots_[id - 1U];
        const std::uint32_t generation = reused.generation + 1U;
        reused                         = Snapshot{};
        reused.generation              = generation;
    } else {
        snapshots_.emplace_back();
        id = static_cast<SnapshotId>(snapshots_.size());
    }
    Snapshot& record = snapshots_[id - 1U];
    record.anchor       = request.anchor;
    record.frontier     = request.frontier;
    record.tail_len     = static_cast<std::uint8_t>(tail_len);
    std::copy(request.tail_tokens.begin(), request.tail_tokens.end(), record.tail_tokens.begin());
    record.tail_pages    = request.tail_pages;
    record.tail_state    = tail_len > 0U ? Residency::Resident : Residency::Absent;
    record.device_slot   = slot;
    record.device_state  = Residency::Resident;
    record.kind          = request.kind;
    record.superseded    = false;
    record.superseded_at = 0;
    record.gdsf_h        = request.priority;
    record.freq          = 1;
    record.last_hit_tick = ++tick_;
    record.live          = true;

    slot_owner_[static_cast<std::size_t>(slot)] = id;
    nodes_[request.anchor - 1U].snapshots.push_back(id);
    inc_live_below(request.anchor);
    ++live_snapshots_;
    return id;
}

SlotId PrefixIndex::reserve_snapshot_slot(std::uint64_t priority) { return acquire_slot(priority); }

void PrefixIndex::release_reserved_slot(SlotId slot) noexcept {
    if (slot == kInvalidSlot) { return; }
    free_slots_.push_back(slot);
}

void PrefixIndex::supersede(SnapshotId id) noexcept {
    Snapshot* record = id == kInvalidSnapshot ? nullptr
                     : (id <= snapshots_.size() && snapshots_[id - 1U].live ? &snapshots_[id - 1U]
                                                                            : nullptr);
    if (record == nullptr || record->superseded) { return; }
    record->superseded    = true;
    record->superseded_at = ++tick_;
}

void PrefixIndex::hit(SnapshotId id) noexcept {
    if (id == kInvalidSnapshot || id > snapshots_.size()) { return; }
    Snapshot& record = snapshots_[id - 1U];
    if (!record.live) { return; }
    record.superseded    = false;
    record.superseded_at = 0;
    ++record.freq;
    record.last_hit_tick = ++tick_;
}

SlotId PrefixIndex::detach_snapshot(SnapshotId id) {
    if (id == kInvalidSnapshot || id > snapshots_.size()) { return kInvalidSlot; }
    Snapshot& record = snapshots_[id - 1U];
    if (!record.live) { return kInvalidSlot; }

    if (record.tail_state == Residency::Resident && sink_ != nullptr) {
        sink_->release_tail_pages(record.tail_pages);
    }
    const SlotId slot = record.device_slot;

    if (valid(record.anchor)) {
        std::vector<SnapshotId>& list = nodes_[record.anchor - 1U].snapshots;
        list.erase(std::remove(list.begin(), list.end(), id), list.end());
    }
    dec_live_below(record.anchor, 1U);

    record.live = false;
    free_snapshots_.push_back(id);
    --live_snapshots_;
    return slot;
}

void PrefixIndex::delete_snapshot(SnapshotId id) {
    const SlotId slot = detach_snapshot(id);
    if (slot == kInvalidSlot) { return; }
    if (sink_ != nullptr) { sink_->release_snapshot_slot(slot); }
    slot_owner_[static_cast<std::size_t>(slot)] = kInvalidSnapshot;
    free_slots_.push_back(slot);
}

void PrefixIndex::remove_snapshot(SnapshotId id) { delete_snapshot(id); }

NodeId PrefixIndex::evict_device_lru() {
    if (lru_head_ == kInvalidNode) { return kInvalidNode; }
    const NodeId victim = lru_head_;
    const BlockNode& record = nodes_[victim - 1U];
    if (record.active_refs != 0U || record.restore_pins != 0U) { return kInvalidNode; }
    unlink_lru(victim);
    erase_subtree(victim);
    return victim;
}

void PrefixIndex::erase_subtree(NodeId node) {
    if (!valid(node) || node == root_) { return; }

    std::vector<NodeId> order;
    std::vector<NodeId> stack{node};
    while (!stack.empty()) {
        const NodeId current = stack.back();
        stack.pop_back();
        const BlockNode& record = nodes_[current - 1U];
        if (!record.live || record.active_refs != 0U || record.restore_pins != 0U) { return; }
        order.push_back(current);
        for (NodeId child : record.children) { stack.push_back(child); }
    }

    for (NodeId current : order) {
        BlockNode& record = nodes_[current - 1U];
        const std::vector<SnapshotId> anchored = record.snapshots;
        for (SnapshotId id : anchored) { delete_snapshot(id); }
    }

    const NodeId parent = nodes_[node - 1U].parent;
    if (valid(parent)) {
        std::vector<NodeId>& siblings = nodes_[parent - 1U].children;
        siblings.erase(std::remove(siblings.begin(), siblings.end(), node), siblings.end());
    }

    for (NodeId current : order) {
        BlockNode& record = nodes_[current - 1U];
        if (record.lru_linked) { unlink_lru(current); }
        if (record.device_state == Residency::Resident) {
            if (sink_ != nullptr) { sink_->release_block_pages(record.device_pages); }
            record.device_state = Residency::Absent;
        }
        free_node(current);
    }
}

} // namespace ninfer::prefix_cache
