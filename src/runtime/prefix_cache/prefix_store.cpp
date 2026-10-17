#include "runtime/prefix_cache/prefix_store.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <system_error>
#include <vector>

namespace ninfer::prefix_cache {
namespace {

constexpr std::uint64_t kStoreMagic   = 0x4E46'4B56'5354'3032ULL; // "NFKVST02"
constexpr std::uint32_t kStoreVersion = 2;

// Header fields written in fixed order. Layout is host-endian and version-bound.
struct Header {
    std::uint64_t magic                  = kStoreMagic;
    std::uint32_t version                = kStoreVersion;
    std::uint32_t tp                     = 0;
    std::uint32_t device_snapshot_slots  = 0;
    std::uint32_t block_slab_count       = 0;
    std::uint32_t snapshot_slab_count    = 0;
    std::uint32_t reserved               = 0;
    std::uint64_t block_rank_bytes       = 0;
    std::uint64_t snapshot_rank_bytes    = 0;
    std::uint64_t block_slab_bytes       = 0;
    std::uint64_t snapshot_slab_bytes    = 0;
    std::uint64_t disk_file_uid          = 0;
    std::uint64_t tick                   = 0;
    std::uint64_t node_count             = 0;
    std::uint64_t live_node_count        = 0;
    std::uint64_t snapshot_count         = 0;
    std::uint64_t live_snapshot_count    = 0;
    std::uint64_t payload_bytes          = 0;
    std::uint64_t checksum               = 0;
};

// FNV-1a 64: a cheap integrity tag, not a security boundary. Catches truncation and gross
// corruption of the small header/record region.
std::uint64_t fnv1a(std::uint64_t hash, const void* data, std::size_t bytes) noexcept {
    const auto* p = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
        hash ^= p[i];
        hash *= 0x100'0000'01B3ULL;
    }
    return hash;
}

// 4 (magic u64 -> wait) ... count explicitly: u64 + 4x u32 + u32 = 8 + 16 + 4 = 28, then 12 u64 =
// 96, total 124, plus the checksum u64 = 132.
constexpr std::size_t kHeaderBytes = 132;

void encode_header(const Header& h, std::array<std::uint8_t, kHeaderBytes>& out) noexcept {
    std::size_t off = 0;
    const auto put = [&](const void* src, std::size_t n) {
        std::memcpy(out.data() + off, src, n);
        off += n;
    };
    put(&h.magic, sizeof(h.magic));
    put(&h.version, sizeof(h.version));
    put(&h.tp, sizeof(h.tp));
    put(&h.device_snapshot_slots, sizeof(h.device_snapshot_slots));
    put(&h.block_slab_count, sizeof(h.block_slab_count));
    put(&h.snapshot_slab_count, sizeof(h.snapshot_slab_count));
    put(&h.reserved, sizeof(h.reserved));
    put(&h.block_rank_bytes, sizeof(h.block_rank_bytes));
    put(&h.snapshot_rank_bytes, sizeof(h.snapshot_rank_bytes));
    put(&h.block_slab_bytes, sizeof(h.block_slab_bytes));
    put(&h.snapshot_slab_bytes, sizeof(h.snapshot_slab_bytes));
    put(&h.disk_file_uid, sizeof(h.disk_file_uid));
    put(&h.tick, sizeof(h.tick));
    put(&h.node_count, sizeof(h.node_count));
    put(&h.live_node_count, sizeof(h.live_node_count));
    put(&h.snapshot_count, sizeof(h.snapshot_count));
    put(&h.live_snapshot_count, sizeof(h.live_snapshot_count));
    put(&h.payload_bytes, sizeof(h.payload_bytes));
    const std::uint64_t checksum = fnv1a(0xCBF2'9CE4'8422'2325ULL, out.data(), off);
    put(&checksum, sizeof(checksum));
}

bool decode_header(const std::array<std::uint8_t, kHeaderBytes>& in, Header& h) noexcept {
    std::size_t off = 0;
    const auto get = [&](void* dst, std::size_t n) {
        std::memcpy(dst, in.data() + off, n);
        off += n;
    };
    get(&h.magic, sizeof(h.magic));
    get(&h.version, sizeof(h.version));
    get(&h.tp, sizeof(h.tp));
    get(&h.device_snapshot_slots, sizeof(h.device_snapshot_slots));
    get(&h.block_slab_count, sizeof(h.block_slab_count));
    get(&h.snapshot_slab_count, sizeof(h.snapshot_slab_count));
    get(&h.reserved, sizeof(h.reserved));
    get(&h.block_rank_bytes, sizeof(h.block_rank_bytes));
    get(&h.snapshot_rank_bytes, sizeof(h.snapshot_rank_bytes));
    get(&h.block_slab_bytes, sizeof(h.block_slab_bytes));
    get(&h.snapshot_slab_bytes, sizeof(h.snapshot_slab_bytes));
    get(&h.disk_file_uid, sizeof(h.disk_file_uid));
    get(&h.tick, sizeof(h.tick));
    get(&h.node_count, sizeof(h.node_count));
    get(&h.live_node_count, sizeof(h.live_node_count));
    get(&h.snapshot_count, sizeof(h.snapshot_count));
    get(&h.live_snapshot_count, sizeof(h.live_snapshot_count));
    get(&h.payload_bytes, sizeof(h.payload_bytes));
    const std::uint64_t stored = [&] {
        std::uint64_t v = 0;
        std::memcpy(&v, in.data() + off, sizeof(v));
        return v;
    }();
    const std::uint64_t expect = fnv1a(0xCBF2'9CE4'8422'2325ULL, in.data(), off);
    h.checksum = stored;
    return stored == expect;
}

// A node record is a live flag followed, when persisted, by fixed fields.
constexpr std::size_t kNodePayloadBytes =
    sizeof(std::uint32_t) + sizeof(std::int32_t) + sizeof(std::uint64_t) + sizeof(std::uint64_t) +
    static_cast<std::size_t>(kBlockTokens) * sizeof(TokenId) + sizeof(std::uint64_t) +
    sizeof(std::int32_t) + sizeof(std::uint32_t);
constexpr std::size_t kSnapshotPayloadBytes =
    sizeof(std::uint32_t) + sizeof(std::uint32_t) + sizeof(std::uint8_t) +
    static_cast<std::size_t>(kBlockTokens - 1) * sizeof(TokenId) + sizeof(std::uint8_t) +
    sizeof(std::uint32_t) + sizeof(std::uint64_t) + sizeof(std::uint64_t) + sizeof(std::int32_t) +
    sizeof(std::uint32_t);

// Deserialized record staging. Small relative to the tree itself.
struct NodeRec {
    bool live = false;
    std::uint32_t parent = kInvalidNode;
    std::int32_t depth = -1;
    std::uint64_t lookup_hash = 0;
    std::uint64_t extra = 0;
    std::array<TokenId, kBlockTokens> tokens{};
    std::uint64_t last_access_tick = 0;
    std::int32_t disk_page = kInvalidDiskPage;
    std::uint32_t disk_generation = 0;
};

struct SnapRec {
    bool live = false;
    std::uint32_t anchor = kInvalidNode;
    std::uint32_t frontier = 0;
    std::uint8_t tail_len = 0;
    std::array<TokenId, kBlockTokens - 1> tail_tokens{};
    std::uint8_t kind = 0;
    std::uint32_t freq = 1;
    std::uint64_t gdsf_h = 0;
    std::uint64_t last_hit_tick = 0;
    std::int32_t disk_page = kInvalidDiskPage;
    std::uint32_t disk_generation = 0;
};

struct Writer {
    std::FILE* f = nullptr;
    bool ok = true;
    void raw(const void* data, std::size_t bytes) {
        if (!ok || bytes == 0) { return; }
        if (std::fwrite(data, 1, bytes, f) != bytes) { ok = false; }
    }
    template <class T>
    void value(const T& v) {
        raw(&v, sizeof(T));
    }
};

struct Reader {
    std::FILE* f = nullptr;
    bool ok = true;
    void raw(void* data, std::size_t bytes) {
        if (!ok || bytes == 0) { return; }
        if (std::fread(data, 1, bytes, f) != bytes) { ok = false; }
    }
    template <class T>
    void value(T& v) {
        raw(&v, sizeof(T));
    }
};

} // namespace

std::filesystem::path resolve_prefix_store_file(const std::filesystem::path& path) {
    if (path.empty()) { return path; }
    std::error_code ec;
    if (std::filesystem::is_directory(path, ec)) {
        return path / "prefix-cache.bin";
    }
    // A trailing separator names a directory that need not exist yet.
    if (!path.has_filename()) {
        return path / "prefix-cache.bin";
    }
    return path;
}

std::filesystem::path resolve_prefix_store_slabs_file(const std::filesystem::path& index_path) {
    std::filesystem::path slabs = index_path;
    slabs.replace_extension(".slabs");
    return slabs;
}

bool PrefixStore::save(const std::filesystem::path& raw_path, const PrefixIndex& index,
                       const DiskSlabPool& disk_pool, const PrefixStoreGeometry& geometry) {
    if (!disk_pool.enabled() || geometry.disk_file_uid == 0 ||
        disk_pool.file_uid() != geometry.disk_file_uid ||
        disk_pool.block_slab_count() != geometry.block_slab_count ||
        disk_pool.snapshot_slab_count() != geometry.snapshot_slab_count ||
        disk_pool.block_slab_bytes() != geometry.block_slab_bytes ||
        disk_pool.snapshot_slab_bytes() != geometry.snapshot_slab_bytes) {
        return false;
    }

    const std::filesystem::path path = resolve_prefix_store_file(raw_path);
    const std::size_t node_count     = index.nodes_.size();
    const std::size_t snap_count     = index.snapshots_.size();
    if (node_count == 0) { return false; } // an index always owns a root

    const std::size_t snapshot_base = geometry.block_slab_count;

    // Persist only disk-backed nodes whose whole ancestry is disk-backed, plus every disk-backed,
    // non-superseded, block-aligned snapshot on a fully persisted path. A record whose disk slot no
    // longer validates is dropped with its subtree.
    std::vector<std::uint8_t> keep_node(node_count, 0);
    std::vector<std::int32_t> block_owner(geometry.block_slab_count, -1);
    keep_node[index.root_ - 1U] = 1;
    std::vector<NodeId> queue{index.root_};
    while (!queue.empty()) {
        const NodeId cur = queue.back();
        queue.pop_back();
        for (NodeId child : index.nodes_[cur - 1U].children) {
            const BlockNode& c = index.nodes_[child - 1U];
            if (!c.live || c.disk_state != Residency::Disk ||
                c.disk_page < 0 ||
                static_cast<std::size_t>(c.disk_page) >= geometry.block_slab_count) {
                continue;
            }
            const std::size_t page = static_cast<std::size_t>(c.disk_page);
            if (block_owner[page] != -1 || keep_node[child - 1U] != 0) { continue; }
            if (!disk_pool.validate_slot(c.disk_page, disk_pool.generation(c.disk_page))) {
                continue;
            }
            keep_node[child - 1U] = 1;
            block_owner[page]     = static_cast<std::int32_t>(child);
            queue.push_back(child);
        }
    }

    std::vector<std::uint8_t> keep_snap(snap_count, 0);
    std::vector<std::int32_t> snap_owner(geometry.snapshot_slab_count, -1);
    for (std::size_t j = 0; j < snap_count; ++j) {
        const Snapshot& s = index.snapshots_[j];
        if (!s.live || s.superseded || s.tail_len != 0 || s.disk_state != Residency::Disk ||
            s.disk_page < 0 || s.anchor == kInvalidNode || s.anchor > node_count ||
            keep_node[s.anchor - 1U] == 0) {
            continue;
        }
        const std::size_t page = static_cast<std::size_t>(s.disk_page);
        if (page < snapshot_base || page >= snapshot_base + geometry.snapshot_slab_count) {
            continue;
        }
        const std::size_t idx = page - snapshot_base;
        if (snap_owner[idx] != -1) { continue; }
        if (!disk_pool.validate_slot(s.disk_page, disk_pool.generation(s.disk_page))) { continue; }
        keep_snap[j]       = 1;
        snap_owner[idx]    = static_cast<std::int32_t>(j + 1U);
    }

    std::uint64_t kept_nodes = 0;
    for (std::size_t i = 0; i < node_count; ++i) {
        if (keep_node[i] != 0) { ++kept_nodes; }
    }
    std::uint64_t kept_snaps = 0;
    for (std::size_t j = 0; j < snap_count; ++j) {
        if (keep_snap[j] != 0) { ++kept_snaps; }
    }

    const std::uint64_t payload_bytes =
        static_cast<std::uint64_t>(node_count) + kept_nodes * kNodePayloadBytes +
        static_cast<std::uint64_t>(snap_count) + kept_snaps * kSnapshotPayloadBytes;

    Header header;
    header.tp                    = geometry.tp;
    header.device_snapshot_slots = geometry.device_snapshot_slots;
    header.block_slab_count      = geometry.block_slab_count;
    header.snapshot_slab_count   = geometry.snapshot_slab_count;
    header.block_rank_bytes      = geometry.block_rank_bytes;
    header.snapshot_rank_bytes   = geometry.snapshot_rank_bytes;
    header.block_slab_bytes      = geometry.block_slab_bytes;
    header.snapshot_slab_bytes   = geometry.snapshot_slab_bytes;
    header.disk_file_uid         = geometry.disk_file_uid;
    header.tick                  = index.tick_;
    header.node_count            = node_count;
    header.live_node_count       = kept_nodes;
    header.snapshot_count        = snap_count;
    header.live_snapshot_count   = kept_snaps;
    header.payload_bytes         = payload_bytes;

    std::error_code ec;
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, ec);
        if (ec) { return false; }
    }
    std::filesystem::path tmp = path;
    tmp += ".tmp";

    std::FILE* f = std::fopen(tmp.string().c_str(), "wb");
    if (f == nullptr) { return false; }

    bool ok = true;
    std::array<std::uint8_t, kHeaderBytes> hbuf{};
    encode_header(header, hbuf);
    Writer w{f};
    w.raw(hbuf.data(), hbuf.size());

    for (std::size_t i = 0; i < node_count && w.ok; ++i) {
        const std::uint8_t live = keep_node[i];
        w.value(live);
        if (live == 0) { continue; }
        const BlockNode& n = index.nodes_[i];
        w.value(n.parent);
        w.value(n.depth);
        w.value(n.lookup_hash);
        w.value(n.extra);
        w.raw(n.tokens.data(), n.tokens.size() * sizeof(TokenId));
        w.value(n.last_access_tick);
        w.value(n.disk_page);
        const std::uint32_t generation = disk_pool.generation(n.disk_page);
        w.value(generation);
    }
    for (std::size_t j = 0; j < snap_count && w.ok; ++j) {
        const std::uint8_t live = keep_snap[j];
        w.value(live);
        if (live == 0) { continue; }
        const Snapshot& s = index.snapshots_[j];
        w.value(s.anchor);
        w.value(s.frontier);
        w.value(s.tail_len);
        w.raw(s.tail_tokens.data(), s.tail_tokens.size() * sizeof(TokenId));
        const std::uint8_t kind = static_cast<std::uint8_t>(s.kind);
        w.value(kind);
        w.value(s.freq);
        w.value(s.gdsf_h);
        w.value(s.last_hit_tick);
        w.value(s.disk_page);
        const std::uint32_t generation = disk_pool.generation(s.disk_page);
        w.value(generation);
    }
    ok = w.ok;
    if (ok && std::fflush(f) != 0) { ok = false; }
    if (std::fclose(f) != 0) { ok = false; }
    if (!ok) {
        std::error_code rm;
        std::filesystem::remove(tmp, rm);
        return false;
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::error_code rm;
        std::filesystem::remove(tmp, rm);
        return false;
    }
    return true;
}

bool PrefixStore::load(const std::filesystem::path& raw_path, PrefixIndex& index,
                       DiskSlabPool& disk_pool, const PrefixStoreGeometry& geometry,
                       PrefixStoreStats* stats) {
    const auto reject = [&]() {
        index.clear();
        disk_pool.clear();
        return false;
    };
    if (!disk_pool.enabled() || geometry.disk_file_uid == 0 ||
        disk_pool.file_uid() != geometry.disk_file_uid ||
        disk_pool.block_slab_count() != geometry.block_slab_count ||
        disk_pool.snapshot_slab_count() != geometry.snapshot_slab_count ||
        disk_pool.block_slab_bytes() != geometry.block_slab_bytes ||
        disk_pool.snapshot_slab_bytes() != geometry.snapshot_slab_bytes ||
        index.device_snapshot_slots() != geometry.device_snapshot_slots) {
        return reject();
    }

    const std::filesystem::path path = resolve_prefix_store_file(raw_path);
    std::FILE* f = std::fopen(path.string().c_str(), "rb");
    if (f == nullptr) { return reject(); }

    std::array<std::uint8_t, kHeaderBytes> hbuf{};
    if (std::fread(hbuf.data(), 1, hbuf.size(), f) != hbuf.size()) {
        std::fclose(f);
        return reject();
    }
    Header header;
    if (!decode_header(hbuf, header) || header.magic != kStoreMagic ||
        header.version != kStoreVersion) {
        std::fclose(f);
        return reject();
    }
    if (header.tp != geometry.tp || header.device_snapshot_slots != geometry.device_snapshot_slots ||
        header.block_rank_bytes != geometry.block_rank_bytes ||
        header.snapshot_rank_bytes != geometry.snapshot_rank_bytes ||
        header.block_slab_bytes != geometry.block_slab_bytes ||
        header.snapshot_slab_bytes != geometry.snapshot_slab_bytes ||
        header.block_slab_count != geometry.block_slab_count ||
        header.snapshot_slab_count != geometry.snapshot_slab_count ||
        header.disk_file_uid != geometry.disk_file_uid) {
        std::fclose(f);
        return reject();
    }
    if (header.node_count == 0 ||
        header.node_count > (1ULL << 26U) || header.snapshot_count > (1ULL << 24U)) {
        std::fclose(f);
        return reject();
    }

    const auto file_size = [&]() -> std::uint64_t {
        const long cur = std::ftell(f);
        if (cur < 0 || std::fseek(f, 0, SEEK_END) != 0) { return 0; }
        const long end = std::ftell(f);
        (void)std::fseek(f, cur, SEEK_SET);
        return end < 0 ? 0 : static_cast<std::uint64_t>(end);
    }();
    if (file_size != kHeaderBytes + header.payload_bytes) {
        std::fclose(f);
        return reject();
    }

    Reader r{f};
    std::vector<NodeRec> nodes(header.node_count);
    for (auto& n : nodes) {
        r.value(n.live);
        if (!r.ok || !n.live) { continue; }
        r.value(n.parent);
        r.value(n.depth);
        r.value(n.lookup_hash);
        r.value(n.extra);
        r.raw(n.tokens.data(), n.tokens.size() * sizeof(TokenId));
        r.value(n.last_access_tick);
        r.value(n.disk_page);
        r.value(n.disk_generation);
    }
    std::vector<SnapRec> snaps(header.snapshot_count);
    for (auto& s : snaps) {
        r.value(s.live);
        if (!r.ok || !s.live) { continue; }
        r.value(s.anchor);
        r.value(s.frontier);
        r.value(s.tail_len);
        r.raw(s.tail_tokens.data(), s.tail_tokens.size() * sizeof(TokenId));
        r.value(s.kind);
        r.value(s.freq);
        r.value(s.gdsf_h);
        r.value(s.last_hit_tick);
        r.value(s.disk_page);
        r.value(s.disk_generation);
    }
    std::fclose(f);
    if (!r.ok) { return reject(); }

    // Rebuild children adjacency and keep the disk-backed subtree rooted at the root node (id 1).
    const std::size_t node_count = nodes.size();
    std::vector<std::vector<std::uint32_t>> children(node_count + 1);
    for (std::size_t i = 0; i < node_count; ++i) {
        if (!nodes[i].live) { continue; }
        const std::uint32_t id = static_cast<std::uint32_t>(i + 1U);
        if (nodes[i].parent == kInvalidNode) {
            if (id != 1U) { return reject(); }
            continue;
        }
        if (nodes[i].parent == 0 || nodes[i].parent > node_count || nodes[i].parent == id) {
            return reject();
        }
        children[nodes[i].parent].push_back(id);
    }
    if (node_count < 1 || nodes[0].live == false) { return reject(); }

    const std::size_t snapshot_base = geometry.block_slab_count;
    std::vector<std::uint8_t> keep_node(node_count, 0);
    std::vector<std::int32_t> block_owner(geometry.block_slab_count, -1);
    keep_node[0] = 1;
    std::vector<std::uint32_t> queue{1U};
    while (!queue.empty()) {
        const std::uint32_t cur = queue.back();
        queue.pop_back();
        for (std::uint32_t child : children[cur]) {
            const NodeRec& c = nodes[child - 1U];
            if (c.disk_page < 0 ||
                static_cast<std::size_t>(c.disk_page) >= geometry.block_slab_count ||
                block_owner[static_cast<std::size_t>(c.disk_page)] != -1) {
                continue;
            }
            if (!disk_pool.validate_slot(c.disk_page, c.disk_generation)) { continue; }
            keep_node[child - 1U] = 1;
            block_owner[static_cast<std::size_t>(c.disk_page)] = static_cast<std::int32_t>(child);
            queue.push_back(child);
        }
    }

    std::vector<std::uint8_t> keep_snap(snaps.size(), 0);
    std::vector<std::int32_t> snap_owner(geometry.snapshot_slab_count, -1);
    for (std::size_t j = 0; j < snaps.size(); ++j) {
        const SnapRec& s = snaps[j];
        if (!s.live || s.tail_len != 0 || s.anchor == kInvalidNode || s.anchor > node_count ||
            keep_node[s.anchor - 1U] == 0 || s.disk_page < 0) {
            continue;
        }
        const std::size_t page = static_cast<std::size_t>(s.disk_page);
        if (page < snapshot_base || page >= snapshot_base + geometry.snapshot_slab_count) {
            continue;
        }
        const std::size_t idx = page - snapshot_base;
        if (snap_owner[idx] != -1) { continue; }
        if (!disk_pool.validate_slot(s.disk_page, s.disk_generation)) { continue; }
        keep_snap[j]    = 1;
        snap_owner[idx] = static_cast<std::int32_t>(j + 1U);
    }

    index.clear();
    disk_pool.clear();
    index.nodes_.assign(node_count, BlockNode{});
    index.snapshots_.assign(snaps.size(), Snapshot{});
    index.free_nodes_.clear();
    index.free_snapshots_.clear();
    index.live_nodes_     = 0;
    index.live_snapshots_ = 0;
    index.lru_head_       = kInvalidNode;
    index.lru_tail_       = kInvalidNode;
    index.tick_           = header.tick;
    index.root_           = 1U;

    std::uint64_t live_nodes = 0;
    for (std::size_t i = 0; i < node_count; ++i) {
        BlockNode& n = index.nodes_[i];
        if (keep_node[i] == 0) {
            n.live = false;
            index.free_nodes_.push_back(static_cast<NodeId>(i + 1U));
            continue;
        }
        n = BlockNode{};
        n.parent       = nodes[i].parent;
        n.depth        = nodes[i].depth;
        n.lookup_hash  = nodes[i].lookup_hash;
        n.extra        = nodes[i].extra;
        n.tokens       = nodes[i].tokens;
        n.device_state = Residency::Absent;
        n.host_state   = Residency::Absent;
        n.disk_page    = nodes[i].disk_page;
        n.disk_state   = Residency::Disk;
        n.last_access_tick = nodes[i].last_access_tick;
        n.live         = true;
        ++live_nodes;
    }
    std::uint64_t live_snaps = 0;
    for (std::size_t j = 0; j < snaps.size(); ++j) {
        Snapshot& s = index.snapshots_[j];
        if (keep_snap[j] == 0) {
            s.live = false;
            index.free_snapshots_.push_back(static_cast<SnapshotId>(j + 1U));
            continue;
        }
        s = Snapshot{};
        s.anchor       = snaps[j].anchor;
        s.frontier     = snaps[j].frontier;
        s.tail_len     = snaps[j].tail_len;
        s.tail_tokens  = snaps[j].tail_tokens;
        s.device_slot  = kInvalidSlot;
        s.device_state = Residency::Absent;
        s.host_state   = Residency::Absent;
        s.disk_page    = snaps[j].disk_page;
        s.disk_state   = Residency::Disk;
        s.kind         = static_cast<SnapshotKind>(snaps[j].kind);
        s.freq         = snaps[j].freq != 0 ? snaps[j].freq : 1U;
        s.gdsf_h       = snaps[j].gdsf_h;
        s.last_hit_tick = snaps[j].last_hit_tick;
        s.live         = true;
        ++live_snaps;
    }
    index.live_nodes_     = static_cast<std::size_t>(live_nodes);
    index.live_snapshots_ = static_cast<std::size_t>(live_snaps);

    // Rebuild the derived structure: parent -> children, anchor -> snapshots, and every node's
    // live_snapshots_below.
    for (std::size_t i = 1; i < node_count; ++i) {
        BlockNode& n = index.nodes_[i];
        if (!n.live || n.parent == kInvalidNode) { continue; }
        index.nodes_[n.parent - 1U].children.push_back(static_cast<NodeId>(i + 1U));
    }
    for (std::size_t j = 0; j < snaps.size(); ++j) {
        if (keep_snap[j] == 0) { continue; }
        Snapshot& s = index.snapshots_[j];
        index.nodes_[s.anchor - 1U].snapshots.push_back(static_cast<SnapshotId>(j + 1U));
        for (NodeId cur = s.anchor; cur != kInvalidNode; cur = index.nodes_[cur - 1U].parent) {
            ++index.nodes_[cur - 1U].live_snapshots_below;
        }
    }

    // The disk pool adopts the persisted eviction metadata; node slots the tree no longer reaches
    // from a live snapshot are marked dead, matching dec_live_below's cold-start policy.
    for (std::size_t i = 0; i < node_count; ++i) {
        if (keep_node[i] == 0) { continue; }
        BlockNode& n = index.nodes_[i];
        const bool dead = n.live_snapshots_below == 0U;
        disk_pool.occupy_loaded(n.disk_page, dead, false, 1U, n.last_access_tick,
                                nodes[i].disk_generation);
    }
    for (std::size_t j = 0; j < snaps.size(); ++j) {
        if (keep_snap[j] == 0) { continue; }
        const Snapshot& s = index.snapshots_[j];
        disk_pool.occupy_loaded(s.disk_page, false, false, s.freq, s.last_hit_tick,
                                snaps[j].disk_generation);
    }

    index.slot_owner_.assign(geometry.device_snapshot_slots, kInvalidSnapshot);
    index.free_slots_.clear();
    for (std::int32_t slot = static_cast<std::int32_t>(geometry.device_snapshot_slots) - 1;
         slot >= 0; --slot) {
        index.free_slots_.push_back(slot);
    }
    index.evictable_pages_ = 0;

    if (stats != nullptr) {
        stats->nodes     = live_nodes;
        stats->snapshots = live_snaps;
        stats->slabs     = live_nodes + live_snaps;
        stats->bytes     = header.payload_bytes;
    }
    return true;
}

bool save_prefix_store(const std::filesystem::path& path, const PrefixIndex& index,
                       const DiskSlabPool& disk_pool, const PrefixStoreGeometry& geometry) {
    return PrefixStore::save(path, index, disk_pool, geometry);
}

bool load_prefix_store(const std::filesystem::path& path, PrefixIndex& index,
                       DiskSlabPool& disk_pool, const PrefixStoreGeometry& geometry,
                       PrefixStoreStats* stats) {
    return PrefixStore::load(path, index, disk_pool, geometry, stats);
}

} // namespace ninfer::prefix_cache
