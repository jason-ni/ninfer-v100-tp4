#pragma once

// M3a disk-tier index container: a versioned, validated metadata container that persists the
// hybrid prefix cache across process restarts (docs/maintainer/hybrid-prefix-cache-plan.md,
// section "M3a").
//
// The container holds only index metadata: block-tree topology, the snapshot table, and, for every
// persisted record, the disk slot that owns its bytes plus that slot's generation and eviction
// metadata. The KV bytes themselves live in the disk slab file (DiskSlabPool), which is written
// continuously at runtime. This file is rewritten (atomically, temp + rename) on a debounce, so it
// can lag the slab file but never references a slot the slab file does not contain:
//
//  - a slot written after the last index rewrite is either still free, or referenced at an older
//    generation -- the loader's header check rejects the record;
//  - a slab file that was reinitialized (different geometry, or replaced) carries a different
//    per-creation uid, so the whole index is rejected, never partially applied.
//
// The on-disk layout is host-endian and tightly bound to the running configuration (tp, KV layout,
// slab geometry). A container whose stored geometry (or slab uid) does not match the current engine
// is rejected whole rather than partially applied.

#include "runtime/prefix_cache/disk_slab_pool.h"
#include "runtime/prefix_cache/prefix_index.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace ninfer::prefix_cache {

// Immutable storage facts a container is bound to. Both sides must agree exactly for a load to be
// accepted.
struct PrefixStoreGeometry {
    std::uint32_t tp                    = 0;
    std::uint32_t device_snapshot_slots = 0;
    std::uint64_t block_rank_bytes      = 0;
    std::uint64_t snapshot_rank_bytes   = 0;
    std::uint64_t block_slab_bytes      = 0;
    std::uint64_t snapshot_slab_bytes   = 0;
    std::uint32_t block_slab_count      = 0;
    std::uint32_t snapshot_slab_count   = 0;
    // Identity of the disk slab file this index binds to (DiskSlabPool::file_uid()). 0 disables
    // disk-tier persistence.
    std::uint64_t disk_file_uid         = 0;
};

struct PrefixStoreStats {
    std::uint64_t nodes     = 0;
    std::uint64_t snapshots = 0;
    std::uint64_t slabs     = 0;
    std::uint64_t bytes     = 0;
};

// Resolves the configured path to the concrete index file. A path that names an existing directory
// (or ends with a separator) stores the index as `<dir>/prefix-cache.bin`; any other path is used
// verbatim.
[[nodiscard]] std::filesystem::path resolve_prefix_store_file(const std::filesystem::path& path);

// The slab file paired with an index path: the index path with its extension replaced by `.slabs`
// (e.g. `/cache/prefix-cache.bin` -> `/cache/prefix-cache.slabs`).
[[nodiscard]] std::filesystem::path
resolve_prefix_store_slabs_file(const std::filesystem::path& index_path);

// Serialize the disk-resident subset of the cache to `path` atomically (temp file + rename). Only
// records whose disk slot is currently valid are written; a record with no disk copy (a node or
// snapshot that was never spilled, or already evicted from the disk tier) is dropped together with
// its subtree. Returns false and leaves any existing container untouched on any failure. The
// Program must spill host-resident records before calling this (the flush sweep does so each tick).
bool save_prefix_store(const std::filesystem::path& path, const PrefixIndex& index,
                       const DiskSlabPool& disk_pool, const PrefixStoreGeometry& geometry);

// Rebuild the disk-resident index from `path` into `index`. `disk_pool` must already own the
// matching slab file (matching geometry and uid). Returns false on a missing file, an unsupported
// version, a geometry/uid mismatch, or structural corruption; in that case `index` is cleared back
// to empty. Records whose disk slot fails validation are dropped with their subtrees. `stats` is
// filled on success when non-null.
bool load_prefix_store(const std::filesystem::path& path, PrefixIndex& index,
                       DiskSlabPool& disk_pool, const PrefixStoreGeometry& geometry,
                       PrefixStoreStats* stats = nullptr);

// Implementation detail: the friend that reaches into PrefixIndex's private tree. Kept separate so
// the index header only needs a forward declaration.
struct PrefixStore {
    static bool save(const std::filesystem::path& path, const PrefixIndex& index,
                     const DiskSlabPool& disk_pool, const PrefixStoreGeometry& geometry);
    static bool load(const std::filesystem::path& path, PrefixIndex& index,
                     DiskSlabPool& disk_pool, const PrefixStoreGeometry& geometry,
                     PrefixStoreStats* stats);
};

} // namespace ninfer::prefix_cache
