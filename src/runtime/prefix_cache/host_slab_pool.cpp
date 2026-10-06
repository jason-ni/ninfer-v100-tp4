#include "runtime/prefix_cache/host_slab_pool.h"

#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>

namespace ninfer::prefix_cache {

namespace {

std::size_t slab_offset(std::size_t block_count) noexcept { return block_count; }

} // namespace

HostSlabPool::HostSlabPool(HostSlabPoolOptions options)
    : options_(options), block_slab_count_(options.block_slab_count),
      snapshot_slab_count_(options.snapshot_slab_count) {
    if (options.total_bytes == 0 || options.block_slab_count == 0 ||
        options.snapshot_slab_count == 0 || options.block_slab_bytes == 0 ||
        options.snapshot_slab_bytes == 0) {
        return; // disabled
    }

    void* memory = nullptr;
    cudaError_t err = cudaHostAlloc(&memory, options.total_bytes, cudaHostAllocDefault);
    if (err != cudaSuccess || memory == nullptr) {
        return; // allocation failed, pool disabled
    }

    block_slabs_ = static_cast<HostSlab*>(memory);
    char* cursor = static_cast<char*>(memory) + (options.block_slab_count + options.snapshot_slab_count) * sizeof(HostSlab);
    std::size_t offset = (options.block_slab_count + options.snapshot_slab_count) * sizeof(HostSlab);

    for (std::size_t i = 0; i < options.block_slab_count; ++i) {
        block_slabs_[i].data = cursor;
        block_slabs_[i].bytes = options.block_slab_bytes;
        cursor += options.block_slab_bytes;
        offset += options.block_slab_bytes;
    }

    snapshot_slabs_ = block_slabs_ + options.block_slab_count;
    for (std::size_t i = 0; i < options.snapshot_slab_count; ++i) {
        snapshot_slabs_[i].data = cursor;
        snapshot_slabs_[i].bytes = options.snapshot_slab_bytes;
        cursor += options.snapshot_slab_bytes;
        offset += options.snapshot_slab_bytes;
    }

    if (offset > options.total_bytes) {
        // Misconfiguration: slabs do not fit in the total budget. Free and disable.
        cudaFreeHost(memory);
        block_slabs_ = nullptr;
        snapshot_slabs_ = nullptr;
        block_slab_count_ = 0;
        snapshot_slab_count_ = 0;
    }
}

HostSlabPool::~HostSlabPool() {
    if (block_slabs_ != nullptr) {
        cudaFreeHost(block_slabs_);
    }
}

std::size_t HostSlabPool::used_bytes() const noexcept {
    std::size_t used = 0;
    for (std::size_t i = 0; i < block_slab_count_; ++i) {
        if (block_slabs_[i].occupied) {
            used += block_slabs_[i].bytes;
        }
    }
    for (std::size_t i = 0; i < snapshot_slab_count_; ++i) {
        if (snapshot_slabs_[i].occupied) {
            used += snapshot_slabs_[i].bytes;
        }
    }
    return used;
}

std::size_t HostSlabPool::free_bytes() const noexcept {
    return options_.total_bytes - used_bytes();
}

void HostSlabPool::reset_slab(HostSlab& slab) noexcept {
    slab.occupied = false;
    slab.pinned = false;
    slab.dead = false;
    slab.dead_at = 0;
    slab.superseded = false;
    slab.superseded_at = 0;
    slab.freq = 0;
    slab.last_hit_tick = 0;
}

void HostSlabPool::occupy(HostPageId page) noexcept {
    HostSlab* slab = slab_mut(page);
    if (slab == nullptr) {
        return;
    }
    slab->occupied = true;
    slab->pinned = false;
    slab->dead = false;
    slab->dead_at = 0;
    slab->superseded = false;
    slab->superseded_at = 0;
    slab->freq = 1;
    slab->last_hit_tick = ++tick_;
}

HostPageId HostSlabPool::find_free(std::size_t offset, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t idx = offset + i;
        const HostSlab* s = slab(static_cast<HostPageId>(idx));
        if (s != nullptr && !s->occupied) {
            return static_cast<HostPageId>(idx);
        }
    }
    return kInvalidHostPage;
}

HostPageId HostSlabPool::pick_victim(std::size_t offset, std::size_t count) noexcept {
    // Pass 1: dead slabs (oldest first).
    std::uint64_t oldest = UINT64_MAX;
    HostPageId best = kInvalidHostPage;
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t idx = offset + i;
        const HostSlab* s = slab(static_cast<HostPageId>(idx));
        if (s == nullptr || !s->occupied || s->pinned || !s->dead) {
            continue;
        }
        if (s->dead_at < oldest) {
            oldest = s->dead_at;
            best = static_cast<HostPageId>(idx);
        }
    }
    if (best != kInvalidHostPage) {
        return best;
    }

    // Pass 2: superseded slabs (oldest first).
    oldest = UINT64_MAX;
    best = kInvalidHostPage;
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t idx = offset + i;
        const HostSlab* s = slab(static_cast<HostPageId>(idx));
        if (s == nullptr || !s->occupied || s->pinned || !s->superseded) {
            continue;
        }
        if (s->superseded_at < oldest) {
            oldest = s->superseded_at;
            best = static_cast<HostPageId>(idx);
        }
    }
    if (best != kInvalidHostPage) {
        return best;
    }

    // Pass 3: GDSF — lowest frequency, then oldest recency.
    std::uint64_t min_freq = UINT64_MAX;
    std::uint64_t min_tick = UINT64_MAX;
    best = kInvalidHostPage;
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t idx = offset + i;
        const HostSlab* s = slab(static_cast<HostPageId>(idx));
        if (s == nullptr || !s->occupied || s->pinned) {
            continue;
        }
        if (s->freq < min_freq ||
            (s->freq == min_freq && s->last_hit_tick < min_tick)) {
            min_freq = s->freq;
            min_tick = s->last_hit_tick;
            best = static_cast<HostPageId>(idx);
        }
    }
    return best;
}

SlabAllocation HostSlabPool::allocate_block() {
    SlabAllocation result;
    if (!enabled()) {
        return result;
    }

    HostPageId page = find_free(0, block_slab_count_);
    if (page == kInvalidHostPage) {
        // No free slab: evict one. The victim's previous owner must be reconciled by the caller
        // before the new data is written.
        page = pick_victim(0, block_slab_count_);
        if (page == kInvalidHostPage) {
            return result; // all block slabs pinned
        }
        result.evicted = page; // previous owner existed at this index
    }
    occupy(page);
    result.page = page;
    return result;
}

SlabAllocation HostSlabPool::allocate_snapshot() {
    SlabAllocation result;
    if (!enabled()) {
        return result;
    }

    const std::size_t offset = slab_offset(block_slab_count_);
    HostPageId page = find_free(offset, snapshot_slab_count_);
    if (page == kInvalidHostPage) {
        // No free slab: evict one. The victim's previous owner must be reconciled by the caller
        // before the new data is written.
        page = pick_victim(offset, snapshot_slab_count_);
        if (page == kInvalidHostPage) {
            return result; // all snapshot slabs pinned
        }
        result.evicted = page; // previous owner existed at this index
    }
    occupy(page);
    result.page = page;
    return result;
}

void HostSlabPool::release(HostPageId page) noexcept {
    HostSlab* slab = slab_mut(page);
    if (slab == nullptr) {
        return;
    }
    reset_slab(*slab);
}

void* HostSlabPool::data(HostPageId page) const noexcept {
    const HostSlab* s = slab(page);
    if (s == nullptr) {
        return nullptr;
    }
    return s->data;
}

std::size_t HostSlabPool::bytes(HostPageId page) const noexcept {
    const HostSlab* s = slab(page);
    if (s == nullptr) {
        return 0;
    }
    return s->bytes;
}

void HostSlabPool::pin(HostPageId page) noexcept {
    HostSlab* slab = slab_mut(page);
    if (slab != nullptr && slab->occupied) {
        slab->pinned = true;
    }
}

void HostSlabPool::unpin(HostPageId page) noexcept {
    HostSlab* slab = slab_mut(page);
    if (slab != nullptr && slab->occupied) {
        slab->pinned = false;
    }
}

void HostSlabPool::mark_dead(HostPageId page) noexcept {
    HostSlab* slab = slab_mut(page);
    if (slab != nullptr && slab->occupied && !slab->dead) {
        slab->dead = true;
        slab->dead_at = ++tick_;
    }
}

void HostSlabPool::mark_alive(HostPageId page) noexcept {
    HostSlab* slab = slab_mut(page);
    if (slab != nullptr && slab->occupied) {
        slab->dead = false;
        slab->dead_at = 0;
    }
}

void HostSlabPool::supersede(HostPageId page) noexcept {
    HostSlab* slab = slab_mut(page);
    if (slab != nullptr && slab->occupied && !slab->superseded) {
        slab->superseded = true;
        slab->superseded_at = ++tick_;
    }
}

void HostSlabPool::clear_superseded(HostPageId page) noexcept {
    HostSlab* slab = slab_mut(page);
    if (slab != nullptr && slab->occupied) {
        slab->superseded = false;
        slab->superseded_at = 0;
    }
}

void HostSlabPool::hit(HostPageId page, std::uint64_t tick) noexcept {
    HostSlab* slab = slab_mut(page);
    if (slab == nullptr || !slab->occupied) {
        return;
    }
    ++slab->freq;
    slab->last_hit_tick = tick;
}

const HostSlab* HostSlabPool::slab(HostPageId page) const noexcept {
    if (page < 0 || page >= static_cast<HostPageId>(block_slab_count_ + snapshot_slab_count_)) {
        return nullptr;
    }
    if (static_cast<std::size_t>(page) < block_slab_count_) {
        return &block_slabs_[page];
    }
    return &snapshot_slabs_[page - static_cast<HostPageId>(block_slab_count_)];
}

HostSlab* HostSlabPool::slab_mut(HostPageId page) noexcept {
    if (page < 0 || page >= static_cast<HostPageId>(block_slab_count_ + snapshot_slab_count_)) {
        return nullptr;
    }
    if (static_cast<std::size_t>(page) < block_slab_count_) {
        return &block_slabs_[page];
    }
    return &snapshot_slabs_[page - static_cast<HostPageId>(block_slab_count_)];
}

} // namespace ninfer::prefix_cache
