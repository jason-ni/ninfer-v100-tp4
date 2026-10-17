#include "runtime/prefix_cache/host_slab_pool.h"

#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>

namespace ninfer::prefix_cache {

namespace {

std::size_t slab_offset(std::size_t block_count) noexcept { return block_count; }

} // namespace

HostSlabPool::HostSlabPool(HostSlabPoolOptions options) : options_(options) {
    if (options.total_bytes == 0 || options.block_slab_count == 0 ||
        options.snapshot_slab_count == 0 || options.block_slab_bytes == 0 ||
        options.snapshot_slab_bytes == 0) {
        return; // disabled by configuration
    }

    // Pin exactly the slab data. The HostSlab metadata lives in heap vectors, so it never eats
    // into -- or invalidates -- the pinned budget (the previous in-pool header carve-out could
    // push the layout past total_bytes and silently free the whole pool).
    const std::size_t data_bytes = options.block_slab_count * options.block_slab_bytes +
                                   options.snapshot_slab_count * options.snapshot_slab_bytes;
    void* memory = nullptr;
    cudaError_t err = cudaHostAlloc(&memory, data_bytes, cudaHostAllocDefault);
    if (err != cudaSuccess || memory == nullptr) {
        std::fprintf(stderr,
                     "[host-cache] cudaHostAlloc(%zu MiB) failed: %s; host tier disabled\n",
                     data_bytes >> 20U, cudaGetErrorString(err));
        return;
    }

    block_slabs_.resize(options.block_slab_count);
    snapshot_slabs_.resize(options.snapshot_slab_count);
    char* cursor = static_cast<char*>(memory);
    for (HostSlab& slab : block_slabs_) {
        slab.data  = cursor;
        slab.bytes = options.block_slab_bytes;
        cursor += options.block_slab_bytes;
    }
    for (HostSlab& slab : snapshot_slabs_) {
        slab.data  = cursor;
        slab.bytes = options.snapshot_slab_bytes;
        cursor += options.snapshot_slab_bytes;
    }
    host_memory_  = memory;
    pinned_bytes_ = data_bytes;
}

HostSlabPool::~HostSlabPool() {
    if (host_memory_ != nullptr) {
        cudaFreeHost(host_memory_);
    }
}

std::size_t HostSlabPool::used_bytes() const noexcept {
    std::size_t used = 0;
    for (const HostSlab& slab : block_slabs_) {
        if (slab.occupied) {
            used += slab.bytes;
        }
    }
    for (const HostSlab& slab : snapshot_slabs_) {
        if (slab.occupied) {
            used += slab.bytes;
        }
    }
    return used;
}

std::size_t HostSlabPool::free_bytes() const noexcept {
    return pinned_bytes_ - used_bytes();
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

    HostPageId page = find_free(0, block_slabs_.size());
    if (page == kInvalidHostPage) {
        // No free slab: evict one. The victim's previous owner must be reconciled by the caller
        // before the new data is written.
        page = pick_victim(0, block_slabs_.size());
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

    const std::size_t offset = slab_offset(block_slabs_.size());
    HostPageId page = find_free(offset, snapshot_slabs_.size());
    if (page == kInvalidHostPage) {
        // No free slab: evict one. The victim's previous owner must be reconciled by the caller
        // before the new data is written.
        page = pick_victim(offset, snapshot_slabs_.size());
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

void HostSlabPool::occupy_loaded(HostPageId page, bool dead, bool superseded, std::uint64_t freq,
                                 std::uint64_t last_hit_tick) noexcept {
    HostSlab* slab = slab_mut(page);
    if (slab == nullptr) {
        return;
    }
    slab->occupied = true;
    slab->pinned   = false;
    slab->dead     = dead;
    slab->dead_at  = dead ? ++tick_ : 0;
    slab->superseded    = superseded;
    slab->superseded_at = superseded ? ++tick_ : 0;
    slab->freq          = freq;
    slab->last_hit_tick = last_hit_tick;
    if (last_hit_tick > tick_) { tick_ = last_hit_tick; }
}

void HostSlabPool::clear() noexcept {
    for (HostSlab& slab : block_slabs_) { reset_slab(slab); }
    for (HostSlab& slab : snapshot_slabs_) { reset_slab(slab); }
    tick_ = 0;
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
    if (page < 0) {
        return nullptr;
    }
    const auto idx = static_cast<std::size_t>(page);
    if (idx < block_slabs_.size()) {
        return &block_slabs_[idx];
    }
    const std::size_t snap = idx - block_slabs_.size();
    if (snap < snapshot_slabs_.size()) {
        return &snapshot_slabs_[snap];
    }
    return nullptr;
}

HostSlab* HostSlabPool::slab_mut(HostPageId page) noexcept {
    if (page < 0) {
        return nullptr;
    }
    const auto idx = static_cast<std::size_t>(page);
    if (idx < block_slabs_.size()) {
        return &block_slabs_[idx];
    }
    const std::size_t snap = idx - block_slabs_.size();
    if (snap < snapshot_slabs_.size()) {
        return &snapshot_slabs_[snap];
    }
    return nullptr;
}

} // namespace ninfer::prefix_cache
