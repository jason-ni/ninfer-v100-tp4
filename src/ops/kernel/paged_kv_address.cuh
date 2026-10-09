#pragma once

#include "core/paged_kv_cache.h"

#include <cstdint>
#include <cstdio>

namespace ninfer::ops {

inline constexpr int kPagedKVPageShift = 6;
inline constexpr int kPagedKVPageMask  = kPagedKVPageSize - 1;

static_assert(kPagedKVPageSize == (1 << kPagedKVPageShift));

extern __device__ unsigned long long g_kv_dbg_base[16];
extern __device__ int g_kv_dbg_pos[16];
extern __device__ int g_kv_dbg_tag[16];

__device__ __forceinline__ void kv_debug_probe(int tag, const void* base, int page, int pos,
                                               int lane) {
    if (lane == 0 && page >= 8 && page < 16) {
        g_kv_dbg_base[page] = static_cast<unsigned long long>(
            reinterpret_cast<std::uintptr_t>(base));
        g_kv_dbg_pos[page] = pos;
        g_kv_dbg_tag[page] = tag;
    }
}

__device__ __forceinline__ std::int32_t paged_kv_physical_page(const std::int32_t* block_table,
                                                               std::int32_t position) {
    return block_table[position >> kPagedKVPageShift];
}

template <int LeadingExtent, int HeadExtent>
__device__ __forceinline__ std::int64_t paged_kv_page_head_offset(std::int32_t physical_page,
                                                                  std::int32_t head) {
    return static_cast<std::int64_t>(LeadingExtent) * kPagedKVPageSize *
           (static_cast<std::int64_t>(head) +
            static_cast<std::int64_t>(HeadExtent) * physical_page);
}

template <int LeadingExtent, int HeadExtent>
__device__ __forceinline__ std::int64_t
paged_kv_element_offset(std::int32_t physical_page, std::int32_t head, std::int32_t page_offset,
                        std::int32_t leading) {
    return paged_kv_page_head_offset<LeadingExtent, HeadExtent>(physical_page, head) +
           static_cast<std::int64_t>(LeadingExtent) * page_offset + leading;
}

template <int LeadingExtent, int HeadExtent>
__device__ __forceinline__ std::int64_t
paged_kv_element_offset(const std::int32_t* block_table, std::int32_t head, std::int32_t position,
                        std::int32_t leading) {
    return paged_kv_element_offset<LeadingExtent, HeadExtent>(
        paged_kv_physical_page(block_table, position), head, position & kPagedKVPageMask, leading);
}

} // namespace ninfer::ops
