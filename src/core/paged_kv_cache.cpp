#include "core/paged_kv_cache.h"

#include "core/device.h"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer {
namespace {

std::int32_t checked_i32(std::uint32_t value, const char* label) {
    if (value == 0 ||
        value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(std::string(label) + " must fit positive int32");
    }
    return static_cast<std::int32_t>(value);
}

std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
    if (tokens == 0) { return 0; }
    return 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

void validate_distinct_pools(std::span<const PagedKVReservation> reservations) {
    for (std::size_t i = 0; i < reservations.size(); ++i) {
        if (reservations[i].pool == nullptr || reservations[i].page_entitlement == 0) {
            throw std::invalid_argument("Paged KV reservation must name a pool and pages");
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (reservations[i].pool == reservations[j].pool) {
                throw std::invalid_argument("Paged KV bundle contains the same pool twice");
            }
        }
    }
}

} // namespace

PagedKVPoolLayout plan_paged_kv_pool(LayoutBuilder& builder, const PagedKVPoolSpec& spec) {
    const std::int32_t physical_pages = checked_i32(spec.page_group_count, "Paged KV page count");
    const std::int32_t logical_pages =
        checked_i32(spec.logical_page_capacity, "Paged KV logical page capacity");
    if (spec.table_rows <= 0) {
        throw std::invalid_argument("Paged KV table row count must be positive");
    }
    if (spec.planes.empty()) { throw std::invalid_argument("Paged KV pool must contain planes"); }

    PagedKVPoolLayout layout;
    layout.spec = spec;
    layout.planes.reserve(spec.planes.size());
    for (std::size_t index = 0; index < spec.planes.size(); ++index) {
        const PagedKVPlaneSpec& plane = spec.planes[index];
        if (plane.leading_extent <= 0 || plane.head_extent <= 0) {
            throw std::invalid_argument("Paged KV plane extents must be positive");
        }
        const std::string label = "Paged KV plane " + std::to_string(index);
        PagedKVPlaneLayout planned;
        planned.spec = plane;
        if (spec.plane_order == PagedKVPlaneOrder::PageMajor) {
            planned.storage = builder.add_tensor(
                plane.dtype,
                {plane.leading_extent, kPagedKVPageSize, plane.head_extent, physical_pages},
                plane.alignment, label);
        } else {
            planned.storage = builder.add_tensor(
                plane.dtype,
                {plane.leading_extent, kPagedKVPageSize, physical_pages, plane.head_extent},
                plane.alignment, label);
        }
        layout.planes.push_back(planned);
    }
    layout.block_tables = builder.add_tensor(DType::I32, {logical_pages, spec.table_rows}, 256,
                                             "Paged KV block tables");
    return layout;
}

std::size_t PagedKVPoolLayout::payload_bytes() const noexcept {
    std::size_t total = 0;
    for (const PagedKVPlaneLayout& plane : planes) { total += plane.storage.region.bytes; }
    return total;
}

std::size_t PagedKVPoolLayout::metadata_bytes() const noexcept { return block_tables.region.bytes; }

PagedKVPool::PagedKVPool(DeviceSpan backing, const PagedKVPoolLayout& layout)
    : spec_(layout.spec), block_tables_(layout.block_tables.bind(backing)),
      row_in_use_(static_cast<std::size_t>(layout.spec.table_rows), false) {
    if (layout.planes.size() != spec_.planes.size() || layout.planes.empty()) {
        throw std::invalid_argument("Paged KV layout plane inventory is inconsistent");
    }
    if (block_tables_.dtype != DType::I32 ||
        block_tables_.ne[0] !=
            checked_i32(spec_.logical_page_capacity, "Paged KV logical page capacity") ||
        block_tables_.ne[1] != spec_.table_rows) {
        throw std::logic_error("Paged KV block-table layout is inconsistent");
    }

    planes_.reserve(layout.planes.size());
    for (std::size_t index = 0; index < layout.planes.size(); ++index) {
        const PagedKVPlaneLayout& plane = layout.planes[index];
        if (plane.spec.dtype != spec_.planes[index].dtype ||
            plane.spec.leading_extent != spec_.planes[index].leading_extent ||
            plane.spec.head_extent != spec_.planes[index].head_extent) {
            throw std::logic_error("Paged KV plane layout does not match its spec");
        }
        planes_.push_back(plane.storage.bind(backing));
    }

    free_page_ids_.reserve(spec_.page_group_count);
    for (std::uint32_t page = 0; page < spec_.page_group_count; ++page) {
        free_page_ids_.push_back(static_cast<std::int32_t>(page));
    }
}

std::uint32_t PagedKVPool::page_group_count() const noexcept { return spec_.page_group_count; }

std::uint32_t PagedKVPool::logical_page_capacity() const noexcept {
    return spec_.logical_page_capacity;
}

std::int32_t PagedKVPool::table_row_count() const noexcept { return spec_.table_rows; }

std::size_t PagedKVPool::plane_count() const noexcept { return planes_.size(); }

const Tensor& PagedKVPool::plane(std::size_t index) const { return planes_.at(index); }

const Tensor& PagedKVPool::block_tables() const noexcept { return block_tables_; }

Tensor PagedKVPool::block_table_row(std::int32_t row) const {
    if (row < 0 || row >= table_row_count()) {
        throw std::out_of_range("Paged KV block-table row out of range");
    }
    return block_tables_.slice(1, row, 1).view(
        {static_cast<std::int32_t>(logical_page_capacity())});
}

namespace {

// A plane is `head_extent` contiguous runs of (leading_extent * page_size) elements per page.
// The plane order decides where those runs sit:
//   - PageMajor: [page][head][token][leading], leading fastest; run h of page p starts at
//     LE * PS * (h + HE * p). This is the convention ops/kernel/paged_kv_address.cuh addresses
//     (every attention read and KV write goes through paged_kv_element_offset).
//   - HeadMajor: [head][page][token][leading]; run h of page p starts at LE * PS * (p + P * h),
//     matching the DFlash full-context pool's bidirectional attention and append kernels.
std::size_t per_plane_page_bytes(const PagedKVPlaneSpec& spec) noexcept {
    return static_cast<std::size_t>(spec.leading_extent) * kPagedKVPageSize *
           static_cast<std::size_t>(spec.head_extent) * dtype_size(spec.dtype);
}

template <bool DeviceToHost>
bool copy_plane_page(const PagedKVPlaneSpec& spec, std::int32_t page_id, void* host_base,
                     char* device_base, std::size_t physical_pages, PagedKVPlaneOrder order,
                     cudaStream_t stream) noexcept {
    const std::size_t leading = static_cast<std::size_t>(spec.leading_extent);
    const std::size_t heads   = static_cast<std::size_t>(spec.head_extent);
    const std::size_t run_bytes = leading * kPagedKVPageSize * dtype_size(spec.dtype);
    const std::size_t page      = static_cast<std::size_t>(page_id);
    char* host = static_cast<char*>(host_base);
    for (std::size_t h = 0; h < heads; ++h) {
        const std::size_t run_index =
            order == PagedKVPlaneOrder::PageMajor ? h + heads * page : page + physical_pages * h;
        char* dev_run = device_base + run_bytes * run_index;
        if (DeviceToHost) {
            if (cudaMemcpyAsync(host, dev_run, run_bytes, cudaMemcpyDeviceToHost, stream) !=
                cudaSuccess) {
                return false;
            }
        } else {
            if (cudaMemcpyAsync(dev_run, host, run_bytes, cudaMemcpyHostToDevice, stream) !=
                cudaSuccess) {
                return false;
            }
        }
        host += run_bytes;
    }
    return true;
}

} // namespace

std::size_t PagedKVPool::per_page_bytes() const noexcept {
    std::size_t total = 0;
    for (const PagedKVPlaneSpec& spec : spec_.planes) {
        total += per_plane_page_bytes(spec);
    }
    return total;
}

bool PagedKVPool::copy_page_d2h(void* host_dst, std::int32_t page_id,
                                cudaStream_t stream) const noexcept {
    char* out = static_cast<char*>(host_dst);
    for (std::size_t i = 0; i < planes_.size(); ++i) {
        if (!copy_plane_page<true>(spec_.planes[i], page_id, out,
                                   static_cast<char*>(planes_[i].data), spec_.page_group_count,
                                   spec_.plane_order, stream)) {
            (void)cudaGetLastError();
            return false;
        }
        out += per_plane_page_bytes(spec_.planes[i]);
    }
    return true;
}

bool PagedKVPool::copy_page_h2d(const void* host_src, std::int32_t page_id,
                                cudaStream_t stream) const noexcept {
    char* in = const_cast<char*>(static_cast<const char*>(host_src));
    for (std::size_t i = 0; i < planes_.size(); ++i) {
        if (!copy_plane_page<false>(spec_.planes[i], page_id, in,
                                    static_cast<char*>(planes_[i].data), spec_.page_group_count,
                                    spec_.plane_order, stream)) {
            (void)cudaGetLastError();
            return false;
        }
        in += per_plane_page_bytes(spec_.planes[i]);
    }
    return true;
}

std::uint32_t PagedKVPool::entitled_pages() const noexcept { return entitled_pages_; }

std::uint32_t PagedKVPool::mapped_pages() const noexcept { return mapped_pages_; }

std::uint32_t PagedKVPool::free_pages() const noexcept {
    return static_cast<std::uint32_t>(free_page_ids_.size());
}

std::uint32_t PagedKVPool::cached_pages() const noexcept { return cached_pages_; }

bool PagedKVPool::can_reserve(std::uint32_t page_entitlement) const noexcept {
    // Signed headroom: `entitled + cached` may not exceed the pool, and any future accounting
    // drift must reject the reservation rather than underflow `page_group_count() - used` into a
    // huge unsigned value that accepts everything.
    const std::int64_t headroom = static_cast<std::int64_t>(page_group_count()) -
                                  static_cast<std::int64_t>(entitled_pages_) -
                                  static_cast<std::int64_t>(cached_pages_);
    return page_entitlement != 0 && page_entitlement <= logical_page_capacity() &&
           headroom >= 0 && static_cast<std::int64_t>(page_entitlement) <= headroom;
}

bool PagedKVPool::can_replace_entitlement(std::uint32_t old_pages,
                                          std::uint32_t new_pages) const noexcept {
    if (old_pages > entitled_pages_ || new_pages > logical_page_capacity()) { return false; }
    const std::int64_t headroom = static_cast<std::int64_t>(page_group_count()) -
                                  static_cast<std::int64_t>(entitled_pages_ - old_pages) -
                                  static_cast<std::int64_t>(cached_pages_);
    return headroom >= 0 && static_cast<std::int64_t>(new_pages) <= headroom;
}

PagedKVAllocation PagedKVPool::reserve(std::uint32_t page_entitlement) {
    if (!can_reserve(page_entitlement)) { throw std::bad_alloc(); }
    PagedKVAllocation allocation(*this, page_entitlement);
    add_entitlement(page_entitlement);
    return allocation;
}

void PagedKVPool::zero_pages(std::span<const std::int32_t> page_ids, cudaStream_t stream) {
    if (page_ids.empty()) { return; }

    std::vector<std::int32_t> sorted(page_ids.begin(), page_ids.end());
    std::sort(sorted.begin(), sorted.end());
    if (sorted.front() < 0 || sorted.back() >= static_cast<std::int32_t>(page_group_count())) {
        throw std::out_of_range("Paged KV physical page is out of range");
    }
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        throw std::invalid_argument("Paged KV physical pages must be distinct");
    }

    const auto zero_run = [&](std::int32_t first, std::int32_t count) {
        for (const Tensor& plane : planes_) {
            auto* base = static_cast<unsigned char*>(plane.data);
            if (spec_.plane_order == PagedKVPlaneOrder::PageMajor) {
                CUDA_CHECK(cudaMemsetAsync(base + static_cast<std::int64_t>(first) * plane.nb[3], 0,
                                           static_cast<std::size_t>(count) * plane.nb[3], stream));
            } else {
                CUDA_CHECK(cudaMemset2DAsync(base + static_cast<std::int64_t>(first) * plane.nb[2],
                                             plane.nb[3], 0,
                                             static_cast<std::size_t>(count) * plane.nb[2],
                                             static_cast<std::size_t>(plane.ne[3]), stream));
            }
        }
    };

    std::size_t begin = 0;
    while (begin < sorted.size()) {
        std::size_t end = begin + 1;
        while (end < sorted.size() && sorted[end] == sorted[end - 1] + 1) { ++end; }
        zero_run(sorted[begin], static_cast<std::int32_t>(end - begin));
        begin = end;
    }
}

std::vector<std::int32_t> PagedKVPool::take_pages(std::uint32_t count,
                                                  std::int32_t preferred_first) {
    if (count == 0) { return {}; }
    if (count > free_page_ids_.size()) {
        throw std::logic_error("Paged KV entitlement could not be materialized");
    }

    const auto run_at = [&](std::size_t begin) {
        if (begin + count > free_page_ids_.size()) { return false; }
        for (std::uint32_t offset = 1; offset < count; ++offset) {
            if (free_page_ids_[begin + offset] !=
                free_page_ids_[begin] + static_cast<std::int32_t>(offset)) {
                return false;
            }
        }
        return true;
    };

    std::size_t selected = free_page_ids_.size();
    if (preferred_first >= 0) {
        const auto it =
            std::lower_bound(free_page_ids_.begin(), free_page_ids_.end(), preferred_first);
        if (it != free_page_ids_.end() && *it == preferred_first) {
            const auto begin = static_cast<std::size_t>(it - free_page_ids_.begin());
            if (run_at(begin)) { selected = begin; }
        }
    }
    if (selected == free_page_ids_.size()) {
        for (std::size_t begin = 0; begin + count <= free_page_ids_.size(); ++begin) {
            if (run_at(begin)) {
                selected = begin;
                break;
            }
        }
    }

    std::vector<std::int32_t> out;
    out.reserve(count);
    if (selected != free_page_ids_.size()) {
        const auto first = free_page_ids_.begin() + static_cast<std::ptrdiff_t>(selected);
        const auto last  = first + static_cast<std::ptrdiff_t>(count);
        out.insert(out.end(), first, last);
        free_page_ids_.erase(first, last);
    } else {
        const auto last = free_page_ids_.begin() + static_cast<std::ptrdiff_t>(count);
        out.insert(out.end(), free_page_ids_.begin(), last);
        free_page_ids_.erase(free_page_ids_.begin(), last);
    }
    mapped_pages_ += count;
    return out;
}

void PagedKVPool::return_pages(std::span<const std::int32_t> pages) noexcept {
    if (pages.empty()) { return; }
    free_page_ids_.insert(free_page_ids_.end(), pages.begin(), pages.end());
    std::sort(free_page_ids_.begin(), free_page_ids_.end());
    mapped_pages_ -= static_cast<std::uint32_t>(pages.size());
}

void PagedKVPool::add_entitlement(std::uint32_t pages) noexcept { entitled_pages_ += pages; }

void PagedKVPool::replace_entitlement(std::uint32_t old_pages, std::uint32_t new_pages) noexcept {
    entitled_pages_ = entitled_pages_ - old_pages + new_pages;
}

std::vector<std::int32_t> PagedKVPool::acquire_cache_pages(std::uint32_t count) {
    if (count == 0) { return {}; }
    if (count > free_page_ids_.size()) {
        throw std::bad_alloc();
    }
    std::vector<std::int32_t> pages = take_pages(count, -1);
    cached_pages_ += count;
    return pages;
}

void PagedKVPool::release_cache_pages(std::span<const std::int32_t> page_ids) noexcept {
    if (page_ids.empty()) { return; }
    // A cache page was already removed from mapped_pages_ when it was promoted, so releasing it
    // only moves it back to the free list and out of the cache. Decrementing mapped here (as
    // return_pages does) would double-count and drive mapped_pages_ negative under eviction.
    cached_pages_ -= static_cast<std::uint32_t>(page_ids.size());
    free_page_ids_.insert(free_page_ids_.end(), page_ids.begin(), page_ids.end());
    std::sort(free_page_ids_.begin(), free_page_ids_.end());
}

void PagedKVPool::promote_to_cache(std::span<const std::int32_t> page_ids) {
    if (page_ids.empty()) { return; }
    if (page_ids.size() > mapped_pages_) {
        throw std::invalid_argument("Paged KV promotion exceeds the mapped page count");
    }
    // page_ids are physical page-group ids in [0, page_group_count()); they are NOT bounded by the
    // per-allocation logical capacity (M >= L). A resumed long-context request can own physical
    // pages above L, so validating against logical_page_capacity() rejects valid promotions.
    const auto capacity = static_cast<std::int32_t>(page_group_count());
    for (const std::int32_t page : page_ids) {
        if (page < 0 || page >= capacity) {
            throw std::invalid_argument("Paged KV promotion physical page is out of range");
        }
    }
    const auto count = static_cast<std::uint32_t>(page_ids.size());
    // A promoted page leaves the owning allocation's reservation and becomes cache-owned: it must
    // no longer be counted in entitled_pages_. Counting it in both `entitled` and `cached` makes
    // `page_group_count() - entitled - cached` underflow (unsigned), which silently defeats every
    // capacity predicate and over-admits until materialization runs out of pages.
    if (entitled_pages_ < count) {
        throw std::logic_error("Paged KV promotion exceeds the entitled page count");
    }
    entitled_pages_ -= count;
    cached_pages_ += count;
    mapped_pages_ -= count;
}

void PagedKVPool::acquire_row(std::int32_t row) {
    if (row < 0 || row >= table_row_count()) {
        throw std::out_of_range("Paged KV block-table row out of range");
    }
    if (row_in_use_[static_cast<std::size_t>(row)]) {
        throw std::logic_error("Paged KV block-table row is already bound");
    }
    row_in_use_[static_cast<std::size_t>(row)] = true;
}

void PagedKVPool::release_row(std::int32_t row) noexcept {
    row_in_use_[static_cast<std::size_t>(row)] = false;
}

PagedKVAllocation::PagedKVAllocation(PagedKVPool& pool, std::uint32_t page_entitlement)
    : pool_(&pool), page_entitlement_(page_entitlement) {
    page_ids_.reserve(page_entitlement);
}

PagedKVAllocation::~PagedKVAllocation() { release(); }

PagedKVAllocation::PagedKVAllocation(PagedKVAllocation&& other) noexcept
    : pool_(other.pool_), page_ids_(std::move(other.page_ids_)),
      cache_owned_(std::move(other.cache_owned_)),
      page_entitlement_(other.page_entitlement_),
      bound_row_(other.bound_row_) {
    other.pool_             = nullptr;
    other.page_entitlement_ = 0;
    other.bound_row_        = -1;
}

PagedKVAllocation& PagedKVAllocation::operator=(PagedKVAllocation&& other) noexcept {
    if (this == &other) { return *this; }
    release();
    pool_                   = other.pool_;
    page_ids_               = std::move(other.page_ids_);
    cache_owned_            = std::move(other.cache_owned_);
    page_entitlement_       = other.page_entitlement_;
    bound_row_              = other.bound_row_;
    other.pool_             = nullptr;
    other.page_entitlement_ = 0;
    other.bound_row_        = -1;
    return *this;
}

bool PagedKVAllocation::valid() const noexcept { return pool_ != nullptr; }

std::uint32_t PagedKVAllocation::page_entitlement() const noexcept { return page_entitlement_; }

std::uint32_t PagedKVAllocation::mapped_page_count() const noexcept {
    return static_cast<std::uint32_t>(page_ids_.size());
}

std::uint32_t PagedKVAllocation::owned_page_count() const noexcept {
    std::uint32_t owned = 0;
    for (bool cache_owned : cache_owned_) {
        if (!cache_owned) { ++owned; }
    }
    return owned;
}

std::uint32_t PagedKVAllocation::borrowed_page_count() const noexcept {
    std::uint32_t cached = 0;
    for (bool cache_owned : cache_owned_) {
        if (cache_owned) { ++cached; }
    }
    return cached;
}

std::uint32_t PagedKVAllocation::mapped_token_capacity() const noexcept {
    return mapped_page_count() * static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::int32_t PagedKVAllocation::bound_row() const noexcept { return bound_row_; }

std::span<const std::int32_t> PagedKVAllocation::page_ids() const noexcept { return page_ids_; }

bool PagedKVAllocation::belongs_to(const PagedKVPool& pool) const noexcept {
    return pool_ == &pool;
}

bool PagedKVAllocation::is_cache_owned(std::uint32_t page_index) const noexcept {
    return page_index < cache_owned_.size() && cache_owned_[page_index];
}

void PagedKVAllocation::borrow_prefix(std::span<const std::int32_t> page_ids) {
    if (!valid()) {
        throw std::logic_error("Cannot borrow prefix pages into an empty Paged KV allocation");
    }
    if (!page_ids_.empty()) {
        throw std::logic_error("Paged KV prefix pages must be borrowed before materialization");
    }
    if (page_ids.empty()) { return; }
    page_ids_.assign(page_ids.begin(), page_ids.end());
    cache_owned_.assign(page_ids.size(), true);
}

void PagedKVAllocation::promote_page_at(std::uint32_t index) {
    if (!valid() || index >= page_ids_.size()) {
        throw std::invalid_argument("Paged KV promotion index is out of range");
    }
    if (cache_owned_[index]) {
        throw std::invalid_argument("Paged KV page is already cache-owned");
    }
    // The promoted page becomes cache-owned rather than owned: it keeps its id and stays published
    // in the block table but leaves this allocation's entitlement, so the pool's entitled count
    // matches the sum of the allocations' entitlements after every promotion. A non-cache-owned
    // index is an owned page, hence within the current entitlement, so the decrement is safe.
    const std::int32_t page = page_ids_[index];
    pool_->promote_to_cache(std::span<const std::int32_t>(&page, 1));
    cache_owned_[index] = true;
    --page_entitlement_;
}

void PagedKVAllocation::set_page_entitlement(std::uint32_t pages) {
    if (!valid() || pages < owned_page_count()) {
        throw std::invalid_argument("Paged KV entitlement is smaller than owned pages");
    }
    if (!pool_->can_replace_entitlement(page_entitlement_, pages)) { throw std::bad_alloc(); }
    page_ids_.reserve(pages);
    pool_->replace_entitlement(page_entitlement_, pages);
    page_entitlement_ = pages;
}

void PagedKVAllocation::cancel_unmapped_entitlement() noexcept {
    if (!valid()) { return; }
    const std::uint32_t owned = owned_page_count();
    pool_->replace_entitlement(page_entitlement_, owned);
    page_entitlement_ = owned;
}

void PagedKVAllocation::materialize_pages(std::uint32_t pages, cudaStream_t stream) {
    if (!valid() || pages < mapped_page_count()) {
        throw std::invalid_argument("Paged KV materialize extent is outside entitlement");
    }
    const std::uint32_t old_count = mapped_page_count();
    const std::uint32_t count     = pages - old_count;
    if (count == 0) { return; }
    // New pages are owned, so the owned count grows by `count` and must stay within entitlement.
    if (owned_page_count() + count > page_entitlement_) {
        throw std::invalid_argument("Paged KV materialize extent is outside entitlement");
    }
    const std::int32_t preferred =
        page_ids_.empty() ? -1 : static_cast<std::int32_t>(page_ids_.back() + 1);
    std::vector<std::int32_t> acquired = pool_->take_pages(count, preferred);
    page_ids_.insert(page_ids_.end(), acquired.begin(), acquired.end());
    cache_owned_.resize(page_ids_.size(), false);
    if (bound_row_ >= 0) { publish_range(old_count, count, stream); }
}

void PagedKVAllocation::materialize_tokens(std::uint32_t tokens, cudaStream_t stream) {
    materialize_pages(pages_for_tokens(tokens), stream);
}

void PagedKVAllocation::trim_pages(std::uint32_t pages) {
    if (!valid()) { throw std::logic_error("Cannot trim an empty Paged KV allocation"); }
    if (pages > mapped_page_count()) {
        throw std::invalid_argument("Paged KV trim extent exceeds mapped pages");
    }
    // The trimmed tail must be all owned pages; a cache-owned page (borrowed prefix or adopted
    // block) is still live in the block table and must not be returned to the pool.
    for (std::uint32_t i = pages; i < mapped_page_count(); ++i) {
        if (cache_owned_[i]) {
            throw std::invalid_argument("Paged KV trim extent would discard cache-owned pages");
        }
    }
    if (pages == mapped_page_count()) { return; }
    pool_->return_pages(std::span<const std::int32_t>(
        page_ids_.data() + pages, static_cast<std::size_t>(mapped_page_count() - pages)));
    page_ids_.resize(pages);
    cache_owned_.resize(pages);
}

void PagedKVAllocation::trim_tokens(std::uint32_t tokens) { trim_pages(pages_for_tokens(tokens)); }

void PagedKVAllocation::bind_row(std::int32_t row, cudaStream_t stream) {
    if (!valid()) { throw std::logic_error("Cannot bind an empty Paged KV allocation"); }
    if (bound_row_ >= 0) { throw std::logic_error("Paged KV allocation is already bound"); }
    pool_->acquire_row(row);
    bound_row_ = row;
    publish_mapping(stream);
}

void PagedKVAllocation::publish_mapping(cudaStream_t stream) const {
    if (bound_row_ < 0) { throw std::logic_error("Paged KV allocation is not bound"); }
    publish_range(0, mapped_page_count(), stream);
}

void PagedKVAllocation::publish_range(std::uint32_t first_page, std::uint32_t page_count,
                                      cudaStream_t stream) const {
    if (page_count == 0) { return; }
    Tensor row         = pool_->block_table_row(bound_row_);
    auto* destination  = static_cast<std::int32_t*>(row.data) + first_page;
    const auto* source = page_ids_.data() + first_page;
    CUDA_CHECK(cudaMemcpyAsync(destination, source,
                               static_cast<std::size_t>(page_count) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
}

void PagedKVAllocation::unbind_row() noexcept {
    if (bound_row_ < 0) { return; }
    pool_->release_row(bound_row_);
    bound_row_ = -1;
}

Tensor PagedKVAllocation::block_table() const {
    if (bound_row_ < 0) { throw std::logic_error("Paged KV allocation is not bound"); }
    return pool_->block_table_row(bound_row_);
}

void PagedKVAllocation::release() noexcept {
    if (!valid()) { return; }
    unbind_row();
    // Return only the owned pages to the pool. Cache-owned pages (borrowed prefix / adopted
    // blocks) belong to the prefix cache, stay out of this release, and are reclaimed through the
    // index's own eviction path instead.
    std::vector<std::int32_t> owned;
    owned.reserve(page_ids_.size());
    for (std::size_t i = 0; i < page_ids_.size(); ++i) {
        if (!cache_owned_[i]) { owned.push_back(page_ids_[i]); }
    }
    if (!owned.empty()) {
        pool_->return_pages(std::span<const std::int32_t>(owned.data(), owned.size()));
    }
    pool_->replace_entitlement(page_entitlement_, 0);
    page_ids_.clear();
    cache_owned_.clear();
    page_entitlement_ = 0;
    pool_             = nullptr;
}

std::vector<PagedKVAllocation>
reserve_paged_kv_bundle(std::span<const PagedKVReservation> reservations) {
    validate_distinct_pools(reservations);
    for (const PagedKVReservation& reservation : reservations) {
        if (!reservation.pool->can_reserve(reservation.page_entitlement)) {
            throw std::bad_alloc();
        }
    }

    std::vector<PagedKVAllocation> allocations;
    allocations.reserve(reservations.size());
    for (const PagedKVReservation& reservation : reservations) {
        allocations.push_back(reservation.pool->reserve(reservation.page_entitlement));
    }
    return allocations;
}

void resize_paged_kv_bundle(std::span<const PagedKVResize> changes) {
    for (std::size_t i = 0; i < changes.size(); ++i) {
        const PagedKVResize& change = changes[i];
        if (change.allocation == nullptr || !change.allocation->valid()) {
            throw std::invalid_argument("Paged KV resize must name a live allocation");
        }
        const std::uint32_t borrowed = change.allocation->borrowed_page_count();
        if (change.mapped_pages < borrowed ||
            change.mapped_pages > change.allocation->mapped_page_count() ||
            change.mapped_pages - borrowed > change.page_entitlement) {
            throw std::invalid_argument("Paged KV resize extents are inconsistent");
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (change.allocation == changes[j].allocation ||
                change.allocation->pool_ == changes[j].allocation->pool_) {
                throw std::invalid_argument("Paged KV resize contains the same pool twice");
            }
        }
        if (!change.allocation->pool_->can_replace_entitlement(change.allocation->page_entitlement_,
                                                               change.page_entitlement)) {
            throw std::bad_alloc();
        }
    }
    // Complete every potentially throwing host allocation before changing any pool accounting.
    for (const PagedKVResize& change : changes) {
        change.allocation->page_ids_.reserve(change.page_entitlement);
    }
    for (const PagedKVResize& change : changes) {
        change.allocation->trim_pages(change.mapped_pages);
        change.allocation->set_page_entitlement(change.page_entitlement);
    }
}

} // namespace ninfer
