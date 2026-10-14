#include "core/device.h"
#include "core/paged_kv_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

struct PlannedPagedCache {
    ninfer::PagedKVPoolLayout layout;
    std::size_t bytes = 0;
};

PlannedPagedCache
plan_paged_cache(std::uint32_t pages, std::uint32_t logical_pages, std::int32_t rows,
                 std::vector<ninfer::PagedKVPlaneSpec> planes,
                 ninfer::PagedKVPlaneOrder order = ninfer::PagedKVPlaneOrder::PageMajor) {
    ninfer::LayoutBuilder builder;
    auto layout = ninfer::plan_paged_kv_pool(builder, {.page_group_count      = pages,
                                                       .logical_page_capacity = logical_pages,
                                                       .table_rows            = rows,
                                                       .plane_order           = order,
                                                       .planes                = std::move(planes)});
    return PlannedPagedCache{std::move(layout), builder.finish(256)};
}

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_size(std::size_t actual, std::size_t expected, const char* label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int check_shape(const ninfer::Tensor& tensor, const std::int32_t (&expected)[4],
                const char* label) {
    int failures = 0;
    for (int i = 0; i < 4; ++i) {
        if (tensor.ne[i] != expected[i]) {
            ++failures;
            std::cerr << label << ".ne[" << i << "] expected " << expected[i] << ", got "
                      << tensor.ne[i] << '\n';
        }
    }
    return failures;
}

int expect_page_ids(std::span<const std::int32_t> actual,
                    std::initializer_list<std::int32_t> expected, const char* label) {
    if (actual.size() == expected.size() &&
        std::equal(actual.begin(), actual.end(), expected.begin(), expected.end())) {
        return 0;
    }
    std::cerr << label << " page IDs differ\n";
    return 1;
}

int expect_device_page_ids(const ninfer::Tensor& row, std::initializer_list<std::int32_t> expected,
                           const char* label) {
    std::vector<std::int32_t> actual(expected.size());
    const cudaError_t err = cudaMemcpy(
        actual.data(), row.data, actual.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " copy failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    return expect_page_ids(actual, expected, label);
}

int expect_zeroed_pages(ninfer::PagedKVPool& pool, ninfer::PagedKVPlaneOrder order,
                        std::span<const std::int32_t> pages, cudaStream_t stream,
                        const char* label) {
    const ninfer::Tensor& plane = pool.plane(0);
    cudaError_t err             = cudaMemsetAsync(plane.data, 0x5a, plane.bytes(), stream);
    if (err != cudaSuccess) {
        std::cerr << label << " setup failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    pool.zero_pages(pages, stream);
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << label << " synchronization failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }

    std::vector<unsigned char> actual(plane.bytes());
    err = cudaMemcpy(actual.data(), plane.data, actual.size(), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " copy failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    std::vector<unsigned char> expected(actual.size(), 0x5a);
    for (const std::int32_t page : pages) {
        if (order == ninfer::PagedKVPlaneOrder::PageMajor) {
            const std::size_t begin = static_cast<std::size_t>(page * plane.nb[3]);
            std::fill(expected.begin() + static_cast<std::ptrdiff_t>(begin),
                      expected.begin() + static_cast<std::ptrdiff_t>(begin + plane.nb[3]), 0);
        } else {
            for (std::int32_t head = 0; head < plane.ne[3]; ++head) {
                const std::size_t begin =
                    static_cast<std::size_t>(head * plane.nb[3] + page * plane.nb[2]);
                std::fill(expected.begin() + static_cast<std::ptrdiff_t>(begin),
                          expected.begin() + static_cast<std::ptrdiff_t>(begin + plane.nb[2]), 0);
            }
        }
    }
    if (actual == expected) { return 0; }
    std::cerr << label << " cleared bytes outside the selected physical pages\n";
    return 1;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    int failures = 0;
    ninfer::DeviceContext ctx(0);

    auto paged_plan = plan_paged_cache(10, 10, 2,
                                       {{ninfer::DType::I8, 64, 2},
                                        {ninfer::DType::I8, 64, 2},
                                        {ninfer::DType::FP16, 1, 2},
                                        {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena paged_arena(paged_plan.bytes);
    ninfer::PagedKVPool paged_pool({paged_arena.base(), paged_arena.capacity()}, paged_plan.layout);
    failures += expect_size(paged_pool.plane_count(), 4, "paged plane count");
    failures += check_shape(paged_pool.plane(0), {64, 64, 2, 10}, "paged code plane");
    failures += check_shape(paged_pool.plane(2), {1, 64, 2, 10}, "paged scale plane");
    failures += check_shape(paged_pool.block_table_row(0), {10, 1, 1, 1}, "paged block-table row");
    if (!paged_pool.plane(0).is_contiguous() || !paged_pool.plane(2).is_contiguous() ||
        !paged_pool.block_table_row(0).is_contiguous()) {
        ++failures;
        std::cerr << "Paged planes or block-table row are not contiguous\n";
    }
    const std::size_t expected_payload =
        2 * ninfer::Tensor(nullptr, ninfer::DType::I8, {64, 64, 2, 10}).bytes() +
        2 * ninfer::Tensor(nullptr, ninfer::DType::FP16, {1, 64, 2, 10}).bytes();
    failures +=
        expect_size(paged_plan.layout.payload_bytes(), expected_payload, "paged payload bytes");
    failures += expect_size(paged_plan.layout.metadata_bytes(), 10 * 2 * sizeof(std::int32_t),
                            "paged metadata bytes");
    const std::int32_t selected_pages[] = {1, 2, 6};
    failures += expect_zeroed_pages(paged_pool, ninfer::PagedKVPlaneOrder::PageMajor,
                                    selected_pages, ctx.stream, "page-major selective zero");

    auto head_major_plan = plan_paged_cache(10, 10, 1, {{ninfer::DType::BF16, 128, 8}},
                                            ninfer::PagedKVPlaneOrder::HeadMajor);
    ninfer::DeviceArena head_major_arena(head_major_plan.bytes);
    ninfer::PagedKVPool head_major_pool({head_major_arena.base(), head_major_arena.capacity()},
                                        head_major_plan.layout);
    failures += check_shape(head_major_pool.plane(0), {128, 64, 10, 8}, "head-major paged plane");
    failures += expect_zeroed_pages(head_major_pool, ninfer::PagedKVPlaneOrder::HeadMajor,
                                    selected_pages, ctx.stream, "head-major selective zero");

    auto allocation_a = paged_pool.reserve(3);
    auto allocation_b = paged_pool.reserve(3);
    allocation_a.materialize_pages(3);
    allocation_b.materialize_pages(3);
    allocation_a.bind_row(0);
    allocation_b.bind_row(1);
    failures += expect_device_page_ids(allocation_a.block_table(), {0, 1, 2}, "allocation A");
    failures += expect_device_page_ids(allocation_b.block_table(), {3, 4, 5}, "allocation B");
    allocation_a.release();

    auto allocation_c = paged_pool.reserve(6);
    allocation_c.materialize_pages(6);
    allocation_c.bind_row(0);
    failures +=
        expect_page_ids(allocation_c.page_ids(), {0, 1, 2, 6, 7, 8}, "fragmented allocation C");
    failures +=
        expect_device_page_ids(allocation_c.block_table(), {0, 1, 2, 6, 7, 8}, "fragmented row C");
    failures += expect_device_page_ids(allocation_b.block_table(), {3, 4, 5}, "isolated row B");

    allocation_c.trim_pages(2);
    if (paged_pool.can_reserve(2)) {
        ++failures;
        std::cerr << "Unmapped entitlement was exposed as reservable capacity\n";
    }
    allocation_c.cancel_unmapped_entitlement();
    if (!paged_pool.can_reserve(5)) {
        ++failures;
        std::cerr << "Cancelled entitlement did not return reservable capacity\n";
    }

    allocation_c.unbind_row();
    const std::int32_t retained_prefix[] = {allocation_c.page_ids()[0], allocation_c.page_ids()[1]};
    const ninfer::PagedKVResize claim[]  = {
        {.allocation = &allocation_c, .mapped_pages = 2, .page_entitlement = 5}};
    ninfer::resize_paged_kv_bundle(claim);
    allocation_c.bind_row(0);
    allocation_c.materialize_pages(5);
    if (allocation_c.page_ids()[0] != retained_prefix[0] ||
        allocation_c.page_ids()[1] != retained_prefix[1]) {
        ++failures;
        std::cerr << "Retained allocation moved its prefix pages\n";
    }
    allocation_c.trim_tokens(65);
    failures += expect_size(allocation_c.mapped_page_count(), 2, "partial-tail mapped pages");
    allocation_c.trim_tokens(64);
    failures += expect_size(allocation_c.mapped_page_count(), 1, "page-aligned mapped pages");

    auto main_plan    = plan_paged_cache(4, 4, 1, {{ninfer::DType::BF16, 16, 1}});
    auto backend_plan = plan_paged_cache(2, 2, 1, {{ninfer::DType::BF16, 16, 1}});
    ninfer::DeviceArena main_arena(main_plan.bytes);
    ninfer::DeviceArena backend_arena(backend_plan.bytes);
    ninfer::PagedKVPool main_pool({main_arena.base(), main_arena.capacity()}, main_plan.layout);
    ninfer::PagedKVPool backend_pool({backend_arena.base(), backend_arena.capacity()},
                                     backend_plan.layout);
    const ninfer::PagedKVReservation impossible_bundle[] = {
        {.pool = &main_pool, .page_entitlement = 3},
        {.pool = &backend_pool, .page_entitlement = 3},
    };
    try {
        auto unused = ninfer::reserve_paged_kv_bundle(impossible_bundle);
        (void)unused;
        ++failures;
        std::cerr << "Impossible multi-pool reservation succeeded\n";
    } catch (const std::bad_alloc&) {}
    failures += expect_size(main_pool.entitled_pages(), 0, "failed bundle main entitlement");
    failures += expect_size(backend_pool.entitled_pages(), 0, "failed bundle backend entitlement");

    const ninfer::PagedKVReservation possible_bundle[] = {
        {.pool = &main_pool, .page_entitlement = 2},
        {.pool = &backend_pool, .page_entitlement = 2},
    };
    auto bundle = ninfer::reserve_paged_kv_bundle(possible_bundle);
    bundle[0].materialize_pages(2);
    bundle[1].materialize_pages(2);
    const ninfer::PagedKVResize impossible_resize[] = {
        {.allocation = &bundle[0], .mapped_pages = 1, .page_entitlement = 1},
        {.allocation = &bundle[1], .mapped_pages = 1, .page_entitlement = 3},
    };
    try {
        ninfer::resize_paged_kv_bundle(impossible_resize);
        ++failures;
        std::cerr << "Impossible multi-pool resize succeeded\n";
    } catch (const std::bad_alloc&) {}
    failures += expect_size(bundle[0].mapped_page_count(), 2, "failed resize main mapping");
    failures += expect_size(bundle[0].page_entitlement(), 2, "failed resize main entitlement");
    failures += expect_size(bundle[1].mapped_page_count(), 2, "failed resize backend mapping");
    failures += expect_size(bundle[1].page_entitlement(), 2, "failed resize backend entitlement");

    auto cache_plan = plan_paged_cache(6, 6, 1, {{ninfer::DType::BF16, 16, 1}});
    ninfer::DeviceArena cache_arena(cache_plan.bytes);
    ninfer::PagedKVPool cache_pool({cache_arena.base(), cache_arena.capacity()}, cache_plan.layout);

    std::vector<std::int32_t> cached = cache_pool.acquire_cache_pages(2);
    failures += expect_page_ids(cached, {0, 1}, "cache-owned pages");
    failures += expect_size(cache_pool.cached_pages(), 2, "cached page count");
    failures += expect_size(cache_pool.free_pages(), 4, "free pages after cache ownership");
    if (cache_pool.can_reserve(5) || !cache_pool.can_reserve(4)) {
        ++failures;
        std::cerr << "Cache-owned pages were not subtracted from reservable capacity\n";
    }

    auto borrowed = cache_pool.reserve(2);
    borrowed.borrow_prefix(cached);
    borrowed.materialize_pages(4);
    failures += expect_size(borrowed.borrowed_page_count(), 2, "borrowed prefix pages");
    failures += expect_size(borrowed.owned_page_count(), 2, "owned suffix pages");
    failures += expect_size(borrowed.mapped_page_count(), 4, "borrowed mapped pages");
    failures += expect_page_ids(borrowed.page_ids(), {0, 1, 2, 3}, "borrowed page order");
    borrowed.bind_row(0);
    failures += expect_device_page_ids(borrowed.block_table(), {0, 1, 2, 3}, "borrowed block table");

    bool trim_rejected = false;
    try {
        borrowed.trim_pages(1);
    } catch (const std::invalid_argument&) {
        trim_rejected = true;
    }
    if (!trim_rejected) {
        ++failures;
        std::cerr << "Trimming into the borrowed prefix was not rejected\n";
    }
    borrowed.trim_pages(2);
    failures += expect_page_ids(borrowed.page_ids(), {0, 1}, "trim keeps borrowed prefix");
    borrowed.unbind_row();
    borrowed.release();
    failures +=
        expect_size(cache_pool.free_pages(), 4, "borrowed pages were not returned to the pool");
    failures += expect_size(cache_pool.cached_pages(), 2, "borrowed release dropped cache pages");

    cache_pool.release_cache_pages(cached);
    failures += expect_size(cache_pool.cached_pages(), 0, "cache release count");
    failures += expect_size(cache_pool.free_pages(), 6, "cache release restored free pages");

    cached    = cache_pool.acquire_cache_pages(2);
    auto share_a = cache_pool.reserve(1);
    auto share_b = cache_pool.reserve(1);
    share_a.borrow_prefix(cached);
    share_b.borrow_prefix(cached);
    share_a.materialize_pages(3);
    share_b.materialize_pages(3);
    failures += expect_page_ids(share_a.page_ids(), {0, 1, 2}, "shared borrower A");
    failures += expect_page_ids(share_b.page_ids(), {0, 1, 3}, "shared borrower B");
    share_a.release();
    failures += expect_size(cache_pool.free_pages(), 3, "shared borrower A release");
    failures += expect_size(cache_pool.cached_pages(), 2, "shared borrower A kept cache pages");

    share_b.unbind_row();
    share_b.release();
    failures += expect_size(cache_pool.free_pages(), 4, "shared borrower B release");

    auto promoter = cache_pool.reserve(3);
    promoter.borrow_prefix(cached);
    promoter.materialize_pages(4);
    failures += expect_page_ids(promoter.page_ids(), {0, 1, 2, 3}, "promoter page order");
    // A page already borrowed from the cache is not owned and cannot be promoted again.
    bool promote_rejected = false;
    try {
        promoter.promote_page_at(0);
    } catch (const std::invalid_argument&) {
        promote_rejected = true;
    }
    if (!promote_rejected) {
        ++failures;
        std::cerr << "Promoting a borrowed (cache-owned) page was not rejected\n";
    }
    promoter.promote_page_at(2);
    promoter.promote_page_at(3);
    // Accessor backing the capture loop's skip guard: a borrowed/promoted index reports cache-owned,
    // and re-promoting it must be rejected so capture can skip pages it already handed to the cache.
    if (!promoter.is_cache_owned(0) || !promoter.is_cache_owned(1) || !promoter.is_cache_owned(2) ||
        !promoter.is_cache_owned(3)) {
        ++failures;
        std::cerr << "is_cache_owned did not report borrowed/promoted pages\n";
    }
    bool double_promote_rejected = false;
    try {
        promoter.promote_page_at(2);
    } catch (const std::invalid_argument&) {
        double_promote_rejected = true;
    }
    if (!double_promote_rejected) {
        ++failures;
        std::cerr << "Re-promoting an already promoted (cache-owned) page was not rejected\n";
    }
    failures += expect_size(promoter.borrowed_page_count(), 4, "promoted pages joined the prefix");
    failures += expect_size(promoter.owned_page_count(), 0, "promotion left no owned pages");
    failures += expect_size(cache_pool.cached_pages(), 4, "promotion counted cache pages");
    promoter.unbind_row();
    promoter.release();
    failures +=
        expect_size(cache_pool.free_pages(), 2, "promoted pages were not returned to the pool");
    failures += expect_size(cache_pool.cached_pages(), 4, "promoted pages stayed cache-owned");
    const std::int32_t all_promoted[] = {0, 1, 2, 3};
    cache_pool.release_cache_pages(all_promoted);
    failures += expect_size(cache_pool.cached_pages(), 0, "promoted cache release");
    failures += expect_size(cache_pool.free_pages(), 6, "promoted cache release restored free pages");

    // Regression: page ids are physical page-group ids bounded by page_group_count(), not by the
    // per-allocation logical capacity. With M > L, a long-context allocation can own and promote
    // physical pages above L; those must not be rejected. Reproduce by forcing the allocation into
    // the high physical pages (occupy the low ones as cache-owned first).
    auto wide_plan = plan_paged_cache(12, 4, 1, {{ninfer::DType::BF16, 16, 1}});
    ninfer::DeviceArena wide_arena(wide_plan.bytes);
    ninfer::PagedKVPool wide_pool({wide_arena.base(), wide_arena.capacity()}, wide_plan.layout);
    std::vector<std::int32_t> wide_cached = wide_pool.acquire_cache_pages(8);
    failures += expect_page_ids(wide_cached, {0, 1, 2, 3, 4, 5, 6, 7}, "wide cache-owned pages");
    auto wide = wide_pool.reserve(4);
    wide.materialize_pages(4);
    failures += expect_page_ids(wide.page_ids(), {8, 9, 10, 11}, "wide physical page ids");
    bool wide_promoted = true;
    try {
        wide.promote_page_at(0);
        wide.promote_page_at(1);
        wide.promote_page_at(2);
        wide.promote_page_at(3);
    } catch (const std::invalid_argument&) {
        wide_promoted = false;
    }
    if (!wide_promoted) {
        ++failures;
        std::cerr << "Promoting physical pages above the logical capacity was rejected\n";
    }
    failures += expect_size(wide_pool.cached_pages(), 12, "wide promotion counted cache pages");
    wide.unbind_row();
    wide.release();

    // Regression: promotion moves already-mapped pages to cache ownership, so they must leave the
    // owning allocation's entitlement and the pool's entitled count. Counting a promoted page in
    // both `entitled` and `cached` made `page_group_count() - entitled - cached` underflow as
    // unsigned, which defeated `can_reserve` and let the engine over-admit requests the pool could
    // not then materialize (std::bad_alloc / "entitlement could not be materialized").
    {
        auto promote_plan = plan_paged_cache(8, 8, 1, {{ninfer::DType::BF16, 16, 1}});
        ninfer::DeviceArena promote_arena(promote_plan.bytes);
        ninfer::PagedKVPool promote_pool({promote_arena.base(), promote_arena.capacity()},
                                         promote_plan.layout);
        auto a = promote_pool.reserve(6);
        a.materialize_pages(6);
        for (std::uint32_t index = 0; index < 6; ++index) { a.promote_page_at(index); }
        failures +=
            expect_size(a.page_entitlement(), 0, "promotion did not release the owned entitlement");
        failures += expect_size(a.borrowed_page_count(), 6, "promotion did not move pages to borrowed");
        failures += expect_size(promote_pool.entitled_pages(), 0, "pool entitled kept promoted pages");
        failures += expect_size(promote_pool.cached_pages(), 6, "pool cached missed promoted pages");
        // 8 pages total, 6 now cache-owned: only 2 are reservable. The pre-fix underflow made the
        // 3-page reservation (and even a 7-page one) succeed.
        if (promote_pool.can_reserve(3)) {
            ++failures;
            std::cerr << "can_reserve ignored cache-owned pages after promotion\n";
        }
        if (!promote_pool.can_reserve(2)) {
            ++failures;
            std::cerr << "can_reserve rejected the genuinely free pages\n";
        }
        bool over_admitted = false;
        try {
            auto bad = promote_pool.reserve(3);
            (void)bad;
        } catch (const std::bad_alloc&) { over_admitted = true; }
        if (!over_admitted) {
            ++failures;
            std::cerr << "reserve over-admitted capacity held by cache-owned pages\n";
        }
        a.unbind_row();
        a.release();
    }

    // Regression: copy_page_d2h/h2d must place each page's head_extent runs at the offset the
    // pool's plane order dictates. A HeadMajor plane keeps its runs one page stride apart
    // (p*nb[2] + h*nb[3]); the copy previously assumed the PageMajor run index (h + HE*p), so a
    // HeadMajor page landed at half its true offset and clobbered neighbouring runs.
    const ninfer::PagedKVPlaneOrder orders[] = {ninfer::PagedKVPlaneOrder::PageMajor,
                                                ninfer::PagedKVPlaneOrder::HeadMajor};
    for (const ninfer::PagedKVPlaneOrder order : orders) {
        auto round_plan = plan_paged_cache(4, 2, 1, {{ninfer::DType::BF16, 8, 2}}, order);
        ninfer::DeviceArena round_arena(round_plan.bytes);
        ninfer::PagedKVPool pool({round_arena.base(), round_arena.capacity()}, round_plan.layout);
        const ninfer::Tensor& plane    = pool.plane(0);
        const std::size_t run_bytes    = 8 * ninfer::kPagedKVPageSize * 2u; // BF16 leading 8
        const std::size_t page_stride  = order == ninfer::PagedKVPlaneOrder::PageMajor
                                             ? static_cast<std::size_t>(plane.nb[3])
                                             : static_cast<std::size_t>(plane.nb[2]);
        const std::size_t head_stride  = order == ninfer::PagedKVPlaneOrder::PageMajor
                                             ? static_cast<std::size_t>(plane.nb[2])
                                             : static_cast<std::size_t>(plane.nb[3]);
        const std::int32_t head_count  = order == ninfer::PagedKVPlaneOrder::PageMajor
                                             ? plane.ne[2]
                                             : plane.ne[3];
        std::vector<unsigned char> source(pool.per_page_bytes());
        for (std::size_t i = 0; i < source.size(); ++i) {
            source[i] = static_cast<unsigned char>(i * 7u + 3u);
        }
        if (cudaMemsetAsync(plane.data, 0x5a, plane.bytes(), nullptr) != cudaSuccess) {
            ++failures;
            std::cerr << "page copy setup failed\n";
            continue;
        }
        const bool wrote = pool.copy_page_h2d(source.data(), 1, nullptr);
        std::vector<unsigned char> device(plane.bytes());
        const cudaError_t err =
            cudaMemcpy(device.data(), plane.data, device.size(), cudaMemcpyDeviceToHost);
        if (!wrote || err != cudaSuccess) {
            ++failures;
            std::cerr << "page copy failed for plane order " << static_cast<int>(order) << '\n';
            continue;
        }
        std::vector<unsigned char> expected(device.size(), 0x5a);
        for (std::int32_t head = 0; head < head_count; ++head) {
            const std::size_t begin = page_stride +
                                      head_stride * static_cast<std::size_t>(head);
            std::copy(source.begin() + static_cast<std::ptrdiff_t>(head) *
                                            static_cast<std::ptrdiff_t>(run_bytes),
                      source.begin() + static_cast<std::ptrdiff_t>(head + 1) *
                                           static_cast<std::ptrdiff_t>(run_bytes),
                      expected.begin() + static_cast<std::ptrdiff_t>(begin));
        }
        if (device != expected) {
            ++failures;
            std::cerr << "page copy placed runs at the wrong offset for plane order "
                      << static_cast<int>(order) << '\n';
        }
    }

    return failures == 0 ? 0 : fail("kv cache test failed");
}
