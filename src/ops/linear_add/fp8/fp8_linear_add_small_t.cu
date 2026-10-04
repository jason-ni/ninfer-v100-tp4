#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"
#include "ops/linear/fp8/fp8_small_t.cuh"
#include "ops/linear_add/fp8/fp8_linear_add_epilogue.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <class Geometry, int ActiveTokens>
struct Fp8LinearAddSmallTProductionSchedule;

// LinearAdd's residual read and fused epilogue shift the register/occupancy crossovers relative to
// contiguous Linear. These are the measured RTX 5090 ranges for [5120,6144].
template <int ActiveTokens>
struct Fp8LinearAddSmallTProductionSchedule<Fp8Residual6144Geometry, ActiveTokens> {
    static_assert(ActiveTokens >= kFp8FirstSmallT && ActiveTokens <= kFp8LastSmallT);
    static constexpr int kWarpsPerCta =
        (ActiveTokens >= 15 && ActiveTokens <= 19) || ActiveTokens == 22 ? 4 : 8;
    static constexpr int kRowsPerWarp   = ActiveTokens <= 5 ? 1 : 2;
    static constexpr int kValuesPerLane = ActiveTokens >= 20 && ActiveTokens <= 23 ? 8 : 16;
    static constexpr int kTokenTile     = ActiveTokens == 24 ? 12 : ActiveTokens;
    static constexpr auto kCodeCache =
        (ActiveTokens >= 12 && ActiveTokens <= 14) || ActiveTokens == 23 ? Fp8CodeCache::Streaming
                                                                         : Fp8CodeCache::Default;
    static constexpr auto kBlockOrder = ActiveTokens == 24
                                            ? Fp8SmallTBlockOrder::TokenTilesContiguous
                                            : Fp8SmallTBlockOrder::RowsContiguous;
    using Type =
        Fp8SmallTSchedule<kWarpsPerCta, kRowsPerWarp, kValuesPerLane, kTokenTile, 1,
                          Fp8SmallTActivationAccess::TokenPacked, kCodeCache, 1, kBlockOrder, 1>;
};

// The longer K of [5120,17408] favors streaming code loads around its two register crossovers;
// keeping the default v16 schedule at T=9..16 avoids an isolated fast point followed by a cliff.
template <int ActiveTokens>
struct Fp8LinearAddSmallTProductionSchedule<Fp8Residual17408Geometry, ActiveTokens> {
    static_assert(ActiveTokens >= kFp8FirstSmallT && ActiveTokens <= kFp8LastSmallT);
    static constexpr int kWarpsPerCta   = ActiveTokens >= 21 && ActiveTokens <= 22 ? 4 : 8;
    static constexpr int kRowsPerWarp   = ActiveTokens <= 5 ? 1 : 2;
    static constexpr int kValuesPerLane = ActiveTokens >= 20 && ActiveTokens <= 23 ? 8 : 16;
    static constexpr int kTokenTile     = ActiveTokens == 24 ? 12 : ActiveTokens;
    static constexpr auto kCodeCache    = (ActiveTokens >= 6 && ActiveTokens <= 8) ||
                                               (ActiveTokens >= 17 && ActiveTokens <= 20) ||
                                               ActiveTokens == 23
                                              ? Fp8CodeCache::Streaming
                                              : Fp8CodeCache::Default;
    static constexpr auto kBlockOrder   = ActiveTokens == 24
                                              ? Fp8SmallTBlockOrder::TokenTilesContiguous
                                              : Fp8SmallTBlockOrder::RowsContiguous;
    using Type =
        Fp8SmallTSchedule<kWarpsPerCta, kRowsPerWarp, kValuesPerLane, kTokenTile, 1,
                          Fp8SmallTActivationAccess::TokenPacked, kCodeCache, 1, kBlockOrder, 1>;
};

// linear_add's tp2 row shards inherit their parent's measured small-T schedule: tuning is
// inherited from the tp1 parent family, never re-measured for the shard.
template <int ActiveTokens>
struct Fp8LinearAddSmallTProductionSchedule<Fp8Residual6144Tp2RowGeometry, ActiveTokens>
    : Fp8LinearAddSmallTProductionSchedule<Fp8Residual6144Geometry, ActiveTokens> {};

template <int ActiveTokens>
struct Fp8LinearAddSmallTProductionSchedule<Fp8Residual17408Tp2RowGeometry, ActiveTokens>
    : Fp8LinearAddSmallTProductionSchedule<Fp8Residual17408Geometry, ActiveTokens> {};

// linear_add's tp4 row shards: the 6144 quarter (K=1536) inherits the parent's schedule; the
// 17408 quarter (K=4352=2^8*17) pins ValuesPerLane=8 because the parent's v16 phase of 512 does
// not divide K, matching the parent family's own Fp8LinearSmallTProductionSchedule rule.
template <int ActiveTokens>
struct Fp8LinearAddSmallTProductionSchedule<Fp8Residual6144Tp4RowGeometry, ActiveTokens>
    : Fp8LinearAddSmallTProductionSchedule<Fp8Residual6144Geometry, ActiveTokens> {};

template <int ActiveTokens>
struct Fp8LinearAddSmallTProductionSchedule<Fp8Residual17408Tp4RowGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kFp8FirstSmallT && ActiveTokens <= kFp8LastSmallT);
    using Parent                        = Fp8LinearAddSmallTProductionSchedule<Fp8Residual17408Geometry,
                                                       ActiveTokens>;
    static constexpr int kWarpsPerCta   = Parent::kWarpsPerCta;
    static constexpr int kRowsPerWarp   = Parent::kRowsPerWarp;
    static constexpr int kValuesPerLane = 8;
    static constexpr int kTokenTile     = Parent::kTokenTile;
    static constexpr auto kCodeCache    = Parent::kCodeCache;
    static constexpr auto kBlockOrder   = Parent::kBlockOrder;
    using Type =
        Fp8SmallTSchedule<kWarpsPerCta, kRowsPerWarp, kValuesPerLane, kTokenTile, 1,
                          Fp8SmallTActivationAccess::TokenPacked, kCodeCache, 1, kBlockOrder, 1>;
};

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    using Schedule = typename Fp8LinearAddSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;
    auto* output              = static_cast<__nv_bfloat16*>(residual.data);
    fp8_small_t_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales),
            Fp8ContiguousOutput{output, Geometry::kOutputRows},
            Fp8AddResidualEpilogue{output, Geometry::kOutputRows});
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, kFp8FirstSmallT + static_cast<int>(Offsets)>...};
}

template <class Geometry>
const auto& launchers() {
    static constexpr auto kLaunchers =
        make_launchers<Geometry>(std::make_index_sequence<kFp8LastSmallT - kFp8FirstSmallT + 1>{});
    return kLaunchers;
}

} // namespace

void fp8_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                   cudaStream_t stream) {
    if (x.ne[1] < kFp8FirstSmallT || x.ne[1] > kFp8LastSmallT) {
        throw std::invalid_argument("fp8 linear_add small-T: unsupported T");
    }
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kFp8FirstSmallT);
    switch (resolve_fp8_problem(weight.n, weight.k)) {
    case Fp8Problem::Residual6144:
        launchers<Fp8Residual6144Geometry>()[index](x, weight, residual, stream);
        return;
    case Fp8Problem::Residual17408:
        launchers<Fp8Residual17408Geometry>()[index](x, weight, residual, stream);
        return;
    case Fp8Problem::Residual6144Tp2Row:
        launchers<Fp8Residual6144Tp2RowGeometry>()[index](x, weight, residual, stream);
        return;
    case Fp8Problem::Residual17408Tp2Row:
        launchers<Fp8Residual17408Tp2RowGeometry>()[index](x, weight, residual, stream);
        return;
    case Fp8Problem::Residual6144Tp4Row:
        launchers<Fp8Residual6144Tp4RowGeometry>()[index](x, weight, residual, stream);
        return;
    case Fp8Problem::Residual17408Tp4Row:
        launchers<Fp8Residual17408Tp4RowGeometry>()[index](x, weight, residual, stream);
        return;
    case Fp8Problem::AttnInput:
    case Fp8Problem::GdnInput:
    case Fp8Problem::MlpGateUp:
    case Fp8Problem::Vocabulary:
    case Fp8Problem::VocabularyTp2Column:
    case Fp8Problem::GdnInputTp2Column:
    case Fp8Problem::MlpGateUpTp2Column:
    case Fp8Problem::AttnInputTp2Column:
    case Fp8Problem::VocabularyTp4Column:
    case Fp8Problem::GdnInputTp4Column:
    case Fp8Problem::MlpGateUpTp4Column:
    case Fp8Problem::AttnInputTp4Column:
        break;
    }
    throw std::invalid_argument("fp8 linear_add small-T: unsupported problem");
}

} // namespace ninfer::ops::detail
