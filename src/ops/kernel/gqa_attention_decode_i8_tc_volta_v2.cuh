#pragma once

// ninfer::ops - sm_70 split-KV causal small-T attention over the INT8 group-64 KV cache,
// key-split variant (8 warps, Bc = 64 keys per shared-memory tile).
//
// Same contract as gqa_attention_small_t_tc_volta_partial_i8_kernel (same append prologue, same
// per-split partial_acc / partial_m / partial_l outputs, so the reduce kernel is shared),
// restricted to row_count = tokens * GroupSize <= 32 (one Volta row tile, no second pass).
//
// Why a second kernel: the phase-1 kernel stages 16 keys per tile and has all four warps consume
// one eight-key QK^T subgroup at a time, with the same FP32 score tile shared through shared
// memory and two block barriers per 16 keys. Long-context profiling of that topology on the
// sibling fork (186K keys, T=4) showed it at ~140 GB/s: removing either the QK^T MMAs or the
// global loads alone only cut ~25-30%, i.e. the cost is the per-tile pipeline (redundant QK^T for
// the dimension-split warps, two barriers per 16 keys, small loads), not one resource.
//
// This variant: 8 warps, Bc = 64 keys per shared-memory tile. Warp w computes QK^T only for its
// own 8 keys [w*8, w*8+8) of the tile (no redundancy), the 8 warps exchange per-row maxima through
// shared memory, write their softmax probabilities P into a shared 32x64 tile, and then each warp
// runs PV over all 64 keys for its own D/8 = 32-column slice (a quarter of the phase-1 kernel's
// register accumulator per warp). Three block barriers per 64 keys instead of eight. The int8
// codes + scales of the next tile are prefetched into registers while the current tile is computed
// (Volta has no cp.async, so this is the only way to overlap HBM latency with the tensor-core work
// of a single resident CTA), and int8 -> fp16 dequantization uses the byte_perm 1024-bias trick
// instead of I2C/I2F, which issues at quarter rate on Volta.
//
// Differences from the sibling fork's version this was ported from: the INT8 cache here stores the
// raw values (no normalized-Hadamard rotation), so no rotation is applied to Q or K, and the
// partials stay BF16 like the phase-1 kernel's so the existing reduce kernel consumes them
// unchanged.

#include "ops/common/volta_mma.cuh"
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

// P x V with fp32 accumulation straight into the output accumulator (D layout). Replaces an fp16
// mma per 8-key group followed by 8 cvt + 8 FADD per thread; the conversion/adds were ~10% of the
// kernel at 186K on the sibling fork, and fp32 accumulation is also the more accurate of the two.
__device__ __forceinline__ void gqa_i8_v2_mma_pv_f32(float (&d)[8], const half2 (&p)[4],
                                                     const half2 (&v)[4]) {
    const int* Pxi = reinterpret_cast<const int*>(p);
    const int* Vxi = reinterpret_cast<const int*>(v);
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
                   "+f"(d[6]), "+f"(d[7])
                 : "r"(Pxi[0]), "r"(Pxi[1]), "r"(Vxi[0]), "r"(Vxi[1]));
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
                   "+f"(d[6]), "+f"(d[7])
                 : "r"(Pxi[2]), "r"(Pxi[3]), "r"(Vxi[2]), "r"(Vxi[3]));
}

inline constexpr int kGqaSmallTI8VoltaV2Warps   = 8;
inline constexpr int kGqaSmallTI8VoltaV2Bc      = 64;
inline constexpr int kGqaSmallTI8VoltaV2Stride  = kGqaHeadDim + 8;
inline constexpr int kGqaSmallTI8VoltaV2PStride = kGqaSmallTI8VoltaV2Bc + 8;

// Physical page ids one split stages, derived from the declared visible-key domain exactly like
// the phase-1 kernels' bound (see kGqaSmallTSplitPageIds): one 64-key tile is one page, and a
// split stages ceil(ceil(window/64)/active_splits)*64 keys plus the trailing tile rounding. The
// fixed 64 the sibling fork hard-codes is not enough for this repo's 560-split Volta policy at
// the 1,048,576-key domain (the 85-split phase-1 comment already records 194 ids), so this
// reuses the computed bound.
template <typename Geometry>
inline constexpr int kGqaSmallTI8VoltaV2Pages =
    kGqaSmallTSplitPageIds<Geometry, kGqaSmallTI8VoltaV2Bc>;

template <typename Geometry>
inline constexpr std::size_t kGqaSmallTI8VoltaV2SmemBytes =
    sizeof(half) * (32 * kGqaSmallTI8VoltaV2Stride                            // q_s
                    + 2 * kGqaSmallTI8VoltaV2Bc * kGqaSmallTI8VoltaV2Stride   // k_s, v_s
                    + 32 * kGqaSmallTI8VoltaV2PStride)                        // p_s
    + sizeof(float) * kGqaSmallTI8VoltaV2Warps * 32                               // red_s
    + sizeof(std::int32_t) * kGqaSmallTI8VoltaV2Pages<Geometry>;                  // pages_s

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput>
__launch_bounds__(kGqaSmallTI8VoltaV2Warps * 32, 1) __global__
    void gqa_attention_small_t_tc_volta_partial_i8_v2_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, std::int8_t* cache_k_i8,
    std::int8_t* cache_v_i8, __half* cache_k_scale, __half* cache_v_scale,
    const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity, float scale,
    __nv_bfloat16* partial_acc, float* partial_m, float* partial_l) {
    static_assert(TokenTile * Geometry::GroupSize <= 32,
                  "the key-split variant covers one 32-row Volta tile; wider token tiles stay on "
                  "the phase-1 kernel");
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
    constexpr int Warps        = kGqaSmallTI8VoltaV2Warps;
    constexpr int Threads      = Warps * 32;
    constexpr int Br           = 32;
    constexpr int Bc           = kGqaSmallTI8VoltaV2Bc;
    constexpr int D            = kGqaHeadDim;
    constexpr int DChunks      = D / 8;
    constexpr int DSlice       = D / Warps;     // 32 PV output columns per warp
    constexpr int DChunksLocal = DSlice / 8;    // 4
    constexpr int KeyGroups    = Bc / 8;        // 8, one per warp for QK^T
    constexpr int Stride       = kGqaSmallTI8VoltaV2Stride;
    constexpr int PStride      = kGqaSmallTI8VoltaV2PStride;
    constexpr int Groups       = D / kGqaKvQuantGroup;
    constexpr float Log2E      = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    static_assert(KeyGroups == Warps, "one 8-key QK^T group per warp");
    static_assert(Bc == kPagedKVPageSize && Threads == 256 && D / 8 == 32,
                  "fast tile load assumes one page per tile and key = warp + 8i");

    extern __shared__ __align__(16) unsigned char smem_raw[];
    half* q_s  = reinterpret_cast<half*>(smem_raw);
    half* k_s  = q_s + Br * Stride;
    half* v_s  = k_s + Bc * Stride;
    half* p_s  = v_s + Bc * Stride;
    float* red_s = reinterpret_cast<float*>(p_s + Br * PStride);
    std::int32_t* physical_pages_s = reinterpret_cast<std::int32_t*>(red_s + Warps * 32);

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    int valid_tokens      = tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
    }

    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * kGqaHeadDim * Geometry::QHeads * tokens *
                       split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }

    const int row_count = tokens * Geometry::GroupSize;

    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                    -CUDART_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = 0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] =
                    __float2bfloat16(0.0f);
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || tokens < 1 || tokens > TokenTile ||
        split_count <= 0) {
        return;
    }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[tokens - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int window = last_pos + 1;
    // The device-side split policy is shared with the phase-1 kernel (it depends only on the
    // window, grid, and token tile), so both kernels agree on which splits are active.
    const int active_split_count =
        gqa_small_t_active_splits<Geometry, true>(window, split_count, TokenTile);
    if (split >= active_split_count) { return; }

    const int span          = window;
    const int logical_tiles = div_up(span, Bc);
    const bool tile_split   = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(span, active_split_count);
    const int split_start = split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);
    const int first_page = first_tile >> kPagedKVPageShift;
    const int page_count = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int page = tid; page < page_count; page += Threads) {
        physical_pages_s[page] = block_table[first_page + page];
    }

    if constexpr (CacheInput::writes_cache) {
        // Quantize this tile's new rows into the cache before any of it is staged back out.
        // One warp owns one (token, 64-group) pair so that the absmax reduction that sets the
        // group's scale is exactly a full-warp reduction: 32 lanes x 2 elements = 64 = one group.
        // This mirrors the phase-1 kernel's append block (and the SIMT kernel's); the codes have
        // to come out bit-identical, since the kernels share a cache.
        for (int pair = warp; pair < valid_tokens * Groups; pair += Warps) {
            const int token = pair / Groups;
            const int grp   = pair - token * Groups;
            const int position = pos[token];
            if (position < split_start || position >= split_end || position < 0 ||
                position >= logical_capacity) {
                continue;
            }
            int physical_page    = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
            physical_page        = __shfl_sync(FullMask, physical_page, 0);
            const int page_offset = position & kPagedKVPageMask;
            const int d0          = grp * kGqaKvQuantGroup + lane;
            const int d1          = d0 + 32;
            const float kv0 = __bfloat162float(input.k[gqa_kv_new_index<Geometry>(kv_head, d0, token)]);
            const float kv1 = __bfloat162float(input.k[gqa_kv_new_index<Geometry>(kv_head, d1, token)]);
            const float vv0 = __bfloat162float(input.v[gqa_kv_new_index<Geometry>(kv_head, d0, token)]);
            const float vv1 = __bfloat162float(input.v[gqa_kv_new_index<Geometry>(kv_head, d1, token)]);
            float kamax = fmaxf(fabsf(kv0), fabsf(kv1));
            float vamax = fmaxf(fabsf(vv0), fabsf(vv1));
            kamax       = warp_max(kamax, FullMask);
            vamax       = warp_max(vamax, FullMask);
            const __half ksh = __float2half_rn(kamax > 0.0f ? kamax / 127.0f : 0.0f);
            const __half vsh = __float2half_rn(vamax > 0.0f ? vamax / 127.0f : 0.0f);
            const float k_inv = __half2float(ksh) > 0.0f ? 1.0f / __half2float(ksh) : 0.0f;
            const float v_inv = __half2float(vsh) > 0.0f ? 1.0f / __half2float(vsh) : 0.0f;
            cache_k_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d0, page_offset)] =
                gqa_kv_quant_code(kv0, k_inv);
            cache_k_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d1, page_offset)] =
                gqa_kv_quant_code(kv1, k_inv);
            cache_v_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d0, page_offset)] =
                gqa_kv_quant_code(vv0, v_inv);
            cache_v_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d1, page_offset)] =
                gqa_kv_quant_code(vv1, v_inv);
            if (lane == 0) {
                const std::int64_t so =
                    gqa_kv_quant_scale_index<Geometry>(physical_page, kv_head, grp, page_offset);
                cache_k_scale[so] = ksh;
                cache_v_scale[so] = vsh;
            }
        }
    }

    // Q tile: 32 rows (rows >= row_count are zero). The INT8 cache stores raw values here (no
    // normalized-Hadamard rotation like the sibling fork), so Q is only widened and rounded.
    for (int row = warp; row < Br; row += Warps) {
        if (row < row_count) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
#pragma unroll
                for (int part = 0; part < 8; ++part) {
                    const int d = lane + 32 * part;
                    q_s[row * Stride + d] =
                        __float2half(__bfloat162float(q[gqa_q_index<Geometry>(q_head, d, token)]));
                }
                continue;
            }
        }
#pragma unroll
        for (int part = 0; part < 8; ++part) { q_s[row * Stride + lane + 32 * part] = __float2half(0.0f); }
    }
    __syncthreads(); // Q tile, page table, and appended cache rows are visible

    // Per-thread row bookkeeping (identical in every warp: same mma layout).
    const int r_lo = volta_d_get_i(0) & ~2;
    const int r_hi = volta_d_get_i(0) | 2;
    int q_head_lo = 0, tok_lo = 0, q_head_hi = 0, tok_hi = 0;
    gqa_small_t_tc_row_to_qt<Geometry>(r_lo, tokens, kv_head, q_head_lo, tok_lo);
    gqa_small_t_tc_row_to_qt<Geometry>(r_hi, tokens, kv_head, q_head_hi, tok_hi);
    const int qabs_lo = (r_lo < row_count) ? pos[tok_lo] : -1;
    const int qabs_hi = (r_hi < row_count) ? pos[tok_hi] : -1;

    float acc_f[DChunksLocal][8];
#pragma unroll
    for (int c = 0; c < DChunksLocal; ++c) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { acc_f[c][i] = 0.0f; }
    }
    float m_lo = -CUDART_INF_F, m_hi = -CUDART_INF_F;
    float l_lo = 0.0f, l_hi = 0.0f; // this warp's share of the row sums (same running max)

    // Register double-buffering: the int8 codes + scales of tile kb+1 are loaded into registers
    // while tile kb is being computed, then dequantized into shared memory at the top of the next
    // iteration. Volta has no cp.async, so this is the only way to overlap HBM latency with the
    // tensor-core work of a single resident CTA.
    constexpr int ChunksPerThread = Bc * (D / 8) / Threads; // 8
    static_assert(Bc * (D / 8) % Threads == 0);
    int2 k_codes[ChunksPerThread], v_codes[ChunksPerThread];
    half k_scales[ChunksPerThread], v_scales[ChunksPerThread];
    auto load_tile = [&](int kb) {
        const int k0   = first_tile + kb * Bc;
        const int page = physical_pages_s[(k0 >> kPagedKVPageShift) - first_page];
        // Interior tile: a 64-key tile is exactly one page, and thread tid owns key (warp + 8i)
        // and columns lane*8..+7, so every address is the page base plus a constant.
        if (k0 >= split_start && k0 + Bc <= split_end) {
            const std::int64_t code_base =
                paged_kv_page_head_offset<kGqaHeadDim, Geometry::KVHeads>(page, kv_head) +
                warp * kGqaHeadDim + lane * 8;
            const std::int64_t scale_base =
                paged_kv_page_head_offset<kGqaKvQuantGroups, Geometry::KVHeads>(page, kv_head) +
                warp * kGqaKvQuantGroups + (lane * 8) / kGqaKvQuantGroup;
#pragma unroll
            for (int i = 0; i < ChunksPerThread; ++i) {
                k_codes[i]  = load_vec<int2>(cache_k_i8 + code_base + i * 8 * kGqaHeadDim);
                v_codes[i]  = load_vec<int2>(cache_v_i8 + code_base + i * 8 * kGqaHeadDim);
                k_scales[i] = cache_k_scale[scale_base + i * 8 * kGqaKvQuantGroups];
                v_scales[i] = cache_v_scale[scale_base + i * 8 * kGqaKvQuantGroups];
            }
            return;
        }
#pragma unroll
        for (int i = 0; i < ChunksPerThread; ++i) {
            const int chunk = tid + i * Threads;
            const int key_l = chunk / (D / 8);
            const int d     = (chunk - key_l * (D / 8)) * 8;
            const int key   = k0 + key_l;
            if (key >= split_start && key < split_end) {
                const int page_offset = key & kPagedKVPageMask;
                const std::int64_t code_off =
                    gqa_kv_quant_code_index<Geometry>(page, kv_head, d, page_offset);
                const std::int64_t scale_off = gqa_kv_quant_scale_index<Geometry>(
                    page, kv_head, d / kGqaKvQuantGroup, page_offset);
                k_codes[i]  = load_vec<int2>(&cache_k_i8[code_off]);
                v_codes[i]  = load_vec<int2>(&cache_v_i8[code_off]);
                k_scales[i] = cache_k_scale[scale_off];
                v_scales[i] = cache_v_scale[scale_off];
            } else {
                k_codes[i]  = make_int2(0, 0);
                v_codes[i]  = make_int2(0, 0);
                k_scales[i] = __float2half(0.0f);
                v_scales[i] = __float2half(0.0f);
            }
        }
    };
    // int8 -> fp16 without I2F: Volta's integer->float conversions issue at quarter rate. Bias
    // each signed byte to unsigned (xor 0x80), splice it into the mantissa of fp16 1024.0
    // (0x64xx = 1024 + byte), subtract 1152 (exact), then multiply by the fp16 scale. fp16*fp16
    // is exact in fp32, so this rounds identically to the reference float(c) * scale -> half path.
    auto dequant8 = [](int2 raw, half sc) {
        const __half2 scale2 = __half2half2(sc);
        const __half2 bias2  = __float2half2_rn(1152.0f);
        const unsigned lo = static_cast<unsigned>(raw.x) ^ 0x80808080u;
        const unsigned hi = static_cast<unsigned>(raw.y) ^ 0x80808080u;
        unsigned h[4];
        h[0] = __byte_perm(lo, 0x64646464u, 0x5150);
        h[1] = __byte_perm(lo, 0x64646464u, 0x5352);
        h[2] = __byte_perm(hi, 0x64646464u, 0x5150);
        h[3] = __byte_perm(hi, 0x64646464u, 0x5352);
        __half2 out[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            out[j] = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&h[j]), bias2), scale2);
        }
        return *reinterpret_cast<const int4*>(out);
    };

    load_tile(0);
    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;

        // ---- Dequantize the prefetched tile into shared memory, then prefetch the next one. ----
#pragma unroll
        for (int i = 0; i < ChunksPerThread; ++i) {
            const int chunk = tid + i * Threads;
            const int key_l = chunk / (D / 8);
            const int d     = (chunk - key_l * (D / 8)) * 8;
            store_vec(&k_s[key_l * Stride + d], dequant8(k_codes[i], k_scales[i]));
            store_vec(&v_s[key_l * Stride + d], dequant8(v_codes[i], v_scales[i]));
        }
        if (kb + 1 < key_blocks) { load_tile(kb + 1); }
        __syncthreads(); // (1) K/V tile staged

        // ---- QK^T for this warp's 8 keys, full D contraction. ----
        const int sub_k0 = k0 + warp * 8;
        float d_score[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int c = 0; c < DChunks; ++c) {
            half2 qf[4];
            volta_load_qp(qf, reinterpret_cast<const half2*>(&q_s[c * 8]), Stride / 2);
            half2 kf[4];
            volta_load_k(kf, reinterpret_cast<const half2*>(&k_s[warp * 8 * Stride + c * 8]),
                         Stride / 2);
            volta_mma_qk(d_score, qf, kf);
        }
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int row  = volta_d_get_i(l);
            const int key  = sub_k0 + volta_d_get_j(l);
            const int qabs = ((l & 2) == 0) ? qabs_lo : qabs_hi;
            const bool ok  = row < row_count && key >= split_start && key < split_end && key <= qabs;
            d_score[l]     = ok ? d_score[l] * scale : -CUDART_INF_F;
        }
        float bm_lo = fmaxf(fmaxf(d_score[0], d_score[1]), fmaxf(d_score[4], d_score[5]));
        float bm_hi = fmaxf(fmaxf(d_score[2], d_score[3]), fmaxf(d_score[6], d_score[7]));
        bm_lo       = fmaxf(bm_lo, __shfl_xor_sync(FullMask, bm_lo, 2, 32));
        bm_hi       = fmaxf(bm_hi, __shfl_xor_sync(FullMask, bm_hi, 2, 32));
        // lanes with (lane & 2) == 0 own r_lo, the others own r_hi; each row gets written by
        // several lanes with the identical value.
        red_s[warp * 32 + (((lane & 2) == 0) ? r_lo : r_hi)] = ((lane & 2) == 0) ? bm_lo : bm_hi;
        __syncthreads(); // (2) per-warp row maxima visible

        float tm_lo = -CUDART_INF_F, tm_hi = -CUDART_INF_F;
#pragma unroll
        for (int w = 0; w < Warps; ++w) {
            tm_lo = fmaxf(tm_lo, red_s[w * 32 + r_lo]);
            tm_hi = fmaxf(tm_hi, red_s[w * 32 + r_hi]);
        }
        const float new_m_lo = fmaxf(m_lo, tm_lo);
        const float new_m_hi = fmaxf(m_hi, tm_hi);
        const float alpha_lo =
            (m_lo == -CUDART_INF_F) ? 0.0f : exp2_approx((m_lo - new_m_lo) * Log2E);
        const float alpha_hi =
            (m_hi == -CUDART_INF_F) ? 0.0f : exp2_approx((m_hi - new_m_hi) * Log2E);
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const float new_m = ((l & 2) == 0) ? new_m_lo : new_m_hi;
            d_score[l] = (new_m > -CUDART_INF_F && d_score[l] > -CUDART_INF_F)
                             ? exp2_approx((d_score[l] - new_m) * Log2E)
                             : 0.0f;
        }
        float bl_lo = d_score[0] + d_score[1] + d_score[4] + d_score[5];
        float bl_hi = d_score[2] + d_score[3] + d_score[6] + d_score[7];
        bl_lo       = bl_lo + __shfl_xor_sync(FullMask, bl_lo, 2, 32);
        bl_hi       = bl_hi + __shfl_xor_sync(FullMask, bl_hi, 2, 32);
        l_lo        = l_lo * alpha_lo + bl_lo;
        l_hi        = l_hi * alpha_hi + bl_hi;
        m_lo        = new_m_lo;
        m_hi        = new_m_hi;

#pragma unroll
        for (int l = 0; l < 8; ++l) {
            p_s[volta_d_get_i(l) * PStride + warp * 8 + volta_d_get_j(l)] = __float2half(d_score[l]);
        }
        __syncthreads(); // (3) full 32x64 P tile visible; red_s free again

        // ---- PV: this warp's 32 output columns over all 64 keys. ----
        // acc_f is in the mma D layout: element i belongs to row r_lo when (i & 2) == 0.
#pragma unroll
        for (int c = 0; c < DChunksLocal; ++c) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { acc_f[c][i] *= ((i & 2) == 0) ? alpha_lo : alpha_hi; }
        }
#pragma unroll
        for (int g = 0; g < KeyGroups; ++g) {
            half2 p[4];
            *reinterpret_cast<int4*>(p) =
                *reinterpret_cast<const int4*>(&p_s[lane * PStride + g * 8]);
#pragma unroll
            for (int c = 0; c < DChunksLocal; ++c) {
                half2 vf[4];
                volta_load_v(vf,
                             reinterpret_cast<const half2*>(
                                 &v_s[g * 8 * Stride + warp * DSlice + c * 8]),
                             Stride / 2);
                gqa_i8_v2_mma_pv_f32(acc_f[c], p, vf);
            }
        }
        __syncthreads(); // (4) k_s/v_s/p_s may be overwritten by the next tile
    }

    // ---- Combine the per-warp row sums, then write partials. ----
    red_s[warp * 32 + (((lane & 2) == 0) ? r_lo : r_hi)] = ((lane & 2) == 0) ? l_lo : l_hi;
    __syncthreads();
    const int row      = lane;
    const float own_m  = ((lane & 2) == 0) ? m_lo : m_hi;
    if (row < row_count) {
        int q_head = 0;
        int token  = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            if (warp == 0) {
                float own_l = 0.0f;
#pragma unroll
                for (int w = 0; w < Warps; ++w) { own_l += red_s[w * 32 + row]; }
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = own_m;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = own_l;
            }
        }
    }
    // acc_f is in the mma D layout (row volta_d_get_i(i), column volta_d_get_j(i) of the chunk).
#pragma unroll
    for (int c = 0; c < DChunksLocal; ++c) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = volta_d_get_i(i);
            if (r >= row_count) { continue; }
            int qh = 0;
            int tk = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(r, tokens, kv_head, qh, tk);
            if (!gqa_valid_q_head<Geometry>(kv_head, qh)) { continue; }
            const int d = warp * DSlice + c * 8 + volta_d_get_j(i);
            partial_acc[gqa_partial_acc_index<Geometry>(qh, d, tk, split, tokens)] =
                __float2bfloat16(acc_f[c][i]);
        }
    }
#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
}

} // namespace ninfer::ops
