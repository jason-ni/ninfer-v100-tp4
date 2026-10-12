#pragma once

// ninfer::ops - IQ4_NL-G32 KV cache codec (shared device helpers).
//
// Storage contract (docs/maintainer/iq4nl-kv-plan.md): each 32-value group along the head dim
// carries one FP16 scale plus 16 packed code bytes; a 4-bit code indexes the 16-entry asymmetric
// LUT kvalues_iq4nl (physical layout identical to GGUF Q4_0). Logical dequantization is
// FP32(FP16 scale) * LUT[code]; the encode rule is the deterministic reference pipeline
// (argmin-to-LUT codes, weighted-LSQ scale, -7..+7 adjacent-scale retry, FP16 RNE at the stored
// boundary) so a CPU oracle replicates it bit-for-bit.
//
// Quantization (append) and dequantization (staged reads) are FUSED into the consuming kernels
// exactly as for the INT8-G64 codec (gqa_attention_kv_quant.cuh): this header only provides the
// index math, the shared LUT, and the encode/dequant primitives they all reuse. The primitives
// read or write through caller-provided shared-memory LUT slices because the hot dequant path
// runs every thread of every tile; the LUT writer must complete before the first append write or
// staged read, i.e. before the same barrier that already exists in each consumer.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaI4NLGroup = 32;
inline constexpr int kGqaI4NLGroups  = kGqaKvQuantHeadDim / kGqaI4NLGroup; // 8 groups per row
inline constexpr int kGqaI4NLHeadCodes = kGqaKvQuantHeadDim / 2;           // code bytes per row

// kvalues_iq4nl from the sibling fork (ggml-common.h): 8 negative / 8 positive, positive-biased
// for attention activation statistics. Every consumer carries a CTA-local shared copy and seeds
// it through this accessor (a device function rather than a header-level constant array: the
// device pass of this toolchain does not resolve inline constexpr arrays pulled in late).
__device__ __forceinline__ std::int8_t gqa_i4nl_value_at(int i) {
    switch (i) {
    case 0: return (std::int8_t)-127;
    case 1: return (std::int8_t)-104;
    case 2: return (std::int8_t)-83;
    case 3: return (std::int8_t)-65;
    case 4: return (std::int8_t)-49;
    case 5: return (std::int8_t)-35;
    case 6: return (std::int8_t)-22;
    case 7: return (std::int8_t)-10;
    case 8: return 1;
    case 9: return 13;
    case 10: return 25;
    case 11: return 38;
    case 12: return 53;
    case 13: return 69;
    case 14: return 89;
    default: return 113;
    }
}

template <int GeometryKVHeads>
__device__ __forceinline__ std::int64_t gqa_i4nl_code_index(int physical_page, int kv_head, int d,
                                                            int page_offset) {
    return paged_kv_element_offset<kGqaI4NLHeadCodes, GeometryKVHeads>(physical_page, kv_head,
                                                                       page_offset, d >> 1);
}

template <int GeometryKVHeads>
__device__ __forceinline__ std::int64_t gqa_i4nl_scale_index(int physical_page, int kv_head,
                                                             int group, int page_offset) {
    return paged_kv_element_offset<kGqaI4NLGroups, GeometryKVHeads>(physical_page, kv_head,
                                                                    page_offset, group);
}

// Bit-exact replica of the reference best_index_int8(16, kvalues_iq4nl, x): clamp to the table
// bounds, binary search with the midpoint taken low, then take the lower index on an exact
// distance tie. No floating point other than the two distance terms.
__device__ __forceinline__ int gqa_i4nl_best_code(float x, const std::int8_t* lut) {
    if (x <= static_cast<float>(lut[0])) { return 0; }
    if (x >= static_cast<float>(lut[15])) { return 15; }
    int lo = 0, hi = 15;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (x < static_cast<float>(lut[mid])) { hi = mid; } else { lo = mid; }
    }
    const float dl = x - static_cast<float>(lut[hi - 1]);
    const float dh = static_cast<float>(lut[hi]) - x;
    return (dl < dh) ? hi - 1 : hi;
}

// Warp-wide encode of one 32-group: one source value per lane (lane == value index), x already
// converted from BF16. On completion lanes [0,16) hold the packed code byte (byte b carries
// code(2b) in the low nibble and code(2b+1) in the high nibble, matching gqa_i4nl_code_index and
// gqa_i4nl_dequant_x8) and every lane holds the FP16 scale.
//
// The pipeline mirrors the reference quantize_row_iq4_nl(single-block) case in exact arithmetic
// order: the sequential strict-greater absmax scan's winner (FIRST index, on |value| ties) is
// reproduced by a butterfly reduce that carries the winning index and keeps the smaller index on
// ties; the objective is the weighted (x^2) LSQ fit d = sum(q*w*x)/sum(q^2*w); the -7..+7
// integer-offset scale retry accepts a candidate only when sum_qx^2 strictly beats the running
// best measured against that candidate's own sum_q2; and the final codes are re-derived from 1/d
// (the pre-rounding optimal scale), never from the retried offsets. The stored scale rounds d to
// FP16 RNE at the storage boundary.
__device__ __forceinline__ void gqa_i4nl_encode_warp(float x, unsigned full_mask, const std::int8_t* lut,
                                     std::uint32_t& code_byte, half& scale_out) {
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    float amax       = fabsf(x);
    float ext        = x;
    int   index      = lane; // original lane of the current absmax winner (first-index tie-break)
    #pragma unroll
    for (unsigned m = 16; m > 0; m >>= 1) {
        const float o_amax  = __shfl_xor_sync(full_mask, amax, m);
        const float o_ext   = __shfl_xor_sync(full_mask, ext, m);
        const int   o_index = __shfl_xor_sync(full_mask, index, m);
        if (o_amax > amax || (o_amax == amax && o_index < index)) {
            amax  = o_amax;
            ext   = o_ext;
            index = o_index;
        }
    }
    if (amax < 1e-15f) {
        code_byte = 0u;
        scale_out = __float2half(0.0f);
        return;
    }

float best_q    = 0.0f;
    float best_q2   = 0.0f;
    float d         = 0.0f;
    {
        // Initial pass: id = 127/ext matches the reference -ext/values[0] at ntry > 0. The
        // division and its product with x stay uncontracted by the explicit intermediates.
        const float id     = 127.0f / ext;
        const float al     = id * x;
        const int l        = gqa_i4nl_best_code(al, lut);
        const float q      = static_cast<float>(lut[l]);
        const float w      = x * x;
        const float sum_qx = warp_sum(w * q * x, full_mask);
        const float sum_q2 = warp_sum(w * q * q, full_mask);
        d                  = (sum_q2 > 0.0f) ? sum_qx / sum_q2 : 0.0f;
        best_q             = d * sum_qx;
        best_q2            = sum_q2;
    }
    #pragma unroll
    for (int itry = -7; itry <= 7; ++itry) {
        // Reference order: id = (itry + values[0])/max, then al = id*xb[j].
        const float id     = (static_cast<float>(itry) - 127.0f) / ext;
        const float al     = id * x;
        const int l        = gqa_i4nl_best_code(al, lut);
        const float q      = static_cast<float>(lut[l]);
        const float w      = x * x;
        const float sum_qx = warp_sum(w * q * x, full_mask);
        const float sum_q2 = warp_sum(w * q * q, full_mask);
        if (sum_q2 > 0.0f && sum_qx * sum_qx > best_q * best_q2) {
            d        = sum_qx / sum_q2;
            best_q   = d * sum_qx;
            best_q2  = sum_q2;
        }
    }
    const float id_final = (d != 0.0f) ? 1.0f / d : 0.0f;
    const float al_final = id_final * x;
    const int my_code    = gqa_i4nl_best_code(al_final, lut);
    // Adjacent-pair packing so byte b pairs values 2b and 2b+1 (the gqa_i4nl_code_index /
    // gqa_i4nl_dequant_x8 layout): lane b gathers the codes of lanes 2b and 2b+1.
    const int c_lo = __shfl_sync(full_mask, my_code, (2 * lane) & 31);
    const int c_hi = __shfl_sync(full_mask, my_code, (2 * lane + 1) & 31);
    code_byte = (lane < 16)
                    ? (static_cast<std::uint32_t>(c_lo) |
                       (static_cast<std::uint32_t>(c_hi) << 4))
                    : 0u;
    scale_out = __float2half_rn(d);
}

// Dequantize eight contiguous head-dim values (four code bytes / eight nibbles, d % 32 == 0 so
// the eight values share one 32-group scale) to eight FP16 values packed as int4, in the same
// rounding order as the INT8-G64 staged dequant: exact fp16 expansion of the LUT entry and one
// RNE fp16 multiply by the exact stored scale (the byte_perm 1024-bias construction is the
// quarter-rate I2F workaround, so it operates on the expanded LUT entries, not the nibbles).
__device__ __forceinline__ int4 gqa_i4nl_dequant_x8(std::uint32_t nibbles4 /*4 bytes, little-endian d0..d7*/,
                                                    half sc, const std::int8_t* lut) {
    const unsigned r = nibbles4;
    std::int8_t c[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) { c[i] = lut[(r >> (4 * i)) & 0xF]; }
    // Pack the expanded LUT bytes as unsigned (a sign-extended int8 would poison the higher
    // bytes of the OR), matching the raw little-endian byte order the INT8 codec feeds here.
    const auto byte = [](std::int8_t v) {
        return static_cast<unsigned>(static_cast<std::uint8_t>(v));
    };
    const int2 packed{static_cast<int>(byte(c[0]) | (byte(c[1]) << 8) |
                                        (byte(c[2]) << 16) | (byte(c[3]) << 24)),
                      static_cast<int>(byte(c[4]) | (byte(c[5]) << 8) |
                                        (byte(c[6]) << 16) | (byte(c[7]) << 24))};
    const __half2 scale2 = __half2half2(sc);
    const __half2 bias2  = __float2half2_rn(1152.0f);
    const unsigned lo    = static_cast<unsigned>(packed.x) ^ 0x80808080u;
    const unsigned hi    = static_cast<unsigned>(packed.y) ^ 0x80808080u;
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
}

// Staging counterpart for kernels that read the group scale in FP32 once per 8-wide unit (the
// phase-1 tensor-core path): one RNE fp16 rounding of each exact F32 product, matching the
// INT8-G64 staged rule of gqa_kv_dequant_i8x8_f16_from. The exact-product argument holds since
// the LUT entry times the FP32 scale is representable without additional rounding before the
// single final conversion in either formulation.
__device__ __forceinline__ int4 gqa_i4nl_staged_f16x8(std::uint32_t nibbles4, float s,
                                                      const std::int8_t* lut) {
    const unsigned r = nibbles4;
    __half2 out[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(lut[(r >> (8 * i)) & 0xF]) * s;
        const float x1 = static_cast<float>(lut[(r >> (8 * i + 4)) & 0xF]) * s;
        out[i]         = __floats2half2_rn(x0, x1);
    }
    return *reinterpret_cast<const int4*>(out);
}

} // namespace ninfer::ops