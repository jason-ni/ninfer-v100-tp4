# Plan — IQ4_NL KV cache storage (from the llama.cpp sibling fork)

Status: **active design/feasibility record, 2026-10-12.** P1-P3 delivered and passing (IQ4_NL-G32
codec, decode route, VoltaFlash prefill route, A1/A2/A3 conformance incl. TP2 head-local). P4
(end-to-end quality/performance qualification + remaining docs) pending. The P1-P4 close-outs below
remain the exit criteria.

## Objective

Adopt the sibling fork's (`../llama`, branch `port/sm70-d256`) IQ4_NL KV-cache type into NInfer as
a third `--kv-dtype` option for the qwen3.6/qwen3.8 family text route, so the V100X2 target
(180k capacity, TP2/TP4) can cut growing-KV occupancy ~45% vs INT8-G64 with no further end-to-end
quality loss and committed decode throughput at or above the ~45 tok/s @ ~85k occupied-context
acceptance line. Weights, artifact identities, and the Q4_K/Q6_K packed bytes are untouched: KV
is runtime state, not checkpoint data.

## Codec definition (authoritative)

The codec is defined by the fork and adopted verbatim as NInfer's **IQ4_NL-G32** storage contract:

- Block: 32 values, one FP16 scale + 16 bytes of nibbles (physical layout identical to Q4_0,
  18 B/block), with **adjacent-pair packing**: byte b carries `code(2b)` in the low nibble and
  `code(2b+1)` in the high nibble, matching `gqa_i4nl_code_index` / `gqa_i4nl_dequant_x8`.
  Head dim 256 → 8 blocks/row = 144 B per K or V row vs INT8-G64's 264 B:
  **−45.5% per plane, −45.5% total**.
- LUT: the 16-entry asymmetric int8 table `kvalues_iq4nl =
  {-127,-104,-83,-65,-49,-35,-22,-10, 1,13,25,38,53,69,89,113}` (fork `ggml/src/ggml-common.h`).
- Decode (oracle boundary): `value[i] = FP32(FP16 scale) * LUT[nibble(i)]`.
- Encode: per 32-block — argmin-to-LUT codes, weighted (`x^2`) LSQ scale, ±7 adjacent-scale retry
  accepted on a strict objective improvement, FP16 RNE applied once at the stored scale boundary.
  The deterministic rule the Op registers is NInfer's own pipeline: its warp cross-checks the LSQ
  sums with a butterfly (xor-tree) order and picks the *first* index on an `abs`max tie, so the CPU
  oracle in `tests/ops/gqa_attention_fixture.h` replicates **that** order bit-for-bit. The sibling
  fork's sequential CPU accumulation is provenance and a closeness check, **not** the gate.

Source references (fork `d8742f743` + working tree):

| Item | Fork location |
|---|---|
| LUT / struct layout | `ggml/src/ggml-common.h`, `block_iq4_nl` (d + qs[16]) |
| CPU ref encode/decode | `ggml/src/ggml-quants.c` (`quantize_row_iq4_nl_ref`, `dequantize_row_iq4_nl`) |
| CUDA vec-KQ expand+dp4a | `ggml-src/ggml-cuda/fattn-common.cuh` (`iq4_nl_expand_lut`, `vec_dot_fattn_vec_KQ_iq4_nl`) |
| Vectorized contiguous→F16 + strided(view) kernels | `ggml/src/ggml-cuda/convert.cu` (`dequantize_block_iq4_nl_f16`) |
| V100 measurements + rejected alternatives | `FA-IQ4_NL-V100-STATUS.md` |

NInfer owns its own copy of this contract (oracle + docs); the fork is provenance, not a runtime
dependency. If the kernel-side encode is later profiled down to ntry ∈ {0,1}, that changes the
semantic contract and must land as a named contract change (oracle, op header prose, these docs
together), never a silent tuning knob.

## What already works

- `PagedKVPool` / `PagedKVAllocation` are dtype-agnostic: page=64 tokens, raw-byte
  `copy_page_d2h/h2d`, prefix retention/borrowing, block tables all carry over unchanged.
- Quantization/dequantization is already fused into the GQA attention kernels (no standalone
  quant op exists to disturb); adding the new codec follows the exact seam INT8-G64 used
  (`gqa_attention_kv_quant.cuh` shared helpers + per-route kernel branches).
- An IQ4 block spans 32 head dims inside one (token, head) row, and a decode 64-key tile equals
  one physical page: **blocks never cross pages or tokens**, so the fork's two rejected
  fixes/non-fixes (used-KV pad-to-256, vec-window OOB) do not exist here.
- DFlash full pool is fixed BF16 (`dflash_context_impl.h`), DFlash2 local sliding KV is a
  separate cyclic store, Vision keeps its own BF16 KV, MTP reuses the same op family — none of
  these are touched by this plan.

## Change inventory

### P1 — plumbing (small)

1. `include/ninfer/types.h`: `KvCacheStorage::Int4NLGroup32`.
2. `apps/cli/options.cpp`, `src/serve/serve_options.cpp`: accept `--kv-dtype iq4nl`
   (default stays `int8`); serving options mirror; `tools/v100/ninfer-v100x2.sh` default
   untouched (override via `NINFER_V100X2_KV_DTYPE=iq4nl`).
3. `core/dtype.{h,cpp}`: new logical marker `DType::I4NL` (element size 1, code-plane granularity
   only; view-level, mirroring how `DType::I8` marks the INT8 row), used by `PagedKVLayerView` /
   `PagedKVBatchLayerView` with `quant_group=32`. Physical pool planes stay U8 codes + FP16 scales.
4. `src/targets/qwen3_6/impl/runtime/layouts_impl.h` (~L979–980) and
   `src/targets/qwen3_6/impl/state/decoder_state.cpp` (pool-spec push, validation): enum ↔ DType
   mapping; plane specs `U8 [64, 128, Nheads]` + `FP16 [64, 8, Nheads]` (per plane, per K/V);
   `program_impl.h` reverse mapping.
5. Wrapper validation/routing/workspace API dtype branches:
   `src/ops/wrapper/gqa_attention.cpp` (`validate`, `volta_flash_route_possible`,
   `gqa_attention_split_capacity`, `gqa_attention_workspace_capacity_bytes`),
   `src/ops/wrapper/kv_cache_append_prefix.cpp` if its code-dtype derivation participates (DFlash
   route is BF16-only today; verify no branch needed).

Gate: builds clean; pool sizing/planning unit checks show expected per-page byte deltas;
`--help`/CLI and serving schema tests pass with the new value rejected or accepted exactly where
the contract says.

### P2 — codec header + decode route (main effort)

1. New `src/ops/kernel/gqa_attention_kv_iq4nl.cuh` (shared device helpers, same role as
   `gqa_attention_kv_quant.cuh`): `__constant__` LUT; paged index math (code offset
   `base + (d>>1)` with nibble parity; scale group-32 offsets); encode primitive
   (BF16 → codes+scale, LSQ+ntry) shared by prefill fill and A2; staged decode
   (`load 64-bit` 8 nibbles → LUT gather → ×scale → pack fp16x2/bf16x2), interface-shaped on
   `gqa_kv_dequant_i8x8_f16_from` so call sites swap locally.
2. Decode kernels: add IQ4 variants of
   `gqa_attention_decode_i8_tc_volta_v2.cuh` (the production long-context split-KV TC route),
   phase-1 `_tc_volta` and the SIMT `_i8` small-T families. The v2 register-prefetch topology
   (codes+scales prefetched, staged to smem FP16) is kept; only the dequant step and scale
   indexing change (group 64 → 32). Cover the MTP-verify rows selected at
   `gqa_attention_decode.cu` L62–70 (tokens=5/6, window ≤512/≤160/(5000,8198]).
3. `gqa_kv_append` (A2) standalone write path with the encode kernel.
4. Launcher dispatch (`src/ops/launcher/gqa_attention_decode.cu`) + route names.

Gate: A3 (`gqa_attention_cached`) conformance passes against the independent FP64 ideal oracle
with IQ4-decoded logical values, at real geometries (24|4 global; 12|2 head-local under TP2),
token extents including MTP verify widths, and the registered envelope domain. This is the fastest
numerical-correctness evidence and lands before prefill work.

### P3 — prefill routes

1. `VoltaFlash` route (`gqa_attention_volta_flash.cu`): its gather stage currently converts BF16
   or INT8-G64 cache rows to FP16 once per layer; add an IQ4 gather/dequant kernel. Port the
   fork's vectorized contiguous kernel plus a **page-aware strided (view) version** — the only
   genuinely new kernel shape, since the fork operates on contiguous sequence caches.
2. `ChunkedSmallT` route: append prologue switches to the IQ4 encode primitive; historical reads
   reuse the P2 staged decode.
3. A1 (append+compute fused) conformance, then A2.

Gate: A1/A2 conformance at real prompt shapes incl. masked tails and `valid_columns`;
`gqa_attention_workspace_capacity_bytes` exact for the new dtype.

### P4 — end-to-end qualification + docs

1. Quality: TP2, 180k, standard corpus, MTP3 greedy — record actual token counts, acceptance,
   committed throughput; compare perplexity/acceptance vs the int8 baseline run on the same box.
   Acceptance against the LM Studio baseline is per `docs/performance.md` methodology (recorded
   token counts + sampling, not short-prompt extrapolation).
2. Performance: committed decode at ≥85k occupied context (acceptance line ~45 tok/s) and at the
   long tail; also record `--kv-capacity auto` resolved capacity delta vs int8 as the
   demonstrated occupancy saving. Expected shape (fork, V100, 9B, `-fa on`): ALU-bound ≤~32k
   (iq4 possibly 5–15% behind int8/f16), KV-bandwidth-bound ≥~70k (iq4 ≈ int8); the 27B/256-head
   dim balance may differ — measure, do not extrapolate.
3. Docs (together, same change): `docs/cli.md` + `docs/serving.md` (--kv-dtype option),
   `docs/performance.md` (new measured results), `docs/maintainer/paged-kv-cache.md` +
   `tensor-formats.md` (IQ4_NL-G32 layout + codec prose), `gqa_attention.h` shared-numerical
   contract block (three storage profiles now), op-development admission/qualification record.

**Close-out status (2026-10-12).** Docs 1-3 except `performance.md` are landed: `--kv-dtype iq4nl`
is documented in `cli.md`/`serving.md`, the IQ4_NL-G32 plane geometry and bytes/token are in
`paged-kv-cache.md`, and the op header carries the third numerical profile. `tensor-formats.md` is a
weight-format registry and does not own a KV codec, so it is unchanged.

The **acceptance measurement is blocked on this machine**: the registered V100X2 artifact
(`/Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer`) and its source GGUF are absent, and the target
of the comparison is the user's LM Studio Q4_K_M baseline. Preliminary Engine-level evidence only
(NOT acceptance), through the public `.ninfer` Engine on the registered `qwen3.8-27b/nvfp4` artifact,
TP2, MTP draft window 3, one request:

| occupied tokens | KV dtype | KV payload | prefill tok/s | decode tok/s | MTP accept |
|---|---|---|---|---|---|
| 22368 | int8-g64 | 0.60 GiB | 1872.7 | 113.09 | (not recorded) |
| 22368 | iq4nl-g32 | 0.33 GiB | 1858.4 | 111.19 | (not recorded) |
| 85724 | int8-g64 | 2.19 GiB | 1387.2 | 84.01 | 55.6% |
| 85724 | iq4nl-g32 | 1.20 GiB | 1379.3 | 98.08 | 78.8% |

IQ4_NL cuts KV payload 45% and, at the acceptance-scale 85k occupation, leads committed decode by
~17%; at 22k it trails by ~1.7% (the ALU-bound short-context regime). The two dtypes generate
different text, so their MTP acceptance and thus their decode throughput are not an isolated
bandwidth comparison. Re-run on the registered Q4_K_M artifact with recorded sampling before
publishing any performance claim.

## Tests and numerical contract

- `include/ninfer/ops/gqa_attention.h` contract block gains the third profile: encoding =
  bit-exact CPU replica of the fork's `quantize_row_iq4_nl_ref` (LSQ, ntry=7, FP16 RNE at its
  stated boundaries) as the registered codec oracle — analogous in nature to the GGUF weight-codec
  oracle, so it does not copy any production kernel staging; decoding = LUT × FP16 scale; ideal
  attention oracle uses decoded FP32 logical values with FP64 arithmetic, exactly as the INT8-G64
  profile does today.
- `tests/ops/test_gqa_attention.cpp` fixture matrix extended with the new cache dtype across the
  real geometry set; block-index off-by regressions (the fork burned itself on a `k/QI4_NL` vs
  `k/QI8_1` chunk error) are caught by multi-extent T=1..conformance runs. No new test files for
  plumbing/getters; no coverage-driven tests.
- Long-context boundary qualifications (envelope 1,048,575/1,048,576) inherit the existing
  criterion handling only if the reduction-length contract claims it; state any limitation
  explicitly rather than silently inheriting.

## Risks

| Risk | Assessment / mitigation |
|---|---|
| Short/mid ctx decode 5–15% slower than int8 (LUT gather replaces the byte_perm 1024-bias trick in an ALU-bound regime) | Confirmed effect class in the fork; the acceptance workload (≥85k occupied) sits in the bandwidth regime. Measure both regimes; the MTP-verify small-window rows must not be left on a degraded path. |
| End-to-end quality loss beyond the per-op tolerance | LUT asymmetry matches attention activation statistics; community benchmarks put iq4_nl ≈ q8_0. Behavioral criterion = MTP acceptance + perplexity diff on the corpus, recorded, not asserted. |
| Prefill encode cost (argmin+LSQ+7 retries per 32 values) | Prefill is matmul-dominated; expect low share. Full-fidelity first; a reduced-retry variant is a named contract change (§Codec definition). |
| CUDA Graph address stability | Unaffected: page mappings unchanged, only in-register decode logic differs. |
| Fork measurements transfer wrong (different model scale / head dim shifts the ALU↔bandwidth crossover) | P4 measures the claim at the claimed scope on this box; fork numbers are provenance only. |

## Environment

- Build: `build-v100/`, sm_70, CUDA 12.8, `-DCMAKE_CUDA_ARCHITECTURES=70`; keep aggregate host CPU
  usage under the user's ~85% ceiling (build concurrency/quota accordingly).
- GPUs: **run all GPU work with `CUDA_VISIBLE_DEVICES=2,3`.** The four cards are otherwise idle; these
  two are the user-designated test pair. (CUDA runtime enumeration order does not match
  `nvidia-smi` index order on this box: the K620 is a separate low-capability device and must never
  be selected for `sm_70` kernels.)
- Artifacts/commands:
  ```bash
  MODEL=/Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf
  NINFER_WEIGHTS=/Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer
  CUDA_VISIBLE_DEVICES=2,3 NINFER_V100X2_KV_DTYPE=iq4nl tools/v100/ninfer-v100x2.sh --prompt ...
  ```

## Out of scope

DFlash drafter KV (fixed BF16 cyclic + full pools), Vision tower KV, additional codec variants
(e.g. IQ4_XS), CPU fallback decode path, host-tier-specific optimizations beyond raw-byte page
copy, and any change to OpenAI/Anthropic schemas beyond the documented `kv_cache storage` option
plumbing already mirrored by `--kv-dtype`.