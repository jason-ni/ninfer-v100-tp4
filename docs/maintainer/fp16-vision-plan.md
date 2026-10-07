# Plan — native FP16 weight format, tensor-parallel FP16 vision tower

Status: phases 1–3c done + GPU-verified at tp2 and tp4 against the re-converted artifact; docs
(phase 4) remain
Scope: vision tower only

Decisions:

1. FP16 only for the vision tower (the text tower keeps NVFP4/FP8/GGUF).
2. Fused in-kernel BF16→FP16 (option b) — best performance, no staging buffer.
3. Accept the BF16→FP16 exponent-range narrowing.
4. Consumer arch: V100 / sm_70 only (sm_86/89 FP16-vision is a follow-up).
5. The vision tower is tensor-parallel (sharded) — option C.
6. `ops::vision_attention` is parameterized in place for head-sharding.
7. The single-device ViT path is preserved for `tp == 1`; the sharded schedule is
   used only at `tp > 1`.
8. **`patch_embedding` is replicated, not column-parallel + allgather** (refines §5).
   Rationale from code investigation: (a) there is no physical transpose op, and
   `ops::allgather_rows` gathers along `ne[1]` while a feature-sharded output has its
   sharded axis on `ne[0]`, so the plan's allgather would need a new transpose kernel or a
   per-patch gather loop; (b) replicating the 3.5 MB/rank `patch_embedding` is negligible
   against the ~460 MB/rank saved by sharding the large matrices, so it does not change the KV
   result in §6; (c) it matches the text path's replicated token-embedding and makes the
   residual stream full-width on every rank from the first op. **Consequence: the only
   collective the sharded schedule needs is `allreduce_sum`** (the row-parallel joins).

## 1. Goal

Add a native `FP16` tensor format to `.ninfer`, store the vision-tower weight
matrices as FP16, compute them directly on Volta FP16 tensor cores
(`mma.sync.m8n8k4`) with no dequant and no staging, and shard the tower across
the TP ranks so it does not consume a full cross-rank KV budget.

## 2. Context and measured sizes

- V100 (sm_70) tensor cores accelerate only FP16: `mma_bf16` is gated to
  `__CUDA_ARCH__ >= 800`, and the BF16 linear on Volta is the CUTLASS
  `OpClassSimt` path (CUDA cores). So a BF16 vision tower is the slow path.
- The original converter quantized the vision tower (Q4/Q5/Q6 + W8 merger) to
  reach the FP16 tensor-core route (`q5_volta_mma_gemm.cuh`). That route is
  lossy.

| vision tower form | bytes/param | size per rank |
|---|---|---|
| BF16 (source) / FP16 | 2 | ~921 MB (measured: `model.visual.*` bf16 = 921.5 MB, ~460 M params) |
| current quantized Q4/Q5/Q6/W8 | 0.53–1.06 | ~295 MB |

FP16 therefore costs ~+626 MB/rank versus the current quantized tower, and
~+921 MB/rank versus no vision. Sharding (decision 5) brings the per-rank cost
back down to ~921/tp MB.

## 3. FP16 linear kernel (fused, option b) — DONE

`src/ops/linear/fp16/{fp16_launch.h,fp16_dispatch.h,fp16_dispatch.cpp,
fp16_volta_mma_gemm.cuh,fp16_volta_mma_gemm.cu}` mirror the q5 files: the
q5 tile with the codec removed and a direct 16-byte FP16 weight load.
Schedule: 4 warps × 8 output rows/CTA, `kKStep=32`, `kTTile=32`, `kThreads=128`,
double-buffered, `__launch_bounds__(128,8)`. Single-split only, so
`linear_workspace_capacity_bytes(FP16_CTRL) = 0`.

Wired into `src/ops/linear/linear.cpp` (`case QType::FP16_CTRL` in
`dispatch_linear` and `linear_workspace_capacity_bytes`) and both CMake lists.
`select_fp16_launch` returns the Volta kernel under `NINFER_VOLTA_BUILD` and
requires `k % 8 == 0` (see §5.4 for why that matters at tp=4).

**Verified:** `tests/ops/linear/test_fp16_a16.cpp` passes against an FP64 oracle
at every registered vision shape (whole + shard), including the fc2 `k=4304`
K-tail zero-pad and a column-parallel / row-parallel shard pair.

## 4. Format registry — DONE

### 4.1 C++

- `src/artifact/reader.h`: `NumericFormat::FP16`; `reader.cpp` `parse_format` +
  the v3 kind map accept `"FP16"`/`"fp16"`.
- `src/artifact/storage_layouts.cpp`: `format_name` → `"FP16"`;
  `direct_word_bytes(FP16)` → 2; admitted in `ContiguousLeV1`.
- `src/core/tensor.h`: `QType::FP16_CTRL = 5` (subsequent values renumbered).
- `src/artifact/typed_binding.cpp`: `storage_layout_for` → `ContiguousLeV1`;
  `qtype_for` → `FP16_CTRL`; `dtype_for` → `DType::FP16`.
- Materializer: no change (contiguous-le copies payload bytes).

### 4.2 Python converter

- `tools/artifact/numeric.py`: `FP16 = DirectFormat("FP16", 2)` in
  `DIRECT_FORMATS`/`__all__`.
- `tools/artifact/layouts.py`: `"FP16"` in `CONTIGUOUS_LE_V1.formats`;
  `"FP16": torch.float16` in `DIRECT_DTYPES`; allowed in encode/decode/size.
- `tools/convert/qwen3_6/common/inventory.py`: `FP16` constant, `DIRECT_FORMATS`,
  `FORMAT_NAMES`; `build_vision_specs` matrices → `FP16` (biases/norms/position
  stay BF16). Shared by the 27B and 35B converters.
- `tools/convert/qwen3_6/common/recipe.py`: vision matrices wrapped in
  `Cast(..., FP16)`; `materialize_expression` handles the FP16 cast target.

**Verified:** the 27B groupwise-int / NVFP4 and the qwen3.8-27b NVFP4 inventory
tests all report `"FP16": 111` (the 111 vision matrices) with the Q/W8 counts
dropped accordingly.

## 5. Tensor-parallel vision tower (option C)

### 5.1 Design

The ViT is treated as a standard TP transformer; the residual stream `x`
(`[1152,P]`, `ne[0]=hidden` contiguous, `ne[1]=P`) stays full-width on every rank.
With decision 8 (replicated `patch_embedding`) the residual is full-width from the
first op, and **the only collective is `allreduce_sum`**.

| op | split | per-device shape | collective |
|---|---|---|---|
| patch_embedding `[1152,1536]` | **replicated** | `[1152,1536]` | — (full `x` produced on every rank) |
| norms, position_embedding, row-parallel OUTPUT biases (`output_bias`,`fc2_bias`) | replicated | full | — |
| column-parallel N biases (`qkv_bias`,`fc1_bias`,merger `fc1_bias`) | sharded exactly like their weight | equals the sharded projection width | — |
| qkv `[3456,1152]` | column-parallel **by head** (3 planes × `(16/tp)·72` rows) | `[3·(16/tp)·72,1152]` | — |
| rope | per-rank heads | — | — |
| vision_attention | **head-sharded** (`heads = 16/tp`) | `[72,(16/tp),P]` | — |
| o_proj `[1152,1152]` | row-parallel (K `1152/tp`) | in `[1152/tp,P]`, out partial `[1152,P]` | `allreduce_sum` |
| fc1 `[4304,1152]` | column-parallel (N mod-8 split) | `[4304/tp,P]` | — |
| fc2 `[1152,4304]` | row-parallel (K mod-8 split) | in `[4304/tp,P]`, out partial `[1152,P]` | `allreduce_sum` |
| merger fc1 `[4608,4608]` | column-parallel (N `4608/tp`) | `[4608/tp,T]` | — |
| merger fc2 `[5120,4608]` | row-parallel (K `4608/tp`) | in `[4608/tp,T]`, out partial `[5120,T]` | `allreduce_sum` → full on every rank |

The residual adds (after o_proj and fc2) are elementwise on the full-width stream,
per rank, no collective. This is exactly the text TP mixer pattern
(`run_layers_parallel`), so the sharded encode mirrors it.

### 5.2 Where the sharding lives

- **Shard map** (object → per-device ranges + axis): added to the 27B target's
  `shard_mapping` (`src/targets/qwen3_6_27b/impl/load/bindings.cpp`), which is what
  the installed `binder.set_shard_resolver` calls at tp>1. The 35B target is tp==1-only
  (no shard resolver) and needs no map. Vision geometry comes from the family
  `VisionBackboneConfig` (`src/targets/qwen3_6/export/ninfer/targets/qwen3_6/vision.h`),
  already included by the 27B `bindings.h`.
- **Per-device materialization**: `qwen3_6::materialize_vision_common` gains
  `int device, int tp` parameters and passes the sharded shape (per-device N or K)
  to `artifact::materialized_weight`, which checks it against the bytes actually placed
  for that (object, device). The 27B `build_device_view` drops its `tp != 1` throw and
  calls it per device; `merger_fc2` is materialized per device the same way.

### 5.3 Shard map (concrete)

Geometry: `hidden=1152`, `intermediate=4304`, `heads=16`, `head_dim=72`,
`merger_hidden=4608`, `patch_dim=1536`. Matched by suffix on the artifact object name.

- **Replicated** (empty plan): `vision/patch_embedding`, `vision/position_embedding`,
  every `*/output_bias|fc2_bias|norm1/*|norm2/*`, `vision/merger/norm/*`,
  `vision/merger/fc2_bias`. (The row-parallel OUTPUT biases stay full width because their
  projection's `allreduce_sum` reconstructs the full width before the bias is applied.)
- **Rows (column-parallel)**: a column-parallel N bias is sharded exactly like the weight that
  produces its output, so each rank's `add_bias` width equals its sharded projection:
  - `*/attention/qkv`: three head-aligned row blocks — one per plane,
    `append_column_block(offset∈{0,1152,2304}, size=1152, divisor=16)`. Per device
    `[3·(16/tp)·72, 1152]`, concatenated q|k|v.
  - `*/attention/qkv_bias`: the identical three-plane head split as `*/attention/qkv`.
  - `*/mlp/fc1`: `append_block_mod8(N=4304)`; `*/mlp/fc1_bias`: the same split.
  - `vision/merger/fc1`: `append_block_mod8(N=4608)` (even: `4608/tp`); `vision/merger/fc1_bias`:
    the same split.
- **Columns (row-parallel)**:
  - `*/attention/output` (o_proj): `append_row_mod8(K=1152)` (even: `1152/tp`).
  - `*/mlp/fc2`: `append_row_mod8(K=4304)`.
  - `vision/merger/fc2`: `append_row_mod8(K=4608)` (even: `4608/tp`).

`append_block_mod8` / `append_row_mod8` split a `total` (a multiple of 8) into `tp`
contiguous parts **each a multiple of 8**, distributing the remainder in whole 8-units:
`units=total/8, base=units/tp, rem=units%tp, count(d)=(base+(d<rem?1:0))*8`. This is the
same per-device extent the materializer must use, so both sides derive it from one helper.

### 5.4 The `k % 8` alignment rule and tp=4

The FP16 kernel loads both operands in `uint4` (8-half) units; a weight row starts at
`w + row·k`, which is 16-byte aligned for every row only when `k % 8 == 0`. Hence
`select_fp16_launch` requires the contraction `K` to be a multiple of 8.

- tp=2: every row-parallel K (`1152/2=576`, `4304/2=2152`, `4608/2=2304`) is a
  multiple of 8 → even splits are fine.
- **tp=4: `fc2`'s K = `4304/4 = 1076`, and `1076 % 8 == 4`** — an even split is invalid.
  The mod-8 uneven split fixes it: `fc2` K per device = `[1080,1080,1072,1072]`, and
  `fc1`'s N (its input) uses the identical split so the column-parallel output width
  matches the row-parallel contraction width on every rank. `o_proj` (`288`) and
  `merger_fc2` (`1152`) stay even at tp=4.

### 5.5 VisionContext / schedule changes

- `VisionContext` becomes execution-context-based: it holds a per-rank weight-shard
  array (index r = rank r's shard, mirroring `TextContext::full_peer_`), the shared
  `ExecutionContext`, and the shared `CollectiveComms`. Rank 0 uses its primary members;
  peers use the array. At `tp == 1` the sharded arrays are unused and the existing
  single-device `encode` runs unchanged.
- New sharded `encode` (tp>1) mirrors `run_layers_parallel`: loop over the 27 blocks;
  per-rank workspace scopes; column-parallel leaves via `ops::linear_column_parallel`,
  row-parallel leaves via `ops::linear_row_parallel` (each carries its single
  `allreduce_sum`); head-local `ops::vision_attention` + `ops::rope` per rank; residual
  adds full-width per rank. Cross-rank ordering is by NCCL stream semantics (no host
  sync, no inter-layer barrier); `synchronize_all()` only at the encode boundary.
- `ops::vision_attention`: parameterize `heads = q.ne[1]` (currently hard-coded
  `kHeads=16` in `src/ops/wrapper/vision_attention.cpp` and the
  `allocate_vision_flash_workspace` shapes); add a `heads` argument, validate
  `16 % tp == 0`. `ops::rope` already operates per head.
- Workspace: per-rank sharded intermediate shapes (qkv `3·1152/tp`, attended
  `72·(16/tp)`, up `4304/tp`); full-width residual/output unchanged.
- Drive the sharded encode from the multimodal prefill path
  (`TextContext::prefill_chunk(..., VisionPrefillSession&, ...)` currently throws at
  tp>1, `text_context_impl.h:1456`): add a parallel overload that runs the sharded encode
  in lockstep (chunk boundaries are token-driven and identical across ranks) then the
  parallel text prefill. The merged output lands full-width on every rank, so the
  existing `x[r]` scatter/assembly is unchanged.

### 5.6 Numerical criterion

Per-rank shards + `allreduce_sum` reproduce the tp1 ViT within the same BF16
split-reduction bound the text path uses. Oracle: an independent FP64 ViT at the real
shapes; the sharded route is checked against it directly, with the tp1 route as
supplementary evidence.

## 6. KV impact

The pool is sized by the tightest device (`resolve_kv_capacity_symmetric`,
`src/runtime/engine/kv_capacity.cpp:144-158`), and KV addressing is symmetric, so
replicating (A2) vs rank-0-only (B) gives the **same** capacity; only sharding
(C) reduces the per-rank footprint.

Context-scaled KV is the 16 full-attention layers (48 GDN layers are fixed
per-slot state), INT8 group-64: 16.9 KB/token (tp2), 8.45 KB/token (tp4).

| design | vision bytes on bottleneck rank | KV lost (tp2) | KV lost (tp4) |
|---|---|---|---|
| A2 replicate | 921 MB | ~55 k | ~109 k |
| B rank-0 only | 921 MB | ~55 k | ~109 k |
| **C shard (this plan)** | 921/tp + 3.5 MB (replicated patch_embedding) | ~27 k | ~27 k |

## 7. Artifact and tests

- Re-convert `/media/zt/models/orcarouter/qwen3_8_27b_nvfp4.ninfer`; verify vision
  object sizes = `2·N·K` whole and per shard (per-device extents from §5.3). DONE.
- `tests/ops/linear/test_fp16_a16.cpp` — DONE (whole + shard shapes, FP64 oracle).
- `tests/ops/test_vision_attention.cpp` — DONE (heads=16/8/4 vs FP64 oracle + non-divisor
  rejection), covering the head-sharded attention leaf.
- `tests/targets/qwen3_6_27b/test_sharded_materialization.cpp` — DONE: binds tp2 with Vision and
  checks `vision/layers/0/mlp/fc1` lands as a proper (sub-full, equal-width) shard on both
  devices; the full text-shard byte audit passes against the re-converted artifact.
- End-to-end ground truth replaces a tp1 parity test: the NVFP4 artifact's ~21 GiB of weights do
  not fit one 16 GB card, so a tp1 ViT run is infeasible. `examples/cli/messages/image_natural.json`
  returns the correct answer (mailbox `24`, sun on the right) at both tp2 and tp4 (the mod-8
  uneven MLP split), which is the available parity evidence.

## 8. Docs / contract

- `docs/maintainer/tensor-formats.md`: add FP16 to the registry table; remove it from
  the §10 exclusion list.
- `docs/maintainer/storage-layouts.md`: contiguous-le-v1 admits FP16.
- `docs/maintainer/qwen3.8-27b-artifact.md` / `qwen3.6-27b-artifact.md` /
  `qwen3.6-35b-a3b-artifact.md`: vision formats, inventory tables, source mapping, and the new
  §7.4 shard-geometry table.
- `AGENTS.md`: vision tower is native FP16 and split across `tp` ranks; names the replicated
  `patch_embedding`, position embedding, norms, and row-parallel output biases (decision 8).
- `apps/cli/options.cpp`: the `--tp` help now lists `--vision` (matching `serve_options.cpp`).

## 9. Risks

- **K-tile alignment at tp=4** — RESOLVED by the mod-8 uneven split (§5.4); fc2/fc1
  use `[1080,1080,1072,1072]` per device. The fused kernel's K-tail zero-pad handles a
  non-32-multiple K within a tile; the mod-8 split guarantees aligned row starts.
- Partial row tiles: predicate `n0 + r < n` (the q5/fp16 kernel already does).
- Shard boundary alignment for FP16 contiguous-le: even/8-aligned K and N boundaries;
  each shard shape is a valid kernel problem by construction (§5.3–5.4).
- BF16→FP16 exact only within [~6e-5, 65504]; accepted (decision 3).
- Lockstep of the sharded encode across ranks relies on identical token-driven chunk
  boundaries (same guarantee the text TP path uses); verify in the parity test.

## 10. Phasing / status

1. **Format registry (C++ + Python) + reader/converter round-trip.** DONE — verified
   (§4).
2. **Fused FP16 linear kernel + FP64 oracle.** DONE — `test_fp16_a16` passes (§3).
3a. **FP16 C++ vision bindings** (family + 27B/35B `merger_fc2`). DONE — compiles;
   tp==1 vision is now FP16.
 3b. **Vision shard map + per-device materialization.** DONE + GPU-verified — added the §5.3 map
     to the 27B `shard_mapping` (`vision/` block: qkv by head, o_proj/fc2/merger_fc2 row-parallel,
     fc1/merger_fc1 column-parallel, everything else replicated); removed both tp>1 vision throws
     (`bind_artifact`, `build_device_view`); threaded `device`/`tp` through
     `materialize_vision_common` (family + 27B + 35B call sites) with per-device extents from the
     shared `vision_mod8_shard_count`. Verified: `test_shard_map` passes with full vision coverage
     at tp=2 and tp=4 (including the mod-8 uneven `[1080,1080,1072,1072]` fc1/fc2 split);
     `test_sharded_materialization` passes against the re-converted artifact — every object's
     per-device shard byte-matches the independently recomputed slice, and tp2+vision now binds and
     shards (fc1 verified sub-full and equal on both devices) instead of being rejected. The tp1
     single-device baseline was re-captured for the FP16 tower (+~625 MiB arena) and skips on a
     16 GB card where the ~20 GiB arena cannot fit.
 3c. **Sharded schedule.** DONE + GPU-verified. `ops::vision_attention` is head-parameterized (head
     count from `q.ne[1]`, must divide 16; `test_vision_attention` passes heads=16/8/4 + non-divisor
     rejection). `VisionContext` holds per-rank `VisionWeights` (`bind_rank`) built from the TP
     constructor `(device, model, ec, events, peer_models)`; `build_workspace_layout` takes
     `(tp, rank)` (qkv `3*hidden/tp`, mod-8 fc1/merger_fc1, per-rank `stage_hidden`/`stage_out`);
     the sharded `encode(item, out[r], work[r])` uses `linear_column_parallel`/`linear_row_parallel`
     (A16-only) for the projections, head-local RoPE + attention, and replicated residual adds, with
     every kernel-issuing per-rank loop wrapped in `for_each_rank(ec, ...)` so each rank's kernels
     run under `cudaSetDevice`. `VisionPrefillSession` grew a `VisionTpSetup` (per-rank `tp_work`
     arenas + `tp_embed_storage` peer copies, `tp_embeddings_`); the Program builds it at `tp>1`.
     The multimodal parallel prefill (`prefill_impl_parallel`) encodes the sharded vision, scatters
     each rank's replicated embeddings into its own residual, and runs the 3-axis MRoPE text layers.
     Two e2e bugs fixed: per-rank kernels issued without device switching, and column-parallel biases
     (`attention/qkv_bias`, `mlp/fc1_bias`, `merger/fc1_bias`) replicated while their projections were
     sharded — they are now sharded exactly like their weight (shard map + materialization extents),
     so each rank's `add_bias` width equals its sharded projection width.
 3d. **Reconvert + parity.** DONE — `/media/zt/models/orcarouter/conv_fp16vision/
     qwen3_8_27b_nvfp4.ninfer` reports `FP16: 111` (the vision matrices; Q4/Q5/W8 vision counts
     dropped accordingly). End-to-end verified on `examples/cli/messages/image_natural.json`:
     tp2 and tp4 (the mod-8 uneven `[1080,1080,1072,1072]` MLP split) both return the ground-truth
     answer (mailbox `24`, sun on the right) with no CUDA/shape errors. A tp1 comparison is not
     runnable for this artifact — its ~21 GiB of weights exceed one 16 GB card — so the tp2/tp4
     ground-truth result is the parity evidence.
 4. **Docs / contract.** DONE — `tensor-formats.md` registers `FP16` as the tenth persistent format
    (direct table + §3.1 semantics + removed from the exclusion list); `storage-layouts.md` admits
    `FP16` under `contiguous-le-v1` (identity table, word-byte table, decode boundary);
    `qwen3.6-27b-artifact.md`, `qwen3.8-27b-artifact.md`, and `qwen3.6-35b-a3b-artifact.md` move the
    Vision linear matrices to `FP16`, keep biases/norms/position BF16, and add a §7.4 Vision
    tensor-parallel shard-geometry table; `AGENTS.md` states the native-FP16 sharded tower contract.
    The CLI `--tp` help now lists `--vision` (matching `serve_options`). No C++ behavior remains
    undocumented in the plan's §8 list.
