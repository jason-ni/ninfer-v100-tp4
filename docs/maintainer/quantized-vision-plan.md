# Plan — quantized (Q4/Q5/Q6/W8) Vision weights + host-staged streaming load

Status: in progress (started 2026-10-13)
Scope: Vision tower only (text/MTP/DFlash recipes untouched)

Motivation: the published orcarouter checkpoint
`/media/zt/models/orcarouter/Qwen3.8-27B-Coder390-EfficientThink-W4A4-W8A8-MTP-DFlash2.ninfer`
stores its Vision matrices as grouped signed-integer weights (Q4/Q5/Q6 + W8 merger), while this
project binds every Vision matrix as native `FP16` (`src/targets/qwen3_6/impl/vision/bindings.cpp`
hard-codes `NumericFormat::FP16`). Two independent features land here:

1. **Quantized Vision weights**: bind, shard, materialize, and execute grouped-integer Vision
   weights on the existing TP routes, so such artifacts load.
2. **Host-staged streaming Vision load (required)**: Vision shards are not resident on the GPU.
   They live in pinned host memory, stream to the device once per media-item encode, and release
   afterwards. Steady-state device residency becomes ~0 whether the tower is FP16 or quantized,
   returning the full Vision footprint to the KV pool. Quantization still matters: it cuts the
   per-encode H2D payload (~150 MB vs ~464 MB per rank at tp2) and matches the published
   artifact's recipe.

## 1. Findings (measured, 2026-10-13)

### 1.1 Artifact facts

| artifact | ver | Vision tower payload |
|---|---|---|
| `.../conv_fp16vision/qwen3_8_27b_nvfp4.ninner` current FP16 tower | v2 | 921.5 MB (FP16 915.3 + BF16 6.1) |
| `.../qwen3_8_27b_nvfp4.ninfer` original local conversion | v2 | 295.7 MB (quantized) |
| `Qwen3.8-27B-Coder390-EfficientThink-W4A4-W8A8-MTP-DFlash2.ninfer` | v3 | 295.7 MB (quantized, byte-identical recipe sizes) |

Quantized per-role recipe (identical in the local original conversion and the Coder390 artifact):

| role (logical) | shape `[N,K]` ×count | format | layout |
|---|---|---:|---|
| `vision/patch_embedding` | [1152,1536] ×1 | `Q6G64_F16S` | `row-split-k128-v1` |
| `vision/layers/*/attention/qkv` (fused q\|k\|v) | [3456,1152] ×27 | `Q4G64_F16S` | `row-split-k128-v1` |
| `vision/layers/*/attention/output` (o_proj) | [1152,1152] ×27 | `Q5G64_F16S` | `row-split-k128-v1` |
| `vision/layers/*/mlp/fc1` | [4304,1152] ×27 | `Q4G64_F16S` | `row-split-k128-v1` |
| `vision/layers/*/mlp/fc2` | [1152,4304] ×27 | `Q5G64_F16S` | `row-split-k128-v1` |
| `vision/merger/fc1` | [4608,4608] ×1 | `W8G32_F16S` | `row-split-k128-v1` |
| `vision/merger/fc2` | [5120,4608] ×1 | `W8G32_F16S` | `row-split-k128-v1` |
| all biases, norms, `position_embedding`, `patch_embedding_bias`, `qkv_bias` | — | `BF16` | `contiguous-le-v1` |

Every one of these formats, layouts, and codecs is already registered in the current
`tensor-formats.md` / `storage-layouts.md` registries; **no new persistent numeric format is
added**. What differs between the FP16 and quantized tower is only the checkpoint numeric
recipe's per-role format assignment (tensor-formats.md §2.5 owns exactly this).

Other measured facts:

- Current loader rejects quantized-Vision artifacts outright. Baseline reproduced:
  `ninfer .../qwen3_8_27b_nvfp4.ninfer` (original, quantized Vision) fails with
  `error: tensor descriptor does not match target contract: vision/patch_embedding` even in
  text-only mode (`require_tensor` validates the format for `ValidateOnly` placement too), and
  the Coder390 v3 artifact fails earlier with `error: NInfer v3 compatibility is limited to
  qwen3.8-27b` (its `metadata.name` is `qwen3.8-27b-coder390-w4a4-w8a8-mtp-dflash2`).
- The Coder390 text recipe (64 layers, 16 full-attention + 48 GDN pattern) is role-for-role
  identical to our `qwen3.8-27b/nvfp4` recipe: attention/GDN projections
  `FP8_E4M3FN_ROW_BF16S`, MLPs `NVFP4`, norms BF16, FP32 control fields. It additionally carries
  the same `mtp` and `dflash2` component sets our v3 reader already adapts
  (`src/artifact/reader.cpp:385-418,497-543`).
- Coder390 frontend resources differ from ours: `tokenizer.json` (11071114 vs 12809320 bytes),
  `chat_template.jinja`, `tokenizer_config.json`, and `generation_config.json` hashes all
  differ; the two Vision preprocessor configs are hash-identical. Coder390 is therefore a
  distinct model identity, not a re-packaging of `qwen3.8-27b`.
- All-K extents are multiples of 128: 1152 = 9×128, 4304 = 34×128, 4608 = 36×128, 1536 = 12×128,
  satisfying the `row-split-k128-v1` shard rule below.

### 1.2 Runtime machinery that already exists

- **Binding**: `artifact::bind_tensor` → `require_tensor(name, format, layout, shape,
  placement)` (`src/artifact/typed_binding.cpp:168`) is strictly format-explicit; the format
  literal lives only in the family vision binder (§3.1). `qtype_for`/`storage_layout_for`
  already map all four grouped formats.
- **Sharding geometry**: the 27B shard map resolves per-device `Rows`/`Cols` ranges; the
  materializer turns them into payload copies via `tensor_row_slice` / `tensor_column_slice`
  (`src/artifact/storage_layouts.cpp:402,501`).
  - Row (N) slicing of `row-split-k128-v1` is legal at *any* row boundary and emits a valid
    standalone three-plane payload with parent `K` preserved (layout doc §3.7).
  - Column (K) slicing requires **one contiguous range whose begin and count are multiples of
    128** (`storage_layouts.cpp:559-580`); the shard's own `K_pad` then equals its `K`.
- **Encodings are self-contained**: whether a sliced shard lands in the device arena or in a
  pinned host arena, its encoded layout (base/high/scale planes concatenated with recomputed
  plane offsets, `TensorSlice.encoded_bytes`) is identical and contiguous — which is exactly why
  the §3.4 bulk H2D is one memcpy per rank rather than per plane.
- **Kernels**: `ops::linear` dispatch is format-generic (`src/ops/linear/linear.cpp:100`). The
  q4/q5/q6/w8 families already register exactly these Vision shapes in their tp1 launch tables
  (legacy pre-FP16 tower): `case 1152` in `q4_dispatch.cpp:113` (n ∈ {3456, 4304}),
  `case 1152/4304` in `q5_dispatch.cpp:89,104`, `case 1536` in `q6_dispatch.cpp:61`,
  `case 4608/5120` in `w8_dispatch.cpp:125-138`. On `NINFER_VOLTA_BUILD` the non-SIMT launches
  degrade through the existing `*_launch_needs_volta_fallback` routes (e.g.
  `q4_dispatch.cpp:153-167`); Q4/Q5/Q6/W8 linears need **no split-K workspace**
  (`linear_workspace_capacity_bytes`, `linear.cpp:180-191`).
- **Schedule**: the sharded `encode` (`src/targets/qwen3_6/impl/runtime/vision_context_impl.h`)
  calls `ops::linear` / `linear_column_parallel` / `linear_row_parallel` on `Weight` views; it
  contains no format branch. Activation workspaces are quantization-independent.
- **KV capacity**: sized from residual device memory per rank
  (`src/runtime/engine/kv_capacity.cpp:144`, symmetric over ranks), so freed Vision bytes
  convert automatically into capacity. Context-scaled KV is 16.9 KB/token (tp2), 8.45 KB/token
  (tp4) for the INT8 group-64 pool (fp16-vision-plan.md §6).

### 1.3 Memory / KV impact (computed from measured payloads)

Per-bottleneck-rank Vision residency:

| | tp2 | tp4 |
|---|---:|---:|
| FP16 sharded, device-resident (today) | ~464 MB | ~232 MB |
| quantized sharded, device-resident | ~150 MB | ~82 MB |
| **FP16 or quantized sharded, host-staged (§3.4)** | **~0 MB steady-state** | **~0 MB steady-state** |
| H2D cost per encode (host-staged) | ~150 MB (quantized) / ~464 MB (FP16) | / 4 per rank |

With host staging the KV delta versus no-Vision collapses to the encode window itself; the
realized context capacity is reported in §4. Encode latency changes (dequant/SIMT projection
routes on sm_70 instead of fused FP16 mma, plus one H2D, overlapable with preprocessing);
measured in §4.

## 2. Design decisions

1. **Recipe-driven Vision formats.** The Vision per-role format table becomes an explicit part of
   the checkpoint numeric recipe, selected by `WeightsProfile`, mirroring how text-layer format
   maps are already profile-owned. No runtime sniffing: an artifact's Vision matrices must bind
   exactly against the profile's formats.
2. **New identities; no reinterpretation of existing ones.**
   - `qwen3.8-27b-coder390` (new model id) with weights id `nvfp4-qvis` (name provisional):
     text/MTP/DFlash2 bound exactly like `Qwen38Nvfp4`; Vision bound to the §1.1 table.
     Registered as a peer of the existing identity per AGENTS.md product-contract amendment
     (an additional checkpoint target, explicitly requested).
   - Optionally (phase P5), a local sibling flavor of plain `qwen3.8-27b` re-converted from the
     official BF16 mmproj source with the canonical `MAXABS_F16_RECIP_RNE_V1` encoder (weights
     id e.g. `nvfp4-qvis`) — same profile codepath, no new machinery.
   - The existing `qwen3.8-27b/nvfp4` identity (FP16 tower) and the GGUF / groupwise-int
     identities keep their text behavior; their towers also become host-staged (format-agnostic
     placement change, §3.4).
3. **Host staging is the default for `--vision`.** No extra flag. Rationale: the per-media cost
   is one ~150 MB H2D (~15 ms PCIe3, overlappable with decode/preprocessing that precedes the
   encode) that buys ~315–465 MB of per-rank steady-state KV capacity; repeated media in one
   request re-stream at that cost, which remains a small fraction of the ViT forward. Pinned
   host budget is accounted separately from `--host-cache-mib`.
4. **Encode-scoped lifetime.** Device weight blocks are taken inside a per-encode arena scope on
   that rank and released when the encode boundary synchronizes; no request/session state pins
   them. Vision runs prefill-side and never inside a captured CUDA Graph on this target
   (decode-only graphs; eager at tp>1 anyway), so address stability obligations do not apply.
   Concurrent requests use per-session scopes, so two encodes never alias a weight block.
5. **Quality prior.** The lossy Vision recipe is the one the model publisher shipped in the
   cited checkpoint (AGENTS.md: reliance on publisher/upstream evidence for lossy recipes).
   Local obligations are codec/operator numeric qualification against the independent decoded
   FP32/FP64 oracle, e2e behavior, and reporting the FP16-vs-quantized difference (§4).
6. **No new kernels beyond launch-table entries.** Reuse the existing grouped-integer linear
   stack; only missing shard geometries get registered (§3.6). No repacking, no dequant cache,
   no format-dependent schedule branches.

## 3. Concrete changes

### 3.1 Vision recipe table (new)

Small per-role `NumericFormat` table owned by the 27B package (two instantiations: the current
all-FP16 map and the §1.1 quantized map), passed into the family binders:

- `src/targets/qwen3_6/impl/vision/bindings.cpp`: `bind_vision_backbone`,
  `bind_vision_merger_input`, and the materialization wrappers take the table instead of the
  `NumericFormat::FP16` literals (8 sites). Biases/norms/`position_embedding` stay `BF16` in
  both tables. Shapes are unchanged.
- `src/targets/qwen3_6_27b/impl/load/bindings.cpp` (~line 1117 and ~1355): the
  `vision/merger/fc2` bind and per-device weight sites take the table's `merger_fc2` entry.

### 3.2 Identity plumbing

- `src/targets/qwen3_6_27b/impl/package.cpp`: new `WeightsProfile` case(s);
  `resolve_weights` maps `(qwen3.8-27b-coder390, nvfp4-qvis)` → text binding
  `bind_nvfp4_text_layers` (structure verified identical) + quantized Vision recipe; sampling
  defaults read from the Coder390 `generation_config.json` at registration time.
- `src/targets/registry.cpp:266-279`: map the new model id to the same 27B package.
- `src/artifact/reader.cpp` v3 adapter: accept the Coder390 `metadata.name`, and derive the
  projected identity from the directory (model id from metadata; weights id asserted from the
  observed per-role format table) instead of the hard-coded `{qwen3.8-27b, nvfp4}`. Reject
  anything not matching one of the registered recipe signatures. The structural object-count
  check is expected to pass unchanged since the Coder390 block structure matches ours.

### 3.3 Shard geometry

FP16 contiguous shards use the mod-8 uneven split (`vision_mod8_shard_count`). Quantized
`row-split-k128-v1` K-slices must instead be **multiples of 128** (layout rule,
`storage_layouts.cpp:565`). Add the layout-aware K-split helper (unit 128) and emit, from the
27B shard map, per-profile splits. These splits feed BOTH device placement (for any
device-resident consumer) and the host-shard placement of §3.4 — the shard extents are identical
in either case:

| matrix (split axis) | tp1 | tp2 | tp4 |
|---|---|---|---|
| qkv (N, by head) | 3456 | 1728 | 864 |
| output o_proj (K) | 1152 | 640 / 512 | 384 / 256 / 256 / 256 |
| fc1 (N) = fc2 (K) | 4304 | 2176 | 1152 / 1152 / 1024 / 1024 |
| merger_fc1 (N) = merger_fc2 (K) | 4608 | 2304 | 1152 ×4 |
| patch_embedding, norms, biases, position (Replicated) | full | full | full |

Properties: every K extent stays a multiple of 128 ⇒ shard `K_pad == K` and the kernel guard
`w.k == padded_k && k % kGroupK == 0` holds; fc1's N split and fc2's K split are generated from
the same counts so column-parallel output width equals row-parallel contraction width on every
rank. qkv_bias/fc1_bias N-splits follow their weight (unchanged mechanics).

### 3.4 Host-staged placement and per-encode upload (the new machinery)

- **Binder/placement**: add a host-sharded placement kind alongside `Device`/`ValidateOnly`
  (`src/artifact/binder.h`). For it, `place()` computes the same per-device `TensorSlice`
  (row/column rules of §3.3) but records destinations in a per-device **pinned host arena**
  instead of a device arena. The per-device shard is written with its standalone encoded layout
  (`encoded_bytes`), packed back-to-back in a single flat pinned buffer per rank, so the whole
  rank shard set is one contiguous region.
- **Materializer** (`src/artifact/materializer.{h,cpp}`): `MaterializedArtifact` gains per-device
  pinned host arenas (allocated with `cudaHostAlloc` at `cudaSetDevice(device)`, owned and freed
  like the device arenas), `host_data(handle, device)` / `host_bytes(handle, device)` accessors,
  and stats (`per_device_host_bytes`). For host placement the source bytes are gathered from the
  Reader's file mapping straight into the host arena (no O_DIRECT staging, no H2D at load).
  Load-time H2D bytes reported therefore exclude Vision; the engine summary reports the pinned
  host bytes alongside device capacity. Allocation failure surfaces as an explicit load error.
- **Target views**: `build_device_view` no longer wraps Vision handles in `Weight` against
  device pointers. Instead it builds a `HostStagedVisionBank`: per rank, the ordered descriptor
  list {handle, format, shard shape `[N,K']` or vector extent, offset within the rank buffer} and
  the bank's total encoded size. The family materialization wrapper
  (`materialize_vision_common` and the `merger_fc2` site) returns descriptors, not device-backed
  views; a small shared helper mirrors `row_split_weight`/`contiguous_weight` geometry (same
  padded-shape/qtype math) against a caller-supplied base pointer.
- **Runtime** (`src/targets/qwen3_6/impl/runtime/vision_context_impl.h`): `VisionContext` holds
  the banks; at each `encode` (both the tp1 and sharded paths), per rank: open a scope in that
  rank's per-session vision-weight arena (capacity = bank total, known at bind time), issue ONE
  bulk H2D of the rank buffer (async, on the rank's stream, before its first encode kernel),
  resolve the `VisionCommonWeights`-shape view struct against the scoped base address, run the
  existing schedule unchanged, and let the scope retire at the encode-boundary sync.
  `VisionPrefillSession` reserves the per-rank weight-arena capacity in its workspace setup;
  the activation layout gains no quantization-aware terms.
- **Conformance**: with `--vision` off the bank is never built (existing `ValidateOnly` path),
  so text-only deployments pay zero Vision cost as today. CUDA Graph capture never observes the
  staged pointers. NCCL ordering is unchanged (uploads precede kernels on per-rank streams under
  the existing `for_each_rank` issue order).

### 3.5 Dispatcher launch tables (bounded additions)

The tp2/tp4 Vision shard geometries are not in the `select_*_shard_launch` tables (they
currently list only text shards). Add the finite Vision extents to q4 (k=1152 rows 1728/864;
fc1/fc2 rows 2176 and tp4 1152/1024), q5 (o_proj k ∈ {640,512,384,256}, fc2 k ∈ {2176,1152,1024}),
q6 (none sharded: patch stays full), w8 (merger k ∈ {2304,1152}) to
`src/ops/linear/{q4,q5,q6,w8}/*_dispatch.cpp`, routed to each family's existing generic
launcher (SIMT/fallback on sm_70). Also verify the tp1 Vision-shape T gates
(`q4_dispatch.cpp:114` `t%4==0, 4<=t<=131072` etc.) against the encoder's segment patch counts
and pad segment T in `VisionContext::encode` if any path can violate one (small-image low-T
regime included).

### 3.6 No-change list (explicit)

`ops::vision_attention`, rope, residual adds, serve protocol, CLI flags (`--vision` / `--tp`
unchanged; staging is implicit). Decode path, KV pool addressing, prefix cache, MTP/DFlash. The
GGUF identity still rejects Vision execution.

## 4. Verification

1. **Codec/oracle** (operator level): extend the existing linear family tests
   (`tests/ops/linear/test_*_a16*`) with the whole and §3.3 shard Vision shapes for Q4/Q5/Q6/W8
   against the independent decoded-weight FP32/FP64 oracle (decode stored codes with the exact
   stored FP16 scales; tensor-formats.md §6.2), including the uneven tp4 splits and the W8
   G32 group boundary inside the 128-column slab.
2. **Materialization byte audit**: extend `tests/targets/qwen3_6_27b/
   test_sharded_materialization.cpp` with a quantized-Vision artifact; every per-rank host shard
   must byte-match an independently recomputed plane copy (three-plane offsets per §1.2), for
   both N-rows and K-columns splits.
3. **Lifetime**: a session-level test runs two consecutive media encodes plus an interleaved
   decode round, confirming the encode-scoped weight region is released between encodes
   (arena high-water unchanged across the second encode) and that two concurrent sessions
   resolve disjoint weight regions.
4. **End-to-end**: `examples/cli/messages/image_natural.json` must return the ground-truth
   answer at tp2 and tp4 on the Coder390 artifact (same bar as the FP16 tower validation,
   fp16-vision-plan.md §7). Same-media comparison between the FP16 and quantized towers: report
   merged-Vision-embedding cosine and the prefill/encode wall time (including the measured H2D
   duration) so the quality/cost trade is recorded. Report realized `--kv-capacity auto`
   numbers for: FP16 resident (baseline), quantized resident, and host-staged quantized, with
   the command + hardware provenance for each.
5. **Negative cases**: a v3 artifact whose Vision formats match neither recipe table is rejected
   at bind/adapt time (loud failure per op-development admission rules).

## 5. Phasing

| phase | content | exit criterion |
|---|---|---|
| P0 | study + baselines | done (2026-10-13) |
| P1 | quantized binding: §3.1–§3.3, §3.5 (profile, recipe table, geometry, adapter, launch tables) | build green; quantized artifact binds at tp1/2/4 (device-resident interim mode acceptable) |
| P2 | host staging: §3.4 (placement kind, host arenas, bank, per-encode upload, session setup) | bank byte-audit passes; encode consumes staged views; §4.3 lifetime test |
| P3 | e2e + numbers: §4.1, §4.4–§4.5 on Coder390 at tp2/tp4 (+ FP16 baseline re-run) | all criteria pass; reported table committed with provenance |
| P4 | docs: `qwen3.8-27b-artifact.md` (new identity + recipe table), `storage-layouts.md` (document the row-split-k128 column-slice rule, currently code-only), `AGENTS.md` (identity list; replace "native FP16" with recipe-owned Vision formats + host-staged residency contract), `concurrent-inference-architecture.md` (staged-weight lifetime), cli/serving docs only if user-visible text changes | link/stale-ref review + `git diff --check` |
| P5 (optional) | local re-convert of the BF16 mmproj source into a plain-`qwen3.8-27b` `nvfp4-qvis` flavor (canonical encoder) + parity vs published recipe | §4.1–4.4 on the local artifact |

## 6. Savings summary

| option | steady-state device cost (tp2, bottleneck rank) | notes |
|---|---:|---|
| `--vision` off (today) | 0 MB | no images |
| FP16 sharded resident (today, `--vision`) | ~464 MB | −27 k tokens vs none |
| quantized sharded resident | ~150 MB | ~−9 k tokens; +18 k vs FP16 |
| **host-staged (this plan, default for `--vision`)** | **~0 MB** | −~150 MB H2D per media encode, overlapable; quality per the §2.5 recipe |

Driving every remaining matrix role to Q4 (information floor of the current format registry,
~251 MB tower total) would shave only ~25 MB more from the resident variant and is NOT adopted
— it weakens the publisher quality prior without moving the host-staged result (the H2D shrinks
to ~125 MB, a second-order effect).

## 7. Risks / open questions

- **sm_70 encode speed**: the grouped-integer projection routes run SIMT (CUDA cores) on V100
  for these shapes (fallback policy in `q4_dispatch.cpp:153`). Expect noticeably slower Vision
  encode than the FP16 mma tower; acceptable a priori (prefill-side, once per media item) but
  P3 measures it. If it matters, promoting a fused dequant-mma route is a follow-up operator
  task, not part of this contract.
- **Low-T gates**: §3.5's T alignment check at tiny patch counts is the main runtime surprise
  class; kept explicit rather than silently widened.
- **PCIe contention**: a ~150 MB burst H2D overlaps other requests' decode traffic; bounded and
  measured in P3, with the rank-local upload deliberately placed just before the encode so it
  cannot delay another request's decode kernels on the same stream.
- **Unbalanced splits** (o_proj tp2/tp4) make the resident variant's realized per-rank saving
  slightly less than half/quarter of the raw 625.8 MB delta; §1.3 already prices the bottleneck
  rank. Host staging makes this moot for steady-state capacity.
- **Pinned host budget**: ~150 MB quantized (≈ 300 MB at tp2) or ~464 MB FP16 per rank,
  accounted beside (not pooled with) `--host-cache-mib`; allocation failure is a loud load error.
- **Coder390 sampling defaults / thinking behavior** ("EfficientThink"): its
  `generation_config.json` differs from ours; P3 captures the delivered defaults explicitly.