# Plan — L2, varlen mixed prefill+decode forward

Status: **proposed, not started.** No code, kernels, or contracts changed yet. This is a
research + design doc; implementation is phased in §9.
Scope: the Qwen3.6-family hybrid text path (GQA full-attention + GDN linear-attention) at
`src/targets/qwen3_6`, targeting the V100X2 acceptance workload. 27B/35B-A3B peer variants share
the design; only the 27B Q4_K_M artifact is required to land the acceptance claim.

Decisions:

1. **L2 is a flat varlen ("ragged") batch, not a padded one.** One forward pass consumes a single
   flat `[hidden, T_total]` token stream where `T_total = B·1 + c` (B decode rows, 1 token each,
   plus one prefill row of `c` tokens), with per-token `seq_id`, per-request query offsets and
   per-request KV/seq length. No padding of the decode rows to width `c`; the ragged structure is
   carried only by the two mixer ops (GQA + GDN), every other layer stays a flat `[hidden, T]` GEMM.
2. **The prefill row is the owner's chunk, folded into the decode round.** L2 does not schedule a
   separate prefill unit when decode-ready rows exist; it attaches the owner's next chunk (`c`
   tokens) to the maximal decode membership as one ragged row. One-prefill-owner is preserved —
   there is exactly one such row, and it is the owner.
3. **Two new ragged mixer ops** (`gqa_attention_varlen`, `gated_delta_net_varlen`) are added to
   `include/ninfer/ops/` as explicit new call signatures; the existing A1/A2/A3 and
   `gated_delta_net`/`_snapshot`/`_replay_record` are **kept** for the pure-decode (graph) and
   speculative-verify paths, which L2 does not subsume.
4. **Decode-only rounds still capture the existing exact-B graph.** Any round containing a
   prefill row (query_len > 1) runs the **eager varlen** pass (vLLM's split). A mixed unit is never
   graph-captured; only the uniform all-`W=1` decode rounds are (and that topology is unchanged).
5. **New `TextPhase::Mixed`** is the third `run_layers` phase (alongside `Prefill` and `Verify`)
   at `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/runtime.h:21`. It threads the ragged
   frame through `run_layers`/`attn_mix`/`gdn_mix`; per-layer linear leaves are phase-agnostic.
6. **Numerical contract is per-request identity.** A mixed batch must reproduce each request's
   isolated prefill/decode result to the same per-request FP32/FP64/GQA/GDN oracle; mixing changes
   batching, not math. The prefill row's GDN conv+recurrence over `c` tokens is verified byte-
   parity against the isolated `Phase::Prefill` GDN, not only the oracle (decision 12).
7. **Backend is fixed at startup** (ordinary / MTP / DFlash). M1 lands ordinary; M3 reuses the
   same varlen substrate to coexist the one prefill row with speculative decode rows (decision 14).
8. **The paged KV substrate is not reworked.** Per-sequence non-contiguous blocks + per-row
   block-table selection already exist (`src/core/paged_kv_cache.h:42`, `PagedKVBatchLayerView.
   block_tables`); L2 only adds a per-token `seq_id` and per-row write-address plumbing at the
   op-call/driver level.

---

## 1. Goal

Make the engine run prefill and decode as **one FLAT ragged/varlen token stream** in a single
forward, instead of the current **separate, alternating, fixed-shape** passes. This lifts the
current non-goal — *"多请求 batched prefill 或 prefill/decode mixed forward
(prefill/decode mixed forward)"* — `docs/maintainer/concurrent-inference-architecture.md:32`
(§1.2), and re-opens §7.1's two-unit model and §8's uniform `[D,1,B]` whole-batch layout for the
mixed case.

The three designs on the table:

| Design | Batch shape | Pre-fill work for the owner's chunk | Mechanism | Notes |
|---|---|---|---|---|
| **L0 (today)** | `DecodeRound` `[W=1,B]` + `PrefillChunk` `[W=chunk,B=1]`, alternating `DecodeRound → PrefillChunk → …` (§7.3:799) | a separate eager `PrefillChunk` | uniform `[W,B]` + `valid_columns` | No mixed pass; the alternation is the blocker. |
| **L1 (padded mixed)** | one uniform `[W=c, B_total]` batch, decode rows padded to width `c` | folded into the same batch | **pad** B decode rows to width `c`, mask with `valid_columns[b]` | Reuses the existing uniform layout + `valid_columns` seam; no padding-free path. Waste is `B·(c-1)` GQA slots + `B·(c-1)` GDN columns of dead work; for `c=128, B=8` that is up to 992 dead token-columns per mixer layer. |
| **L2 (this plan, flat varlen)** | one **flat** `[hidden, T_total]` stream, `T_total = B + c`, per-token `seq_id` | folded into the same batch | **no padding**: per-request `query_start_loc` (=`cu_seqlens_q`) + `seqused_k` + block-table | The high-fidelity path; matches vLLM/llama.cpp. The ragged layout is consumed only by GQA and GDN; linears stay flat. |

L2 wins over L1 by eliminating the padded GQA/GDN dead columns (both mixer ops are the expensive,
per-request ones; the flat linears are unaffected by either). L2 loses over L0 in engineering cost
(it crosses the `W≤16 / B` op-domain seam and the uniform-`[W,B]` layout for both mixer ops). L2 is
strictly more general than L1: L1 is L2 with `query_start_loc = {0,c,2c,…,B_total·c}` and
`valid_columns`.

**Acceptance** (per AGENTS.md "Performance work"): above the LM Studio baseline — ~45 committed
decode tok/s at ~85 000 occupied context — with **no further quality loss**, i.e. per-request
continuation identity vs the L0 control.

---

## 2. Current behavior & the structural blocker

The hybrid model is dispatched per layer by `ModelConfig::is_full(layer)` in
`TextContext::run_layers(Tensor& x, Phase)` — `src/targets/qwen3_6/impl/runtime/text_context_impl.
h:1084`, where `x` is a flat `[hidden, T]` tensor and `ModelConfig::is_full` →
`TextConfig::is_full_attention` → `qwen3_6::is_full_attention_layer`
(`src/targets/qwen3_6/export/ninfer/targets/qwen3_6/hybrid_topology.h:9`,
`(layer+1) % 4 == 0`). For the 27B target this is 64 layers laid out `[GDN,GDN,GDN,Full]×16`
— 16 full-attention + 48 GDN (`src/targets/qwen3_6_27b/impl/config.h:67-68`, static_asserts).
The linear leaves (q/k/v/out projections, MLP, `out_proj`) are **flat `[hidden, T]` GEMMs** and
do not care about B-vs-W. The ragged structure lives only in the two mixer ops:

- **GQA** — `attn_mix` at `text_context_impl.h:908` → `ops::gqa_attention` (`:956` and the
  single-request `:960`). Contract `include/ninfer/ops/gqa_attention.h:119`
  (`gqa_attention`), inputs uniform-`[256, nheads, W, B]`, `positions` I32 `[W,B]`,
  `valid_columns` I32 `[B]` (masked prefix) *or empty*, `kv_table_rows` I32 `[B]`.
  Qualified domain is **`B=1 → any W` (prefill) / `B=2..8 → W=1..16` (verify)**
  (`gqa_attention.h:102-103`). Envelope `kGqaAttentionMaximumVisibleKeys` = 1 048 576
  (`:34`). Companion `gqa_kv_append` (`A2`, `:131`, single-seq `[T]` sequential positions) and
  `gqa_attention_cached` (`A3`, `:140`).
- **GDN** — `gdn_mix` at `text_context_impl.h:969` → `ops::gated_delta_net_snapshot`
  (`:1055`) / `gated_delta_net` (`:1063`) / `gated_delta_net_replay_record` (`:1050`).
  `include/ninfer/ops/gated_delta_net.h`: the snapshot form (`:78`) is "B independent
  recurrences", uniform `[128,H,W,B]` + `g/beta` FP32 `[Hv,W,B]` + per-row initial/snapshot state
  slots + `valid_columns[B]`, with the **same domain** "`B=1` any `W`; `B=2..8` `W=1..16`"
  (`:70`).

There are exactly **two** `run_layers` phases, both uniform-`[W,B]` with per-row `valid_columns`
masking — **not** flat varlen:

- `Phase::Prefill` — B=1, W=chunk. Workspace sized `target_body(text_prefill, 1, chunk, Prefill,
  Prefill, 1, 1, chunk, …)` at `src/targets/qwen3_6/impl/runtime/layouts_impl.h:452`. The single-
  request GDN state/conv use `linear_state_current_slot_` / `conv_slot` (single slot, flat
  `causal_conv1d_silu` over the whole chunk, `gdn_mix:1019-1026`); attention uses `gqa_attention`
  with an **empty** `valid_columns` (`attn_mix:960`, `active_sequence_batch_ == 0`).
- `Phase::Verify` — B∈[2,8], W≤16. Sized ordinary-decode at `layouts_impl.h:462`
  (`target_body(ordinary, batch, batch, Verify, Snapshot, batch, 1, 1, …)`) and MTP/DFlash target
  verify at `:511`/`:615` (`ReplayRecord`). Here `active_sequence_batch_ != 0`, so `attn_mix:945`
  and `gdn_mix:985-1059` view the flat projection buffers as `[., ., width, batch]` and call the
  batched ops with per-row `valid = active_valid_columns_`.

The **two things L2 must cross** (the blockers): (a) the **uniform-W layout** — both mixer ops
view their operands as `[heads, W, B]` and read per-row `valid_columns[B]`; a mixed batch has
*non-uniform* per-row width (decode rows W=1 + one prefill row W=c) that is not representable as
`[W,B_total]` without padding; and (b) the **`W≤16 / B≥2` op-domain seam** — the mixed batch is
`B_total∈[2,8]` but the prefill row's width `c` is 128 (MTP/ordinary `kPrefillChunkAlignment=128`,
`src/targets/qwen3_6/impl/runtime/text_context.h:92`) or 1024 (DFlash), far past the 16 that the
`B≥2` branch qualifies. The op is **W-dependent** in its launch (the `B=1` prefill kernel is the
split-KV / full-causal path; the `B≥2` decode kernel is the split-KV decode path with the
`kGqaSmallTSplitPageIds` shared-memory page-id staging bound from `kGqaAttentionMaximumVisibleKeys`,
`gqa_attention.h:16-18`).

The **framework** is `OrdinaryDecodeIngress`/`Egress`,
`src/targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h:31-42` — `kMaximumConcurrency
(=8, include/ninfer/types.h:21)` rows × 1 token each (`tokens[], cache_positions[], rope_
positions[], text_kv_table_rows[], lanes[], sampling[]`). The MTP/DFlash frames
(`round_state.h:46-88`) **already carry `B·width` flat arrays + per-row `*_valid_columns`**
(e.g. `current_drafts[B*W]`, `target_valid_columns[B]`) — so **per-row ragged width machinery
already exists in-tree** for speculative verify. The ordinary decode frame has *no* width dim.

The **decode body** is `ordinary_batch_body` — `src/targets/qwen3_6/impl/runtime/decode_impl.h:
13-77` (H2D ingress → `TextContext::ordinary_decode_batch` → `ops::scatter` continuation hidden →
`ops::sample` over B rows → D2H egress); `capture_ordinary_decode_batch`/`ordinary_decode_batch`
(`:81-92`) wrap it for graph-capture vs eager. `TextContext::ordinary_decode_batch`
(`text_context_impl.h:741-783`) does one `ops::embedding` → `run_layers(x, Phase::Verify, tap)`
→ `rmsnorm` + `lm_head`. The **prefill body** is `TextContext::prefill_impl`
(`text_context_impl.h:1139+`, `run_layers(x, Phase::Prefill, tap)` at `:1276`; the bonus/logit on
the final chunk only at `:1286-1301`), with the per-step extent `min(prefill_chunk, remaining)` in
`src/targets/qwen3_6/impl/runtime/program_impl.h:3124-3139`.

**KV substrate** is paged: `src/core/paged_kv_cache.h` +
`docs/maintainer/paged-kv-cache.md`. A1 consumes `PagedKVBatchLayerView` (shared planes + the
block-table matrix `block_tables` I32 `[logical_pages, table_rows]`, `:42`) with
`kv_table_rows[b]` selecting one row per sequence. The substrate already supports per-sequence
non-contiguous blocks + per-row table selection; L2 changes nothing here but the op-call/driver
(per-token `seq_id`, per-row query+KV offsets) — decision 8.

**Tensor parallel** at TP>1 runs the head-split pool on the same shape per rank; the vocab-split
output head gathers to rank 0 and `ops::sample` runs there alone (peers never sample,
`decode_impl.h:24-51` comment + `run_layers_parallel` `text_context_impl.h:1908`). V100X2 TP>1
uses **eager** decode.

---

## 3. Core idea (the varlen invariant)

**One flat token stream; per-token `seq_id`; per-request query/KV offsets.** A mixed forward is a
single `run_layers` over a flat `x : [hidden, T_total]` where `T_total = B + c`. The invariant:

- Every token in the stream belongs to exactly one request; the mapping is the per-token
  `seq_id[T_total]` (equivalently the ragged offset array `query_start_loc[B_total+1]`
  = `cu_seqlens_q`, where `query_start_loc[r+1]-query_start_loc[r]` is request `r`'s new
  query length).
- Every request contributes its own KV/seq length `seqused_k[r]` (=`seq_lens` in vLLM terms) and
  its own paged block-table row. Decode row `r`: `query_len 1`, `seqused_k = frontier+1`. Prefill
  row: `query_len c`, `seqused_k = reuse_base + c`.
- Every **linear** layer sees only `[hidden, T_total]` — it is the exact same flat GEMM the
  isolated prefill/decode already run, so it is unchanged (phase-agnostic).
- **Only** the GQA and GDN leaves consume the ragged indices: GQA via
  `query_start_loc` + `seqused_k` + `kv_table_rows`; GDN via `query_start_loc` (row boundaries of
  the recurrence) + per-row initial/final state slot.

This is the vLLM V1 `FlashAttentionMetadata` shape (vLLM `v1/attention/backends/flash_attn.py`,
`mamba_attn.py`: `query_start_loc [num_reqs+1]`, `seq_lens`/`seqused_k [num_reqs]`,
`block_table [max_reqs, max_blocks]`, with a `num_decode_tokens`/`num_prefills` split) and the
llama.cpp server `pre_decode` merged prefill+decode batch. NInfer keeps it native: the ragged
indices are device data published per round; no host per-row kernel loop; one whole-batch schedule.

---

## 4. Design decisions (numbered, binding)

1. Add `gqa_attention_varlen` to `include/ninfer/ops/gqa_attention.h`; keep A1/A2/A3.
2. Add `gated_delta_net_varlen` to `include/ninfer/ops/gated_delta_net.h`; keep
   `gated_delta_net`/`_snapshot`/`_replay_record`.
3. The varlen op family accepts a `B_total ∈ [1, kMaximumConcurrency]` ragged batch with
   **per-row query_len ∈ [1, prefill_chunk]** (decode rows 1; the at-most-one prefill row `c`;
   MTP/DFlash rows `draft+1`). `B_total=1` with `query_len=c` must reduce to the prefill A1.
4. Add `TextPhase::Mixed` (`runtime.h:21`). `run_layers(x, Phase::Mixed, tap)` threads the ragged
   frame; `attn_mix`/`gdn_mix` gain a `Mixed` branch calling the varlen ops.
5. The decode-only exact-B **graph** is unchanged and still captured (`layouts_impl.h` ordinary
   profile). The mixed unit is **eager-only** and never captured.
6. Sampler consumes the **B decode sub-rows** (gather over the decode token flat-positions) plus,
   only when the prefill row's chunk **finalizes**, the prefill row's final flat column (the bonus
   token). The c prefill non-final columns are not sampled.
7. The mixed round **always** includes all decode-ready rows (maximal decode, §2.1); the prefill
   row is the single owner (§2.5); the owner's chunk `c` and reuse-base are attached by the
   scheduler (`build_round_membership` + a new mixed unit).
8. The paged substrate is reused as-is; only per-token `seq_id` + per-row write-address/placement
   plumbing is added at the driver.
9. `query_start_loc` is a device I32 `[B_total+1]` published host→device once per round;
   `seqused_k`, `kv_table_rows`, GDN initial/final slots are device I32 `[B_total]`; per-decode-row
   `lanes` and `sampling` are host SoA for the egress gather.
10. The varlen GDN uses the **per-row (snapshot-form) conv + recurrence** for the prefill row over
    `c` tokens (not the flat `causal_conv1d_silu` single-state path), and is verified byte-parity
    with the isolated `Phase::Prefill` GDN for that row (decision 6, §8).
11. Prefill rows are always placed **after** the decode rows in the flat stream (decode-first
    ordering) so the sampler gather over decode sub-rows is a contiguous prefix `[0, B)`.
12. The mixed forward computes `lm_head` over a **gathered** subset of columns — the B decode
    columns (+ the prefill-final column when finalizing) — not over all `T_total`, to avoid a
    `vocab × (B+c)` GEMM (`vocab = 248 320`, `c` up to 128/1024).
13. `query_start_loc`, `seqused_k`, `kv_table_rows`, GDN slots must be address-stable across the
    decode-only graph path: the decode-only round never reads the varlen frame, so the frame
    backing can be a separate region and the graph's `OrdinaryDecodeIngress`/`Egress` layout is
    byte-identical to today (no decode regression).
14. M3 generalizes the *same* varlen substrate so a speculative (MTP/DFlash) decode row has
    `query_len = draft+1` in the same flat stream; L2's `Mixed` phase is the general ragged
    batch, and `Phase::Verify` for speculative verify *coexists* (reused where the prefill row's
    `query_len>1` path and decode rows' `query_len=1` path overlap the prefill/decode kernels).

---

## 5. The varlen batch representation

**Worked example — B=3 decode rows + 1 prefill row of `c=4` tokens.** (Real `c` is 128/1024; 4 is
used for a compact table.) Decode rows are the decode-ready lanes; the prefill row is the owner,
resumed from a reused prefix at `reuse_base=2048`.

Flat stream (`T_total = 3 + 4 = 7`), decode-first (decision 11):

| flat pos `t` | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| `input_ids[t]` | tok(r0) | tok(r1) | tok(r2) | tok(r3,0) | tok(r3,1) | tok(r3,2) | tok(r3,3) |
| `seq_id[t]` | 0 | 1 | 2 | 3 | 3 | 3 | 3 |
| `position[t]` | 1000 | 502 | 88 | 2048 | 2049 | 2050 | 2051 |
| `rope_position[t]` | 1000 | 502 | 88 | 2048 | 2049 | 2050 | 2051 |

Per-request (B_total=4) device vectors:

| request | `seqused_k` (visible keys) | `kv_table_row` (paged bound_row) | GDN init slot→advance | sampler? |
|---|---|---|---|---|
| r0 (decode) | 1001 | alloc_row(r0) | slot(r0) → +1 | yes (flat col 0) |
| r1 (decode) | 503 | alloc_row(r1) | slot(r1) → +1 | yes (flat col 1) |
| r2 (decode) | 89 | alloc_row(r2) | slot(r2) → +1 | yes (flat col 2) |
| r3 (prefill) | 2052 | alloc_row(r3) | slot(r3) → +4 | bonus col 6 **only if** chunk finalizes |

`query_start_loc[5] = {0, 1, 2, 3, 7}`. `rope_position == position` here (no `rope_delta`, no
media); with a rewrite-checkpoint delta the 1-axis rope positions and (Vision) the 3-axis
`[3,T]` MRoPE positions are the ragged form of today's `rope_positions`.

**Flow through the model.** `x = [5120, 7]` = `ops::embedding(input_ids)`. For each of the 64
layers, `is_full(layer)` selects the leaf; all linear leaves (qkv `2560=2·(24·256 + 4·256)`→
proj → `o_proj 5120×6144`/TP, GDN projections `key_dim 2048`, `value_dim 6144`, `gate`, MLP
`17408`, `out_proj`) run **flat over 7 columns** — identical to the isolated prefill/decode. The
two mixer leaves consume the ragged indices:

- **GQA** (16 layers): query tokens at `query_start_loc`; each token attends to its own
  `seqused_k` keys under per-request causal-by-absolute-position. The r3 first new token
  (pos 2048) attends to keys `[0, 2048]` (full continued-context); a decode token (pos 1000) to
  `[0, 1000]`. New K/V for the 7 tokens are appended into the paged pool (per-row block table,
  per-token position) *before* the attention reads, as A1 does.
- **GDN** (48 layers): the recurrence runs independently per request — r0/r1/r2 advance by 1
  token each (decode snapshot semantics), r3 advances by 4 tokens (prefill recurrence) from its
  own state slot; intermediate conv+recurrent state for r3 (prefix capture / snapshot) is written
  by the per-row snapshot path.

After the final `rmsnorm`, decision 12 gathers the **B decode columns** (flat `[0, B)`) — and the
r3-final column (flat 6) only on a finalize chunk — into a compact `[5120, B(+1)]`, runs `lm_head`
on that subset, and `ops::sample` over the B decode rows (rank-0 only at TP>1). Continuation
hidden is published per decode lane exactly as `decode_impl.h:70` does today.

---

## 6. Per-component design

### 6.1 Varlen GQA op

New signature (added to `include/ninfer/ops/gqa_attention.h`, alongside A1 `:119`):

```
void gqa_attention_varlen(const Tensor& q, const Tensor& k, const Tensor& v,     // flat [256,nheads,T]
                          const Tensor& positions,                              // flat I32 [T] (absolute)
                          const Tensor& query_start_loc,                        // I32 [Btot+1]
                          const Tensor& seqused_k,                              // I32 [Btot] (visible keys)
                          const Tensor& kv_table_rows,                          // I32 [Btot]
                          float scale, PagedKVBatchLayerView cache,             // shared planes + block-table matrix
                          GqaExecutionEnvelope envelope,                        // host launch bound over max seqused_k
                          WorkspaceArena& workspace, Tensor& out, cudaStream_t s);
```

- `q/k/v/out` are flat BF16 `[256, nheads, T]` (request-major, decode-first, decision 11).
  `query_start_loc[0]=0`, `query_start_loc[Btot]=T`. `Btotal ∈ [1,8]`; per-row
  `query_len ∈ [1, prefill_chunk]`. `Btotal=1` reduces to the prefill A1 (empty `valid_columns`).
- Envelope: `GqaExecutionEnvelope{min,max}` is a **host launch-resource promise over the batch
  maximum** `seqused_k` (`gqa_attention.h:112-114`); `max = max_r seqused_k[r] ≤ capacity`.
  A new capacity fn `gqa_attention_varlen_workspace_capacity_bytes(q_heads, cache_dtype,
  envelope, Btotal, max_query_len)` sizes from the prefill row (its `c` dominates), so the decode
  rows' split-KV staging bound is never the limit.
- **Kernel**: two inner paths selected per request/CTA by `query_len`: `query_len==1` → the
  existing **split-KV decode** inner loop (page-id staging from `kGqaSmallTSplitPageIds`,
  `src/ops/kernel/gqa_attention_decode.cuh`); `query_len>1` → the existing **prefill causal**
  tiling (`src/ops/kernel/gqa_attention_prefill_bf16.cuh`). This is FlashAttention-2 varlen
  (vLLM handles both in one varlen kernel). The W≤16 / B-domain seam (`gqa_attention.h:102-103`)
  is crossed by treating `query_len` as ragged per row rather than a single `W`.
- **New-KV append folded in** (A1 semantics): the 7 flat tokens' K/V are written to
  `positions[t]` in the paged pool per their `seq_id`'s block table *before* the attention read.
- **Contract change points**: `gqa_attention.h` (new decl + doc block; new capacity fn); wrapper
  `src/ops/wrapper/gqa_attention.cpp` (`validate_attention_tensors` ragged branch); launch
  `src/ops/launcher/gqa_attention_*.cu`; kernels `src/ops/kernel/gqa_attention_{decode,prefill_
  bf16}.cuh` (a unified varlen launcher over the two inner loops); the GQA conformance test
  `tests/ops/test_gqa_attention_long_context.cpp` gains ragged cases.

### 6.2 Varlen GDN op

New signature (added to `include/ninfer/ops/gated_delta_net.h`, alongside
`gated_delta_net_snapshot` `:78`):

```
void gated_delta_net_varlen(const Tensor& q, const Tensor& k, const Tensor& v,  // flat BF16 [128,Hqk,T]
                            const Tensor& g, const Tensor& beta,                // flat FP32 [Hv,T]
                            float scale, bool normalize_qk,                     // = true (matches prefill)
                            const Tensor& query_start_loc,                      // I32 [Btot+1]
                            const Tensor& initial_state_slots,                  // I32 [Btot] (abs pool slot)
                            const Tensor& final_state_slots,                    // I32 [Btot] (abs pool slot)
                            Tensor& conv_states,                                // per-row conv history pool
                            Tensor& ssm_states,                                 // FP32 [128,128,Hv,Slots]
                            WorkspaceArena& ws, Tensor& out, cudaStream_t s);    // flat BF16 [128,Hv,T]
```

- Per-row `[offset=len]` from `query_start_loc`; row `r` runs the DeltaNet recurrence over its
  `len` tokens starting from `ssm_states[initial_state_slots[r]]` and writes the final state to
  `ssm_states[final_state_slots[r]]` (a decode row: `final==initial`, advance by 1; the prefill
  row: advance by `c`). `conv_states` is addressed per row (per-row conv history) and **advanced
  over the row's `len` tokens** — decision 10.
- **Numerically critical (decision 10)**: the prefill row must use the **per-row (snapshot-form)
  conv + recurrence**, not the flat single-state `causal_conv1d_silu` prefill path
  (`text_context_impl.h:1019-1026`). Both must match the FP64 oracle; the plan verifies
  **byte-parity** (`initial→final` state + `out` columns) between the varlen prefill row and the
  isolated `Phase::Prefill` GDN for the same `c` tokens and initial state, to catch any conv-tail
  / boundary divergence (a realistic regression, §11).
- **State-snapshot semantics**: decode row advances its recurrent state by 1; prefill row by `c`;
  an optional per-column snapshot stride (as `gated_delta_net_snapshot`'s
  `snapshot_base_slots`, `gated_delta_net.h:72-76`) is reused for the hybrid prefix-cache taps at
  a chunk boundary (M2).
- **Contract change points**: `gated_delta_net.h` (new decl + doc); the GDN projection wrapper
  (`gdn_input_projection_snapshot` at `text_context_impl.h:1011` → a varlen per-row variant);
  GDN kernel under `src/ops/` (a varlen recurrence launcher over rows of variable length sharing
  the single-state / snapshot inner recurrence); GDN numeric test gains ragged + prefill-parity
  cases.

### 6.3 KV write

The paged substrate (`src/core/paged_kv_cache.h`) already stores per-sequence non-contiguous
blocks and per-row block-table selection (`PagedKVBatchLayerView.block_tables`, `:42`;
`block_table_row`, `:110`). L2 adds, at the op-call/driver level:

- A **varlen append** (a ragged `gqa_kv_append`, or folded into `gqa_attention_varlen` per §6.1)
  that writes the 7 flat tokens' K/V to `positions[t]` in the pool of `seq_id[t]`'s block-table
  row. Per-token `seq_id` + per-row `kv_table_rows` are the only new inputs.
- The **prefill row's** append extends the owner's allocation past `reuse_base` (the M2 restore
  case) exactly as `materialize_sequence_kv` does today (`program_impl.h:3469`) — one call for
  the owner's chunk, plus `c` KV slots.
- No plane/table-matrix base address moves; only the owner's tail block table and the
  per-row placement change. This confirms decision 8.

### 6.4 New frame

`OrdinaryDecodeIngress`/`Egress` (`round_state.h:31-42`) stay **byte-identical** and are still the
pinned/address-stable transfer format for the decode-only graph path (decision 13). A new
**mixed frame** (host SoA + device region), e.g. `MixedDecodeIngress`/`Egress`, is added:

```
input_ids[Tmax]                // Ttotal = B + c ; Tmax = kMaximumConcurrency + prefill_chunk
positions[Tmax], rope_positions[Tmax]  (+ [3,Tmax] MRoPE for Vision in M5)
query_start_loc[Btot+1]; seqused_k[Btot]; kv_table_rows[Btot]; gdn_init[Btot]; gdn_final[Btot]
lanes[B]; sampling[B]                              // per-decode-row, for egress gather
egress: sampled_tokens[B]; bonus_token[1] (only on finalize); (publish continuation hidden per lane via scatter)
```

`Tmax = kMaximumConcurrency + prefill_chunk` (e.g. `8 + 128 = 136`; DFlash `8 + 1024 = 1032`).
Only the exact-[B, Btot, c] prefix is consumed; the inactive tail is never read (mirrors
§8.2 of the architecture doc). The prefill row's fields (reuse-base, `c`) are host-computed from
the owner's `SequenceState` and staged with the decode rows in one whole-batch publication
(§8.3). **What stays pinned for the decode-only graph**: `OrdinaryDecodeIngress`/`Egress`, the
`OrdinaryDecodeState` backing, the decode `logits/hidden/sampled_tokens` regions, and all
state-pool bases — all unchanged, so the exact-B decode graph replays against the same addresses
with no decode regression.

### 6.5 Traversal + Phase

- `enum class TextPhase { Prefill, Verify, Mixed }` (`runtime.h:21`).
- New `TextContext::mixed_prefill_decode_batch(…)` (sibling of `ordinary_decode_batch`
  `:741`, `prefill_impl` `:1139`). It assembles the flat frame (embedding), binds the ragged
  selectors, and calls `run_layers(x, Phase::Mixed, tap)`.
- `run_layers(Tensor& x, Phase::Mixed, tap)` (`text_context_impl.h:1084`): identical layer loop;
  the `is_full`/GDN dispatch is unchanged; only `attn_mix`/`gdn_mix` gain a `Mixed` branch
  (a new scoped state `active_mixed_` holding `query_start_loc/seqused_k/kv_table_rows/gdn init+
  final`), routing to `gqa_attention_varlen` / `gated_delta_net_varlen` (§6.1/§6.2). Linears and
  `mlp_tail` (`:1074`) are phase-agnostic flat GEMMs over `T = x.ne[1]`.
- **GQA vs GDN interleave**: the `[GDN,GDN,GDN,Full]×16` schedule is unchanged; each of the 48 GDN
  leaves consumes the ragged indices and its per-row state; each of the 16 GQA leaves consumes
  the ragged indices and its per-row block table. No cross-layer dependency is introduced; the
  flat residual `x` remains contiguous `[hidden, T_total]` on every rank.
- **Bonus/finalization**: `mixed_prefill_decode_batch` computes logits over the gathered
  decode(+finalize) columns only (decision 12); the r3 bonus token is sampled **only when the
  owner's chunk finalizes** (`staged.cursor + c == prompt_tokens`, mirroring `program_impl.h:
  3139` `final_candidate`). Non-final chunks produce no bonus and the owner keeps ownership for
  its next chunk.
- **Relation to `Phase::Verify` (MTP/DFlash)**: L2 does **not** subsume speculative verify.
  `Phase::Verify` (MTP/DFlash target verify, `target_verify_batch_impl` `:785`, uniform-W +
  `valid_columns` + argmax licensing over the draft window + replay records,
  `gated_delta_net_replay_record`) is a distinct math (draft acceptance, not autoregressive
  decode). The two **coexist**: a mixed round has B speculative rows (`query_len = draft+1`,
  M3) + one prefill row; the varlen `Mixed` substrate generalizes the decode half of Verify, but
  the argmax-licensing/replay machinery stays on the Verify path. M1/M2 (ordinary) never touch
  Verify.

### 6.6 Sampling

Only the **B decode sub-rows** are sampled. Because decode rows are a contiguous flat prefix
`[0, B)` (decision 11), the sampler reads `logits[:, 0:B]` (gathered from the compact lm_head
output, decision 12) — this is exactly today's `ops::sample(logits, sampled, …)` over B rows at
`decode_impl.h:71`; the only change is that the logits come from the gathered mixed output rather
than the decode-only path. The `c` prefill non-final tokens and the non-final prefill columns are
**never** sampled. On a finalize chunk, the r3-final column (flat `B+c-1`) is sampled with
`ops::kSamplePurposePrefill` (`include/ninfer/ops/sampling.h:15`) to produce the owner's bonus
token, distinct from the decode RNG purpose. Continuation hidden is published per decode lane
via `ops::scatter(hidden, lanes, continuation_hidden_store, stream)` exactly as `:70`.

### 6.7 Tensor parallel

At TP>1 each rank runs its **head-split** pool on the **same ragged shape** (decision: the
varlen indices `query_start_loc/seqused_k/kv_table_rows` are identical across ranks — they are
request/logical, not head-sharded). Per-rank `q/k/v/n_kv` and GDN head/channel extents are divided
by `tp` (`shard_kv_heads`, `ModelConfig:text_context.h:73-79`), and `run_layers_parallel`
(`text_context_impl.h:1908`) becomes the TP mixed path: each rank runs its own eager varlen pass
under `cudaSetDevice`, with NCCL `allreduce_sum` on the row-parallel joins (the tower/text `o_
proj`, MLP) — the same pattern the TP prefill/decode paths already use. The vocab-split `lm_head`
output gathers to **rank 0** and `ops::sample` runs there alone; peers never sample
(`decode_impl.h:24-51`). **Eager TP>1 interaction**: V100X2 TP>1 decode is already eager; the
mixed unit is eager varlen on every rank (no graph capture for the mixed shape, §7), so TP>1 adds
no graph-topology change — only per-rank frame copies (each rank's own pinned mixed ingress,
mirroring `peer_host_ingress`, `schedule.h:158-161`). M4 covers this.

### 6.8 Scheduler

`worker_loop` (`src/runtime/engine/concurrent_executor.h:1237`, alternation at `:1266-1292`)
changes from `DecodeRound ⇄ PrefillChunk` to a **mixed unit**:

```
membership = build_round_membership()           // all decode-ready rows (§2.1), :686
if (prefill_lane_ && !membership.empty()):
    run_mixed_unit(membership, prefill_lane_)   // NEW: decode rows + owner's chunk as one varlen pass
    previous_unit_was_decode = true
elif (prefill_lane_):
    run_prefill_step()                          // pure prefill (no decode rows) — :754, unchanged
elif (!membership.empty()):
    run_decode_round(membership)                // :1092, graph-or-eager unchanged
else: try_admit_one() / idle
```

- `build_round_membership` (`:686`) is unchanged (maximal decode); the **owner's lane + chunk `c`
  + reuse-base** are attached by `run_mixed_unit`, which routes through a new
  `ProgramImplCore::decode_mixed_batch(lanes, budgets, prefill_lane, chunk)` (sibling of
  `decode_ordinary_batch`, `program_impl.h:3414`). That fills the B decode ingress rows exactly as
  `:3457-3470` does today, *plus* assembles the owner's flat prefill segment (its `input_ids[base:
  base+c]`, `positions`, the ragged selectors) — reusing the per-step extent logic
  (`program_impl.h:3124-3139`).
- **Invariants preserved**: §2.1 maximal decode (the membership is the full decode-ready set, the
  owner is *added*, never a decode row dropped); §2.5 one prefill owner (exactly one ragged prefill
  row = the owner's chunk); §5.6 admission-turn gate (a mixed unit still runs one bounded prefill
  chunk; the owner does not become decode-ready until its chunk finalizes, so it stays the owner
  across mixed rounds exactly as `run_prefill_step` does today — `:754`, `:737`); §6.2 state
  growth (the round's growth = B·1 + c tokens is bounded by the owner's entitlement; the mixed
  unit only converts reservation→used, matching §6.2:660 *"Prefill/decode 推进只把该 request 已有
  的 logical reservation 转换成 used capacity"*; the owner's reservation already covers its prompt).
- **§7.1 re-opened**: the unit vocabulary gains `MixedUnit(decode-ready ∪ owner-chunk)` alongside
  `PrefillChunk` and `DecodeRound`; all on one execution lane, serial. **§7.3 policy** replaces
  the `Decode → Prefill → …` alternation with `Mixed → Mixed → …` when both are runnable, falling
  back to pure `Prefill`/`Decode` when only one is.

---

## 7. CUDA graph / fixed-shape impact

- **Decode-only** (all rows `W=1`, no prefill row): still captures the existing exact-B graph
  (`capture_ordinary_decode_batch`, `decode_impl.h:81`; profiles `layouts_impl.h` ordinary). The
  varlen frame is **not** read by that path (decision 13), so the graph's node topology, ingress
  addresses, and state-pool bases are unchanged → zero decode regression.
- **Any round with a prefill row** (a ragged `query_len>1` row) runs the **eager varlen** pass:
  `T_total = B + c` varies with `c` (128/1024 / residual tail) and with the decode membership `B`,
  so there is no fixed shape to capture. This is vLLM's split (decode→graph, mixed/prefill→eager).
  The mixed unit is an **eager executable**, exactly like today's eager `PrefillChunk`
  (which is not graph-captured). No new `Definition[family, B, profile]` is added for the mixed
  shape; §9.1 of the architecture doc (exact-B decode/MTP/DFlash definitions) is unaffected.
- **TP1 (graph-captured decode)**: L2 adds one eager entry point (the mixed varlen pass); the
  decode-only graph is untouched. **TP>1 (already eager)**: the mixed varlen pass is new but eager;
  per-rank, NCCL-ordered, no graph. Net: L2's graph footprint is *one eager function*; it does not
  change any captured graph's topology.

---

## 8. Numerical-correctness plan

Oracle per AGENTS.md "Numerical correctness": each Op is checked against a single independent
naive FP32/FP64 oracle that evaluates the complete logical formula from the represented public
inputs (for INT8-G64 KV, the exact decoded code/scale; FP64 GDN recurrence from the FP32 initial
state). Mixing changes **batching, not math**, and the per-request argument is:

- **GQA is per-request over its own KV.** In `gqa_attention_varlen`, request `r`'s query tokens
  read only `seqused_k[r]` keys of `r`'s own block-table row; the causal mask is by absolute
  position within `r`. No cross-request key is ever read (separate block-table rows + ragged
  `query_start_loc` boundaries), so request `r`'s attention result is **identical** to the isolated
  prefill (r3) or isolated decode (r0..r2) result — the flat varlen kernel is the same math as
  A1 prefill/decode, just indexed per row. Checked directly against the GQA FP64 oracle at the
  mixed ragged shapes; A1-vs-varlen parity is supplementary.
- **GDN is per-request over its own recurrent state.** In `gated_delta_net_varlen`, row `r` runs
  its recurrence from `ssm_states[initial_state_slots[r]]` over its `len` tokens and writes to
  `ssm_states[final_state_slots[r]]`; rows never share state. So r0..r2 equal the isolated
  snapshot-decode state and r3 equals the isolated prefill state.
- **Per-request oracle to check**: the *existing* GQA oracle (`tests/ops/test_gqa_attention_
  long_context.cpp`) and GDN oracle, extended at **mixed ragged shapes** (B_total with a decode-
  row batch `query_len=1` plus one prefill row `query_len=c`). The criterion for the varlen kernels
  is the same suite-owned GQA/GDN criterion (gross terms + per-case relative-L2); the FP32 GDN
  state boundary and FP64 recurrence are unchanged.
- **Prefill-row GDN conv-parity (decision 10)**: verify **byte-parity** of the varlen prefill
  row's `out` columns + final conv/recurrent state against the isolated `Phase::Prefill` GDN for
  the same `c` tokens and initial state — this is the one place where L2 could differ from
  L0 (per-row snapshot conv vs flat `causal_conv1d_silu`), and it is caught *directly* (not by
  end-to-end plausibility).
- **End-to-end per-request identity**: a 2-request workload (one mid-decode, one mid-prefill
  resumed from a reused prefix) run **mixed** must produce, for both requests, the same committed
  continuation as running them **isolated / L0** (greedy, same sampling). This is the acceptance
  "no further quality loss" proof.
- **Graph/eager + TP regression**: decode-only graph replay vs eager (unchanged) and tp2/tp4
  ground-truth (M4) must hold; the mixed round's INT8-KV and FP32-GDN-state bytes are unchanged by
  the batching.

---

## 9. Phased implementation + effort/risk

Milestones are independently verifiable; the two ragged kernels dominate cost and land in M1.

- **M1 — TP1, text-only, ordinary decode + one prefill row (core).**
  Scope: `gqa_attention_varlen` + `gated_delta_net_varlen` + varlen KV append (per-token
  seq_id); `TextPhase::Mixed` + `mixed_prefill_decode_batch` + `run_layers` Mixed branch +
  workspace recipe (`layouts_impl.h` mixed profile); the mixed frame (§6.4); the sampler gather
  (§6.6); scheduler `run_mixed_unit` + `ProgramImplCore::decode_mixed_batch` (§6.8). Invariants
  checked: §2.1 / §2.5 / §6.2 (a decode row is never dropped, one owner, reservation→used).
  **Exit**: mixed GQA/GDN kernels pass the FP64 oracle at mixed ragged shapes **and** the prefill-
  row GDN conv-parity (decision 10); greedy end-to-end per-request identity vs the L0 control on a
  2-request (mid-decode + mid-prefill) workload; decode-only graph path byte-identical.
- **M2 — prefix / retained-hit interaction.** Scope: the prefill row resumed from a restored
  block snapshot (`reuse_base = frontier`), KV append extending the owner past the reuse frontier,
  per-column GDN snapshot stride feeding the hybrid prefix-cache taps at a chunk boundary
  (§6.2/§6.3, `docs/maintainer/hybrid-prefix-cache-plan.md`). **Exit**: a returning request with a
  reused prefix run *mixed* == cold-run control (continuation-identical).
- **M3 — MTP/DFlash coexistence.** Scope: a speculative decode row (`query_len = draft+1`) shares
  the flat stream with the prefill owner; argmax-licensing/replay stays on the Verify path
  (decision 14). **Exit**: MTP/DFlash acceptance + per-request continuation parity vs isolated.
- **M4 — TP>1.** Scope: eager varlen on every head-split rank, per-rank frame copies, rank-0
  vocab gather + `ops::sample` (`:71`), NCCL allreduce on the joins (§6.7). **Exit**: tp2/tp4
  ground-truth parity + decode-only graph/eager regressions green.
- **M5 — Vision.** Scope: the multimodal prefill row in the mixed batch (3-axis MRoPE ragged,
  vision scatter into the flat residual, sharded VP encode folded into the owner's unit,
  `text_context_impl.h:2121`). **Exit**: vision-grounded parity (the `image_natural.json`
  mailbox-24 case) with a coexisting decode round.

| # | Component | Change point (file:line) | Effort | Risk |
|---|---|---|---|---|
| 1 | Varlen GQA op/kernel | `gqa_attention.h:119` (new decl) + `src/ops/kernel/gqa_attention_{decode,prefill_bf16}.cuh`, `src/ops/launcher/gqa_attention_*.cu`, `src/ops/wrapper/gqa_attention.cpp` | **L** | High (split-KV vs prefill branch, FP oracle, INT8-KV) |
| 2 | Varlen GDN op/kernel | `gated_delta_net.h:78` (new decl) + GDN projection wrapper `text_context_impl.h:1011`, GDN recurrence kernel | **L** | High (per-row conv+recurrence, state slots, **prefill conv-parity**) |
| 3 | Varlen KV append | `gqa_attention.h:131` (ragged A2) / folded into #1; per-token `seq_id` | **M** | Medium (paged substrate already OK, decision 8) |
| 4 | Paged substrate | `src/core/paged_kv_cache.h:42` (unchanged) | **S** | Low |
| 5 | Mixed frame | `round_state.h:31` (keep) + new `MixedDecodeIngress/Egress` + persistent region | **M** | Medium (decode path must stay pinned/address-stable) |
| 6 | Traversal + Phase | `runtime.h:21` (`TextPhase`), `text_context.h:372/375`, `text_context_impl.h:908/969/741` (new Mixed branch + `mixed_prefill_decode_batch`), `layouts_impl.h:448-467` (mixed workspace) | **L** | Medium (phase threading, workspace recipe) |
| 7 | Sampling gather | `decode_impl.h:71` (gather decode sub-rows + bonus) | **S–M** | Low–Medium |
| 8 | Scheduler + Program | `concurrent_executor.h:686/1092/1266-1292` (`run_mixed_unit`), `program_impl.h:3414` (new `decode_mixed_batch`) + `:3124-3139` (folded chunk) | **M–L** | Medium (§2.1/§2.5/§5.6/§6.2 invariant preservation) |
| 9 | Tensor parallel | `text_context_impl.h:1908/2445` (mixed parallel), rank-0 gather, `peer_host_ingress` (`schedule.h:158-161`) | **M–L** | High (eager TP>1, NCCL in eager varlen) |

**Effort picture**: M1 is the bulk — the two ragged kernels (#1, #2) are each ~1–2 engineer-weeks,
with #6/#8/#5 a further ~1–2 weeks; **M1 ≈ 4–6 engineer-weeks**. M2 ≈ 1 wk; M3 ≈ 2 wks (Verify
coexistence is the hard part); M4 ≈ 1–2 wks (eager TP>1 only); M5 ≈ 1–2 wks. **Total ≈ 9–14
engineer-weeks**, front-loaded in M1. The critical path is the GQA + GDN varlen kernels and the
prefill-GDN conv-parity proof (the one place L2 can differ numerically from L0).

---

## 10. Trade-offs (vs L0 / L1)

- **Vs L0 (today, alternating)**: L2 removes the `Decode → Prefill → Decode` alternation when both
  are runnable — a mid-decode round no longer stalls on a separate eager prefill unit, and the
  owner's first decode token overlaps its final prefill chunk. Wins: TTFT for admitted requests +
  smoother decode occupancy. Loses: engineering cost + two ragged kernels + the prefill-GDN conv-
  parity proof; a slightly more complex scheduler unit.
- **Vs L1 (padded mixed)**: L2 is strictly better — no padded GQA/GDN dead columns (L1 wastes up
  to `B·(c−1)` token-columns per mixer layer, e.g. `8·127 = 1016` at `c=128, B=8`; L2's flat stream
  is exactly `B+c`). L1 reuses the existing uniform layout + `valid_columns` and is cheaper to
  prototype (no new op; just pad + mask), so it is the natural **stopgap** if M1 slips (L2 generalizes
  L1, decision 3). L2's cost is the op-domain seam crossing (§2).
- **Net**: L2 is the high-effort, high-fidelity, no-dead-work path requested here; L1 is the
  fallback; L0 is the baseline. L2 preserves every §8 invariant (mixed is still a single serial,
  bounded, whole-batch unit on one lane).

---

## 11. Risks and open questions

1. **Prefill-GDN conv-parity (decisions 6/10)** — the varlen prefill row uses the per-row
   snapshot conv, not the flat `causal_conv1d_silu`. If the two conv paths differ at the chunk
   tail/boundary, the prefill row's recurrent state diverges from L0. *Mitigation*: M1 exit
   requires byte-parity `out`+state vs isolated `Phase::Prefill`; if a true divergence is found,
   either (a) make the varlen conv exactly reproduce the prefill conv, or (b) accept the per-row
   conv as the *new* qualified profile and qualify the GDN oracle at the ragged shapes (AGENTS: a
   fused route need not reproduce the unfused materialization, but the *observable* state must
   match the oracle). **This is the top open question.**
2. **GQA varlen kernel on V100 (sm_70)** — unifying the split-KV decode inner loop and the prefill
   causal tiling in one varlen launcher is new Volta work; the prefill row's long `c` over long
   `seqused_k` must not regress the split-KV decode bound. *Mitigation*: keep the two inner loops;
   select per request/CTA; qualify the envelope over the prefill row.
3. **Fixed-shape vs eager for the mixed unit** — `T_total = B + c` is not fixed (c is 128/1024 /
   residual). Confirmed eager-only (§7). *Open*: whether a small set of *padded* mixed graph
   shapes (L1-style) is worth capturing for the common `c=128` case; deferred (L1 is the fallback,
   not the target).
4. **Scheduler unit growth** — `run_mixed_unit` fuses owner-chunk assembly with decode
   membership; a bug here could drop a decode row (violating §2.1) or double-advance the owner
   (§2.5). *Mitigation*: reuse `build_round_membership` unchanged; the owner path reuses
   `resolve_prefill_step`'s ownership bookkeeping (`concurrent_executor.h:737`).
5. **Sampling RNG separation** — the r3 bonus (`kSamplePurposePrefill`) and decode
   (`kSamplePurposeDecode`) RNG must not collide when sampled in the same round. *Open*: confirm
   the sampler keys by (purpose, position) so the mixed round's two purposes are disjoint.
6. **Workspace peak** — the mixed profile's peak must cover a prefill-row GQA/GDN stage at `c`
   tokens in the *same* call as the B decode columns; verify `WorkspacePlan.capacity` (max over
   profiles) covers `B + c` (the current `ordinary_round` only covers `B`). *Mitigation*: a new
   `mixed_round` profile sized at `B + prefill_chunk`, added to the `max` at
   `layouts_impl.h:635`.

---

## Relationship to the architecture doc

L2 lifts **§1.2** (the "prefill/decode mixed forward / multi-request batched prefill" non-goal at
`concurrent-inference-architecture.md:32`) for the mixed case while leaving the other non-goals
(prepemption, general continuous batching, dynamic serving-time graph capture) intact. It
re-opens **§7.1** (the two-unit model `PrefillChunk | DecodeRound` at `:765`) by adding a third
serial unit `MixedUnit(decode-ready ∪ owner-chunk)`, and rewrites the **§7.3** decode/prefill
alternation (`:799`) to `Mixed → Mixed` when both are runnable with pure-unit fallback. It extends
**§8** (batched model execution): the whole-batch `current_tokens[B] → [D,1,B]` layout (§8.5 `:999`,
§8.2 `:889`) becomes a flat ragged `[hidden, B+c]` with `query_start_loc`/`seqused_k` for the mixer
operands (§8.6 `:1018-1019` already names "ragged batch" for full attention and per-row recurrent
handles for GDN — L2 makes that the actual mechanism rather than the uniform-`[W,B]`+`valid_
columns` shortcut), and **§8.8** (`:1056`) is superseded: the prefill unit now *does* hold a
temporary folded row inside a decode round (rather than a standalone single-sequence unit that
joins decode only at the next boundary). L2 **preserves** §2.1 (maximal batched decode), §2.5
(one prefill owner), §5.6 (admission-turn gate), §6.2 (per-request entitlement bounds the round's
state growth, incl. the folded prefill row), and the CUDA-graph model §9 (decode-only exact-B
graphs unchanged; the mixed unit is eager, uncaptured). `docs/maintainer/paged-kv-cache.md` is
unchanged (decision 8) — the substrate already supports the per-sequence/per-row addressing the
varlen ops consume.