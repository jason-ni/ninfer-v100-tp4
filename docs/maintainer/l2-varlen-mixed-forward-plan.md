# Plan — L2, varlen mixed prefill+decode forward

Status: **M1 delivered, TP2-validated; the path is rank-generic for TP4.** The scheduler unit, mixed
frame, workspace profile, and `TextPhase::Mixed` traversal are implemented; the two mixer leaves
dispatch the existing batched-decode op on the decode sub-batch and the existing single-prefill op
on the owner sub-batch instead of the two new ragged kernels this plan originally proposed
(decision 3 updated below, §9). That op-reuse pivot removes the numerics-critical varlen work (the
owner's prefill row runs the exact same prefill ops as an isolated chunk, so the §11.1 conv-parity
question is designed out) and lands per-request identity end to end. TP4 was not run on the
development box: only CUDA devices 1 and 2 are free there. The fused
`gqa_attention_varlen`/`gated_delta_net_varlen` kernels (M1f) are **dropped**: a kernel-time
profile of the mixed unit (TP2, NVFP4 27B, 12.46 s total GPU time) puts the entire mixer family the
fused kernels would replace at ~3.4% (GDN recurrence 2.5%, all GQA attention routes ~0.9%), so
fusing the two sub-batch dispatches into one launch each can only remove launch overhead (<0.5%
end to end) while reintroducing the §11.1 conv-parity risk — the dominant cost is weight
dequantization + GEMM + collectives. M2 (prefix/retained-hit) is delivered (see §9). **L2 is closed
here:** M3 (MTP/DFlash coexistence) and M5 (Vision) are deliberately not pursued inside L2. M3 is a
`MixedMtpUnit` that places the speculative verify slab (uniform width `W_max` at or below the batched
ops' qualified bound, per-row `valid_columns`) and the owner's prefill slab in the same flat stream
as ONE shared forward: the linear leaves (weight dequantization + GEMM) run once over
`W·B + c`, and the mixers dispatch per slab to the existing ops — the MTP slab is the exact batched
verify / replay-record invocation of an isolated MTP round, the prefill slab the exact prefill ops
of an isolated chunk. The dropped varlen kernels are only the mechanism for a general per-row-
arbitrary-length stream; the uniform-slab form needs no new kernel. The previously recorded
"two separate passes, sharing no linears" stopgap is superseded by this: it keeps two full weight
dequantization/read passes over the mixed unit and therefore forfeits the sharing payoff. M3's
remaining work is frame/transaction wiring plus the M1/M2-style identity gate, ~2 weeks; it does not
touch the single-request acceptance (which never mixes) and only pays while ≥2 requests are active
with overlapping prefill/decode phases. M5 is a similar-scale capability for the NVFP4/groupwise
identities only. **M3 is delivered** (2026-10-11): the mixed MTP unit folds the speculative verify
slab and the owner's next chunk into one shared flat pass, dispatches the mixers per slab, and runs
the MTP accept/alignment/proposal pipeline over the slab afterward; when the owner chunk finalizes,
the unit also samples the owner's bonus token and runs the owner's MTP alignment (draft seeding)
inside the unit. Only the reused-suffix `BeforeSuffix` bridge still requires the standalone step.
Gate: `ninfer_qwen3_8_27b_v100x2_mtp_mixed_real_test` (TP2, NVFP4) — a single-chunk (finalize-only)
owner and a multi-chunk owner each match their isolated MTP run.
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
   speculative-verify paths, which L2 does not subsume. *Superseded for M1 correctness:* the
   delivered M1 pass is a flat `[hidden, B+c]` stream whose linear leaves run once over `T` while
   each mixer leaf dispatches its two sub-batches to the EXISTING ops (batched-decode snapshot op on
   columns `[0,B)`, single-prefill op on `[B,T)`). The two varlen kernels were the performance
   milestone (§9 M1f); they are **dropped** on measured evidence (kernel-time profile, see §9/§11.1),
   because they would replace the pair of sub-batch dispatches with one ragged launch each for
   <0.5% end-to-end gain at added numerics risk.
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
7. **Backend is fixed at startup** (ordinary / MTP / DFlash). M1 lands ordinary; M3 coexists the one
   prefill row with speculative decode rows in the same flat stream (decision 14).
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
14. M3 places the speculative (MTP/DFlash) decode rows in the same flat stream as the prefill owner,
    as a **uniform verify slab**: every row has `query_len = draft+1 ≤ W_max`, masked by per-row
    `valid_columns` exactly as `Phase::Verify` runs today, so the slab sits inside the existing
    batched ops' qualified domain and the mixers dispatch per slab, as M1 does — no new kernel. The
    dropped varlen kernels are the mechanism for the general per-row-arbitrary-length stream only.
    `Phase::Verify`'s argmax licensing / replay-record machinery is unchanged and runs
    **after** the shared pass (its inputs — the slab's target logits — come out of that pass),
    settling only the speculative rows' state.

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

The varlen op shapes in §6.1/§6.2 are the original M1f design; they are **not built** (M1f dropped,
§9). M1 instead dispatches the existing ops per sub-batch (§6.5). M3 (decision 14) follows the same
pattern with the speculative rows as a uniform verify slab and likewise needs no new kernel; the
varlen kernels remain the only (unbuilt) mechanism for a general flat stream whose per-row query
lengths are arbitrary.

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
  `Phase::Verify` (MTP/DFlash target verify, uniform-W + `valid_columns` + argmax licensing over the
  draft window + replay records, `gated_delta_net_replay_record`) is a distinct math (draft
  acceptance, not autoregressive decode). The two **coexist**: the M3 mixed round carries the
  speculative rows as one uniform verify slab (`query_len = draft+1`, width `W_max`, per-row
  `valid_columns`) plus one prefill row in ONE shared flat pass — linears run once over
  `W·B + c`, mixers dispatch per slab (decision 14); the argmax-licensing / replay-record machinery
  is unchanged and settles the speculative rows after the pass. M1/M2 (ordinary) never touch Verify.

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

Milestones are independently verifiable. M1 is delivered in the op-reuse form described in the
status header; the ragged kernels that would have been its performance follow-up (M1f) are dropped
on measured evidence (§9 M1f).

- **M1 — TP1..TP4, text-only, ordinary decode + one prefill row (core).** *Delivered; TP2
  validated (the rank-generic path covers TP4, unrun on this box).* Scope:
  `TextPhase::Mixed` + `run_layers` Mixed branch + `mixed_prefill_decode_batch` (single-device and
  per-rank forms) + per-sub-batch mixer dispatch (`attn_mix_mixed` / `gdn_mix_mixed` single-device;
  `attn_mix_parallel` / `gdn_mix_parallel` Mixed branches for TP>1); the mixed frame (§6.4) with one
  flat host staging uploaded into every rank; the sampler gather (§6.6, decode rows group-0, owner
  bonus via `logits_parallel` at TP>1); `run_mixed_unit` + `ProgramImplCore::decode_mixed_batch`
  (§6.8); `mixed_round` workspace profile. At TP>1 every rank reads its own frame slices and the
  existing column/row-parallel leaves plus their allreduce joins carry the pass, including the
  rank-0 vocabulary gather. Invariants checked: §2.1 / §2.5 / §6.2 (a decode row is never dropped,
  one owner, reservation→used).
  **Exit** (met 2026-10-10, TP2, NVFP4 27B artifact): greedy end-to-end per-request identity for a
  2-request (mid-decode + mid-prefill) workload vs each request's isolated run; NVTX confirms the
  unit engages (`mixed.*` ranges over ~10 rounds); the MTP acceptance gate and the ordinary
  concurrent gates are unchanged; the decode-only graph path is untouched. TP1 was not runnable
  here because the 27B artifacts exceed one 16 GB V100; TP4 was not run because the development box
  permits only two of its four V100s (`CUDA_VISIBLE_DEVICES=1,2`).
- **M1f — fused varlen kernels (performance).** *Dropped on measured evidence.* A `cuda_gpu_kern_sum`
  profile of the delivered mixed unit (TP2, NVFP4 27B, 46 mixed rounds, 12.46 s total GPU kernel
  time) attributes ~3.4% to the whole mixer family the fused kernels would replace — GDN recurrence
  (`recurrent_direct_kernel` 1.9% + `recurrent_snapshot_kernel` 0.3% + prepare 0.1%) and every GQA
  attention route (`gqa_attention_decode_i8_tiled`, `..._small_t...`, `flash_attn_ext_f16`) ~0.9%.
  The remaining ~96% is weight dequantization (~25%), cutlass GEMMs (~33%), and all-reduce (~6%).
  Replacing the two sub-batch dispatches with one ragged launch each removes launch overhead only
  (<0.5% end to end) while reintroducing the §11.1 conv-parity exposure, so it is not pursued.
- **M2 — prefix / retained-hit interaction.** *Delivered.* A reused-prefix prefill owner
  (`staged.base > 0`) is folded into a mixed round correctly: the mixed path's owner frontier and
  per-token KV positions are the absolute cursor, not `base + cursor` (the delivered M1 form
  double-counted the reused prefix, which only manifests with reuse). The owner's rewrite-checkpoint
  capture (raw hidden at the frontier + GDN state image) and the hybrid prefix-cache block-commit /
  tap capture now run through a shared `ProgramImplCore::capture_prefix_after_chunk`, so a prefill
  folded into a mixed round grows the cache exactly as an isolated chunk; a boundary-aligned
  checkpoint frontier is materialized in the mixed pass. A checkpoint frontier strictly inside the
  chunk still needs the standalone internal split and is handled by the standalone prefill step
  (the mixed unit does not force a split, which would perturb the decode batch composition). The
  block-aligned final-chunk tap cut is shared with the standalone path. **Exit** (met, TP2, NVFP4
  27B, `ninfer_qwen3_8_27b_v100x2_mixed_reuse_real_test`): the same reused-prefix owner run mixed
  (with a concurrent decoder forcing the suffix chunks through mixed rounds) reproduces the
  single-lane reused control exactly (24/24); the same test fails when the base double-count is
  restored. Along the way the delivered `prefix_tap_slots.reserve()` (no clear) stale-slot bug in
  the hybrid prefix cache was fixed — a retained lane kept a consumed `kInvalidSlot` at index 0.
- **M3 — MTP/DFlash coexistence.** *Delivered* (2026-10-11, TP2). A `MixedMtpUnit` sharing the
  flat pass: the speculative rows as one uniform verify slab (`query_len = draft+1`, `W_max` at or
  below the batched ops' qualified bound, per-row `valid_columns`, the exact same op invocation as
  an isolated verify) plus the owner's prefill slab, with the linear leaves (weight dequantization +
  GEMM) running once over `W·B + c`; the argmax-licensing / replay-record machinery stays on the
  Verify semantics after the pass (decision 14). **No new kernel** — the previously recorded "two
  separate passes, sharing no linears" stopgap (interleaving only) is superseded by the shared-slab
  form, because the stopgap keeps two full weight dequantization/read passes over the mixed unit and
  forfeits the sharing payoff. Delivered as: the `MtpDecodeState` mixed frame (flat stream + owner
  regions + control), the `mixed_mtp_batch` traversal (`TextPhase::Mixed`, slab half using the
  verify op path with `RecordForReplay` GDN, owner half the prefill path), the shared
  `mtp_propose_round` accept/alignment/proposal pipeline, `ProgramImplCore::decode_mixed_mtp_batch`
  with the speculative settlement, `run_mixed_unit` reused unchanged, and the scheduler's
  `mixed_owner_eligible` predicate. **Scope:** the unit folds every bridge-free owner chunk, the
  tail included. The owner's tail additionally runs the owner MTP alignment
  (`TextContext::owner_mtp_align`, via `mtp_prefill_chunk(_parallel)` + the AR proposal loop) inside
  the unit and settles `mtp_kv_valid`/`mtp_draft_count`/`mtp_drafts` in the Program; only a
  reused-suffix `BeforeSuffix` bridge (the suffix's first chunk) still takes the standalone step.
  M3 only pays while ≥2 requests are active with overlapping prefill/decode phases; it does not touch
  the single-request acceptance. Gate: `ninfer_qwen3_8_27b_v100x2_mtp_mixed_real_test` (TP2, NVFP4) —
  a single-chunk (finalize-only) owner and a multi-chunk owner each match their isolated MTP run —
  with the M1/M2 replay gate, the isolated MTP teacher-forced oracle
  (`ninfer_qwen3_8_27b_v100x2_real_test`), and the concurrent-prefix gate unchanged. DFlash follows
  the same slab form (its verify width is at the same qualified bound); not yet wired.
- **M5 — Vision.** *Not pursued* (see the status header): the multimodal prefill row in the mixed
  batch (3-axis MRoPE ragged, vision scatter into the flat residual, sharded VP encode folded into
  the owner's unit) — a similar-scale capability for the NVFP4/groupwise identities only.

TP>1 (originally M4) is delivered inside M1: per-rank frame slices, `run_layers_parallel(Phase::
Mixed)`, the existing split leaves, and the rank-0 bonus gather. No separate milestone remains.

| # | Component | Change point (file:line) | Effort | Risk |
|---|---|---|---|---|
| 1 | Varlen GQA op/kernel (M1f) — **dropped**, §9 | `gqa_attention.h:119` (new decl) + `src/ops/kernel/gqa_attention_{decode,prefill_bf16}.cuh`, `src/ops/launcher/gqa_attention_*.cu`, `src/ops/wrapper/gqa_attention.cpp` | **L** | High (split-KV vs prefill branch, FP oracle, INT8-KV) |
| 2 | Varlen GDN op/kernel (M1f) — **dropped**, §9 | `gated_delta_net.h:78` (new decl) + GDN projection wrapper `text_context_impl.h:1011`, GDN recurrence kernel | **L** | High (per-row conv+recurrence, state slots, **prefill conv-parity**) |
| 3 | Varlen KV append | `gqa_attention.h:131` (ragged A2) / folded into #1; per-token `seq_id` | **M** | Medium (paged substrate already OK, decision 8) |
| 4 | Paged substrate | `src/core/paged_kv_cache.h:42` (unchanged) | **S** | Low |
| 5 | Mixed frame | `round_state.h:31` (keep) + new `MixedDecodeIngress/Egress` + persistent region | **M** | Medium (decode path must stay pinned/address-stable) |
| 6 | Traversal + Phase | `runtime.h:21` (`TextPhase`), `text_context.h:372/375`, `text_context_impl.h:908/969/741` (new Mixed branch + `mixed_prefill_decode_batch`), `layouts_impl.h:448-467` (mixed workspace) | **L** | Medium (phase threading, workspace recipe) |
| 7 | Sampling gather | `decode_impl.h:71` (gather decode sub-rows + bonus) | **S–M** | Low–Medium |
| 8 | Scheduler + Program | `concurrent_executor.h:686/1092/1266-1292` (`run_mixed_unit`), `program_impl.h:3414` (new `decode_mixed_batch`) + `:3124-3139` (folded chunk) | **M–L** | Medium (§2.1/§2.5/§5.6/§6.2 invariant preservation) |
| 9 | Tensor parallel | `text_context_impl.h:1908/2445` (mixed parallel), rank-0 gather, `peer_host_ingress` (`schedule.h:158-161`) | **M–L** | High (eager TP>1, NCCL in eager varlen) |

**Effort picture**: M1 is delivered in its op-reuse form (#3–#9): per-sub-batch mixer dispatch on a
single flat `[hidden, B+c]` stream, the mixed frame, phase, workspace, `run_mixed_unit`, and TP>1.
M2 is delivered (the retained-prefix owner folded through the mixed owner, §9). The two ragged
kernels (#1, #2) were the ~1–2 engineer-weeks-each critical path and are **dropped** on the measured
<0.5% end-to-end payoff (§9 M1f); the prefill-GDN conv-parity proof they would have required is
therefore not on any active path. **M3 is delivered** (shared verify-slab pass, non-tail owner
chunks; decision 14 + §9). Remaining, outside L2: M5 ≈ 1–2 wks (Vision); DFlash may adopt the same
M3 slab form (not yet wired).

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

1. **Prefill-GDN conv-parity (decisions 6/10)** — *Designed out for M1.* The delivered owner half
   runs the existing single-state prefill ops (`gdn_input_projection` + `causal_conv1d_silu` +
   `gated_delta_net`) on the `[B,T)` slice with the owner's own conv/recurrent slot, i.e. exactly
   the arithmetic an isolated `Phase::Prefill` chunk performs. The plan's original per-row snapshot
   conv is gone. The risk would have returned only if M1f replaced this half with a fused varlen
   kernel; M1f is dropped (§9), so the risk is off the active path.
2. **GQA varlen kernel on V100 (sm_70)** — off the active path (M1f dropped, §9). M1 uses the
   existing decode and prefill routes unchanged, so no new Volta kernel is on the critical path.
3. **Fixed-shape vs eager for the mixed unit** — `T_total = B + c` is not fixed (c is 128/1024 /
   residual). Confirmed eager-only (§7). The delivered unit is never graph-captured; decode-only
   rounds keep their exact-B graphs, and the ordinary frame's ingress/egress regions stay at their
   previous offsets so those graphs remain address-stable. *Open*: whether a small set of *padded*
   mixed graph shapes (L1-style) is worth capturing for the common `c=128` case; deferred.
4. **Scheduler unit growth** — `run_mixed_unit` fuses owner-chunk assembly with decode
   membership; a bug here could drop a decode row (violating §2.1) or double-advance the owner
   (§2.5). *Mitigation*: reuse `build_round_membership` unchanged; the owner path reuses
   `resolve_prefill_step`'s ownership bookkeeping (`concurrent_executor.h:737`).
5. **Sampling RNG separation** — the r3 bonus (`kSamplePurposePrefill`) and decode
   (`kSamplePurposeDecode`) RNG must not collide when sampled in the same round. *Open*: confirm
   the sampler keys by (purpose, position) so the mixed round's two purposes are disjoint.
6. **Workspace peak** — *Resolved.* The mixed profile is now an exact model of the mixed body in
   `build_workspace_plan` (`mixed_body`): the T-wide projections and the prefill half's conv
   scratch, plus the decode half's batch-shaped GQA scratch and snapshot-projection scratch, each
   in its own nested scope exactly as the execution nests `work_.scope()`. The earlier
   `target_body(Prefill)` profile under-planned both decode-half scratches (this was latent because
   the 27B does not fit one V100). The TP>1 logits gather columns are planned at
   `max_concurrency`.

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