# sm70 v2 attention + split reduce + TP mailbox port plan

Status: **complete.** All three selected optimizations (#1 INT8 small-T decode attention v2, #5
small-T split reduce, TP mailbox) are ported, built, and measured (+16.5% committed decode at 65K
on the available NVFP4 identity, §Measurements). The one open item is the Q4_K_M 85K acceptance
measurement, which is **permanently blocked on this box**: `/Models` (and the
`qwen3.8-27b/gguf-q4-k-m` artifact and the 85K corpus) does not exist here, and the same limitation
is already recorded in `docs/performance.md`. `docs/maintainer/sm70-optimization-comparison.md`
carries the port status and measurements. This plan is retained as the port's design record because
its named close-out gate cannot be met locally; it is no longer an active working plan.

## Scope

Selected by the user from `sm70-optimization-comparison.md`:

1. **#1 INT8 small-T decode attention v2** — tpx `small_t_i8_volta_v2.cuh` (8 warps, `Bc=64`,
   key-split QK^T with no redundancy, shared-memory max/P exchange, per-warp `D/8` PV slices,
   fp32 PV mma, PRMT int8->fp16 dequant, register double-buffered K/V prefetch).
2. **#5 small-T split reduce** — tpx's branch-free, x8-unrolled group-shared split accumulation.
3. **TP mailbox** — host pinned (`/dev/shm`) mailbox and NVLink device mailbox (CUDA IPC/peer
   access) for BF16 small-message all-reduce, plus the vocabulary-shard row gather, with NCCL
   fallback.

**TP4 is in scope for all three.** V100X2 supports tp in {1,2,4}; the tpx reference implements
TP2 only (two processes, one rank each). This port deliberately generalizes the mailbox protocol
to N ranks in the single-process engine instead of porting the 2-rank-only code.

## Source references (tpx)

| Item | tpx commit / file |
|---|---|
| v2 kernel | `b45e718e` (`small_t_i8_volta_v2.cuh`), `cd13c919` (PRMT dequant), `2b3f16cd` (fp32 PV, page-base tile loads) |
| split reduce | `2c9e20c9` (`small_t.cuh` numerator loop) |
| host mailbox + sharded heads | `2b3f16cd` (`tp2_comm.cpp`, `tp2_mailbox.cu/.h`) |
| NVLink device mailbox + LL | `e107d63d`, `16b23728` |

## Item 1: v2 attention kernel

Files:

- new `src/ops/kernel/gqa_attention_decode_i8_tc_volta_v2.cuh`
- `src/ops/launcher/gqa_attention_decode.cu` dispatch

Adaptations vs tpx (both must be correct in the port):

- This repo's INT8 KV cache stores raw values; there is **no normalized-Hadamard rotation**, so
  the rotation is dropped from Q staging and the K append (tpx rotates both).
- `partial_acc` stays **BF16** (v2 fp32 PV accumulates then rounds once at write-out), so the
  existing `gqa_attention_small_t_reduce_output_kernel` consumes the output unchanged. tpx writes
  f32 partials with its own reduce.
- Names/helpers use the local `gqa_*` vocabulary (`gqa_kv_quant_*`, `paged_kv_page_head_offset`,
  `gqa_small_t_active_splits`, `gqa_small_t_tc_row_to_qt`).
- The append block is the phase-1/SIMT canonical one, so cache codes stay bit-identical across
  all kernels sharing the cache.
- No `key_window` / MTP-window knob (not ported).
- `TokenTile` is a template parameter with `static_assert(TokenTile * GroupSize <= 32)`, and the
  kernel is used only for `tokens * GroupSize <= 32` (one 32-row Volta tile):
  - Gqa27 / Gqa27Tp2 / Gqa27Tp4 (GroupSize 6): T = 3,4,5
  - Gqa35 (GroupSize 8): T = 3,4
  - T = 6 (27B, 36 rows) and T >= 5 (35B) stay on phase-1.
- `Pages` is derived from `kGqaSmallTSplitPageIds<Geometry, 64>` instead of tpx's hard 64: with
  this repo's 560-split Volta policy at the 1,048,576-key domain a split can stage more than 64
  pages (the phase-1 comment already records 194 for 85 splits).
- Dynamic shared memory (~90 KB) with `ensure_func_attr_per_device(MaxDynamicSharedMemorySize)`,
  256 threads, one CTA/SM (`__launch_bounds__(256, 1)`).

TP4 note: the kernel is geometry-generic, but the decode split policy is tuned in units of
phase-1 (`Bc=16`) work; after correctness, re-check/retune the tp4 (6|1, DecodeSplitScale 4) and
tp2 (12|2, scale 2) split counts on hardware, keeping producer/reducer `active_splits` identical.

## Item 2: split reduce

File: `src/ops/kernel/gqa_attention_decode.cuh`
(`gqa_attention_small_t_reduce_output_kernel`).

Port tpx's numerator loop: `Groups = 256 / DChunk` groups each accumulate splits
`g, g+Groups, ...` branch-free with an x8 unroll, then combine group sums through shared memory.
Drops the per-split `weights[split] >= 0` guard (a zero-weight split's partial is 0) and the
shared `values[]` staging. Rank-count independent; applies to tp1/2/4 unchanged.

## Item 3: mailbox (N-rank, single process)

New files: `src/ops/common/tp_mailbox.h`, `src/ops/common/tp_mailbox.cu`; owner hooks in
`CollectiveComms` (`include/ninfer/ops/allreduce.h`, `src/ops/common/allreduce.cu`).

Design:

- One `MailboxPlan` per `CollectiveComms`, created at startup for the actual tp width, owned for
  the communicator's lifetime. All ranks share one decision (single process).
- **Host transport**: one pinned mapped host block, `data[rank][parity][elems]` +
  `flag[src_rank][block]`, per-rank device step counters; every rank publishes its slice, waits
  for all N-1 peer flags, sums all N slices in rank order (bit-identical on every rank).
- **Device transport**: one `Inbox` per rank in its own device memory
  (`data[src_rank][parity][elems]`, `flag[src_rank][block]`); each rank pushes its slice into
  every peer's inbox over NVLink (`cudaDeviceEnablePeerAccess` for every ordered pair), spins on
  flags in its own memory, reads N-1 remote slices from its own inbox. TP2 degenerates to tpx's
  push model. Optional LL variant (data word carries the step) is a follow-up only if the probe
  shows it wins on this box; push + host are the required deliverable.
- **Probe + self-check at init**: exactness self-check per transport (every element must equal the
  integer sum `1+...+tp`), then time NCCL / host / device candidates on a decode-sized message
  (10 KiB and 40 KiB) with events, keep the fastest; fall back to NCCL on any failure. NCCL is
  warmed for the probe sizes before timing (also pre-connects before graph capture).
- `allreduce_sum` uses the plan for contiguous BF16 messages up to 64K elements (128 KB); anything
  larger, non-BF16, or tp == 1 stays on NCCL.
- **Graph safety**: device-side per-block step counters and parity double buffers (tpx design), so
  repeated eager calls and CUDA Graph replays stay correct with no host synchronization.
- **`allgather_rows` fast path**: BF16, all ranks visible, total elements within the gather cap;
  one kernel relocates every rank's contiguous block to its destination row offset and copies the
  N-1 peer blocks from the receiver's inbox. Falls back to NCCL otherwise (uneven splits are
  already generic in NCCL).
- Message sequences must be identical on every rank (same schedule); per-block steps advance in
  lockstep exactly because a single thread drives all ranks.

## Verification

- Build: `build-v100` Volta build (`NINFER_VOLTA_BUILD`, `-DCMAKE_CUDA_ARCHITECTURES=70`).
- v2: existing oracle suites `tests/ops/test_gqa_attention.cpp`,
  `test_gqa_attention_long_context.cpp`, `test_attention_headlocal.cpp` cover T=3..6 (oracle and
  long windows) once the dispatch routes to v2; the A/B knob (`NINFER_SM70_ATTN_V2=0` fallback to
  phase-1) is kept for bisection and future tuning, like tpx's own knob.
- v2 vs phase-1: direct comparison on identical inputs (byte-level cache append check via the
  append cases; output comparison via the registered INT8 criteria).
- reduce: covered by the same suites (all output paths go through the reduce).
- mailbox: extend `tests/ops/test_allreduce.cpp` (it already runs tp2/tp4 and has an FP64 oracle
  plus the bit-identical-rank requirement); add self-check/probe coverage at tp2 and tp4.
- End-to-end: `ninfer-bench`/CLI at 85K occupied tokens, TP2 and TP4, INT8 KV, MTP3, A/B against
  the current build (GPU sharing with the running `ninfer-serve` must be negotiated first).
- Performance acceptance: record real occupied tokens, acceptance, and committed decode tok/s.

## Status

| Item | Status |
|---|---|
| Recon (helpers, TP4 geometry, build, tests) | done |
| v2 kernel file (`gqa_attention_decode_i8_tc_volta_v2.cuh`) | done |
| v2 launcher dispatch + dynamic smem attribute | done |
| split reduce port | done |
| Build + attention oracle tests | done: conformance, head-local, long-context (400K/700K) pass; TP4 6\|1 cases added to the conformance suite |
| v2 TP4 split-policy tuning (6|1/4 and 12|2/2) | correctness verified; tuning after measurement |
| Mailbox N-rank plan: host + device transports, probe, fallback | done |
| Mailbox allreduce fast path | done |
| Mailbox allgather_rows fast path | done |
| test_allreduce + regressions | done: suite passes tp2/tp4 (bit-identical ranks, exact gather, guards); `test_decode_graph_peer` passes tp2/tp4 (graph capture + replay of mailbox collectives); linear_add/linear_swiglu split suites pass |
| A/B measurements | done on the available NVFP4+FP16-vision artifact at 8K and 65K, TP2, eager, MTP3 (see below) |
| Q4_K_M 85K acceptance measurement | blocked: `/Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer` and the 85K corpus do not exist on this machine (`/Models` is absent); the available identities are `qwen3.8-27b/nvfp4` under `/media/zt/models/orcarouter/` |
| TP4 end-to-end measurement | blocked: `ninfer_bench` supports `--tp 1|2` only; tp4 transport is covered by the unit/graph suites and the startup probe |
| Cleanup + doc sync | comparison doc updated with the port status and measurements; `docs/performance.md` sync waits for a Q4_K_M-profile measurement; this plan is removed once that lands |

## Measurements (this box, 2026-10-10)

Artifact `<nvfp4>.ninfer` (`qwen3.8-27b/nvfp4` recipe v1, FP16 vision) because the Q4_K_M
artifact and the 85K corpus are not present locally; TP2 devices 0,1; INT8-G64 KV; MTP3 +
`--lm-head-draft`; eager (`--no-cuda-graph`); `--max-ctx 180000 --prefill-chunk 3072
-pg <P>,512 -r 2 --warmup 0`.

| 65,536-token prompt | prefill tok/s | committed decode tok/s | acceptance | rounds |
|---|---:|---:|---:|---:|
| baseline binary (pre-port: phase-1, old reduce, NCCL) | 1521.60 ± 1.21 | 112.74 ± 0.26 | 0.9794871795 | 260/0 |
| all on | 1520.00 ± 1.94 | **131.33 ± 0.28** | 0.9794871795 | 260/0 |
| `NINFER_SM70_ATTN_V2=0` (phase-1) | 1518.47 ± 0.40 | 115.55 ± 0.17 | 0.9794871795 | 260/0 |
| `NINFER_TP_MAILBOX=0` (NCCL) | 1516.24 ± 2.49 | 129.35 ± 0.19 | 0.9794871795 | 260/0 |

- **#1 v2 attention: +13.7%** committed decode at 65K (115.55 -> 131.33); +1.6% at 8,192 tokens
  (147.77 -> 150.19), i.e. it scales with context. Prefill unchanged (decode-only kernel).
- **mailbox: +1.5%** at 65K (129.35 -> 131.33). In-engine, not probe-level: nsys shows decode
  all-reduce messages at ~12.9 us avg for the NVLink mailbox kernel (10.9 us median) against
  ~17.1 us avg for NCCL ring-LL in the same engine. The startup probe reports a larger gap
  (NCCL ~23-37 us) because it serializes host launches; the honest in-engine saving is ~3-5 us
  per message (128-137 per round).
- **#5 split reduce**: residual ~+1.0% (baseline 112.74 -> phase-1/no-mailbox 115.55 is +2.5%,
  of which the mailbox is +1.5%).
- **Port total: 112.74 -> 131.33 = +16.5%** committed decode at 65K on this identity, with
  identical acceptance and round counts in every configuration.
- nsys kernel counts for the all-on run: `nvlink_allreduce_kernel` 12.9 us avg (10.9 median),
  `nvlink_gather_rows_kernel` 18.3 us avg (head gather fast path active),
  `gqa_attention_small_t_tc_volta_partial_i8_v2_kernel` 29.1 us avg replacing phase-1.

Startup probe on this box: TP2 NCCL ~23-37 us, host mailbox ~50-112 us, NVLink mailbox
~10-22 us; TP4 NCCL ~45-121 us, NVLink ~12-57 us. The host transport is much slower here than
on the tpx box (`cudaHostAlloc` mapped memory over PCIe); it remains the fallback when peer
access is unavailable.

## Progress log

- 2026-10-10: recon complete; plan recorded.
- 2026-10-10: v2 kernel adapted from tpx (Hadamard removed, bf16 partials, computed page-id bound,
  TokenTile static assert), wired into `launch_tc_partial_i8` with the per-device dynamic-smem
  opt-in; #5 reduce numerator loop ported. `test_gqa_attention`, `test_attention_headlocal`,
  `test_gqa_attention_long_context` all pass. Added `qwen3_6_27b_tp4` (6|1) cases to the
  conformance suite for v2 (T=5) and the phase-1 fallback (T=6). Fixed a stale head-local
  negative test that still listed 6 heads as unregistered (pre-existing since TP4 landed).
- 2026-10-10: mailbox implemented as `src/ops/common/tp_mailbox.{h,cu}` (N-rank host + NVLink
  push transports, per-block step counters with parity slots, per-rank event probe against NCCL,
  startup exactness self-check, `NINFER_TP_MAILBOX=0` fallback) and wired into `CollectiveComms`
  (`mailbox_allreduce_bf16` / `mailbox_allgather_rows_bf16` fast paths). Two bugs found by the
  probe/tests and fixed: peer-pointer tables were published before every inbox existed, and the
  device-path fold started from each rank's own value, which is not associative for >2 ranks
  (now every rank folds strictly in source order). Probe on this box (40 KB, contended):
  NCCL 27-40 us, host mailbox 92-112 us, NVLink mailbox 20-22 us -> NVLink selected; tp4 probe
  NCCL 45-121 us, NVLink 12-57 us. Full build passes; `ninfer_allreduce_test` and
  `ninfer_decode_graph_peer_test` pass at tp2 and tp4.
- Pre-existing, unrelated failures observed while running the broader op suites:
  `ninfer_linear_split_test` registry expectations (qtype admission) and an `embed_gather`
  `cudaErrorInvalidResourceHandle` in `ninfer_output_head_split_test` that reproduces with
  `NINFER_TP_MAILBOX=0`, so it is not caused by the mailbox. `ninfer_mtp_split_test` aborts on its
  sm_120a-only NVFP4 A4 cases on this Volta box.
