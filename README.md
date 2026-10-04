# NInfer V100X2

[中文文档](README.zh-CN.md) · [Performance methodology](docs/performance.md)

Performance results are separated into [P2P enabled](#p2p-enabled) and
[P2P disabled](#p2p-disabled); build and launch instructions are shared below.

This fork is tuned for one-request Qwen3.8-27B inference on **Tesla V100-SXM2 16 GB**
(`sm_70`, CUDA 12.8), at tensor-parallel width 2 or 4. Its default profile uses the LM Studio
Q4_K_M-derived `.ninfer` artifact, 180,000-token context capacity, complete INT8 group-64 KV,
TP2, CUDA Graphs, and MTP with up to three drafts (zero accepted drafts is valid). TP4 splits the
same resident model across four cards, quartering the per-GPU weight footfall to free KV headroom;
it uses NCCL collectives and one CUDA Graph per rank. Context capacity is an allocation limit, not
the number of prompt tokens in a benchmark.

The starting point combines the RTX 3060 TP2 work and the Volta implementation from
[`geoffwatts/ninfer-v100`](https://github.com/geoffwatts/ninfer-v100), based on
[Neroued/ninfer](https://github.com/Neroued/ninfer). This README contains V100X2-specific changes
and measurements only; inherited RTX 5090/Ampere/Ada results and general upstream capabilities are
intentionally omitted.

## V100X2 changes

- **Q4_K_M prefill:** cooperative SM70 GGML-K block decoding materializes rows into caller-owned
  FP16 workspace and feeds the Volta CUTLASS Tensor-Core GEMM. GDN control projection has an FP32
  output path. The original GGUF Q4_K/Q6_K codes and scales are kept unchanged.
- **Volta FP8/NVFP4 execution:** wide prefill projections decode weights once per call and use
  SM70 CUTLASS, including the prepacked TP2 matrices. NVFP4 decoding reads complete K16 tuples
  cooperatively. Wide TP2 NVFP4 MLP calls at T≥2048 use a 256-column GEMM tile.
  Decode uses QPN Tensor-Core kernels. Both wide prefill and decode keep gate/up
  projections in FP32 through SwiGLU, rounding only the final activation to BF16.
  The prepacked FP8 output head now uses its matching kernel for single-token calls and chunk
  tails too; reading that layout through row-major GEMV was a correctness bug. NVFP4 A4 is not
  supported on V100.
- **Long-context decode:** the split-KV attention reducer shares softmax weights and reads eight
  split vectors cooperatively, then accumulates in the original FP32 order. It retains every
  history token, the same KV format and the same partial-output precision. The INT8 attention
  kernel also computes each QK score tile once and shares it across the four output-dimension
  warps, preserving their softmax and PV accumulation order.
- **DFlash sliding-window attention:** the Volta path uses FP32 split numerators and a warp-per-row
  route once enough rows are in flight; direct tiny windows retain the CTA route. The supported
  local window is 2,048 or 4,096 positions, and the complete-history target KV is unchanged.
- **GDN prefill:** long normalized inputs prepare each Q/K row once in FP32, avoiding repeated
  normalization across state tiles. The sequential FP32 state transition is unchanged. The
  3,072-token TP2 GDN scratch allocation is 24 MiB.
- **Cross-device collectives (NCCL):** row-parallel projections reduce with NCCL (`ncclAllReduce`,
  plus `ncclAllGather` for the split output head and `ncclSend`/`ncclRecv` on the DFlash rank-0
  paths), one communicator per rank, issued from the single engine thread inside
  `ncclGroupStart`/`ncclGroupEnd`. Each rank records its own collectives into its own CUDA Graph,
  so graph capture/replay is preserved. NCCL selects the fastest transport it can (NVLink where
  present, otherwise PCIe P2P or shared memory); the earlier hand-rolled UVA device-to-device
  copies and event pairs are gone.
- **Volta TP2 residuals:** Q5 row shards now select executable SM70 SIMT, fused MMA, or CUTLASS
  routes with the actual peak workspace reserved. BF16 row shards use the SM70 CUTLASS route.
  Both paths pass their independent FP64 operator checks and the two-card composition tests.
- **NVFP4 v3 container:** the reader accepts the upstream `NINFER\x00\x03` single-file
  `qwen3.8-27b` container and projects its physical objects and logical bindings into the
  registered `qwen3.8-27b/nvfp4` identity without repacking weight bytes. The v3
  `tokenizer_config.json` chat template is exposed verbatim. Text and MTP load through the normal
  Engine path; v3 Vision objects are projected for the existing Vision route. Its optional
  five-layer DFlash2 package is also bound for text-only TP2 experiments.
- **MTP and DFlash:** the Q4_K_M target uses the native MTP route. An optional five-layer BF16
  auxiliary / W8-projection DFlash2 route is integrated with TP2. Greedy target verification
  exchanges shard argmax values/IDs instead of full logits; proposal selection merges each
  rank's exact top-16 keys. Sampled verification retains the full-logit route. Removing its unused
  full-history draft KV pool leaves the all-local v3 route at 98,304-token capacity; 180,000-capacity
  DFlash is not claimed on two 16-GB cards. Narrow BF16 draft projections use direct BF16
  operands with FP32 SIMT accumulation on V100. MTP remains the default.

## Measurements

All NInfer results below are from this V100X2 host. Decode tok/s counts committed output tokens,
not drafted tokens; occupied prompt length is reported separately from maximum capacity.
The two areas distinguish the current direct-P2P configuration from the peer-off control and
earlier staging measurements. NVFP4 and Q4_K_M are different weight artifacts, not a matched
quantization-quality comparison.

### P2P enabled

Measured on 2026-10-01 with two V100-SXM2 16 GB cards at 300 W each, CUDA 12.8, PCIe 3.0 ×16,
PHB topology, `iommu=pt`, and identity IOMMU domains. NVIDIA P2P read/write checks pass and
NInfer automatically enables direct P2P. No inference algorithm or weight changes were needed
for the MTP P2P A/B evaluation. Both transport areas use the same launch commands; startup
qualifies the actual bidirectional copy route. `iommu=pt` alone does not prove working P2P.

#### NVFP4 v3 occupied-context sweep

Official `qwen3.8-27b/nvfp4` v3 artifact, TP2, complete INT8 group-64 KV, greedy MTP3, optimized
draft head, CUDA Graphs, 3,072-token prefill chunks and **180,000 context capacity** (180032 KV
positions allocated). Each cell uses two cold-prompt requests without prefix reuse or an extra
request warmup; graphs are primed before measurement. Every request generates 513 tokens:
one from prefill and **512 timed committed decode tokens**. Rates are mean ± sample SD.

| Actual input tokens | Prefill tok/s | Committed decode tok/s | MTP acceptance |
|---:|---:|---:|---:|
| 3,072 | 1,783.96 ± 43.38 | 105.48 ± 0.01 | 72.97% |
| 8,192 | 1,754.43 ± 18.24 | 114.49 ± 0.01 | 81.98% |
| 16,384 | 1,699.06 ± 9.73 | 108.29 ± 0.003 | 79.47% |
| 32,768 | 1,596.24 ± 2.94 | 101.63 ± 0.03 | 79.69% |
| 65,536 | 1,397.83 ± 0.72 | 88.34 ± 0.04 | 79.12% |
| 85,000 | **1,306.48 ± 3.95** | **83.18 ± 0.13** | **79.12%** |

Both repetitions at every length produced identical output IDs and MTP statistics, without
EOS/EOG. Each prompt preserves the task and assistant suffix; its source body and acceptance
rate differ, so the 8K result does not imply that longer context is intrinsically faster. These
are fixed output windows, not complete-program quality scores.

At 85K, the same-input peer-off control below measured 79.18 tok/s: direct P2P improves committed
decode by **5.05%**, with all 513 output IDs and MTP statistics identical across both paths and
repetitions. The resident request averages 65.061 s prefill, 6.155 s decode and 71.220 s total;
model loading takes 19.662 s once, outside request timing. The historical 78.424 figure below
used the pre-reboot corpus and is not the causal P2P baseline.

#### Qualified NVFP4 prefill update

A matched 85K-input / 180K-capacity MTP3 comparison at chunk=2,560, with two cold requests
per implementation, measures **1,297.05 ± 2.21 → 1,311.92 ± 2.06 prefill tok/s (+1.15%)**
for the wider TP2 MLP tile. All 513 output IDs and acceptance counters match. Decode measures
81.63 → 81.71 tok/s; this does not establish a stable decode improvement. Chunk=2,560 was used
because desktop VRAM occupancy made larger chunks fail the startup allowance during this run.
The earlier chunk=3,072 table is not its matched control. Independent numerical criteria were
not relaxed; no new lossy attention option qualified for delivery.

#### NVFP4 v3 capacity sweep: fixed 512-token input

Only the maximum context changes. TP2, INT8 KV, greedy MTP3, optimized draft head, CUDA Graphs
and 1,024-token chunks stay fixed. Each capacity uses one discarded warmup and three measured
requests, with 512 input tokens and 256 timed decode tokens (257 total output tokens).
Decode here uses first-token-to-completion **wall time**, not just Engine decode-phase time.

| Maximum context | Wall decode tok/s | Prefill tok/s | MTP acceptance |
|---:|---:|---:|---:|
| 1,024 | 106.12 ± 0.01 | 1,443.78 | 70.04% |
| 2,048 | 106.12 ± 0.04 | 1,438.44 | 70.04% |
| 4,096 | 105.96 ± 0.02 | 1,440.81 | 70.04% |
| 8,192 | 105.98 ± 0.01 | 1,442.96 | 70.04% |
| 16,384 | 106.06 ± 0.04 | 1,443.37 | 70.04% |
| 32,768 | 106.00 ± 0.03 | 1,443.77 | 70.04% |
| 65,536 | 106.01 ± 0.06 | 1,443.42 | 70.04% |

All seven capacities were allocated exactly. All 21 measured runs produced the same 257 IDs
and acceptance statistics: 173/247 drafts accepted over 83 rounds per request. This is a
short-input capacity check, not a 64K-filled-context result.

#### Communication and regression checks

The 10 KiB BF16 all-reduce measured 26.6085 µs mean, 25.901 µs p50 and 43.910 µs p99 over
500 host-synchronized iterations. The matched peer-off path measured 47.7961 µs mean:
**44.33% lower communication latency**, not 44.33% end-to-end inference improvement. Both
paths passed exact-transfer, uneven-shape, guard and 64-consecutive-round checks.

Q4_K_M and NVFP4 v3 passed the real TP2 MTP and prefix-cache regressions. Graph/eager outputs,
logits, acceptance and retained frontiers agree; each artifact had zero disagreements at
64 teacher-forcing positions. At a 3,274-token prompt, median cold/cached TTFT was
3.33492 s / 16.8689 ms for Q4_K_M and 3.01954 s / 14.6475 ms for NVFP4.
See [regression qualifications](docs/performance.md#p2p-enabled) for the existing Q4 cold/cache
near-tie differences and the limits of these checks.

#### DFlash2 v3 route (separate capacity)

The optional five-layer v3 drafter is measured separately from the 180K MTP acceptance profile.
The delivered route uses 98,304 capacity, 1,024-token chunks, full target INT8 KV,
TP2, greedy sampling, CUDA Graphs and DFlash7. It is text-only and needs the optional v3 drafter.
On the saved 85,000-token LRU-code input, one cold request per window measures:

| Timed committed tokens | Prefill tok/s | Decode tok/s | Accepted/drafted |
|---:|---:|---:|---:|
| 512 | 1,185.52 | **99.93** | 422/623 (67.74%) |
| 2,048, forced window | 1,182.63 | **95.87** | 1674/2612 (64.09%) |

The 512-token window has no EOS/EOG. The forced 2,048-token window continues past `<|im_end|>`
at total output token 1,150 because benchmark stopping is disabled; it is sustained execution
data, **not useful long-code completion speed**. All 513/2049 IDs and acceptance counters match
the preceding argmax-only route. The single-run sharded-selector differences (0.6–0.8%) do not
establish a stable speedup. Independent selector and real graph/eager / 64-position
teacher-forcing checks pass; these are not universal model-quality scores.

On the separate 85K high-code corpus, single forced 2,048-token windows measure DFlash3/5/7
at 79.30/76.10/87.91 tok/s versus MTP3 at 86.14, all at 98,304 capacity and chunk=1,024.
These windows continue past the first `<|im_end|>` at total output token 942 (917 for DFlash5),
so they are stress measurements, not useful completion speeds. DFlash3/7 match the MTP output
IDs; DFlash5 does not. DFlash7 also takes longer for the complete resident request
(95.40 versus 92.92 seconds). Fresh 1,024-token windows are slower; larger experimental drafts
9/11/15 brought no benefit and were removed. The supported 27B maximum remains seven.

Earlier P2P-enabled v3 snapshots, before these vocabulary-transfer reductions: 3,072 input / 512
timed output tokens measured 143.06 tok/s (two runs). A different 85K code corpus measured DFlash7
73.38 tok/s (47.18% acceptance) versus MTP3 83.13 tok/s (78.56%). Different corpora, windows and
acceptance prevent using those numbers or the peer-off MTP table as the current run's speedup
baseline. See [DFlash method and reproduction](docs/performance.md#dflash2-v3-separate-capacity).

#### Stop-aware non-code sweep

The same v3 artifact was tested with story, translation, 32-record JSONL and logic tasks:
**80 requests**, fixed 98,304 capacity / chunk 1,024 / TP2 / INT8 KV / greedy / CUDA Graphs,
no prefix reuse, and model-default stopping enabled. All requests finish at the first model
end token rather than continuing past EOS. D/M below means **DFlash7 / MTP3**, in committed
decode tok/s. Native prompt sizes are 129 / 395 / 118 / 417 tokens in column order; their
rates are two-run means. Every other cell is one measured request, not a repeated mean.

| Actual input tokens | Story D/M | Translation D/M | JSONL D/M | Logic D/M |
|---:|---:|---:|---:|---:|
| native | 49.20 / 74.83 | 134.68 / 123.00 | 210.50 / 137.09 | 158.88 / 126.10 |
| 1024 | 51.24 / 76.66 | 127.44 / 117.58 | 206.30 / 135.98 | 157.87 / 125.01 |
| 2048 | 51.09 / 74.41 | 128.45 / 116.14 | 200.78 / 133.63 | 173.32 / 126.34 |
| 4096 | 47.60 / 70.98 | 116.31 / 113.49 | 201.09 / 133.82 | 162.94 / 125.98 |
| 8192 | 49.11 / 70.17 | 125.74 / 116.00 | 198.69 / 132.06 | 166.37 / 125.39 |
| 16384 | 44.80 / 68.16 | 112.08 / 110.15 | 192.60 / 128.33 | 161.27 / 120.91 |
| 32768 | 41.44 / 63.23 | 110.56 / 108.44 | 175.01 / 119.59 | 142.03 / 111.83 |
| 65536 | 36.02 / 55.63 | 85.29 / 89.32 | 148.14 / 104.67 | 122.20 / 98.14 |
| 85000 | 32.26 / 51.67 | 79.61 / 84.86 | 137.65 / 98.77 | 111.05 / 91.60 |

JSONL is faster with DFlash at every tested length; at 85K, acceptance is 98.30% and decode
is 39.37% faster with identical output IDs. Story acceptance is only about 12–14%, making
DFlash slower than MTP. DFlash prefill is slower throughout this sweep, so faster decode
does not imply a faster complete request: at 85K, logic takes 78.47 / 76.87 seconds D/M.

All JSONL exact checks and logic CHECKs pass; translations pass section/glossary checks only.
Some stories violate length/literal requirements, including a missing `ORCHID-37` in the
native DFlash story. Only 30 of 36 task/length pairs have identical output IDs, so this is
not a universal token-parity or no-quality-loss result. All four 85K pairs match exactly.
[Full prefill, acceptance, output-length and whole-request tables](docs/performance.md#stop-aware-non-code-workloads-dflash7-versus-mtp3)
include the per-row output qualifications.

### P2P disabled

This area covers verified CUDA-managed UVA D2D staging, not an explicit NInfer pinned-host
copy branch. Before `iommu=pt`, the host's translated `DMA-FQ` domains prevented direct P2P.
The current peer-off A/B control uses a process-local CUDA capability-query shim, without
changing system settings. It is a diagnostic, not an advertised CLI switch.

#### NVFP4 v3 same-input 85K control

The same 85,000 input IDs, 180,000 capacity, 512 timed decode tokens, INT8 KV, TP2, greedy MTP3,
optimized draft head and CUDA Graphs as the P2P-enabled run, with two repetitions:

| Prefill tok/s | Committed decode tok/s | Wall decode tok/s | MTP acceptance |
|---:|---:|---:|---:|
| 1,297.94 ± 2.66 | 79.18 ± 0.014 | 79.151 | 79.12% |

Both communication paths accepted 360/455 drafts over 152 rounds per request. All output IDs
and speculative fields match. The same-input improvement is 5.0459% decode / 0.6578% prefill.

#### Earlier official NVFP4 v3 artifact validation

`neroued/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer` is a 23.72-GB, 1246-physical-object
`NINFER\x00\x03` container. Its projected `qwen3.8-27b/nvfp4` identity loaded on both V100s with
INT8 KV and MTP3 at `--max-context 180000` (180032 allocated KV positions); 671 tensors and six
frontend resources were materialized per the normal Engine path. A short greedy MTP check produced
21 tokens in five rounds with 100% acceptance (120.66 committed tok/s at 1024-token capacity).
The short check is a compatibility smoke test, not a long-context performance claim. A stable
512-prompt/512-output benchmark measured 1,344.24 ± 33.15 prefill tok/s and 98.831 ± 0.027
committed decode tok/s. At the acceptance workload (85,000 occupied tokens, 180,000 capacity,
512 output tokens), the same v3 artifact measured 1,277.61 ± 3.12 prefill tok/s and 78.424 ± 0.0004
committed decode tok/s over two repetitions before P2P was enabled; MTP3 acceptance was 79.12%.
These earlier runs use the pre-reboot corpus, not the current same-input A/B control.
The 120.66 figure is therefore a real short-window peak, not the representative decode rate.

#### Earlier Q4_K_M prefill

TP2, INT8 KV, `prefill_chunk=4096`, and 180,000-token capacity:

| Occupied prompt | Prefill throughput | Measurement |
|---:|---:|---|
| 8,192 tokens | 1,672.9 tok/s | one cold run |
| 85,000 tokens | 1,251.44 ± 2.87 tok/s | two cold runs |

The 85K prompt exceeds the requested 1,000 tok/s target. On the 8K probe, chunk sizes
1,024/2,048/3,072/4,096/5,120/8,192 measured 1,454.4/1,599.3/1,636.1/1,672.9/1,672.3/1,195.2
tok/s. For the 85K corpus, chunks 1,024/2,048/4,096 measured 1,133.0/1,213.4/1,251.0 tok/s.
The 4,096-token chunk reserves 1.49 GiB per device. These are prefill measurements, not decode
rates.

The staging TP2 collective path was also run independently on the two V100s: a 10 KiB BF16
all-reduce (the decode-shaped hidden block) measured 48.16 µs mean, 47.09 µs p50 and 64.92 µs p99
over 500 host-synchronized iterations. The test passed all exact-value, guard, uneven-shape and
64-consecutive-round checks. Weight quantization cannot eliminate the fixed collective schedule;
this microbenchmark alone does not establish a hardware-only decode ceiling.

#### Earlier Q4_K_M decode

With exactly 85,000 occupied prompt tokens, 180,000 capacity, TP2, INT8 KV, MTP3, optimized draft
head and CUDA Graphs, the pre-P2P staging path measured two 128-token windows:

| Repetitions | Prefill rate | Committed decode rate |
|---:|---:|---:|
| 2 | **1,251.44 ± 2.87 tok/s** | **50.68 ± 0.04 tok/s** |

Both windows used the same 85K raw token corpus and produced no EOS/EOG; aggregate MTP acceptance
was 78.07%. A separate
512-token occupied prompt at the same capacity
measured **60.0291 ± 0.0571 tok/s** over three 256-token decode windows; this is not an 85K
occupied-context result.

The superseded explicit pinned-host experiment reported **53.4075 ± 0.0639 tok/s** over three
512-token windows; it is not the retained transport implementation and is historical context only.
An earlier matched diagnostic used LM Studio CUDA 2.33.0's automatic GPU split with the same
85,000 prompt IDs, source GGUF, greedy sampling, Q8 KV, maximum context 180,000 (backend rounded
to 180,224), and max-three/min-zero MTP. It measured 35.4977 tok/s in one run. LM Studio's
single-run result is a diagnostic, not a repeated comparison.
Earlier user-reported 45/57 tok/s figures lacked complete workload metadata and are not used as
measured acceptance results.

#### Earlier Q4_K_M maximum-context capacity sweep

This earlier Q4_K_M sweep changes only the configured maximum context. It uses a 512-token code prompt,
a 256-token decode window, greedy sampling, Q8/INT8 KV and MTP3. Each cell is the mean ± sample
standard deviation of three measured runs after one warmup; prompt occupancy is only 512 tokens.
NInfer uses TP2 and its optimized draft head. LM Studio CUDA 2.33.0 uses automatic two-GPU
splitting with maximum-three/minimum-zero drafts.

| Maximum context | NInfer decode tok/s | LM Studio decode tok/s |
|---:|---:|---:|
| 1,024 | 60.06 ± 0.02 | 62.67 ± 0.24 |
| 2,048 | 60.12 ± 0.02 | 62.78 ± 0.05 |
| 4,096 | 60.11 ± 0.03 | 62.69 ± 0.10 |
| 8,192 | 60.13 ± 0.06 | 62.78 ± 0.07 |
| 16,384 | 60.14 ± 0.04 | 62.63 ± 0.06 |
| 32,768 | 60.16 ± 0.05 | 62.58 ± 0.004 |
| 65,536 | 60.06 ± 0.05 | 62.49 ± 0.16 |

Both engines honored all seven capacities. The output IDs were identical within each engine and all
windows were EOS/EOG-free. With this short prompt neither engine slowed materially as the capacity
increased; LM Studio was about 4% faster. This table is a capacity-setting comparison, not an
85K-filled-context benchmark.

#### Earlier NVFP4 at 512-token input

The corrected NVFP4 implementation uses a 512-token code prompt and 256 timed decode tokens, with one
warmup and three measured requests per capacity. TP2, INT8 KV, greedy MTP3, optimized draft head,
CUDA Graphs and 1,024-token prefill chunks are fixed. Decode rates are mean ± sample standard deviation;
decode uses first-token-to-completion wall time.

| Maximum context | NVFP4 decode tok/s | Prefill tok/s | MTP acceptance |
|---:|---:|---:|---:|
| 8,192 | 97.65 ± 0.08 | 1,369.9 | 70.04% |
| 16,384 | 97.67 ± 0.09 | 1,370.9 | 70.04% |
| 32,768 | 97.76 ± 0.11 | 1,370.8 | 70.04% |
| 65,536 | 97.55 ± 0.15 | 1,369.9 | 70.04% |

All four requested capacities were allocated exactly. All 12 measured runs produced the same
257 output IDs, without EOS/EOG; each accepted 173/247 drafts over 83 rounds. These are short-input
capacity checks, not 8K–64K occupied-context decode. The LM Studio table above uses Q4_K_M;
it is not a matched NVFP4 comparison.

#### Earlier NVFP4 at 85K occupied context

Qwen3.8-27B NVFP4, exactly **85,000 prompt tokens**, **180,000 capacity**, TP2, complete INT8
group-64 KV, greedy sampling, MTP3, optimized draft head, CUDA Graphs and 3,072-token prefill chunks:

| Implementation | Runs | Prefill tok/s | Committed decode tok/s | MTP acceptance |
|---|---:|---:|---:|---:|
| FP32 SwiGLU, shared attention scores/reducer weights, prepared GDN Q/K | 2 | **1,279.44 ± 2.94** | **78.36 ± 0.04** | **79.12%** |

Each run generates 513 tokens: one from prefill and **512 timed decode tokens**. The optimized
runs produce identical IDs to each other, with 360/455 accepted drafts over 152 rounds per run.
These exceed the requested 1,000 prefill / 70 committed decode tok/s targets at 85K occupancy.
The 512-token output limit is a throughput window, not a completed-program quality evaluation.
The complete attention operator passes its independent FP64 oracle, including all 24 heads and
four queries at 85K with a 180K envelope. This verifies the measured numerical and generation
behavior; it is not a general coding-quality evaluation.
The GDN path also passes the independent FP64 recurrence oracle for its outputs and final state,
including 3,072 tokens with the real TP2 head geometry and nonzero initial state.

Earlier NVFP4 results, including **78.90 tok/s**, used a wide SwiGLU path that rounded gate/up to
BF16 before SiLU and multiplication. Expanded independent FP64 tests exposed excessive error in
both FP8 and NVFP4 routes. Keeping those intermediates in FP32 fixes the failing tests, including
both TP2 shards at 3,072 tokens, without widening tolerances. The generated sequence changes;
the old figures are real measurements but are **not the current numerical baseline**. Measurements
taken with the earlier mismatched FP8 output-head path are also excluded. A faster experiment
that reassociated the FP32 sum changed this corpus's generated sequence; the delivered reducer
preserves the original order and introduces no additional quantization or approximate attention.
The 3,072-token chunk reserves 1.40 GiB of workspace per device; 4,096 does not fit this NVFP4
artifact together with 180K capacity on the measured host.

This earlier staging request averages 1.57 ms of prompt preparation, 66.435 s of prefill and 6.534 s of
decode, or 72.974 s total. The same invocation loads the resident model once in 18.65 s, including
16.25 s of upload. These are raw token-ID inputs, so preparation does not include text tokenization.
The complete per-stage GPU breakdown, MTP verify/proposal costs, copy activity, timing gaps and
remaining optimization decisions are in [the staging performance ledger](docs/performance.md#nvfp4-full-request-ledger).
Current P2P reproduction commands are in [performance methodology](docs/performance.md#p2p-enabled).

An earlier NVFP4 TP2 comparison against the `plus1998/Ninfer-V100-Duo` code path at a 3K prompt,
98,304 capacity and MTP3 measured 976 tok/s prefill / 69.22 tok/s decode in this fork versus
973.5 / 68.96 tok/s upstream. This is a short-input cross-check only, not an 85K result. The
upstream repository and its reported numbers should not be treated as measurements of this fork.

#### Historical DFlash2 experiment

The old fixed-85K figures (DFlash3 **25.19 tok/s**, DFlash7 **20.52 tok/s**) used the pre-v3
container and old KV layout. They are retained only as historical context; the current v3 result
is the 98,304-capacity measurement above, and 180K DFlash is intentionally not advertised.

## Build and run

Requirements for this profile: 64-bit Linux, NVIDIA driver, CUDA 12.8, CMake 3.28+, C++20 host
compiler, Ninja, pkg-config, FFmpeg development libraries (`libavformat >= 60`, `libavcodec >= 60`,
`libavutil >= 58`, `libswscale >= 7`) and libcurl >= 7.85. Build dependencies locally when the
system versions do not meet those requirements:

```bash
tools/v100/build_dependencies.sh
PKG_CONFIG_PATH="$PWD/build/_deps/install/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
cmake -S . -B build-v100 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build-v100 -j2
```

Convert the local LM Studio source model once (Python 3.11 with NumPy):

```bash
python3 -m tools.convert.qwen3_8_27b.convert_gguf \
  --model /Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj /Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-BF16.gguf \
  --out /Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer
```

Run a request with the V100X2 defaults (devices `0,1`, 180K capacity, 4K prefill chunks, INT8 KV,
MTP3 and optimized draft head):

```bash
tools/v100/ninfer-v100x2.sh \
  --prompt "Explain prefill and decode in three sentences." \
  --max-new 128 --greedy --no-thinking
```

Set `NINFER_V100X2_ARTIFACT`, `NINFER_V100X2_DEVICES`, `NINFER_V100X2_MAX_CONTEXT`,
`NINFER_V100X2_PREFILL_CHUNK`, `NINFER_V100X2_KV_DTYPE`, or
`NINFER_V100X2_DRAFT_TOKENS` to override the launcher defaults. The launcher does not constrain
host CPU affinity; benchmark CPU use should remain below the operator's 85% ceiling.

The current P2P corpus is local at `profiles/bench/v100-code-85000-iommu-pt.ids`; earlier staging
measurements used `/tmp/v100-code-85000.ids`. Corpora, model artifacts and raw profiler reports
are not included in the repository. See [performance methodology](docs/performance.md) for
corpus generation, benchmark commands and additional qualifications.

## Verification

The CUDA 12.8 `sm_70` build was completed with `-j4`. The all-reduce integration test passed,
including exact cross-device byte probes and the 500-iteration 10 KiB microbenchmark. The focused
`nvfp4_a16`, `fp8_a16`, `swiglu_nvfp4`, and `swiglu_fp8` suites passed. Both SwiGLU suites check
full and TP2 shard outputs directly against the same FP64 mathematical oracle, including wide
prefill and caller-workspace sizing. FP8 A8 and NVFP4 A4 suites intentionally
report unsupported on V100 because those tensor-core routes require newer architectures; they are
not part of the V100X2 execution profile.

The decode update also passes `ninfer_gqa_attention_test` and `ninfer_attention_headlocal_test`,
covering BF16/INT8 caches, TP1/TP2, masks, fragmented pages, cache mutation and the full 85K FP64
oracle. FP8 regression cases cover native and prepacked full/TP2 vocabulary heads, single-token
calls and chunk tails. Both FP8 and NVFP4 prefill tests check native/prepacked real matrix shapes
at T=128/1024/4096 against the original represented weights. Q5 LinearAdd passes the FP64 oracle
at both TP2 row-shard extents, including its interior workspace maximum; BF16 Linear passes at
both row and column shards. The two-card LinearAdd and SwiGLU composition suites pass as well.
CLI, server and benchmark are rebuilt with these changes.

## Acknowledgements and license

V100X2 implementation work builds on [Neroued/ninfer](https://github.com/Neroued/ninfer), the
RTX 3060 TP2 fork, and [geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100).
Volta NVFP4 and TP2 behavior was cross-checked against
[plus1998/Ninfer-V100-Duo](https://github.com/plus1998/Ninfer-V100-Duo). The prefill investigation
consulted [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM). DFlash2 follows
[Inco AI's implementation](https://inco.ai/blog/dflash2/) and its
[Qwen3.8-27B draft checkpoint](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2). KVMem was
reviewed but not integrated: selective-history retrieval changes attention semantics, while this
profile retains full-context attention.

The measured models derive from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
The NVFP4 artifact uses the mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4), packaged by
[Neroued](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer).

NInfer and this fork are licensed under [Apache-2.0](LICENSE). See [NOTICE](NOTICE) for required
attribution; third-party dependencies retain their own license files.
