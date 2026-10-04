# TP4: Tensor Parallelism over up to four V100s

Status: implemented.
Scope: extend the Qwen3.8-27B engine from TP2 (2-GPU tensor parallelism) to TP4,
targeting 4× Tesla V100-SXM2-16 GB (`sm_70`). The same code serves TP1, TP2, and TP4.

This document is the design record: what already generalized, the sites that assumed exactly
two ranks, and the two subsystems — the cross-rank collectives and the CUDA-Graph path — that
had to become genuinely N-way. Two design choices here supersede the original plan and are called
out explicitly:

- cross-rank communication is **NCCL** (one communicator per rank, grouped per-rank calls from the
  single engine thread), replacing the bespoke event-based pull; and
- each rank captures its **own** CUDA Graph, with the NCCL collectives recorded as nodes in it,
  replacing the single cross-device graph that forked a peer stream into rank 0's capture.

---

## 1. Goal and hardware

**Goal.** Run one resident Qwen3.8-27B model across **four** V100s (TP4) through the same public
`.ninfer` Engine route, preserving the existing numerical contract and CUDA-Graph decode path.

**Hardware on this box (`nvidia-smi`).**

| CUDA index | GPU | sm | note |
|---|---|---|---|
| 0 | Quadro K620 (2 GB) | 5.0 | **avoid** — cannot run `sm_70` kernels |
| 1, 2, 3, 4 | Tesla V100-SXM2-16 GB | 7.0 | usable |

A TP4 run uses `CUDA_VISIBLE_DEVICES=1,2,3,4` so the app sees devices `0,1,2,3`.

**Why TP4 helps.** At TP2 the 27B Q4_K_M weights occupy ≈ 10.1 GiB per GPU. Splitting the same
weights four ways drops that to ≈ 5.0 GiB per GPU, freeing roughly 5 GiB/GPU for KV cache →
substantially larger context capacity or higher occupancy on the same cards. TP4 is therefore a
*capacity* play as much as a throughput one. Measured on the NVFP4 identity: weights fall to
5.63 GiB/GPU with 8.29 GiB free after startup.

---

## 2. Headline finding: the model split is already N-generic

The weight/head/state partitioning did not need to be re-derived for TP4. Every sharded quantity
is already expressed as `/tp` and every shard-map builder already loops `for (device < tp)`:

- `even_chunk(total, tp, device)` — `src/targets/qwen3_6_27b/impl/load/bindings.cpp` — is a
  generic contiguous-block splitter; rank `r` owns rows `[r·total/tp, (r+1)·total/tp)`.
- `append_column_block` / `append_row_parallel` / `append_channel_block` loop
  `for (int device = 0; device < tp; ++device)` and only use the head count as a `% tp == 0`
  divisibility guard.
- Every sharded extent in `build_device_view` is written `/tp`: attention output `6144/tp`, GDN
  `a_log`/`dt_bias` `48/tp`, convolution `10240/tp`, GDN output `6144/tp`, `output_head`
  `248320/tp`, `draft_head` `131072/tp`, MTP `10240/tp`, `14336/tp`.
- GDN / linear-attention recurrent state is **sharded per rank, not replicated**:
  `src/targets/qwen3_6/impl/runtime/layouts_impl.h` (`kv_heads = kv_heads/tp`,
  `value_heads = gdn_value_heads/tp`). GDN heads are per-head independent, so each rank updates
  **only its own** head-shard state — **no cross-rank sync on the recurrent state**.

The entire effort was (a) widening a handful of constants, fixed-size arrays, and parser
whitelists that were mechanically pinned to `2`, and (b) generalizing the two bespoke two-party
subsystems.

### Divisibility at tp = 4 (all clean)

Model geometry (`src/targets/qwen3_6_27b/impl/config.h`): `hidden=5120`, `layers=64`
(16 full-attention + 48 GDN), `query_heads=24`, `kv_heads=4`, `head_dim=256`,
`gdn_key_heads=16`, `gdn_value_heads=48`, `output_rows=248320`.

| Quantity | Full | /4 | heads per rank | integral? |
|---|---|---|---|---|
| attention Q section | 6144 | 1536 | 6 of 24 | ✓ |
| attention K/V section | 1024 | 256 | 1 of 4 | ✓ |
| GDN key_dim | 2048 | 512 | 4 of 16 | ✓ |
| GDN value_dim | 6144 | 1536 | 12 of 48 | ✓ |
| GDN convolution_dim | 10240 | 2560 | — | ✓ |
| GDN a_log/dt_bias | 48 | 12 | — | ✓ |
| output_head rows | 248320 | 62080 | — | ✓ |
| draft_head rows | 131072 | 32768 | — | ✓ |

No section lands on a non-integer at `tp=4`, so the existing `% tp == 0` guards accept the plan.
(A future model whose head counts are not divisible by 4 is rejected at plan time by the same
guards — fail-fast.)

### Coexistence: tp ∈ {1, 2, 4} share one implementation

`tp` is a **runtime** value selected at startup. Adding TP4 widened the allowed set to `{1, 2, 4}`
under a compile-time cap of `kMaximumDevices = 4`; it did not replace TP2. A process runs one
model at one tp (`--tp 2 --devices X,Y` or `--tp 4 --devices W,X,Y,Z`).

The design principle that makes coexistence free is **one N-generic implementation per subsystem,
parameterized by the runtime `tp` — not a separate TP2 path plus a TP4 path**:

- The size-`kMaximumDevices` (4) arrays hold both: a tp=2 run uses entries `[0],[1]`, tp=4 uses
  `[0..3]`.
- The NCCL collective (§3.5) runs the same grouped per-rank call at every tp: a two-rank exchange
  at `tp=2`, a no-op at `tp=1`.
- The per-rank graph (§3.6) captures one graph per rank at every tp; a tp=2 run captures two
  graphs, a tp=4 run captures four.

Because the NCCL implementation **replaced** the two-party pull rather than sitting beside it,
TP2 was re-verified after the change (numerics + perf) as a regression gate — see §5.

---

## 3. What changed, layer by layer

Legend: **[M]** mechanical (constant/array/whitelist bump), **[moderate]** structural but
localized, **[hard]** bespoke two-party logic that became genuinely N-way.

### 3.1 Root constants and the fixed-size device arrays — [M]

Three roots drive every fixed-size array. Bumping them cascaded to all the `(a)`-class sites:

| Site | Change |
|---|---|
| `src/artifact/binder.h` | `kMaximumDevices = 2` → `4` |
| `src/core/device.h` | `ExecutionContext::dev` array literal `2` → `4` |
| `include/ninfer/types.h` | public memory-report `devices` array `2` → `4` |

Cascading fixed arrays that size off these auto-fixed once the roots grew to 4:
`ShardPlacement::device_ranges`, `MaterializationPlan::device_capacity_bytes`,
`ObjectStorage` device/arena arrays, `Slot::events`/`Slot::pending`, and the ExecutionContext
materializer overload.

**Not affected:** the byte-movement machinery was already an N-loop. The H2D fan-out, the
dest-disjointness check, and binder placement all iterate `for (device < device_count)`.

### 3.2 Parsers, validation messages, and the startup device checks — [M]

- `apps/cli/options.cpp` `parse_tp` whitelist `{1,2}` → `{1,2,4}`; `parse_devices` cap `> 2` → `> 4`.
  The cross-check `devices.size() == tp` and the `tp==1` synthesis are already generic.
- `src/serve/serve_options.cpp` — identical whitelist/cap change.
- Error strings that hardcoded “1 or 2” now reflect the new bound.
- `src/core/device.cu` ctor message and the pairwise compute-capability check generalized to an
  all-pairs loop over `tp` devices (every pair must agree on `sm_70`).

### 3.3 View management in `LoadedModelData` — [moderate]

“One primary + one optional peer” became “an array of N views”:

- the ctor’s `if (tp != 1 && tp != 2) throw` and the two fixed `build_device_view` calls became a
  loop building one view per device (`src/targets/qwen3_6_27b/impl/load/bindings.cpp`);
- `RuntimeModelView runtime; std::optional<RuntimeModelView> runtime_peer;` became a
  `std::array<RuntimeModelView, kMaximumDevices>` indexed by rank
  (`src/targets/qwen3_6_27b/impl/load/bindings.h`);
- `view(rank)` indexes the array with a bounds check against `tp` instead of the
  `rank==0`/`rank==1` special cases.

`build_device_view` itself is already per-device and needed no change beyond being called in a
loop; rank differences come solely from the `device` argument (which arena + which `/tp` extents).

### 3.4 Execution-leaves typed to exactly two ranks — [moderate, broad]

The execution leaves that were typed to a 2-element array and looped `rank < 2` became
`std::array<Tensor, kMaximumDevices>` with `for (rank < tp)`:

- Op signatures (`include/ninfer/ops/`): `attn_input_proj.h`, `linear.h`, `linear_swiglu.h`,
  `gdn_input_proj.h`, `gdn_gating_proj.h`, `linear_add.h`.
- Implementations (`src/ops/wrapper/`): `attn_input_proj.cpp`, `gdn_input_proj.cpp`,
  `linear_swiglu.cpp`, `gdn_gating_proj.cpp`, `linear_add.cpp`.
- The target split-leaf forms in `src/targets/qwen3_6_27b/impl/variant.{h,cpp}`.

The sibling package `src/targets/qwen3_6_35b_a3b/impl/variant.h` uses the identical pattern and
would need the same generalization if it is to support TP4 (out of scope for this deliverable).

### 3.5 Collectives: NCCL, grouped per rank — [hard]

Cross-rank communication is one rank-generic NCCL implementation in
`include/ninfer/ops/allreduce.h` + `src/ops/common/allreduce.cu`, replacing the previous implicit
two-party exchange over `cudaMemcpyAsync(DeviceToDevice)`.

- **Communicators.** `CollectiveComms` owns one `ncclComm_t` per rank
  (`std::array<ncclComm_t, kMaximumDevices>`), constructed from the `ExecutionContext`’s devices
  via `ncclCommInitRank`. `allreduce_sum` (row-parallel output projection: 2/layer → 128/decode
  token), `allgather_rows` (vocab-split `output_head` gather → 1/token), and the rank-0-owned
  `gather_columns_rank0` / `broadcast_rank0` take `const CollectiveComms&`.
- **Grouped single-thread issue.** Because the engine is single-threaded and drives every
  communicator from that one thread, each collective is issued as
  `ncclGroupStart()` → one call per rank on that rank’s own `DeviceContext::stream` →
  `ncclGroupEnd()`. Grouping is mandatory: without it the first collective deadlocks waiting for a
  peer call the same thread has not yet issued. Each rank’s call names its own rank’s
  communicator/stream, so the same loop is a two-rank exchange at `tp=2`.
- **Transport.** On the V100-SXM2 mesh NCCL uses the NVLink P2P fabric directly (measured
  ~48 GB/s per direction on every ordered pair). This keeps TP2 and TP4 traffic off the PCIe/IOMMU
  staging path the old UVA-D2D route used. (The `enable_peer_access` helper was removed.)
- **Reduction and oracle.** `allreduce_sum` is a plain in-place `ncclAllReduce(..., ncclSum, ...)`
  on BF16. It is **not** guaranteed bitwise-identical across ranks; that is consistent with the Op
  oracle (an FP64 sum of the BF16 partials rounded once to BF16) and with reference tensor-parallel
  implementations. Rank-to-rank lockstep is preserved because every token decision is taken on
  rank 0 and published to the peers with `broadcast_rank0`.
- **Dtype.** `allgather_rows` / `gather_columns_rank0` / `broadcast_rank0` are dtype-generic
  (the originals copied bytes). A `nccl_element_type(DType, op)` helper maps
  BF16/FP16/FP32/I8/U8/I32/I64/FP8_E4M3FN to the matching `ncclDataType_t`; element size is derived
  from the actual tensor dtype. Real callers pass I32/I64 (token ids, top-k indices, DFlash drafts)
  as well as BF16.
- **Capture.** The calls are stream-ordered GPU work with no host round trip or staging, so they
  are CUDA-Graph capturable. Ordering between consecutive collectives is NCCL’s own stream
  semantics; there are no cross-stream `cudaEvent`s, which is what lets each rank be captured into
  its own independent graph.

### 3.6 CUDA-Graph: one graph per rank — [hard]

The original plan kept the single cross-device graph (rank 1’s stream forked into rank 0’s
capture). That is unnecessary once the collectives are NCCL calls that are themselves
capturable: **each rank now captures its own graph**.

- `DecodeGraphDefinition::capture` (`src/core/decode_graph.cpp`) begins capture on every rank’s
  stream with `cudaStreamCaptureModeThreadLocal`, runs the body once, and ends capture per rank,
  yielding `ranks` independent `cudaGraphExec`s. `ThreadLocal` is what keeps a capture from
  hijacking another device’s activity on the same thread.
- The body issues each rank’s slice of the layer schedule on that rank’s stream; the NCCL
  collectives inside the body are recorded as nodes into whichever graph is capturing. Replay
  launches each rank’s executable on its own stream.
- A rank-0-only tail (ingress H2D copy, `ops::scatter`, `ops::sample`) runs after the per-rank
  body; `capture` sets `ScopedDevice::set(devices[0])` before the body so that tail launches on the
  origin device. (Without it, capture over multiple ranks left the current device on the last rank
  and the rank-0 tail launched with an invalid resource handle.)
- The old `DecodeGraphPeerBridge` fork/join event machinery is gone; there is no cross-device
  stream fork and no `cudaErrorStreamCaptureMerge` hazard.

Wiring: `src/targets/qwen3_6/impl/runtime/program_impl.h` builds the graph over `dev[0..tp−1]`
instead of asserting `(tp == 2) == (peer_model != nullptr)`; `graph_impl.h`’s `capture_graph`
loops all ranks and `run_prepared` launches each rank’s executable.

**Graph memory.** Per-rank graphs holding NCCL collective nodes cost more than the old event-based
graph: measured (4×V100, capacity 65536, NVFP4 27B, int8 KV) tp2 ~116 MiB/rank and tp4
~310–314 MiB/rank, versus ~26 MiB for the old tp2 graph. The plan budgets these in
`src/targets/qwen3_6/impl/runtime/layouts_impl.h`: ordinary per-batch allowance
`tp>=4 ? 448 MiB : (tp==2 ? 192 MiB : 12 MiB)` times concurrency; MTP/DFlash per-class allowances
`448 / 192 / (tp1 historical)` MiB. tp1 (no collectives) keeps its historical 12/82/64/96 MiB.

---

## 4. Phased implementation (as executed)

1. **Plumbing [M].** §3.1 + §3.2 + §3.3: widened the three root constants, the arrays, parsers,
   validation messages, all-pairs CC check, and turned `runtime_peer` into an N-view array with a
   `build_device_view` loop. Result: TP4 binds, loads, materializes, and runs eager decode.
2. **N-way collectives [hard].** §3.5: replaced the two-party pull with grouped per-rank NCCL, and
   generalized the op signatures (`std::array<Tensor, kMaximumDevices>`, `rank < tp`) per §3.4.
   Result: eager TP4 numerically correct.
3. **Per-rank graphs [hard].** §3.6: one graph per rank with NCCL nodes, capture mode
   `ThreadLocal`, origin device set before the body. Result: captured TP4 decode replays.
4. **Validation & perf.** §5.

---

## 5. Correctness and verification

- **Collective oracle.** `allreduce_sum` at tp=4: FP64 sum of the four BF16 partials, one
  round-to-nearest-even on store. `allgather_rows`/`gather_columns_rank0`/`broadcast_rank0`: exact
  relocation (no arithmetic).
- **Graph/eager parity + reproducibility.** `tests/targets/qwen3_6_27b/test_graph_tp2.cpp`
  over 128 greedy decode steps at TP2 with the NVFP4 artifact: the captured-and-replayed tokens
  are **identical** to eager, and two graph runs are bit-identical. This is what proves every
  rank’s stream was captured (a dropped rank changes the all-reduce and hence the tokens).
- **Cross-rank numerics.** TP2 and TP4 MTP greedy token streams are identical on the same prompt.
- **TP2 regression gate.** The unified NCCL path serves TP2 too, and the TP2 graph test above
  passes on the rewritten implementation.
- **End-to-end (NVFP4 identity; the Q4_K_M artifact is absent in this checkout).**
  Long context, capacity 65536, INT8 group-64 KV, MTP-3, 200 new tokens, ~60079-token prompt:
  TP4 **97.32 tok/s** eager (48.35% draft acceptance, prefill 2512 tok/s), TP4 85.33 tok/s
  captured, versus TP2 69.38 tok/s eager — ≈1.4× TP2 at length, above the 50 tok/s target.
  At short context TP4 MTP reaches ~164 tok/s eager / ~145 captured. TP4 ordinary (no MTP)
  decode is 54.69 tok/s captured versus TP2 40.14.

---

## 6. Risks and open questions

- **Where the risk was.** The model split (§2) needed no math change; effort and correctness risk
  concentrated in the collectives (§3.5) and the per-rank graph (§3.6). The single-thread NCCL
  grouping requirement and the capture-time current-device handling were the two concrete traps.
- **Divisibility.** tp=4 divides every current section cleanly. A future model whose head counts
  are not divisible by 4 is rejected at plan time by the existing `% tp == 0` guards.
- **Artifacts.** The checkout’s contract names the Q4_K_M identity, but that `.ninfer` artifact and
  its GGUF source are absent here; the TP4 mechanism and performance were therefore validated on
  the NVFP4 identity (same Engine route, same sharding). The exact Q4_K_M acceptance number was
  not measured in this checkout.
- **Out of scope.** The 35B-A3B sibling package’s identical tp2 pattern, and any fallback
  non-NCCL collective path.
