# TP4 Design: Extending Tensor Parallelism from 2 to 4 V100s

Status: implemented (Phases 0-2; see §7 for what shipped and the phase-3 validation
blocker). This document remains the maintainer-level account of the TP4 extension.
Scope: extend the Qwen3.8-27B engine from TP2 (2-GPU tensor parallelism) to TP4,
targeting 4× Tesla V100-SXM2-16 GB (`sm_70`).

This document records what already generalized, the exact sites that assumed exactly
two ranks, and the phased path to a working, graph-captured TP4 decode. All `file:line`
references were against the checkout on branch `local/v100x2-dev` at the time of the
plan; the implementation is now on top of it.

---

## 1. Goal and hardware

**Goal.** Run one resident Qwen3.8-27B model across **four** V100s (TP4) through the same
public `.ninfer` Engine route, preserving the existing numerical contract and CUDA-Graph
decode path.

**Hardware on this box (`nvidia-smi`).**

| CUDA index | GPU | sm | note |
|---|---|---|---|
| 0 | Quadro K620 (2 GB) | 5.0 | **avoid** — cannot run `sm_70` kernels |
| 1, 2, 3, 4 | Tesla V100-SXM2-16 GB | 7.0 | usable |

So four `sm_70` devices are physically present (CUDA indices 1–4). TP4 is hardware-feasible
here; only the software caps the engine at two devices. A TP4 run would use
`CUDA_VISIBLE_DEVICES=1,2,3,4` so the app sees devices `0,1,2,3`.

**Why TP4 helps.** At TP2 the 27B Q4_K_M weights occupy ≈ 10.1 GiB per GPU (measured during
the smoke test). Splitting the same weights four ways drops that to ≈ 5.0 GiB per GPU, freeing
roughly 5 GiB/GPU for KV cache → substantially larger context capacity or higher occupancy on
the same cards. TP4 is therefore a *capacity* play as much as a throughput one.

---

## 2. Headline finding: the model split is already N-generic

The single most important fact for this design: **the weight/head/state partitioning does not
need to be re-derived for TP4.** Every sharded quantity is already expressed as `/tp` and every
shard-map builder already loops `for (device < tp)`. Concretely:

- `even_chunk(total, tp, device)` — `src/targets/qwen3_6_27b/impl/load/bindings.cpp:479` — is a
  generic contiguous-block splitter; rank `r` owns rows `[r·total/tp, (r+1)·total/tp)`.
- `append_column_block` / `append_row_parallel` / `append_channel_block`
  (`bindings.cpp:500–563`) loop `for (int device = 0; device < tp; ++device)` and only use the
  head count as a `% tp == 0` divisibility guard.
- Every sharded extent in `build_device_view` (`bindings.cpp:1078–1272`) is written `/tp`:
  attention output `6144/tp`, GDN `a_log`/`dt_bias` `48/tp`, convolution `10240/tp`, GDN output
  `6144/tp`, `output_head` `248320/tp`, `draft_head` `131072/tp`, MTP `10240/tp`, `14336/tp`.
- GDN / linear-attention recurrent state is **sharded per rank, not replicated**:
  `src/targets/qwen3_6/impl/runtime/layouts_impl.h:125` (`kv_heads = kv_heads/tp`), `:138` and
  `:158` (`value_heads = gdn_value_heads/tp`). GDN heads are per-head independent, so each rank
  updates **only its own** head-shard state — **no cross-rank sync on the recurrent state**.

Because of this, TP4 requires **no change to any model-math or sharding expression**. The entire
effort is (a) widening a handful of constants, fixed-size arrays, and parser whitelists that are
mechanically pinned to `2`, and (b) generalizing **two bespoke two-party subsystems** — the
event-based collectives and the cross-device CUDA-Graph bridge — from 2 ranks to N.

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

No section lands on a non-integer at `tp=4`, so the existing `% tp == 0` guards in the append
helpers will accept the plan. (This is a real constraint to re-check if a future model's head
counts are not divisible by 4.)

### Coexistence: both `tp=2` and `tp=4` are supported at once

`tp` is a **runtime** value selected at startup — the engine already runs tp=1 or tp=2 through the
same code. Adding TP4 widens the allowed set to `{1, 2, 4}` under a compile-time cap of
`kMaximumDevices = 4`; it does not replace TP2. A process runs one model at one tp
(`--tp 2 --devices X,Y` or `--tp 4 --devices W,X,Y,Z`).

The design principle that makes coexistence free is **one N-generic implementation per subsystem,
parameterized by the runtime `tp` — not a separate TP2 path plus a TP4 path**:

- The size-`kMaximumDevices` (4) arrays hold both: a tp=2 run uses entries `[0],[1]`, tp=4 uses
  `[0..3]`.
- The N-way collective (§3.5) **degenerates to the current two-party exchange at `tp=2`** and is a
  no-op at `tp=1`; the N-device graph bridge (§3.6) is origin + 1 peer at `tp=2` (today's exact
  topology) and origin + 3 peers at `tp=4`. One implementation serves tp ∈ {1,2,4}.

Consequence: because we *replace* the bespoke two-party pull with the unified reduce rather than
keeping it alongside, **TP2 must be re-verified after the change** (numerics + perf) as a
regression gate — see §5.

---

## 3. What changes, layer by layer

Legend: **[M]** mechanical (constant/array/whitelist bump), **[moderate]** structural but
localized, **[hard]** bespoke two-party logic that must become genuinely N-way.

### 3.1 Root constants and the fixed-size device arrays — [M]

Three roots drive every fixed-size array. Bumping them cascades to all the `(a)`-class sites:

| Site | Today | Change |
|---|---|---|
| `src/artifact/binder.h:16` | `inline constexpr std::size_t kMaximumDevices = 2;` | → `4` |
| `src/core/device.h:39` | `std::array<std::optional<DeviceContext>, 2> dev;` in `ExecutionContext` (literal `2`, not the constant) | → `4` (and ideally define it off one shared constant) |
| `include/ninfer/types.h:523` | `std::array<DeviceMemoryReport, 2> devices{};` in the public memory-report type | → `4` |

Cascading fixed arrays that size off these and auto-fix once the roots grow to 4:

- `src/artifact/binder.h:44` `ShardPlacement::device_ranges`
- `src/artifact/binder.h:67` `MaterializationPlan::device_capacity_bytes`
- `src/artifact/materializer.h:31,32` `per_device_h2d_bytes`, `per_device_capacity_bytes`
- `src/artifact/materializer.h:67,68,72,73` `ObjectStorage::device/device_bytes`,
  `device_arena_`, `physical_devices_`
- `src/artifact/materializer.cpp:64–65` `Slot::events` / `Slot::pending`
- `src/artifact/materializer.cpp:408` ExecutionContext overload's `devices` array

**Not affected:** the byte-movement machinery is already an N-loop. The H2D fan-out
(`materializer.cpp:149–175`, `:350–380`), the dest-disjointness check (`:263–276`), and binder
placement (`binder.cpp:95–106`) all iterate `for (device < device_count)`. No change there.

### 3.2 Parsers, validation messages, and the startup device checks — [M]

- `apps/cli/options.cpp:44–50` `parse_tp` whitelist `{1,2}` → `{1,2,4}`; `:65–67`
  `parse_devices` cap `> 2` → `> 4`. The cross-check at `:279–281` (`devices.size() == tp`) and
  the `tp==1` synthesis at `:286–288` are already generic.
- `src/serve/serve_options.cpp:75–81` and `:96–98` — identical whitelist/cap; same change.
- Error strings that hardcode “1 or 2” (`options.cpp`, `serve_options.cpp`,
  `binder.cpp:26–28`, `bindings.cpp:889`, `bindings.cpp:1063`) → reflect the new bound.
- `src/core/device.cu:45–49` ctor message “requires 1 or 2 device ids” and `:67–78` the
  `if (tp == 2)` **pairwise** compute-capability check → generalize to an all-pairs loop over
  `tp` devices (every pair must agree on `sm_70`).

### 3.3 View management in `LoadedModelData` — [moderate]

This is where “one primary + one optional peer” becomes “an array of N views”:

- `src/targets/qwen3_6_27b/impl/load/bindings.cpp:1063–1066` ctor:
  `if (tp != 1 && tp != 2) throw` and the two fixed calls
  `build_device_view(plan, 0, runtime); if (tp == 2) build_device_view(plan, 1, runtime_peer.emplace());`
  → replace with a loop building one view per device.
- `src/targets/qwen3_6_27b/impl/load/bindings.h:405–407`: the fields
  `RuntimeModelView runtime; std::optional<RuntimeModelView> runtime_peer;` → a
  `std::array<RuntimeModelView, kMaximumDevices>` (or `std::vector`) indexed by rank.
- `bindings.h:397–401` `view(rank)`: hardcoded `rank==0 → runtime`, `rank==1 → *runtime_peer`
  → index into the array with a bounds check against `tp`.
- `bindings.cpp:888–890` `bind_artifact` “tp must be 1 or 2” and `:863–865` the
  `shard.device < kMaximumDevices` guard auto-widen with the constant.

`build_device_view` itself (`bindings.cpp:1078–1272`) is already per-device and needs no change
beyond being called in a loop; rank differences come solely from the `device` argument (which
arena + which `/tp` shard extents).

### 3.4 Execution Ops typed to exactly two ranks — [moderate, broad]

The weight/head *math* generalizes, but every **execution leaf** is typed to a 2-element array
and loops `rank < 2`. These must become `std::array<Tensor, kMaximumDevices>` (or a span + count)
with `for (rank < tp)`:

- Signatures (`include/ninfer/ops/`):
  - `attn_input_proj.h:198,210,231`
  - `linear.h:164,170,209,215`
  - `linear_swiglu.h:184,194`
  - `gdn_input_proj.h:289,297,303`
  - `gdn_gating_proj.h:127,137`
  - `linear_add.h:138,145`
- Implementations (`src/ops/wrapper/`), each with `for (rank = 0; rank < 2; ++rank)`:
  - `attn_input_proj.cpp:419,432,462,487`
  - `gdn_input_proj.cpp:1314,1326,1351,1369`
  - `linear_swiglu.cpp:338,348`
  - `gdn_gating_proj.cpp:327,340,354,362`
  - `linear_add.cpp:388`
- The target split-leaf forms: `src/targets/qwen3_6_27b/impl/variant.h:129+` (“tp == 2 split
  forms”) and the corresponding `variant.cpp` leaves, all using `std::array<Tensor,2>`.

This is broad but mechanical. The sibling package `src/targets/qwen3_6_35b_a3b/impl/variant.h`
uses the identical pattern and would need the same generalization if it is to support TP4 (out of
scope for the 27B deliverable, noted for completeness).

### 3.5 Collectives: two-party pull → N-way reduce — **[hard]**

This is the first of the two genuinely hard parts. Today all cross-rank communication is a
bespoke **two-party** design in `include/ninfer/ops/allreduce.h` (+ `src/ops/common/allreduce.cu`):

- `PeerEvents` is a fixed `std::array<cudaEvent_t, 2>` (`allreduce.h:141–142`) — one
  `inputs_ready[r]` + one `pull_done[r]` **per rank**, for exactly two ranks.
- Every collective hard-requires `ec.tp == 2` via `require_two_devices`:
  - `allreduce_sum` (`allreduce.h:165`) — row-parallel output projection, **1 per layer**
    (mixer-out + MLP down = 2/layer) → **128/decode token**.
  - `allgather_rows` (`:190`) — vocab-split `output_head` gather → **1/token**.
  - `gather_columns_rank0` (`:201`) and `broadcast_rank0` (`:207`) — DFlash rank-0 paths.
- Transport is `cudaMemcpyAsync(cudaMemcpyDeviceToDevice)` over UVA pointers — **deliberately not**
  `cudaMemcpyPeerAsync`, which CUDA rejects inside a stream-capture region
  (`allreduce.h:22–24`). On this V100 topology direct P2P is disabled (translated IOMMU domain),
  so the driver's UVA-D2D staging path is what actually runs.
- The protocol is a **pull**: rank `r` reads the peer's source into rank-`r`-owned storage on its
  own stream, with two event records + two waits per call (`allreduce.h:29–48`). This “nothing a
  rank owns is ever written by the peer's stream” invariant is what makes back-to-back calls safe
  and the whole sequence graph-capturable.

**Design for N-way (recommendation).** Because batch-1 decode payloads are tiny (a row-parallel
partial is `[1, 5120]` ≈ 10 KiB BF16 / 20 KiB FP32), the reduce is **latency-bound, not
bandwidth-bound**, and rank 0 already plays a special role (`broadcast_rank0`). Two viable options:

- **Option A — star/tree through rank 0 (recommended for the first cut).** Phase 1: ranks 1..N−1
  each D2D-copy their partial to a rank-0-owned accumulator slot; rank 0 sums all N. Phase 2:
  rank 0 D2D-broadcasts the result back to ranks 1..N−1. This is the direct generalization of the
  existing two-party pull, reuses the same UVA-D2D + event pattern, and keeps rank 0 as the single
  aggregator (consistent with `broadcast_rank0`). Cost: rank 0 sees N−1 inbound + N−1 outbound
  copies — negligible at these sizes.
- **Option B — ring allreduce.** Each rank adds its received buffer and forwards; N−1 steps around
  a ring. Bandwidth-balanced (no single hot rank) but needs per-edge event bookkeeping for every
  hop. Preferable only if profiling ever shows the star's rank-0 fan-in as a bottleneck — unlikely
  at decode batch 1.

Whichever option is chosen, the invariants to preserve: UVA-D2D `cudaMemcpyAsync` transport (never
`cudaMemcpyPeerAsync`), all ordering via `cudaEventRecord`/`cudaStreamWaitEvent` (which become
graph **edges**, not nodes), and **no host synchronization between calls** (the post-condition at
`allreduce.h:50–53`). `PeerEvents` becomes an N-device event set (per-edge `inputs_ready` /
`pull_done`, or a per-neighbor event in the ring case).

This is a **single** implementation parameterized by `ec.tp` that **replaces** the two-party pull,
not a second path beside it. At `tp=2` the star reduce degenerates to the existing two-device
exchange (rank 1 → rank 0 gather, rank 0 → rank 1 broadcast) and at `tp=1` it is a no-op, so
tp=2, tp=4 (and tp=1) all run the same code.

The `allreduce_sum` oracle generalizes directly: FP64 sum of the **four** BF16 partials, rounded
once to BF16 on store.

### 3.6 CUDA-Graph bridge: two-device → N-device — **[hard]**

The second hard part. Today a **single `cudaGraph` holds both devices' nodes**: rank 1's stream is
forked into rank 0's capture and joined back, via `DecodeGraphPeerBridge`
(`src/core/decode_graph.cpp:145`, `decode_graph.h:27`) holding exactly `origin_device_` +
`peer_device_` and **three** events `{fork_, join_, gate_}`:

- `fork_peer` (`.cpp:91`) records the fork event on origin, makes the peer stream wait on it;
- `join_peer` (`.cpp:101`) rejoins the peer into origin (mandatory, else
  `cudaErrorStreamCaptureUnjoined`);
- `gate_launch` (`.cpp:202`) gates the replayed graph launch on the peer stream;
- `capture()` (`.cpp:235`) = `BeginCapture(origin) → fork_peer → body() → join_peer → EndCapture`.

The binding constraints (`decode_graph.cpp:3–33`): two live captures cannot be linked
(`cudaStreamWaitEvent` across two captures → `cudaErrorStreamCaptureMerge`), so it is **one**
capture on the origin stream with peer work forked in; event record/wait become graph edges.

**For TP4**, rank 0 still owns the single capture, but **three** peer streams (ranks 1,2,3) must be
forked in and joined back. Generalize the bridge from “origin + 1 peer + 3 events” to
“origin + (N−1) peers + per-peer fork/join event pairs + a replay gate.” The cross-rank
collectives (§3.5) already express their ordering as events, so they fold into the same capture as
edges. Wiring sites:

- `src/targets/qwen3_6/impl/runtime/graph_impl.h:45` `capture_graph` (forks rank 1 → generalize to
  fork ranks 1..N−1) and `:12` `run_prepared` (launch once on rank 0 after the gate).
- `src/targets/qwen3_6/impl/runtime/program_impl.h:292–318`: the consistency check
  `(tp == 2) != (peer_model != nullptr)` and `graph_bridge.emplace(dev[0], dev[1])` → build the
  N-device bridge over `dev[0..N−1]`.

This is where most of the correctness risk lives: the fork/join/event topology for four streams
must be captured as one graph and replay deterministically. It should be built and validated in
isolation before being wired into the full decode program. The generalized bridge likewise serves
both tps from one implementation: origin + 1 peer at `tp=2` (identical to today), origin + 3 peers
at `tp=4` — no separate two-device path is retained.

---

## 4. Phased implementation plan

**Phase 0 — plumbing (mechanical).** §3.1 + §3.2 + §3.3: bump the three root constants, widen the
arrays, parsers, validation messages, all-pairs CC check, and turn `runtime_peer` into an N-view
array with a `build_device_view` loop.
*Exit:* TP4 **binds, loads, materializes**, and runs **eager** (non-graph) decode without
crashing; per-rank shard extents verified to be the `/4` values in §2.

**Phase 1 — N-way collectives.** §3.5: replace the two-party `PeerEvents`/pull with the star reduce
through rank 0 (Option A), generalize the op signatures (`std::array<Tensor, kMaximumDevices>`,
`rank < tp`) per §3.4, and drop the `require_two_devices` → `require_tp` checks.
*Exit:* eager TP4 decode is **numerically correct** — each layer's post-allreduce output matches
the TP2 reference within BF16 tolerance (oracle: FP64 sum of 4 partials).

**Phase 2 — N-device graph bridge.** §3.6: generalize `DecodeGraphPeerBridge` to fork/join three
peer streams with per-peer events; wire `program_impl.h` to build the 4-device bridge.
*Exit:* the captured TP4 decode **graph replays** and its output equals eager TP4 output
(graph/eager parity), across many consecutive tokens reusing the same buffers/events.

**Phase 3 — validation & perf.** Run on `CUDA_VISIBLE_DEVICES=1,2,3,4` (app devices `0,1,2,3`)
with `--tp 4 --devices 0,1,2,3`. Confirm decode coherence, measure tok/s vs the TP2 baseline
(≈ 34 tok/s no-MTP / ≈ 48 tok/s MTP-3 from the V100X2 tests), and confirm the per-GPU weight
footfall drops to ≈ 5 GiB freeing KV headroom.

---

## 5. Correctness and verification

- **Collective oracle.** `allreduce_sum` at tp=4: FP64 sum of the four BF16 partials, one
  round-to-nearest-even on store. `allgather_rows`: exact byte relocation (no arithmetic).
- **Layer-level parity.** For a fixed input, each row-parallel projection's post-reduce output at
  tp=4 must equal the tp=2 (and tp=1) reference within BF16 tolerance — this is the direct check
  that the N-way reduce composes the four shards correctly.
- **Graph/eager parity.** Captured TP4 graph output == eager TP4 output over a long token run with
  reused buffers/events (the exact property the `allreduce.h:50–53` post-condition guarantees for
  tp=2; it must hold for tp=4).
- **End-to-end.** Decode coherence on a real prompt + committed tok/s at the claimed scope.
- **TP2 regression gate.** Because the unified N-way reduce/bridge *replaces* the bespoke
  two-party path, TP2 is re-run after the change and must match the pre-change TP2 reference:
  post-reduce outputs within BF16 tolerance, and committed tok/s still at the V100X2 baseline
  (≈ 34 no-MTP / ≈ 48 MTP-3). This is what keeps tp=2 supported alongside tp=4.

---

## 6. Risks and open questions

- **Risk concentration is in §3.5 + §3.6**, not in the model split. Everything up to Phase 0 is
  low-risk mechanical work; the N-way collective and the N-device graph bridge are where effort
  and correctness risk concentrate. Budget accordingly.
- **Divisibility.** tp=4 divides every current section cleanly (§2). This is a property of the
  27B geometry, not a guarantee — a future model whose head counts are not divisible by 4 would be
  rejected at plan time by the existing `% tp == 0` guards (fail-fast, acceptable).
- **Contract note.** The current product contract names “one device or TP2” as the supported
  identity. TP4 is an explicit extension requested here; adopting it means updating the active
  authorities (this doc, `docs/cli.md`/`docs/serving.md` `--tp` docs, and the `types.h`
  `EngineOptions.tp` doc comment) together when it lands.
- **Out of scope for the 27B deliverable:** the 35B-A3B sibling package's identical tp2 pattern
  (§3.4 note), and any NCCL-based collective (the code deliberately avoids NCCL for graph-capture
  reasons; the manual UVA-D2D + event route is retained).

---

## 7. Implementation status

Phases 0-2 landed on top of the checkout this plan was written against; phase 3 is blocked by the
local environment. `tp ∈ {1, 2, 4}` now share one N-generic implementation throughout.

- **Roots (§3.1/§3.2).** `kMaximumDevices` is 4 (`src/core/device.h`, `src/artifact/binder.h`); the
  fixed arrays, `parse_tp` (`{1,2,4}`), `parse_devices` (1-4), error strings, and the pairwise
  compute-capability check (all-pairs now) follow. The Engine/load/CLI/serve `--tp` docs and the
  `types.h` `EngineOptions.tp` comment were updated together.
- **Views (§3.3).** `LoadedModelData` holds `std::array<std::optional<RuntimeModelView>,
  kMaximumDevices>` and `build_device_view` runs in a rank loop.
- **Execution leaves (§3.4).** Op signatures and implementations take
  `std::array<Tensor, kMaximumDevices>` and loop `rank < ec.tp`; the 27B and 35B-A3B target split
  leaves were widened the same way (35B-A3B still rejects tp > 1, unchanged).
- **Collectives (§3.5).** The two-party pull was **replaced** by the Option A star through rank 0:
  peers stage partials into a rank-0-owned accumulator, rank 0 sums and residual-adds, then
  broadcasts bitwise-identically to every peer. `PeerEvents` is a `2 * tp` event set; transport
  stays UVA `cudaMemcpyAsync` D2D with event ordering and no per-call host sync. `require_tp`
  replaces `require_two_devices`. At `tp == 2` this is exactly the old two-party exchange.
- **Graph bridge (§3.6).** `DecodeGraphPeerBridge` forks/joins every peer stream (`tp - 1` peers,
  per-peer fork/join/transfer events) inside the single origin capture; `program_impl` builds it
  over `dev[0..tp-1]`. Validated in isolation: `build-v100/tests/ninfer_decode_graph_peer_test`
  captures 3 nodes at tp 2 and 9 at tp 4, and replays bitwise-equal to eager across many rounds.
- **MTP stem (beyond the original §3).** The MTP `input_projection` is row-parallel over the packed
  `[embedding; hidden]` input, so at tp 4 ranks 0-1 contract embedding rows and ranks 2-3 hidden
  rows. `mtp_forward_stem_parallel` now derives `chunk = mtp_fc_in / tp` and `emb_ranks =
  hidden / chunk`, allocates an embedding root on every embedding rank, and hands each rank a
  contiguous `[chunk, T]` slice of its normalized half. tp 2 is unchanged.
- **Names.** The parallel methods are suffixed `_parallel` (were `_tp2`) and the `tp2()` guard is
  now `tensor_parallel()`; both meant `tp > 1` all along.
- **Tests.** The full CTest suite compiles. `ninfer_allreduce_test` now also passes at tp 4 with
  four visible V100s. Pre-existing failures unrelated to the widening remain (sm_70 A4/A8 stub
  tests, host-only registry admissions, a vision embed-gather abort).
- **Phase 3 measurements.** End-to-end TP4 ran on four free V100s (physical 1-4;
  `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2,3,4` then `--tp 4 --devices 0,1,2,3`). At a
  95652-token prompt, MTP-3, INT8 KV, greedy, 128 committed tokens: tp2 eager 72.7 / tp2 graph 54.0
  / tp4 eager 86.0 / tp4 graph 35.2 tok/s. TP4 eager is fastest, so eager is now the **default at
  `tp > 1`** with `--cuda-graph` as the explicit opt-in (tp 1 keeps graphs). nsys shows cross-device
  traffic is only ~5-8 ms/token and is not increased by graph mode; the graph regression is GPU idle
  bubbles from multi-device graph scheduling, so the follow-up (§3.5/§3.6) is reducing cross-device
  event edges and serialization.
- **Remaining validation.** The measurements above used the available NVFP4 artifact; the Q4_K_M
  acceptance identity and the tp2 pre-change regression gate (numerics + baseline tok/s) still need
  confirmation once that artifact is present.
