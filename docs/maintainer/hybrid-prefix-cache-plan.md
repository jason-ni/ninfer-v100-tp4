# Hybrid prefix cache: V100X2 adaptation plan

Status: implemented and verified. M1 (device-only block + snapshot cache) and M2 (host tier,
including the host-restore continuation-identity exit criterion) are in-tree and verified; M3
(persistence and keys) has not started. This document adapts the content-addressed prefix cache of
`Wallawalla47/ninfer-custom` (its `docs/maintainer/hybrid-prefix-cache-spec.md`, referred to below
as **HPC**) to this checkout's Qwen3.8-27B V100X2 engine. Sections 0–4 keep the original
reference/motivation framing; §5 onward is the implementation record, and §1.1 lists where the
V100X2 implementation deviates from the HPC reference.

---

## 0. Motivation

Cross-turn prefix reuse is the dominant TTFT lever for agentic clients. The engine currently
reuses state only at an exact retained frontier or a template rewrite checkpoint. A client that
**rewrites earlier context between turns** therefore hard-misses and re-prefills the whole prompt.

Measured on this checkout (TP4, INT8-G64 KV, MTP-3, `preserve_thinking`), against an OpenCode
session whose prompt was 100 219 tokens:

```
[req 11] prompt=111929 full_reset reuse_reason=frontier_prefix_mismatch
         divergence=@85242/100219 window_from=85234
prompt  : ...<tool_response>\n[Compressed conversation section]...
resident: ...<tool_response>\n[Old tool result content cleared]...
```

The common prefix ends at `<tool_response>`; the client had replaced an earlier tool result with a
different placeholder. No exact-frontier scheme can serve this. The fix is to make reuse
**content-addressed and boundary-anchored** rather than tied to one owner's current frontier: cache
the complete reusable states that exist *along the prompt path*, not only at its tail.

---

## 1. Reference design (HPC), in one page

HPC replaces the owner/checkpoint cache with a radix tree of physical KV blocks plus sparse state
snapshots. The essentials (HPC §0, §5, §6, §7):

- **Block tree.** A node covers 64 tokens (our page size is already 64 —
  `src/core/paged_kv_cache.h:15` `kPagedKVPageSize = 64`). Node key is
  `(parent, exact tokens[64], media_extra)`; a chained `xxh3_64` hash only selects a child, and a
  hit always re-compares tokens exactly. Identical blocks are one physical page across sessions.
- **Snapshots.** A snapshot is the complete per-sequence **state image** (GDN recurrent + conv
  history + continuation hidden + DFlash rings) captured at a frontier `F`, anchored to the last
  full block below `F`, optionally with a private partial-tail page bundle. Snapshots are
  deduplicated and may be device- or host-resident.
- **Taps.** Where snapshots are taken (HPC §7.1): explicit client breakpoints (exact); the final
  assistant generation opener (exact); end-of-tools / end-of-system (exact); the prompt tail
  `n−1` (flexible); and a **geometric ladder** `n − G·2^k` snapped to a message boundary
  (flexible). Exact taps split a prefill chunk; flexible taps are taken at the chunk boundary and
  cost only one state-image copy. The expensive zero-split kernel tap is explicitly *not*
  implemented in HPC; the flexible tap is a D2D copy of the committed state image at a chunk
  boundary.
- **Host tier.** A pinned slab pool (`--host-cache-mib`) holds KV blocks and snapshot images. On
  lane release blocks are written through asynchronously; device eviction never copies. Restores
  are issued on a dedicated stream, ordered by forward-pass read order, and overlap the first
  prefill pass.
- **Admission.** Walk the prompt's blocks from the root, stop at the first miss, collect snapshots
  on the matched path, and pick the deepest snapshot with `F ≤ n−1` (so at least one token is
  always prefilled and logits/bridges come from the normal path). Cost = restore bytes + prefill
  of `n−F`; resident-device candidates are preferred. O(prompt blocks).
- **Eviction.** Device: LRU, tail-first, evictable pages count as capacity. Host: superseded
  snapshots first, then GreedyDual-Size-Frequency; dead-KV (unreachable by any snapshot) before
  live blocks. Removing a node's last copy deletes the snapshots in its subtree.

The model constraint HPC is built around is exactly ours: the GDN recurrent state cannot be rolled
back, so a prefix is reusable only where the complete state exists. HPC makes those states cheap
to create (one state copy at a chunk boundary), cheap to store (host tier), and exactly
addressable (block path).

### 1.1 Where the V100X2 implementation deviates from the reference

- **Taps are block-aligned; `tail_len` is always 0.** A tap is captured only when the prefill
  cursor sits exactly on a `kBlockTokens` (64) boundary, and it is published with `frontier ==
  cursor` (§6, capture invariant). Flexible tap positions from the planner are snapped down to
  `floor(pos/64)*64`; positions at or below the reuse base are dropped. Because the prompt tail is
  generally not 64-aligned, the final prefill chunk is cut at `floor(prompt_tokens/64)*64` so that
  tail capture lands on a boundary; the residual `prompt_tokens % 64` tokens are prefilled in one
  extra short chunk, charged as one extra prefill quantum in `service_work_quanta`. A capture
  boundary that lands mid-block for any other reason (e.g. a rewrite-checkpoint split) defers the
  tap to the next aligned boundary. The reference's "private partial-tail page bundle" is therefore
  **not** constructed on this path — the index's tail fields exist but stay unused.
- **State image contents.** `ProgramImplCore::copy_snapshot_slot` copies the per-layer GDN conv
  history (BF16) and recurrent state (FP32) only. Continuation hidden and DFlash rings are not part
  of a tap image: under the acceptance MTP profile the resumed suffix recomputes the continuation
  hidden and re-establishes MTP through the normal bridge, and DFlash2 is a separate (text-only)
  profile.
- **Tap sources.** Only the planner's prompt-tail and geometric-ladder (flexible) taps are produced
  today: `plan_request_for_lane` passes a default-empty `TapHint`, so the explicit client
  breakpoints, generation opener, and end-of-tools taps (the "exact" set) are not yet plumbed
  (M3).

---

## 2. What this checkout has today ("Legacy" in HPC terms)

Reuse unit is an **owner's continuation state**, not a content-addressed block.

| Concern | Site |
|---|---|
| Prompt reuse entry point / toggle | `include/ninfer/types.h:200` `ExecutionOptions::allow_prefix_reuse` |
| Reuse path + divergence diagnostics | `include/ninfer/types.h:429` (`PrefixReusePath`), `:450` (`PrefixDivergence`), `:472-475` |
| Linear-attention state slots per lane | `src/targets/qwen3_6/impl/runtime/layouts_impl.h:105-106`, `:143` (`LinearStateSlots::state_slot_count`) |
| GDN replay state-slot contract | `include/ninfer/ops/gdn_replay.h:14,26` (`linear_state_slot`) |
| Rewrite checkpoint for the current turn | `src/targets/qwen3_6/impl/frontend/chat_template.cpp:407-446` (`RewriteCheckpointByteSpec`, `TurnClosure`), `processor.cpp:439-446,593-607` |
| DFlash local rewrite checkpoint | `src/targets/qwen3_6/impl/runtime/dflash_context_impl.h:47-52`, `layouts_impl.h:172` |
| Continuation hidden | `src/targets/qwen3_6/impl/runtime/layouts_impl.h:238` (`rewrite_checkpoint_hidden`) |
| Physical KV containers / ownership | `src/core/paged_kv_cache.{h,cpp}`; `docs/maintainer/paged-kv-cache.md` |
| Convergence + paging authority | `docs/maintainer/concurrent-inference-architecture.md`, `docs/maintainer/paged-kv-cache.md` |

The state kinds handled today are a small closed set (`TurnClosure`, `ResponseReplay`,
`SessionEndpoint`, long anchors, shared stable prefixes). Each kind is bound to an *owner* and to
catalog/anchor limits, so two sessions with the same 30 k preamble share KV only if a shared prefix
was explicitly published.

---

## 3. Why the current design misses (and the hybrid fixes it)

1. **Ownership vs content.** A snapshot is usable only by requests that resolve to the same owner.
   HPC keys on the token path itself, so any request that shares a prefix shares the physical
   blocks, unconditionally.
2. **One frontier vs the path.** Today the reusable state sits at the *current* turn closure. When
   the next prompt diverges at 85 k (OpenCode rewriting a tool result), the state at the tail is
   useless. HPC keeps snapshots at every diverging boundary (opener, end-of-tools, ladder), so a
   rewrite still finds a valid snapshot *before* it.
3. **GPU-resident or nothing.** Snapshot images are large (see §4); keeping many on the device is
   wasteful. HPC pushes them to pinned host RAM and restores on demand.
4. **Planning search.** HPC admission is O(prompt blocks) arithmetic, with no heuristic search on
   the critical path.

---

## 4. V100X2-specific facts that shape the adaptation

- **Page size 64** (`src/core/paged_kv_cache.h:15`) already matches HPC's block.
- **KV is INT8 group-64** at 180 000 capacity. Per token the 16 full-attention layers × 4 KV heads
  are ≈ 33 KiB (HPC §3.1), so a 64-token page is ≈ 2.06 MiB across the rank; one 64-token block is
  cheap to relocate host↔device.
- **State is TP-sharded.** At TP4 the GDN recurrent state is split per rank
  (`src/targets/qwen3_6/impl/runtime/layouts_impl.h`, `value_heads/tp`), so each rank owns only its
  head shard and never syncs state across ranks. A snapshot is therefore **per-rank** and roughly
  1/tp of the HPC image: for 27B the FP32 GDN recurrent alone is 128·128·48·4 ≈ 144 MiB total → ≈
  36 MiB/rank at TP4 (plus conv history, hidden, and DFlash rings when enabled). Exact sizes must be
  measured against the real layouts before sizing the pool.
- **One capture per rank.** This checkout does **not** hold one graph over all ranks (unlike the
  fork's single-device model): each rank captures its own decode graph and NCCL collectives provide
  cross-rank ordering. A host restore must therefore be expressed as work on each rank's own
  stream, and snapshot capture is per rank.
- **MTP profile.** In the acceptance profile MTP KV is position-indexed and resident; a resume
  performs the existing MTP bridge from the resumed frontier (HPC §7.3.5). Tap snapshots need not
  carry MTP state.
- **OpenCode/agent divergence is inside a tool block.** HPC's tap planner already places snapshots
  at the assistant generation opener and at structural boundaries, which is where this divergence
  lands; the geometric ladder covers edits deeper in history.
- **Vision is rejected** for the GGUF-derived identity, so `media_extra` can be a no-op for the
  active V100X2 workload while remaining part of the key for the Vision-capable identities.

---

## 5. Milestones

The plan deliberately splits the subsystem so each milestone is independently verifiable. The
device-only cut already removes the OpenCode hard-miss within a process; the host tier makes it
survive long sessions and multiple conversations.

### M1 — device-only block + snapshot cache

Status: **complete.** Implemented and verified (see `ninfer_prefix_cache_index_test` and the
OpenCode repro). One latent bug found during M2 review — `clear_lane` released *published* snapshot
slots back to the free list even though the live snapshot still owned them — is fixed as part of M2
(task T3, slot-release fix).

Deliverables:

- A block index over the paged KV: nodes keyed by `(parent, tokens[64])`, hash for lookup with exact
  re-compare; node insertion only when all 64 positions of every enabled pool are committed.
- A tap planner producing the §1 set (explicit markers first, then generation opener, structural
  boundaries, prompt tail, geometric ladder), with the HPC clustering/redundancy rules.
- Snapshot capture: D2D-copy the state image (per-layer GDN conv history + recurrent state) into a
  device snapshot slot at a **block-aligned** chunk boundary; publish with `frontier == cursor` once
  the anchor block is committed. Flexible tap positions are snapped to block boundaries and the
  prompt-tail chunk is cut at `floor(n/64)*64` (§1.1, §6).
- Admission = deepest snapshot on the matched path with `F ≤ n−1`; restore by sharing the path's
  device pages and copying the image into the lane's active slot.
- The admission's `prefix_reuse_path` reporting gains a `BlockSnapshot` path (or equivalent) so
  `--request-log-jsonl` shows which snapshot served the request and at what `F`.

Exit: on the OpenCode repro, request 2 onward reports a non-`FullReset` reuse path anchored before
the rewritten tool block, and the prefill token count equals `n − F`.

### M2 — host tier

Status: **complete (2026-10-09).** M1/M2 remain in-tree; the earlier "complete; GPU-verified" claim
rested on a host-tier gate (`exercise_host_tier`) that only asserted `reused > 0`, never that the
restored continuation matched a cold run. The 2026-10-08 reopen — host restore not
continuation-identical — is resolved: root cause, fix, and verification are recorded under "M2
resolution" at the end of this section. Tracked as tasks T0–T7 below, all done in-tree.
`ninfer_prefix_cache_host_test` and `ninfer_prefix_cache_index_test` are green, the full build
links, and the GPU host-reuse scenario passes on V100X2: `ninfer_qwen3_8_27b_v100x2_hybrid_real_test`
runs the normal, pool-exhaustion, and host-tier cases (a returning conversation restored from the
pinned host tier after device eviction) and reports PASS for all three.

A review while landing T7 found and fixed a real bug: `finish_host_write` / `finish_snapshot_write`
pinned their slab in `begin_*` to protect the in-flight transfer but only *un*pinned it on the
failure path. A successful write-through therefore leaked the pin, and since `pick_victim` skips
pinned slabs, the pool could never evict a successfully-cached slab — the host tier filled with
permanently-pinned slabs and stopped making room. Both success paths now release the in-flight pin.

#### Design decisions (locked)

- **Host slab layout is rank-sliced.** A block slab holds `tp × per_rank_block_bytes` (rank `r` at
  offset `r·per_rank_block_bytes`); a snapshot slab holds `tp × gdn_state_bytes`. Per-rank block
  bytes = text-pool per-page bytes + backend-pool per-page bytes (both pools, all planes). Transfers
  run on each rank's own device/stream into its slice.
- **Snapshot image write-through is tap-time, async.** Right after a tap is captured
  (`add_snapshot` in `advance_prefill`), the state image is D2H-copied to a host slab on the cache
  stream (enqueued, not synced). This deviates from the original "on lane release" wording: a
  snapshot slot is a scarce device resource that GDSF can evict *before* the request ends, so the
  host copy must exist from the moment the snapshot is born to survive slot eviction. The slot is
  exclusively owned by its snapshot record, so the D2H reads nothing contended. Completion is polled
  at round boundaries; `restore_pins` (bumped in `begin_snapshot_write`) keeps GDSF from evicting a
  snapshot whose write is still in flight (`acquire_slot` skips `restore_pins != 0`).
- **Block write-through is lane-release, async.** On `clear_lane`, each path node that is
  device-resident but has no host copy yet is D2H-copied to a host slab (enqueued, not synced).
  Pages are cache-owned and protected by `restore_pins` while in flight. Best-effort: a transfer
  failure rolls the node back to no-host-copy rather than failing the (noexcept) lane teardown.
- **Restore (H2D) is synchronous.** At lane start, all missing blocks (text + backend, every rank)
  and the snapshot image are H2D-copied on the per-rank cache stream, then each rank's stream is
  synchronized once. There is no GPU work between enqueue and the first reader on this path, so
  event-deferred overlap buys nothing; sync keeps restored nodes straight to `Resident` with no
  lingering in-flight state.
- **Device eviction never copies.** A backed node (host copy present) demotes: drop device pages,
  keep host copy. An unbacked node erases its subtree. The earlier eviction-time write-through is
  removed.
- **Host-slab eviction reconciliation.** When the slab pool evicts a slab, the owner loses its host
  copy; if the owner then has no data anywhere and is unpinned, it is erased (nodes) / deleted
  (snapshots). In-flight slabs are `pinned` in the pool so they are never victims.
- **Snapshot slot eviction demotes instead of deleting** when the victim has a host copy (stays
  alive as host-resident; no `dec_live_below`).

#### Tasks

| # | Task | Scope | Status |
|---|------|-------|--------|
| T0 | Update tracking doc | Mark M1 done; add this M2 task table + design decisions | done |
| T1 | Fix physical transfer layer | All-rank transfers, backend pages in block data, snapshot image = per-layer conv+recurrent segments, rank-sliced slab sizing; `PagedKVPool::copy_page_d2h/h2d` rewritten non-throwing on the kernel's real layout; `per_page_bytes()` from plane specs | done |
| T2 | Fix PrefixIndex policy | LRU invariant in `link_lru_tail`; pin/unpin simplification; `begin_snapshot_write` pins anchor node; `delete_snapshot` defers while pinned; `detach_snapshot` releases slab; `hit()` updates GDSF metadata; `evictable_page_counts()`; dead-code removal | done |
| T3 | Write-through (D2H) + slot fix | `issue_block_write_through` / `issue_snapshot_write_through` with per-rank events; `tick_prefix_cache()` polling; tap-loop slot settlement (consumed = `kInvalidSlot`); `clear_lane` write-through before unpin, releases only unconsumed slots | done |
| T4 | Restore (H2D) | Rewrite `start_prefill_lane` BlockSnapshot branch: batch H2D of missing blocks (text+backend, all ranks) + snapshot image via `begin_snapshot_restore`; sync per rank; borrow restored pages; admission in `request_plan_impl.h` picks deeper of `best_device`/`best_host` | done |
| T5 | Wire device eviction to memory pressure | `can_admit_lane` counts evictable cache pages (`active_refs==0 && restore_pins==0`) as reclaimable; `reserve_sequence_kv` + restore-time `acquire_cache_pages` retry through `evict_device_lru()` on `bad_alloc`; fix `release_cached_pages` to release backend pages (all ranks) | done |
| T6 | Config surface + docs | Serve `--host-cache-mib` (`serve_options.{h,cpp}`, `generation_service.cpp`); `docs/cli.md`, `docs/serving.md`; finalize this tracking section | done — `--prefix-cache`/`--host-cache-mib` documented in the CLI and serving option tables |
| T7 | Tests + verification | Extend `tests/test_prefix_cache_host.cpp` (demote-vs-erase, slab reconciliation, dead-KV/supersede marking, slot demote, best_host admission, slot-release fix); GPU host-reuse scenario + regression | done — unit tests green; GPU host-reuse scenario passes (`ninfer_qwen3_8_27b_v100x2_hybrid_real_test`: normal / exhaustion / host-tier); acceptance regression verified on a live TP4 session |

Dependency order: T0 → T2 → T1 → T3/T4 → T5 → T6 → T7 (T1/T2 near-independent).

#### Implementation notes (session 2026-10-06)

Done in-tree:

- **T1 transfer layer** — Key discovery: every KV plane is physically `[page][head][token][leading]`
  with leading fastest (`src/ops/kernel/paged_kv_address.cuh`); the PageMajor addressing path uses
  hardcoded template offsets. Old copy helpers used wrong `nb[]` stride math. Rewritten:
  `PagedKVPool::copy_page_d2h/h2d` are non-throwing bool APIs (per plane: HE contiguous runs of
  LE·PS·dsz bytes at byte offset dsz·LE·PS·(h + HE·p); the rewrite first omitted `dsz`, which the
  M2 resolution below fixed); `per_page_bytes()` = Σ LE·PS·HE·dsz. Program
  side: `rank_backend_cache(rank)`, `copy_cache_block(rank, text_page, backend_page, slab_base, d2h)`
  (both pools, all ranks, rank-sliced slabs), `copy_snapshot_slot(rank, slot, slab_base, d2h)` (per
  layer: conv BF16 + recurrent FP32; slot is the last dim so per-slot data is one contiguous run per
  layer). Constructor slab sizing fixed: block slab = tp·(text+backend per-page bytes); snapshot slab
  = tp·per-slot GDN bytes (NOT `gdn_state_bytes` — that member is the whole pool, all slots). Streams
  created only when a pool exists. New members: `prefix_block_rank_bytes`, `prefix_snapshot_rank_bytes`,
  `InflightCacheWrite` + `prefix_inflight_writes`.
- **T2 index policy** — LRU invariant enforced in `link_lru_tail` (linked ⟺ Resident && active_refs==0
  && restore_pins==0); pin/unpin simplified to bump counter then re-link. `begin_host_write` pins via
  `pin_restore` and marks the slab dead if `live_snapshots_below==0`. All `finish_*` paths unpin;
  `finish_device_fill(ok=false)` resets `device_pages`; `finish_snapshot_restore(ok=false)` releases
  its slot internally. `begin_snapshot_write` pins the anchor node (prevents subtree erase mid-D2H);
  `finish_snapshot_write` unpins it. `delete_snapshot` defers while `restore_pins != 0`.
  `detach_snapshot` releases the host slab on full deletion. `hit()` updates GDSF metadata.
  `evictable_page_counts()` added. Removed dead code: `mark_device_resident`/`mark_host_resident`,
  `PrefixPageSink::write_through_block`. Added `HostSlabPool::clear_superseded()`.
- **T3 write-through** — `issue_block_write_through(node)`: `begin_host_write` + D2H per rank + event
  record + inflight tracking. `issue_snapshot_write_through(id)`: same for snapshot images.
  `tick_prefix_cache()`: polls all ranks' cache events, finishes all pending writes when every rank is
  ready (all-or-nothing drain). `advance_prefill` tap loop: after `add_snapshot`, settles the reserved
  slot (releases it if a duplicate holds a different slot), marks consumed slots `kInvalidSlot`, issues
  the snapshot write-through. `clear_lane`: issues block write-through for all pinned nodes before
  unpinning; releases only unconsumed tap slots (fixes the M1 latent bug: published slots stay owned
  by their live snapshot).

#### Verification status

T7 gates were recorded as met at the time; the host-tier continuation-identity exit criterion was
in fact not exercised by the *hybrid* gate (see "M2 resolution" at the end of this section). It is
now exercised by the concurrent shared-prefix gate, which passes both scenarios. Remaining gates:

1. **GPU host-reuse scenario** — `ninfer_qwen3_8_27b_v100x2_hybrid_real_test` (`exercise_host_tier`;
   TP2 + MTP-3 + `--host-cache-mib`) passes: conversation A cold, a disjoint equally long B evicts A
   from device, A re-issued → `ReusedBlockSnapshot` restored from the host tier. The normal and
   pool-exhaustion cases pass alongside it.
2. **Regression / acceptance** — a live TP4 session (`--prefix-cache --host-cache-mib 28192
   --vision`, TP4, MTP-3) ran the agent OCR workload end to end. The raw-output capture shows 59
   `ReusedBlockSnapshot` reuses, every one anchored on a 64-token block boundary, with no
   leaked/mis-anchored continuations; a `--no-prefix-reuse` run of the same workload is the control
   for the detector. The host tier and multi-rank transfers were exercised throughout the session.

The pin-leak fix noted in the M2 status above was found while writing the T7 slab-reconciliation
test.

Facts the next session needs:

- Qwen3.6-27B: kv_heads=4 total (TP2 → 2/rank), head_dim=256, 16 full-attn layers, 48 GDN layers;
  GDN key_heads=16, value_heads=48, key/value_head_dim=128, conv_state_width=3.
- `gdn_state_bytes` (program member) = the WHOLE linear-attention pool (all slots); per-slot bytes must
  be computed from `LinearAttentionStatePoolSpec` (layers × (conv_channels·conv_width·dsz +
  key_dim·value_heads·value_head_dim·4)) or divided by slot count.
- `copy_kv_block_h2d` no longer exists; anything still referencing it is part of the broken T4 branch.
- Build dir: `build-v100/` (ninja). GPU runs: `CUDA_VISIBLE_DEVICES=1,2 GGML_CUDA_PDL=0`, TP2, MTP-3.

#### Verification (T7)

- **Unit:** new index behaviors above; `ninfer_prefix_cache_host_test` and `ninfer_prefix_cache_index_test` stay green.
- **GPU host-reuse scenario** (`CUDA_VISIBLE_DEVICES=1,2 GGML_CUDA_PDL=0`): TP2, MTP-3, `--host-cache-mib` sized to hold ~1 conversation's snapshots. Run conversation A (~90k tokens) then conversation B (different prefix, ~90k) so A's blocks are demoted/evicted from device; re-issue A's next turn → expect `ReusedBlockSnapshot` with frontier < n and continuation **identical** to a cold run of the same prompt (greedy). This is the M2 exit criterion ("device pool cannot hold every conversation still reuses the host tier").
- **Regression:** `--host-cache-mib=0` byte-for-byte M1 behavior; decode tok/s parity on the acceptance workload vs current baseline; TP4 smoke with host tier enabled (multi-rank transfer correctness).

Exit: a long agent session whose device pool cannot hold every conversation still reuses the host
tier; TTFT for a returning session is dominated by H2D restore (~GB/s), not recompute; graph/eager
parity unchanged.

#### M2 resolution — host restore is continuation-identical (2026-10-09)

Status: **fixed and verified.** The 2026-10-08 reopen ("host restore is not continuation-identical")
came from `copy_plane_page` in `src/core/paged_kv_cache.cpp`, the per-plane transfer helper behind
`PagedKVPool::copy_page_d2h/h2d` and therefore behind every demote and restore.

Root cause. The helper computed each head run's device offset as an *element* count but used it as a
*byte* count:

```cpp
char* dev_run = device_base + leading * kPagedKVPageSize * (h + heads * page_id);
//                                   ^ missing dtype_size(spec.dtype)
```

I8 code planes are one byte per element, so the omission was invisible there; the FP16 K/V **scale
planes** are two bytes per element, so their runs were copied at half their true byte offset. A
demote therefore stored the wrong region of each scale plane in the host slab and a restore wrote it
back to the same wrong place, leaving every restored block's real scale region untouched. Attention
reads scales at the correct (doubled) offset, so a restored prefix ran with stale quantization
scales — the same prompt computed cold was exact, hence
`restore_vs_device=0/32 cold_vs_device=32/32 restore_vs_cold=0/32`. The MTP/backend pool's scale
planes were affected identically.

Correction to the earlier localization. The "content write into the shared prefix's cached device
KV pages" premise was a measurement artifact of this same bug: `dbg_checksum_page` (and the
`[PAGECK]` / `[CHUNK]` / `[RDUMP]` / `[LAYERCK]` watches built on it) also read pages through
`copy_page_d2h`, so for an FP16 scale plane "page q" actually read true page `q/2`'s scale region.
The per-plane canary that settled it showed only the scale planes changing, at exactly the region
the attention's own append wrote legitimately (logical pages 4–7); the K/V code planes and the
block table were always correct, and no registered writer ever wrote a foreign page — which is why
the session-3 tracker saw `tag=-1`. The session-2 "device-resident blocks 8–15 mismatch" in the
RDUMP was the same wrong-region read over the two engines' differing page histories, not a clobber.

Fix. `copy_plane_page` now derives `run_bytes = leading * kPagedKVPageSize * dtype_size(dtype)` and
offsets by `run_bytes * (h + HE*p)`, so every plane dtype copies at its true byte offset. While
reviewing that function, `HeadMajor` pools were also found to be unsupported: the DFlash
full-context pool (`layouts_impl.h`, addressed `p + pages*h` by `bidirectional_gqa_attention` and
`kv_cache_append_prefix`) would have been copied at the PageMajor run index. The helper is now
order-aware; the DFlash full pool has no demote/restore path today, so this was latent.
`tests/test_kv_cache.cpp` gained a round-trip test that places a page for both orders and compares
against the layout's own `nb[]` strides (it fails on the pre-fix formula).

Also landed from the review: the duplicate-commit use-after-free in the prefix commit path. The
batch `promote_owned_pages(count)` was replaced by per-block `promote_page_at(index)` with a
per-page `cache_owned_` vector, and the commit loop promotes only blocks the index actually adopted
(duplicates keep the page owned). The old batch API is removed and `tests/test_kv_cache.cpp` was
updated to the per-page form.

Verification (TP2, MTP-3, NVFP4, `CUDA_VISIBLE_DEVICES=2,3 GGML_CUDA_PDL=0`):

```text
device: 4 lanes reused=1984, agreement 32/32                       -> PASS
host:   restore_vs_device=32/32 cold_vs_device=32/32
        restore_vs_cold=32/32; single-lane 32/32 x4                -> PASS
```

- Gate: `ninfer_qwen3_8_27b_v100x2_concurrent_prefix_real_test`, both scenarios.
- Unit: `ninfer_kv_cache_test` (incl. the new order round-trip), `ninfer_kv_cache_append_prefix_test`,
  `ninfer_prefix_cache_host_test`, `ninfer_prefix_cache_index_test`, `ninfer_admission_policy_test`,
  `ninfer_concurrent_executor_test` all green on freshly built binaries.
- `ninfer_qwen3_8_27b_v100x2_prefix_real_test` (graph/eager outputs, logits, acceptance, frontiers)
  passes.

Remaining:

1. Strengthen `exercise_host_tier` in `tests/targets/qwen3_6_27b/test_engine_v100x2_hybrid_real.cpp`
   to assert restored-continuation identity (not just `reused > 0`), matching this gate's criterion.
2. Remove the session-2/3 investigation instrumentation still in the working tree (write probes +
   tracker + watch checkpoints + row/byte dumps, all behind `NINFER_PAGECK` / `NINFER_PREFIX_WATCH`
   and friends) before committing. The session-4 scaffolding was already removed.

### M3 — persistence and keys

Deliverables:

- `--prefix-cache-file PATH`: load at start, dump on graceful shutdown (SIGINT), with a versioned,
  validated container (HPC §5.5).
- Media digests in the block key for the Vision-capable identities.
- Explicit breakpoints (OpenAI/Anthropic `cache_control`/markers) plumbed from the frontend.

Exit: restarting the server and re-issuing a prefix yields the same reuse as before restart.

---

## 6. Numerical correctness and verification

- **Reuse is exact relocation, not arithmetic.** A restored state must be the *same bytes* as the
  committed state at `F`: snapshot capture and restore are copies, compared byte-for-byte against a
  freshly prefilled state at the same `F` (the exact oracle for a state image).
- **Capture invariant (`frontier == state`).** A tap is published only when the prefill cursor sits
  exactly on the block boundary it names. The copied image is the live state *after* `cursor`
  tokens, so publishing it under `frontier = floor(cursor/64)*64` at a non-aligned `cursor` would
  leave the state `cursor % 64` tokens ahead of its frontier, and a resume would double-consume
  those tokens through the GDN recurrence — the Qwen3.8 "tool output leaks into reasoning" defect.
  Flexible taps are therefore snapped to block boundaries and the prompt-tail chunk is cut at
  `floor(prompt_tokens/64)*64` (§1.1), so every snapshot's image is byte-identical to a freshly
  prefilled state at its `F`.
- **Admission equivalence.** Prefill `n − F` tokens from a restored snapshot must produce the same
  continuation as a cold prefill of all `n` tokens, within the same tolerance the cache-off path
  already uses (BF16/INT8 KV profile). This is the end-to-end property; run it over the OpenCode
  repro and a synthetic diverge-and-resume set.
- **Graph/eager parity** must hold across many consecutive tokens with reused buffers and events.
- **TP2 and TP1 regression gates.** The cache is model-agnostic; TP2/TP1 runs with the cache
  enabled must match their pre-change committed tok/s and reuse behavior.
- **Capacity arithmetic.** Because unpinned cache entries own their page bundle exclusively, evicting
  one returns exactly one page per pool; verify `need_pages ≤ available` equals the realized
  capacity on the real pool.

---

## 7. Configuration surface

Implemented server flags:

| Option | Default | Meaning |
|---|---|---|
| `--prefix-cache` | off | Enable the hybrid content-addressed prefix cache (additive to the legacy owner/checkpoint reuse paths). |
| `--host-cache-mib N` | `0` | Pinned host RAM for the prefix pool; `0` = device-only (M1). |
| `--no-prefix-reuse` | reuse on | Disable all prefix reuse for the request. |

Internal `PrefixCacheOptions` fields. They are not exposed as CLI flags; `ProgramImplCore` derives
these defaults when the field is zero:

| Field | Default | Meaning |
|---|---|---|
| `snapshot_slots` | `0` → `max_concurrency + 1` | Device state-image slots. |
| `max_new_taps` | `0` → `2` | Cap on snapshots captured per request. |
| `ladder_tokens` | `0` → `max(4096, 2·prefill_chunk)` | Ladder spacing `n − G·2^k`. |
| `min_gap` | `0` → `max(1024, prefill_chunk)` | Minimum gap between taps. |

Not yet implemented: `--prefix-cache-file` (M3 persistence), explicit client breakpoints
(OpenAI/Anthropic `cache_control`/markers), and a `--use-original-prefix-caching` switch. The hybrid
cache is **additive**, not a replacement: `admit_block_snapshot` is tried first and, on no usable
hit, planning falls through to the existing owner/checkpoint paths (`AppendAtFrontier`, rewrite
checkpoint) unchanged — unlike HPC §14.1, the legacy machinery is still constructed.

---

## 8. Risks and open questions

- **Scope.** This is a new cache subsystem (HPC is ≈6 k lines plus a host arena), not a patch. M1 is
  the smallest cut that removes the OpenCode hard-miss; do not under-budget M2.
- **TP-sharded snapshots.** HPC is single-GPU. Per-rank snapshot capture/restore and per-rank graph
  capture must be designed here; the cross-rank reduce is unaffected and no state sync is needed.
- **State image size.** The per-rank tap image is the GDN conv history + recurrent state only
  (§1.1). Verify its size against the real layouts before sizing device snapshot slots and the host
  pool.
- **Rewriting clients.** Reuse will still restart at the *deepest surviving boundary* before a
  rewrite. If a client rewrites the very first tool block, only the system/tools preamble is
  reusable — that is the floor, and it is a large improvement over a full reset.
- **Contract.** The cache extended the published CLI/serve surface (`--prefix-cache`,
  `--host-cache-mib`), now documented in `docs/cli.md` and `docs/serving.md`. The
  `include/ninfer/types.h` `PrefixCacheOptions` comments carry the internal fields. If the supported
  identities themselves ever change, update `docs/README.md` with them.
