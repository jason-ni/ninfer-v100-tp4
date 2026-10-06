# Hybrid prefix cache: V100X2 adaptation plan

Status: design / plan (not implemented). This document adapts the content-addressed
prefix cache of `Wallawalla47/ninfer-custom` (its `docs/maintainer/hybrid-prefix-cache-spec.md`,
referred to below as **HPC**) to this checkout's Qwen3.8-27B V100X2 engine. It records *what* the
current engine does, *why* exact-frontier reuse misses agent clients, and the staged path to the
hybrid design. It changes no code.

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
- Snapshot capture: reuse the existing committed-state path (`dflash_context_impl.h:47-52`,
  `rewrite_checkpoint_hidden`) to D2D-copy the state image into a device snapshot slot at a chunk
  boundary; publish when the anchor block is committed.
- Admission = deepest snapshot on the matched path with `F ≤ n−1`; restore by sharing the path's
  device pages and copying the image into the lane's active slot.
- The admission's `prefix_reuse_path` reporting gains a `BlockSnapshot` path (or equivalent) so
  `--request-log-jsonl` shows which snapshot served the request and at what `F`.

Exit: on the OpenCode repro, request 2 onward reports a non-`FullReset` reuse path anchored before
the rewritten tool block, and the prefill token count equals `n − F`.

### M2 — host tier

Status: **code complete; GPU verification pending.** Tracked as tasks T0–T7 below. T0–T7 are done
in-tree (uncommitted): the tree compiles, `ninfer_prefix_cache_host_test` and
`ninfer_prefix_cache_index_test` are green, and the full build links. What remains is the T7 GPU
exit criterion (host-reuse scenario + `--host-cache-mib=0` regression) on real weights.

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
| T3 | Write-through (D2H) + slot fix | `issue_block_write_through` / `issue_snapshot_write_through` with per-rank events; `tick_prefix_cache()` polling; tap-loop slot settlement (consumed = `kInvalidSlot`); `clear_lane` write-through before unpin, releases only unconsumed slots | done (tick call sites partial — see WIP note) |
| T4 | Restore (H2D) | Rewrite `start_prefill_lane` BlockSnapshot branch: batch H2D of missing blocks (text+backend, all ranks) + snapshot image via `begin_snapshot_restore`; sync per rank; borrow restored pages; admission in `request_plan_impl.h` picks deeper of `best_device`/`best_host` | done |
| T5 | Wire device eviction to memory pressure | `can_admit_lane` counts evictable cache pages (`active_refs==0 && restore_pins==0`) as reclaimable; `reserve_sequence_kv` + restore-time `acquire_cache_pages` retry through `evict_device_lru()` on `bad_alloc`; fix `release_cached_pages` to release backend pages (all ranks) | done |
| T6 | Config surface + docs | Serve `--host-cache-mib` (`serve_options.{h,cpp}`, `generation_service.cpp`); `docs/cli.md`, `docs/serving.md`; finalize this tracking section | code done; `docs/cli.md`/`docs/serving.md` prose pending |
| T7 | Tests + verification | Extend `tests/test_prefix_cache_host.cpp` (demote-vs-erase, slab reconciliation, dead-KV/supersede marking, slot demote, best_host admission, slot-release fix); GPU host-reuse scenario + regression | unit tests done + green; **GPU scenario pending** |

Dependency order: T0 → T2 → T1 → T3/T4 → T5 → T6 → T7 (T1/T2 near-independent).

#### WIP status (session 2026-10-06, uncommitted)

Done in-tree:

- **T1 transfer layer** — Key discovery: every KV plane is physically `[page][head][token][leading]`
  with leading fastest (`src/ops/kernel/paged_kv_address.cuh`); the `plane_order` enum is vestigial
  for addressing (kernels hardcode template offsets). Old copy helpers used wrong `nb[]` stride math.
  Rewritten: `PagedKVPool::copy_page_d2h/h2d` are non-throwing bool APIs (per plane: HE contiguous
  runs of LE·PS·dsz bytes at offset LE·PS·(h + HE·p)); `per_page_bytes()` = Σ LE·PS·HE·dsz. Program
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

Remaining (only the T7 GPU exit criterion):

1. **GPU host-reuse scenario** (`CUDA_VISIBLE_DEVICES=1,2 GGML_CUDA_PDL=0`, TP2, MTP-3,
   `--host-cache-mib` sized for ~1 conversation): run conversation A (~90k tokens) then B (different
   prefix, ~90k) so A's blocks demote/evict from device; re-issue A's next turn → expect
   `ReusedBlockSnapshot` with frontier < n and a continuation **identical** to a cold run of the same
   prompt (greedy). See the Verification section below.
2. **Regression:** `--host-cache-mib=0` byte-for-byte M1 behavior; decode tok/s parity on the
   acceptance workload; TP4 smoke with the host tier enabled.

All code tasks (T3 tick call sites, T4 restore, T5 eviction wiring, T6 serve config, T7 unit tests)
are implemented in-tree and the build + unit tests are green. The pin-leak fix noted in the M2 status
above was found while writing the T7 slab-reconciliation test.

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

## 7. Configuration surface (proposed)

| Option | Default | Meaning |
|---|---|---|
| `--host-cache-mib N` | `0` | Pinned host RAM for the prefix pool; `0` = device-only (M1). |
| `--prefix-cache-file PATH` | unset | Persist the host pool across restarts (M3). |
| `--use-original-prefix-caching` | off | Keep the current owner/checkpoint path instead of the hybrid one. |
| `--max-new-taps N` | `8` / `2` without host tier | Cap snapshots per request. |
| `--tap-ladder-tokens G` | e.g. `8192` | Ladder spacing `n − G·2^k`. |
| `--tap-min-gap` | `max(1024, chunk)` | Minimum gap between ladder taps. |

Per HPC §14.1, in hybrid mode the legacy owner/checkpoint machinery is not constructed; it remains
behind `--use-original-prefix-caching` for comparison.

---

## 8. Risks and open questions

- **Scope.** This is a new cache subsystem (HPC is ≈6 k lines plus a host arena), not a patch. M1 is
  the smallest cut that removes the OpenCode hard-miss; do not under-budget M2.
- **TP-sharded snapshots.** HPC is single-GPU. Per-rank snapshot capture/restore and per-rank graph
  capture must be designed here; the cross-rank reduce is unaffected and no state sync is needed.
- **State image size.** Verify the per-rank image against the real layouts before sizing device
  snapshot slots and the host pool; DFlash2 rings can dominate.
- **Rewriting clients.** Reuse will still restart at the *deepest surviving boundary* before a
  rewrite. If a client rewrites the very first tool block, only the system/tools preamble is
  reusable — that is the floor, and it is a large improvement over a full reset.
- **Contract.** Adopting this changes the supported-identity description; update
  `docs/README.md`, `docs/cli.md`/`docs/serving.md` (`--tp`, cache options) and the
  `EngineOptions` comments together when it lands.
