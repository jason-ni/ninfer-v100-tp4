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

Deliverables:

- Pinned host slab pool (`--host-cache-mib`), one allocator for KV blocks and snapshot images.
- Async write-through of blocks and snapshot images on lane release; device eviction never copies.
- Restore batches on a dedicated per-rank stream, ordered by forward-pass read order, overlapping
  the restored lane's first prefill; node/tail/image `Filling` states and pins so no eviction frees
  an in-flight destination.
- Device LRU (tail-first) + host GDSF with superseded-first and dead-KV-first ordering.

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
