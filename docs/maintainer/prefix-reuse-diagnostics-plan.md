# Prefix-reuse diagnostics and demand-aware eviction: plan

Status: **active** — branch `local/prefix-reuse-diagnostics`.

This plan adds two things to the hybrid prefix cache path, motivated by an observed failure mode
in the running opencode/MTP deployment (`run-tp4.sh`, `/media/zt/models/orcarouter/conv_fp16vision/
qwen3_8_27b_nvfp4.ninfer`): re-prefill on every agent turn even though the pinned host tier shows
~half free and the device snapshot slots read as the scarce resource (5/5 at 12:17 on Oct 15). The
cause was not identifiable from logs because every hybrid miss is silently dropped inside
`plan_request_for_lane` and only the effective `PrefixReuseReason` (which the legacy path may
overwrite) reaches the request log.

## Deliverable 1: explain every prefix reuse decision

One value struct carries the complete decision record of a request's prefix-reuse gates from plan
time to the done log. It answers, for any request that recomputed N tokens, *which gate* let them
through and *what the cache actually contained* at that moment.

### New public value: `ninfer::PrefixAdmissionDiagnostics` (`include/ninfer/types.h`)

Captured once per request in `plan_request_for_lane` (the single site that sees every gate).
Sections:

- **Input facts** — always filled: `prompt_tokens`, `eligible_blocks = (n-1)/64`,
  `allow_prefix_reuse`, `identity_reusable`, whether the hybrid index is enabled, whether a
  retained sequence exists (+ its frontier), `mtp_kv_valid` / `tail_hidden_valid` / MTP cache
  presence (speculative-gate inputs), device snapshot slot `total`/`used`, `max_new_taps`.
- **Hybrid walk result** — filled when a match ran: `matched_blocks`, whether the walk completed
  all eligible blocks, stall cause (`no-child` vs `hash-collision`), sibling count under the stall
  parent, per-candidate fate (up to 8 deepest, each: frontier, anchor depth, kind, dev/host
  residency, reject code), and selection veto (chosen snapshot vanished / invalid frontier /
  non-tap tail) when a best candidate was found but rejected.
- **Outcome** — `reuse_base` (== hit tokens), the effective path/reason (mirrored from the existing
  fields; the diag struct adds the detail those enums cannot carry).

No string is stored; presentation lives in serve.

### Plumbing (follows the existing `PrefixDivergence` pattern exactly)

`PrefixIndex::match()` gains the walk/candidate diagnostics on `PrefixMatch` →
`RequestPlanImpl::reuse_diag` (program.h) → staged `Prefill::reuse_diag` (initialized at
program_impl.h:1280) → `runtime::BeginSummary::prefix_admission` (contract/types.h, at the three
begin-summary sites) → `ninfer::GenerationResult::prefix_admission` (types.h, copied in
concurrent_executor.h:510/531) → `serve::GenerationMetrics::prefix_admission`
(generation_service.cpp) → request log.

### Presentation

- **Console done line** (`request_log.cpp`): an extra `px=...` compact segment appended whenever
  `reuse_base < prompt_tokens` (any recompute). Zero added noise on full-hit requests. Segments:
  walk outcome, stall details, candidate summary, selection veto, retained/frontier state,
  MTP-gate shortfall, slot usage, and the disk tier's occupancy plus a `from=disk` marker when the
  consumed snapshot was materialized out of the disk tier. Token windows stay out of the compact line.
- **Verbose explanation**: new serve flag `--prefix-debug` (no env fallback). On every request_done,
  when recompute > 0, a multi-line English block prints the gate-by-gate verdict with values and a
  final "why this request prefilled X tokens" conclusion including the divergence window and the
  snapshot inventory. One block per request, keyed by req id.
- **JSONL** (`request_done` events): structured `result.prefix_admission {…}` object mirroring the
  struct, so machine analysis does not rely on parsing the compact line. Per-candidate fates carry
  `device_resident`/`host_restorable`/`disk_restorable`, and the object carries a `disk_tier`
  occupancy block with `served_from_disk`.

Eviction visibility rides in too: the counters below make "did pressure kill my chain" readable
without guessing.

### Verification

- Unit: extend `tests/test_prefix_cache_index.cpp` with walk-stall causes, candidate fates,
  selection vetoes, and the compact/verbose renderers (string-mapping regression guard).
- Live: after deploy on the running TP4 box, one opencode multi-turn session must show:
  turn ≥ 2 either reuses (diag absent from px=) or shows the exact gate + a readable cause.

## Deliverable 2: demand-aware two-tier device eviction

Current `evict_device_lru()` picks the LRU head (oldest unpinned), passes 1 demotes backed nodes,
pass 2 erases a whole subtree. Observed consequences: after finalize (which unpins shallow→deep),
the LRU head is the *shallowest* block, so a single-conversation tree loses its prefix end first,
and pass 2 ignores snapshots (a live tap dies with its subtree).

New policy (agreed): **demand awareness first, geometry second.** Eviction knows which blocks
incoming requests will need (pending plans' `block_path`s + retained commit paths + active lanes'
pinned sets) and optimizes for that instead of raw recency:

1. **Tier 1 — lossless demote:** prefer demotion (device→host) over any deletion. Demote the
   *deepest* non-demand backed node (older recency tie-break), so hot shared shallows stay
   device-resident while conversation tails shed their faster residency.
2. **Tier 2 — dead-branch GC (lossy, data-freeing):** among unpinned device residents whose branch
   holds no demand, erase the *shallowest* branch root (oldest recency first). Removing the branch
   root atomically frees the whole dead branch — old rewritten tool-call histories, superseded
   renderings — including deep tails and dead snapshots, per call. Branches anchoring a live
   non-superseded snapshot keep their resumability investment and are deferred.
3. **Tier 3 — demand-side trim (capacity genuinely exceeded):** only when tiers 1+2 are exhausted,
   trim ONTO the demand set (or the unbacked remainder with no demanded descent below), deepest
   leaf first, demote-preferred — preserving the head plus the inner geometric-ladder taps
   (resumable at a shallower frontier).

Every call still releases exactly the credited pages (one page per pool through the sink), so the
admission credit invariants (`evictable_page_counts`, the bad_alloc retry loops at
program_impl.h:1717/1769/2995) are unchanged; only victim selection moves. Retained-lane eviction
(`evict_retained_lane`) is untouched but counted and logged.

The executor passes the demand set into the program's admission entry points (it owns the pending
plans); program passes it to the new index API `evict(const Demand&)`. No pin-state changes across
requests are introduced.

### Metrics (new, `PrefixCacheMetrics` + metrics_format)

Cumulative counters, rank 0: `device_blocks_demoted_total`, `subtree_erased_total`,
`branch_gc_removed_nodes_total`, `snapshots_deleted_by_eviction_total`,
`retained_lane_evictions_total`. These convert the previous blind spot ("pool had room") into
measurable behavior.

### Verification

- Unit: policy tests in `test_prefix_cache_index.cpp` — dead branch preferred over demanded tail;
  shallowest-dead wins; snapshot-anchored branch deferred; tier 3 trims deepest demanded leaf;
  credit/ownership invariants hold (page counts returned through the sink equal freed pages).
- Behavioral AB on the TP4 box: same opencode multi-turn workload before/after; compare
  `computed_prefill_tokens` per turn and the reason distribution. A change here is a performance
  claim and gets judged on that measurement only.

## Change map

| Area | Files |
|---|---|
| Public value | `include/ninfer/types.h` (`PrefixAdmissionDiagnostics`, `PrefixCacheMetrics` counters) |
| Index core | `src/runtime/prefix_cache/prefix_index.{h,cpp}` (match diagnostics; `evict(const Demand&)`; counters) |
| Planner capture | `src/targets/qwen3_6/impl/runtime/request_plan_impl.h` (single fill site) |
| Threading | `program.h` (plan + staged), `program_impl.h` (staged init, begin summaries ×3, admission call sites, metrics fill), `src/runtime/contract/types.h` (BeginSummary), `src/runtime/engine/concurrent_executor.h` (result copy, demand assembly), `raw_output_log.{h,cpp}` (debug mirror) |
| Serve | `serve_options.{h,cpp}` (`--prefix-debug`), `generation_service.{h,cpp}`, `request_log.{h,cpp}` (compact/verbose/JSONL), `metrics_format.cpp` (counters) |
| Tests | `tests/test_prefix_cache_index.cpp` (walk/candidate diagnostics, three-tier eviction, demand protection, counters), `tests/test_prefix_cache_host.cpp` (tier-1 deepest demotion + demand case), `tests/test_request_log.cpp` (formatter cases), `tests/test_concurrent_executor.cpp` (fake program demand API) |
| Docs | `docs/cli.md` (`--prefix-debug`), `docs/serving.md` (metrics table), this plan's milestone table |

Out of scope: host-slab eviction policy (its dead/superseded/live priority already mirrors tier 1),
snapshot slot GDSF tuning, and template-render fixes (the diagnostics exist to expose those, not
substitute them).

## Milestones

| # | Item | Status |
|---|---|---|
| D1a | `PrefixAdmissionDiagnostics` + plumbing to done log | done |
| D1b | `PrefixIndex::match` walk/candidate diagnostics | done |
| D1c | Compact line + `--prefix-debug` verbose + JSONL block | done |
| D1d | Eviction/retained counters in /metrics | done |
| D2 | Two-tier demand-aware `evict` + executor demand assembly | done |
| V1 | Unit tests (index + formatters), build, targeted test run | done |
| V2 | Live AB on the TP4 box (reason distribution + computed_prefill_tokens) | pending |
| D3 | docs/cli.md + serving.md metrics + milestone closure | done |