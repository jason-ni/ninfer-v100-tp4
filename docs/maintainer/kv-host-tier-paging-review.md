# Code review: KV cache host-tier paging (device evict / host restore)

Scope: `src/core/paged_kv_cache.{h,cpp}`, `src/runtime/prefix_cache/{host_slab_pool,prefix_index}.{h,cpp}`,
and the program-side callers in `src/targets/qwen3_6/impl/runtime/program_impl.h`.

Status: one high-severity correctness bug (KV clobber on duplicate block commit), plus two
medium-severity issues and three minor ones. The tree already carries an "M2 host-restore clobber
investigation" (`paged_kv_cache.h:151`, `prefix_index.h:355`), and the primary finding below is a
concrete mechanism for exactly that clobber that the current `debug_page_referenced` probe does not
catch.

---

## 1. HIGH — duplicate block commit returns a live sequence page to the free pool (use-after-free)

**Where:** `src/targets/qwen3_6/impl/runtime/program_impl.h:3479-3546`

**What happens**

1. On reaching a new block boundary during prefill, the newly committed pages are promoted to
   cache-owned *before* they are handed to the index:

   ```cpp
   sequence.kv->text.promote_owned_pages(new_blocks);          // program_impl.h:3489
   ```

   `PagedKVAllocation::promote_owned_pages` only moves the pages from the allocation's *owned*
   range into its *borrowed* range (`paged_kv_cache.cpp:558-572`); the page ids stay in
   `page_ids_` and remain published in the device block table.

2. Blocks are then committed one by one. When an equal `(parent, tokens)` child already exists and
   is device-resident, `commit_block` does not take the offered pages and returns
   `adopted == false` (`prefix_index.cpp:368-386`).

3. The caller then does:

   ```cpp
   if (!adopted) {
       ...debug_page_referenced...
       release_cached_pages(pages);                            // program_impl.h:3543
   }
   ```

   `release_cached_pages` -> `PagedKVPool::release_cache_pages` pushes the page back onto
   `free_page_ids_` (`paged_kv_cache.cpp:417-435`).

**Why it is wrong**

`pages.page[0]` is `sequence.kv->text.page_ids()[b]` (`program_impl.h:3507`) — the page promoted in
step 1, which is still in the allocation's `page_ids_` (as borrowed) and still referenced by the
sequence's block table. Returning it to the free list makes it simultaneously "owned by the running
sequence's block table" and "free in the pool". A later allocation or cache node can acquire it and
write unrelated KV; the sequence then reads corrupted history. This is precisely the
`[PATCH] ... CHANGED` (same page, changed content) symptom that `debug_watch_prefix_pages` reports.

The existing probe at `program_impl.h:3534-3543` checks `debug_page_referenced(0, page)`, i.e.
whether an *index node* references the page. It does not, because the live reference here is the
sequence allocation, not an index node — so the probe stays silent and the release looks safe.

**Reachability**

Two concurrent requests sharing a prefix, both prefilling past the `match()` cutoff
`eligible = (n-1)/64` (only blocks `[0, eligible)` are matched), will both attempt to commit the
block at index `eligible`. The second one finds the first one's child already resident and takes the
`adopted == false` branch. `max_concurrency >= 2` on a shared long prefix should reproduce it.

**Fix direction**

Promotion must be conditional on adoption; the current batch promote-before-commit ordering cannot
express that.

- Preferred: commit first, promote per block only when `adopted == true` (and only then add the
  node to `prefix_pinned_nodes` / pin it). On `adopted == false` leave the page *owned* by the
  allocation, so it is not released here and is returned normally by `release()`; the sequence keeps
  using its own (identical-content) page.
- Alternative: add a `demote_from_cache(pages)` that returns the pages from cache-owned back to
  owned, and call that instead of `release_cached_pages`.

Do not simply drop the `release_cached_pages` call without restoring ownership, or the page leaks
(it stays borrowed and `release()` never returns it).

---

## 2. MEDIUM — `mapped_pages_` accounting is asymmetric across cache acquire/release

**Where:** `paged_kv_cache.cpp:365`, `402-435`

`acquire_cache_pages` goes through `take_pages`, which does `mapped_pages_ += count`
(`paged_kv_cache.cpp:365`), and then `cached_pages_ += count` (`:408`). `release_cache_pages`
decrements only `cached_pages_` (`:422`); the comment at `:419-421` assumes the page came from
`promote_to_cache` (which *does* decrement `mapped_pages_` at `:461`). For pages obtained through
`acquire_cache_pages` (the host-restore path uses `acquire_cache_bundle`), `mapped_pages_` is never
decremented, so it drifts upward while cache pages are recycled.

Impact today is limited: capacity predicates (`can_reserve`, `can_replace_entitlement`) use
`free_page_ids_`/`entitled_pages_`/`cached_pages_`, not `mapped_pages_`. The only consumer is the
guard in `promote_to_cache` (`:439`), which the drift makes weaker, plus the `mapped_pages()`
accessor. Still a latent accounting landmine; make acquire/release symmetric (e.g. decrement
`mapped_pages_` in `release_cache_pages` for acquired pages, or track cache pages outside
`mapped_pages_` entirely).

---

## 3. MEDIUM — `evict_device_lru` can report a phantom eviction and spin forever

**Where:** `prefix_index.cpp:644-679`, `711-756`

Pass 2 picks `lru_head_` and calls `erase_subtree(current)`, then `return current` regardless of what
`erase_subtree` did. `erase_subtree` aborts the whole operation (returns without unlinking or
releasing anything) if *any* node in the subtree has `active_refs != 0` or `restore_pins != 0`
(`:720`). The candidate itself is unpinned by construction (it was LRU-linked), but a descendant can
carry a `restore_pins` from an in-flight host write / device fill.

The callers loop:

```cpp
for (;;) {
    if (prefix_index->evict_device_lru() == kInvalidNode) throw;
    try { return acquire_once(); } catch (const std::bad_alloc&) { continue; }
}
```

(`program_impl.h:3030-3040`, `1852-1863`, `1907-1917`) — if the eviction freed nothing, the retry
fails again and the same `lru_head_` is returned, so the loop never terminates.

**Fix direction:** make `erase_subtree` return whether it actually erased, and have
`evict_device_lru` skip (advance past) candidates whose subtree cannot be erased, returning
`kInvalidNode` only when no candidate can make progress. Alternatively confine Pass 2 to evicting a
single unpinned node rather than a whole subtree.

---

## 4. Minor

- **Host slab metadata not accounted / no alignment guarantee.**
  `HostSlabPool` embeds the `HostSlab` array at the head of the same `cudaHostAlloc` buffer
  (`host_slab_pool.cpp:30-47`) but `free_bytes()` counts only slab payload
  (`host_slab_pool.cpp:80-82`). Slab data pointers are `HostSlab`-aligned (8B), not 256B; pinned
  copies are correct but not optimally aligned. If metadata + payload exceed `total_bytes`, the pool
  silently disables itself and the caller cannot tell (`host_slab_pool.cpp:49-56`); log a warning.

- **Silent host-tier disable degrades to device-only with no signal.** Combined with the above, a
  too-small `--prefix-cache-host-mib` loses the host tier without any diagnostic.

- **`release_snapshot_slot` in the sink is a no-op** (`program_impl.h:138`). Fine for the current
  static slot partitioning, but worth a comment so a future dynamic layout does not silently leak.

---

## Suggested verification

1. Run two concurrent requests sharing a long prefix with `NINFER_PREFIX_WATCH=1` and
   `max_concurrency >= 2`, prefill past the block-match boundary. Expect `[PATCH] ... CHANGED` on the
   duplicated block's page, and a nearby `[PCDBG] dup-release` followed by a `[KVDBG] TAKE` of the
   same page.
2. Add a temporary assertion in the `adopted == false` branch that the page is not referenced by the
   current sequence's block table (`sequence.kv->text.page_ids()`), to pin the bug directly.
3. After fixing, re-run the page checksum watch and the existing hybrid-prefix real-artifact tests
   (`tests/targets/qwen3_6_27b/test_engine_v100x2_hybrid_real.cpp`) with two concurrent shared-prefix
   requests.
