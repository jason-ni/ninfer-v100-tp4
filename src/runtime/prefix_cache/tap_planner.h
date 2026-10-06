#pragma once

// Deterministic tap planner for the hybrid prefix cache (HPC section 7.1).
//
// The planner is a pure function of the prompt's structural hints: it selects the positions at
// which the runtime should snapshot reusable state. It never touches tokens or device memory; the
// caller supplies hints the Frontend already knows and receives positions with exact/flexible
// placement. "Exact" taps split the prefill chunk; "flexible" taps are taken at the chunk
// boundary that reaches the position.

#include "runtime/prefix_cache/prefix_index.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace ninfer::prefix_cache {

struct TapHint {
    // Client-named breakpoints. Highest priority, always kept, published as Boundary.
    std::vector<std::uint32_t> explicit_markers;
    // Start of the final assistant generation opener; absent when this request resumed from an
    // endpoint snapshot (its endpoint already serves the next turn). Published as OutputBoundary.
    std::vector<std::uint32_t> generation_openers;
    // End of tools / end of the leading System/Developer block. Published as Boundary.
    std::vector<std::uint32_t> structural_boundaries;
    // Protocol-automatic markers (OpenAI/Anthropic default caching). Published as Tap, not shared.
    std::vector<std::uint32_t> automatic_markers;
    // Message start positions the geometric ladder may snap back onto.
    std::vector<std::uint32_t> message_boundaries;
    // Frontiers of snapshots already present on the matched path.
    std::vector<std::uint32_t> existing_frontiers;
    // Half-open token spans occupied by one Vision item; a frontier inside one is invalid.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> vision_spans;
};

struct TapPlanOptions {
    std::uint32_t prompt_tokens   = 0;    // n
    std::uint32_t base_frontier   = 0;    // F_s (the frontier this request resumes from)
    std::uint32_t ladder_tokens   = 8192; // G
    std::uint32_t min_gap_tokens  = 1024; // ladder spacing on both sides
    std::uint32_t max_new_taps    = 2;
};

struct PlannedTap {
    std::uint32_t position = 0;
    bool exact             = false;
    SnapshotKind kind      = SnapshotKind::Tap;
};

// Returns taps sorted ascending by position, at most `max_new_taps`, all in
// (base_frontier, prompt_tokens-1] and outside every Vision span.
[[nodiscard]] std::vector<PlannedTap> plan_taps(const TapPlanOptions& options,
                                                const TapHint& hints);

} // namespace ninfer::prefix_cache
