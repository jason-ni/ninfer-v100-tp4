#include "runtime/prefix_cache/tap_planner.h"

#include <algorithm>
#include <limits>

namespace ninfer::prefix_cache {
namespace {

struct Candidate {
    std::uint32_t position = 0;
    std::uint8_t priority  = 0;
    bool exact             = false;
    SnapshotKind kind      = SnapshotKind::Tap;
};

std::uint32_t snap_to_vision(
    std::uint32_t position,
    const std::vector<std::pair<std::uint32_t, std::uint32_t>>& spans) {
    bool moved = true;
    while (moved) {
        moved = false;
        for (const auto& span : spans) {
            if (position > span.first && position < span.second) {
                position = span.first;
                moved    = true;
                break;
            }
        }
    }
    return position;
}

// Largest message boundary at or below `position` within `min_gap`, else `position`.
std::uint32_t snap_to_boundary(std::uint32_t position, std::uint32_t min_gap,
                               const std::vector<std::uint32_t>& boundaries) {
    std::uint32_t best = position;
    for (std::uint32_t boundary : boundaries) {
        if (boundary > position) { break; }
        if (position - boundary <= min_gap) { best = boundary; }
    }
    return best;
}

} // namespace

std::vector<PlannedTap> plan_taps(const TapPlanOptions& options, const TapHint& hints) {
    std::vector<PlannedTap> result;
    const std::uint32_t token_count = options.prompt_tokens;
    if (token_count == 0 || options.max_new_taps == 0) { return result; }

    const std::uint32_t hi   = token_count - 1U;
    const std::uint32_t base = options.base_frontier;

    const auto valid = [&](std::uint32_t position) { return position > base && position <= hi; };
    const auto add   = [&](std::uint32_t position, std::uint8_t priority, bool exact,
                         SnapshotKind kind, std::vector<Candidate>& out) {
        position = snap_to_vision(position, hints.vision_spans);
        if (valid(position)) { out.push_back(Candidate{position, priority, exact, kind}); }
    };

    std::vector<Candidate> candidates;
    for (std::uint32_t position : hints.explicit_markers) {
        add(position, 1, true, SnapshotKind::Boundary, candidates);
    }
    for (std::uint32_t position : hints.generation_openers) {
        add(position, 2, true, SnapshotKind::OutputBoundary, candidates);
    }
    for (std::uint32_t position : hints.structural_boundaries) {
        add(position, 3, true, SnapshotKind::Boundary, candidates);
    }
    for (std::uint32_t position : hints.automatic_markers) {
        add(position, 3, true, SnapshotKind::Tap, candidates);
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  if (a.position != b.position) { return a.position < b.position; }
                  return a.priority < b.priority;
              });
    std::vector<Candidate> unique;
    unique.reserve(candidates.size());
    for (const Candidate& candidate : candidates) {
        if (!unique.empty() && unique.back().position == candidate.position) {
            if (candidate.priority < unique.back().priority) { unique.back() = candidate; }
            continue;
        }
        unique.push_back(candidate);
    }
    candidates = std::move(unique);

    std::vector<Candidate> accepted;
    std::vector<std::uint32_t> accepted_positions;

    const auto blocked64 = [&](std::uint32_t position) {
        const auto within = [&](std::uint32_t earlier) {
            return earlier <= position && position - earlier <= kBlockTokens;
        };
        for (std::uint32_t accepted_position : accepted_positions) {
            if (within(accepted_position)) { return true; }
        }
        for (std::uint32_t existing : hints.existing_frontiers) {
            if (within(existing)) { return true; }
        }
        return within(base);
    };

    for (const Candidate& candidate : candidates) {
        if (candidate.priority != 1 && blocked64(candidate.position)) { continue; }
        accepted.push_back(candidate);
        accepted_positions.push_back(candidate.position);
    }

    // Prompt tail: one token is always left to prefill, so a full-retry resumes at n-1.
    {
        const std::uint32_t tail = snap_to_vision(hi, hints.vision_spans);
        if (valid(tail) && !blocked64(tail)) {
            accepted.push_back(Candidate{tail, 4, false, SnapshotKind::Tap});
            accepted_positions.push_back(tail);
        }
    }

    // Geometric ladder n - G*2^k, snapped back to a message boundary within the minimum gap.
    const std::uint32_t gap = std::max<std::uint32_t>(1U, options.min_gap_tokens);
    const auto blocked_gap  = [&](std::uint32_t position,
                                 const std::vector<std::uint32_t>& ladder) {
        const auto near = [&](std::uint32_t other) {
            return other <= position ? position - other < gap : other - position < gap;
        };
        for (std::uint32_t accepted_position : accepted_positions) {
            if (near(accepted_position)) { return true; }
        }
        for (std::uint32_t existing : hints.existing_frontiers) {
            if (near(existing)) { return true; }
        }
        for (std::uint32_t previously : ladder) {
            if (near(previously)) { return true; }
        }
        return near(base);
    };

    std::vector<std::uint32_t> ladder_positions;
    if (options.ladder_tokens > 0) {
        std::uint64_t step = options.ladder_tokens;
        while (step < token_count) {
            const std::uint32_t raw = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(token_count) - step);
            if (raw <= base) { break; }
            std::uint32_t position =
                snap_to_vision(snap_to_boundary(raw, options.min_gap_tokens,
                                                hints.message_boundaries),
                               hints.vision_spans);
            if (valid(position) && !blocked_gap(position, ladder_positions)) {
                accepted.push_back(Candidate{position, 5, false, SnapshotKind::Tap});
                accepted_positions.push_back(position);
                ladder_positions.push_back(position);
            }
            if (step > (std::numeric_limits<std::uint64_t>::max() >> 1U)) { break; }
            step <<= 1U;
        }
    }

    std::sort(accepted.begin(), accepted.end(), [](const Candidate& a, const Candidate& b) {
        if (a.priority != b.priority) { return a.priority < b.priority; }
        return a.position < b.position;
    });
    if (accepted.size() > options.max_new_taps) { accepted.resize(options.max_new_taps); }
    std::sort(accepted.begin(), accepted.end(),
              [](const Candidate& a, const Candidate& b) { return a.position < b.position; });

    result.reserve(accepted.size());
    for (const Candidate& candidate : accepted) {
        result.push_back(PlannedTap{candidate.position, candidate.exact, candidate.kind});
    }
    return result;
}

} // namespace ninfer::prefix_cache
