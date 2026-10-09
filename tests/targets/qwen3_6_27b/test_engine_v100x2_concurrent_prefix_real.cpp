// Concurrent shared-prefix reuse gate: the "several long agent requests at once" shape.
//
// A served agent session sends a growing conversation, so every turn reuses the retained prefix.
// When several such sessions are live at once, their lanes borrow the SAME content-addressed
// prefix pages concurrently while decoding. This leg seeds one shared prefix, then admits one lane
// per distinct suffix in a single batched round -- so all lanes simultaneously restore/borrow the
// same retained blocks -- and requires each lane's greedy answer to equal that lane's single-lane
// reuse answer, reproducibly and with the lanes still distinct.
//
// WHAT THIS CATCHES AND WHY IT IS NOT TRIVIAL. A race in the shared prefix -- a borrowed page
// evicted mid-use, a wrong restore slot, a batch index mapping into the wrong lane's KV/GDN -- does
// not crash; it silently corrupts the borrowed context, which is exactly the failure mode that
// turned a served request into malformed tail output. Requiring the batched-reuse answer to match
// the single-lane-reuse answer catches that immediately on a lane whose shared prefix fully
// determines the greedy continuation.
//
// Batch composition can move a greedy knife edge for last-bit reasons unrelated to any defect (see
// test_engine_tp2_real.cpp's batching leg), so the criterion is: at least one lane must be exact,
// and every lane must at least match its own single-lane prefix. Reproducibility (the same batched
// round twice) is the discriminator for a real race between the two devices' streams.
//
// NINFER_V100X2_ARTIFACT=/path/to/model.ninfer ctest -R v100x2_concurrent_prefix

#include "ninfer/engine.h"
#include "v100x2_test_profile.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kLanes        = 4;
// Per-lane logical ceiling. The explicit pool is capped at kLanes * kContext, so this is sized well
// above the actual ~2k-token sequence to leave the pool headroom the retained shared prefix needs.
constexpr std::uint32_t kContext      = 3072;
constexpr std::uint32_t kOutput       = 32;
constexpr std::uint32_t kSuffixTokens = 8;
using Tokens = std::vector<ninfer::TokenId>;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::EngineOptions engine_options(const char* artifact, std::uint32_t host_mib = 0) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.tp            = 2;
    options.devices       = {0, 1};
    std::uint32_t max_ctx = kContext;
    if (const char* e = std::getenv("NINFER_MAX_CTX")) { max_ctx = static_cast<std::uint32_t>(std::stoul(e)); }
    options.max_context   = max_ctx;
    // Room for all four lanes' full sequences plus the retained prefix they all borrow.
    std::uint64_t kv_tokens = static_cast<std::uint64_t>(kLanes) * max_ctx;
    if (const char* e = std::getenv("NINFER_KV_TOKENS")) { kv_tokens = std::stoull(e); }
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(kv_tokens));
    options.prefill_chunk = 256;
    options.kv_cache      = ninfer::KvCacheStorage::Int8Group64;
    // Eager is the product path at tp > 1 on the V100 target.
    options.use_cuda_graph       = false;
    options.max_concurrency      = kLanes;
    options.max_pending_requests = kLanes;
    options.speculative          = v100x2_test::profile(ninfer::ProposalHead::Optimized);
    if (std::getenv("NINFER_GATE_NOSPEC") != nullptr) {
        options.speculative.backend       = ninfer::SpeculativeBackend::None;
        options.speculative.draft_tokens  = 0;
        options.speculative.proposal_head = ninfer::ProposalHead::Full;
    }
    options.prefix_cache.enabled        = true;
    options.prefix_cache.snapshot_slots = 2 * kLanes;
    options.prefix_cache.max_new_taps   = 4;
    options.prefix_cache.ladder_tokens  = 512;
    options.prefix_cache.min_gap        = 256;
    options.prefix_cache.host_cache_mib = host_mib;
    return options;
}

ninfer::RequestOptions request_options(bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = kOutput;
    options.execution.allow_prefix_reuse      = reuse;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.sampling.presence_penalty  = 0.0F;
    options.execution.sampling.frequency_penalty = 0.0F;
    options.stop.include_model_defaults       = false;
    return options;
}

ninfer::ChatMessage message(ninfer::ChatRole role, std::string text) {
    ninfer::ChatMessage result;
    result.role = role;
    result.parts.push_back({ninfer::MessagePartKind::Text, std::move(text), {}});
    return result;
}

// A ~2k-token shared conversation prefix, long enough to span several ladder taps and to make the
// cached continuation depend on the whole retained block path.
ninfer::PromptInput shared_prefix_input() {
    std::string text = "Review these Python functions. Then implement a bounded FIFO queue "
                       "with push, pop, and unit tests. Explain empty and full behavior.\n\n";
    for (int i = 0; i < 120; ++i) {
        text += "def add_" + std::to_string(i) + "(value):\n    return value + " +
                std::to_string(i) + "\n";
    }
    ninfer::PromptInput input;
    input.messages.push_back(message(ninfer::ChatRole::User, std::move(text)));
    return input;
}

// A disjoint, equally long conversation used to fill the device pool so the shared prefix is
// evicted from device and demoted to the host tier. Its text shares no prefix with `shared_prefix_input`.
ninfer::PromptInput disjoint_input(std::uint32_t variant) {
    std::string text = "Conversation " + std::to_string(variant) +
                       ": analyze the complexity of these routines and write a thread-safe LRU "
                       "cache with get, put, and capacity-eviction tests.\n\n";
    for (int i = 0; i < 150; ++i) {
        text += "def sub" + std::to_string(variant) + "_" + std::to_string(i) +
                "(value):\n    return value - " + std::to_string(i) + "\n";
    }
    ninfer::PromptInput input;
    input.messages.push_back(message(ninfer::ChatRole::User, std::move(text)));
    return input;
}

// One lane per distinct suffix, so the four prompts share the whole prefix and then diverge; the
// lanes must stay distinct through the batched decode.
Tokens lane_tokens(const Tokens& prefix, std::uint32_t lane) {
    Tokens result = prefix;
    for (std::uint32_t j = 0; j < kSuffixTokens; ++j) {
        result.push_back(static_cast<ninfer::TokenId>(3000 + lane * 1000 + j * 17));
    }
    return result;
}

// Seed the shared prefix cold: its block path lands on device and is captured as reusable taps.
void seed(ninfer::Engine& engine, const ninfer::PromptInput& input) {
    auto result = engine.generate(engine.prepare(input), request_options(false));
    require(result.generated_token_ids.size() == kOutput, "seed did not generate its tokens");
    require(result.reused_prompt_tokens == 0, "seed unexpectedly reused a prefix");
}

// Sequential single-lane reuse answer for every lane, each resuming from the shared prefix.
std::array<Tokens, kLanes> reference(ninfer::Engine& engine, const Tokens& prefix) {
    std::array<Tokens, kLanes> ref;
    std::array<std::uint32_t, kLanes> depth{};
    for (std::uint32_t lane = 0; lane < kLanes; ++lane) {
        const auto prompt = lane_tokens(prefix, lane);
        const auto result =
            engine.generate(engine.prepare_tokens(prompt, true), request_options(true));
        require(result.generated_token_ids.size() == kOutput,
                "reference lane did not generate its tokens");
        require(result.reused_prompt_tokens > 0 &&
                    result.reused_prompt_tokens + kSuffixTokens + kOutput <= prompt.size(),
                "reference lane did not resume from the shared prefix");
        ref[lane] = result.generated_token_ids;
        depth[lane] = result.reused_prompt_tokens;
    }
    for (std::uint32_t lane = 0; lane < kLanes; ++lane) {
        require(depth[lane] > 0 && depth[lane] == depth[0],
                "reference lanes resumed from different prefix depths");
    }
    std::cout << "reference: shared depth=" << depth[0] << " lanes distinct" << std::endl;
    return ref;
}

struct LaneOutcome {
    Tokens tokens;
    std::uint32_t reused = 0;
};

struct ConcurrentRound {
    std::array<LaneOutcome, kLanes> lanes;
    std::uint64_t rounds = 0;
    std::uint64_t rows   = 0;
};

// Submit all lanes up front so the scheduler places them in one batched round; every lane reuses
// the shared prefix while it decodes alongside the others.
ConcurrentRound run_concurrent(ninfer::Engine& engine, const Tokens& prefix) {
    const ninfer::RuntimeStats before = engine.runtime_stats();
    std::vector<ninfer::PreparedPrompt> prompts;
    prompts.reserve(kLanes);
    for (std::uint32_t lane = 0; lane < kLanes; ++lane) {
        prompts.push_back(engine.prepare_tokens(lane_tokens(prefix, lane), true));
    }
    std::vector<ninfer::GenerationHandle> handles;
    handles.reserve(kLanes);
    for (auto& prompt : prompts) {
        handles.push_back(engine.submit(std::move(prompt), request_options(true)));
    }
    ConcurrentRound out;
    for (std::uint32_t lane = 0; lane < kLanes; ++lane) {
        auto result             = handles[lane].wait();
        out.lanes[lane].tokens  = std::move(result.generated_token_ids);
        out.lanes[lane].reused  = result.reused_prompt_tokens;
    }
    const ninfer::RuntimeStats after = engine.runtime_stats();
    out.rounds = after.decode_rounds - before.decode_rounds;
    out.rows   = after.decode_row_rounds - before.decode_row_rounds;
    return out;
}

std::size_t prefix_agreement(const Tokens& a, const Tokens& b) {
    std::size_t agreed = 0;
    while (agreed < a.size() && agreed < b.size() && a[agreed] == b[agreed]) { ++agreed; }
    return agreed;
}

// The reference lane answer must be reproduced exactly, or the batched-reuse path changed the
// context. `test_engine_tp2_real.cpp` documents that one prompt can sit on a greedy knife edge
// where any last-bit perturbation flips the continuation, so the leg only has full strength on a
// lane that IS batch-invariant; it still requires every lane to be at least as faithful as the
// single-lane answer (agreement is measured against that answer, so a corrupted lane collapses to
// a short prefix and fails).
void require_consistent(const std::array<Tokens, kLanes>& ref, const ConcurrentRound& round) {
    require(round.rows > round.rounds, "the concurrent round did not batch multiple lanes");
    bool distinct = true;
    for (std::uint32_t i = 0; i < kLanes; ++i) {
        require(round.lanes[i].tokens.size() == kOutput,
                "a concurrent lane did not generate its tokens");
        require(round.lanes[i].reused > 0, "a concurrent lane did not reuse the shared prefix");
        for (std::uint32_t j = i + 1; j < kLanes; ++j) {
            if (round.lanes[i].tokens == round.lanes[j].tokens) { distinct = false; }
        }
    }
    require(distinct, "two concurrent lanes produced the same answer -- state is being shared");

    std::size_t exact = 0;
    for (std::uint32_t lane = 0; lane < kLanes; ++lane) {
        const std::size_t agreed = prefix_agreement(ref[lane], round.lanes[lane].tokens);
        std::cout << "lane " << lane << ": reused=" << round.lanes[lane].reused
                  << " agreement=" << agreed << "/" << kOutput << std::endl;
        if (agreed == kOutput) { ++exact; }
    }
    require(exact == kLanes,
            "batched shared-prefix reuse changed a lane's greedy continuation");
}

struct Reference {
    Tokens prefix;
    std::array<Tokens, kLanes> answers;
};

// The correct answers, computed once on a device-only engine whose lanes reuse the shared prefix
// sequentially. Both subject scenarios are measured against these.
Reference device_reference(const char* artifact, const ninfer::PromptInput& input) {
    ninfer::Engine engine(engine_options(artifact));
    v100x2_test::check_identity(engine);
    Reference out;
    out.prefix = engine.prepare(input).debug_token_ids();
    require(out.prefix.size() >= 1024 && out.prefix.size() + 512 < kContext,
            "shared prefix must be long enough to span several taps and leave capacity headroom");
    seed(engine, input);
    out.answers = reference(engine, out.prefix);
    return out;
}

void require_reproducible(const ConcurrentRound& first, const ConcurrentRound& second) {
    for (std::uint32_t lane = 0; lane < kLanes; ++lane) {
        require(first.lanes[lane].tokens == second.lanes[lane].tokens,
                "batched shared-prefix decode is not reproducible run to run");
    }
    std::cout << "batched round: " << first.rows << " rows over " << first.rounds << " rounds"
              << std::endl;
}

// Device borrow: the four lanes borrow the shared prefix's device pages while decoding together.
void exercise_device(const char* artifact, const ninfer::PromptInput& input, const Reference& ref) {
    ninfer::Engine engine(engine_options(artifact));
    v100x2_test::check_identity(engine);
    seed(engine, input);
    const auto first  = run_concurrent(engine, ref.prefix);
    const auto second = run_concurrent(engine, ref.prefix);
    require_reproducible(first, second);
    require_consistent(ref.answers, first);
    std::cout << "V100X2 concurrent shared-prefix reuse (device): PASS" << std::endl;
}

// Host tier: fill the device pool with disjoint conversations so the shared prefix is evicted from
// device and demoted to the host tier, then have all four lanes restore it concurrently while they
// decode together. The restored context must still reproduce the device-path answers exactly.
void exercise_host(const char* artifact, const ninfer::PromptInput& input, const Reference& ref) {
    ninfer::Engine engine(engine_options(artifact, 24576));
    v100x2_test::check_identity(engine);
    seed(engine, input);
    for (std::uint32_t variant = 0; variant < kLanes; ++variant) {
        auto result =
            engine.generate(engine.prepare(disjoint_input(variant)), request_options(false));
        require(result.generated_token_ids.size() == kOutput,
                "disjoint pool filler did not generate its tokens");
    }
    // Diagnostic split: first the same lanes alone (host-restored sequentially), then concurrently.
    // If the single-lane host path already disagrees with the device answers, the restore is wrong
    // on its own, independent of concurrency.
    const auto host_single = reference(engine, ref.prefix);
    // Triangulation on lane 0: a cold recompute in this same host-tier engine. If cold matches the
    // device answer but the restored answer does not, the restore (not the engine's math) is wrong.
    {
        const auto lane0 = lane_tokens(ref.prefix, 0);
        const auto cold  = engine.generate(engine.prepare_tokens(lane0, false), request_options(false));
        std::cout << "host lane0: restore_vs_device="
                  << prefix_agreement(ref.answers[0], host_single[0]) << "/" << kOutput
                  << " cold_vs_device=" << prefix_agreement(ref.answers[0], cold.generated_token_ids)
                  << "/" << kOutput
                  << " restore_vs_cold="
                  << prefix_agreement(host_single[0], cold.generated_token_ids) << "/" << kOutput
                  << std::endl;
    }
    bool host_single_ok = true;
    for (std::uint32_t lane = 0; lane < kLanes; ++lane) {
        std::cout << "host single lane " << lane
                  << ": agreement=" << prefix_agreement(ref.answers[lane], host_single[lane]) << "/"
                  << kOutput << std::endl;
        if (host_single[lane] != ref.answers[lane]) { host_single_ok = false; }
    }
    require(host_single_ok, "single-lane host restore did not reproduce the device answer");
    const auto first  = run_concurrent(engine, ref.prefix);
    const auto second = run_concurrent(engine, ref.prefix);
    require_reproducible(first, second);
    require_consistent(ref.answers, first);
    std::cout << "V100X2 concurrent shared-prefix reuse (host tier): PASS" << std::endl;
}

void exercise(const char* artifact) {
    const ninfer::PromptInput input = shared_prefix_input();
    const Reference ref              = device_reference(artifact, input);
    exercise_device(artifact, input, ref);
    exercise_host(artifact, input, ref);
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_V100X2_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_V100X2_ARTIFACT is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: V100X2 concurrent prefix gate requires two CUDA devices\n";
        return 77;
    }
    try {
        exercise(artifact);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "V100X2 concurrent prefix: " << error.what() << '\n';
        return 1;
    }
}
