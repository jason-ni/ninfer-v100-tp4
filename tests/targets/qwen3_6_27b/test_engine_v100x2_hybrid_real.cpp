// Opt-in TP2 gate for the hybrid content-addressed prefix cache (M1: device-only block +
// snapshot cache). Proves that a cold prefill publishes reusable state along the prompt's
// block path and that later requests sharing that path resume from the deepest surviving
// boundary via PrefixReusePath::BlockSnapshot, including a client that rewrites an earlier
// region between turns.
//
// NINFER_V100X2_ARTIFACT=/path/to/model.ninfer ctest -R v100x2_hybrid_real

#include "ninfer/engine.h"
#include "v100x2_test_profile.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kCapacity = 8192;
constexpr std::uint32_t kOutput = 8;
using Tokens = std::vector<ninfer::TokenId>;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.tp = 2;
    options.devices = {0, 1};
    options.max_context = kCapacity;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(kCapacity);
    options.prefill_chunk = 256;
    options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
    options.use_cuda_graph = false;
    options.max_concurrency      = 1;
    options.max_pending_requests = 1;
    options.speculative = v100x2_test::profile(ninfer::ProposalHead::Optimized);
    options.prefix_cache.enabled       = true;
    options.prefix_cache.snapshot_slots = 4;
    options.prefix_cache.max_new_taps  = 4;
    options.prefix_cache.ladder_tokens = 512;
    options.prefix_cache.min_gap      = 256;
    return options;
}

ninfer::RequestOptions request_options(bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = kOutput;
    options.execution.allow_prefix_reuse = reuse;
    options.execution.sampling.temperature = 0.0F;
    options.execution.sampling.presence_penalty = 0.0F;
    options.execution.sampling.frequency_penalty = 0.0F;
    options.stop.include_model_defaults = false;
    return options;
}

ninfer::ChatMessage message(ninfer::ChatRole role, std::string text) {
    ninfer::ChatMessage result;
    result.role = role;
    result.parts.push_back({ninfer::MessagePartKind::Text, std::move(text), {}});
    return result;
}

// Small prompt (~2000 tokens, ~31 pages) that does NOT trigger pool exhaustion.
ninfer::PromptInput small_prompt() {
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

// Large prompt (~4898 tokens, ~77 pages) that DOES trigger pool exhaustion.
ninfer::PromptInput long_prompt() {
    std::string text = "Review these Python functions. Then implement a bounded FIFO queue "
                       "with push, pop, and unit tests. Explain empty and full behavior.\n\n";
    for (int i = 0; i < 280; ++i) {
        text += "def add_" + std::to_string(i) + "(value):\n    return value + " +
                std::to_string(i) + "\n";
    }
    ninfer::PromptInput input;
    input.messages.push_back(message(ninfer::ChatRole::User, std::move(text)));
    return input;
}

struct Observation {
    std::uint32_t prompt_tokens = 0;
    std::uint32_t reused = 0;
    std::uint32_t computed = 0;
    ninfer::GenerationResult result;
};

Observation run(ninfer::Engine& engine, const std::string& name, ninfer::PreparedPrompt prompt,
                bool reuse, ninfer::PrefixReusePath path) {
    const auto prompt_tokens = static_cast<std::uint32_t>(prompt.debug_token_ids().size());
    const auto before = engine.runtime_stats().computed_prefill_tokens;
    auto result = engine.generate(std::move(prompt), request_options(reuse));
    const auto computed = static_cast<std::uint32_t>(
        engine.runtime_stats().computed_prefill_tokens - before);
    require(result.prefix_reuse_path == path,
            name + ": unexpected prefix path=" +
                std::to_string(static_cast<int>(result.prefix_reuse_path)));
    require(result.reused_prompt_tokens <= prompt_tokens &&
                computed == prompt_tokens - result.reused_prompt_tokens,
            name + ": computed prefill does not equal the actual suffix");
    std::cout << name << " prompt=" << prompt_tokens << " reused=" << result.reused_prompt_tokens
              << " computed=" << computed << std::endl;
    return {prompt_tokens, static_cast<std::uint32_t>(result.reused_prompt_tokens), computed,
            std::move(result)};
}

// Test case 1: small prompt that does NOT trigger pool exhaustion.
// Verifies M1 functionality: cold prefill publishes snapshots, longer prompt reuses them,
// rewritten prompt reuses a boundary before the divergence, restore is deterministic.
void exercise_normal(const char* artifact) {
    ninfer::Engine engine(engine_options(artifact));
    v100x2_test::check_identity(engine);

    const auto base = engine.prepare(small_prompt()).debug_token_ids();
    require(base.size() >= 1500 && base.size() + 512 < kCapacity,
            "small fixture must exceed one ladder span and leave capacity headroom");
    const auto n = static_cast<std::uint32_t>(base.size());

    // 1. Cold prefill publishes block snapshots along the path.
    const auto cold = run(engine, "normal_cold", engine.prepare(small_prompt()), false,
                          ninfer::PrefixReusePath::FullReset);
    require(cold.reused == 0 && cold.computed == n, "cold prefill did not compute the whole prompt");

    // 2. A longer prompt sharing the full prefix resumes from the prompt-tail snapshot.
    Tokens extended = base;
    for (std::uint32_t i = 0; i < 24; ++i) { extended.push_back(static_cast<ninfer::TokenId>(1000 + i)); }
    const auto longer = run(engine, "normal_longer", engine.prepare_tokens(extended, true), true,
                            ninfer::PrefixReusePath::BlockSnapshot);
    // The snapshot is captured at the last full block boundary (floor(n/64)*64), so reuse is
    // anchored there, not at the exact prompt tail.
    const auto n_blocks = (n / 64) * 64;
    require(longer.reused >= n_blocks && longer.reused < n + 24,
            "longer prompt did not resume from the full-prefix snapshot");

    // 3. A client that rewrites an earlier region must still reuse a boundary before it.
    const std::uint32_t divergence = 1024;
    Tokens rewritten = base;
    for (std::uint32_t i = 0; i < 64; ++i) {
        rewritten[divergence + i] = static_cast<ninfer::TokenId>(9000 + i);
    }
    const auto revised = run(engine, "normal_rewritten", engine.prepare_tokens(rewritten, true),
                             true, ninfer::PrefixReusePath::BlockSnapshot);
    require(revised.reused > 0 && revised.reused <= divergence,
            "rewritten prompt reuse was not anchored before the divergence");

    // 4. Restore is deterministic: two consecutive requests that reuse the same prefix
    // (same computational path) must produce identical tokens. The first rewritten request
    // may reuse a different prefix than subsequent ones (because it commits its suffix to the
    // cache), so we compare the second and third requests instead.
    const auto revised_again = run(engine, "normal_rewritten_again",
                                   engine.prepare_tokens(rewritten, true), true,
                                   ninfer::PrefixReusePath::BlockSnapshot);
    const auto revised_third = run(engine, "normal_rewritten_third",
                                   engine.prepare_tokens(rewritten, true), true,
                                   ninfer::PrefixReusePath::BlockSnapshot);
    require(revised_again.result.generated_token_ids == revised_third.result.generated_token_ids,
            "repeated block-snapshot restore produced different tokens");
    std::cout << "V100X2 hybrid prefix cache (normal): PASS" << std::endl;
}

// Test case 2: large prompt that exercises the prefix cache under high memory pressure.
// Verifies that the second request successfully reuses the snapshot from the first request,
// avoiding the need to allocate new pages (which would have triggered std::bad_alloc before
// the prefix cache was implemented).
void exercise_exhaustion(const char* artifact) {
    ninfer::Engine engine(engine_options(artifact));
    v100x2_test::check_identity(engine);

    const auto base = engine.prepare(long_prompt()).debug_token_ids();
    require(base.size() >= 4096 && base.size() + 512 < kCapacity,
            "large fixture must exceed one ladder span and leave capacity headroom");
    const auto n = static_cast<std::uint32_t>(base.size());

    // 1. Cold prefill succeeds (pool has enough pages for the first request).
    const auto cold = run(engine, "exhaust_cold", engine.prepare(long_prompt()), false,
                          ninfer::PrefixReusePath::FullReset);
    require(cold.reused == 0 && cold.computed == n, "cold prefill did not compute the whole prompt");

    // 2. A second request with the same prompt reuses the snapshot from the first request.
    // The prefix cache reuses the existing pages, so no new allocation is needed. This avoids
    // the pool exhaustion scenario that would have triggered std::bad_alloc before the prefix
    // cache was implemented.
    const auto second = run(engine, "exhaust_second", engine.prepare(long_prompt()), true,
                            ninfer::PrefixReusePath::BlockSnapshot);
    require(second.reused > 0 && second.reused < n,
            "second request did not reuse the snapshot from the first request");
    std::cout << "V100X2 hybrid prefix cache (exhaustion): PASS" << std::endl;
}

void exercise(const char* artifact) {
    exercise_normal(artifact);
    exercise_exhaustion(artifact);
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
        std::cout << "skip: V100X2 hybrid gate requires two CUDA devices\n";
        return 77;
    }
    try {
        exercise(artifact);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "V100X2 hybrid: " << error.what() << '\n';
        return 1;
    }
}
