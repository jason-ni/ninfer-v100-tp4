// L2 M2 mixed + reused-prefix gate. A request whose prefix was restored from the hybrid cache
// (reuse_base > 0) must prefill its suffix correctly when that suffix chunk is folded into a mixed
// round alongside another request's decode. `staged.cursor` is already absolute (it starts at the
// reuse base), so the mixed path must not add the base a second time: doing so offsets every owner
// KV/RoPE position and corrupts the continuation. The isolated control is the SAME prompt run
// single-lane with the SAME reuse, so any difference is attributable to the mixed folding.
//
// NINFER_V100X2_ARTIFACT=/path/to/model.ninfer ctest -R v100x2_mixed_reuse

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Tokens = std::vector<ninfer::TokenId>;

constexpr std::uint32_t kOutput       = 24;
constexpr std::uint32_t kContext      = 4096;
constexpr std::uint32_t kSuffixTokens = 320;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.tp                   = 2;
    options.devices              = {0, 1};
    options.max_context          = kContext;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(2 * kContext);
    options.prefill_chunk        = 128;
    options.kv_cache             = ninfer::KvCacheStorage::Int8Group64;
    options.use_cuda_graph       = false;
    options.max_concurrency      = 2;
    options.max_pending_requests = 2;
    options.speculative.backend       = ninfer::SpeculativeBackend::None;
    options.speculative.draft_tokens  = 0;
    options.speculative.proposal_head = ninfer::ProposalHead::Full;
    options.prefix_cache.enabled        = true;
    options.prefix_cache.snapshot_slots = 32;
    options.prefix_cache.max_new_taps   = 4;
    options.prefix_cache.ladder_tokens  = 512;
    options.prefix_cache.min_gap        = 256;
    options.prefix_cache.host_cache_mib = 0;
    return options;
}

ninfer::RequestOptions request_options(bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens    = kOutput;
    options.execution.allow_prefix_reuse         = reuse;
    options.execution.sampling.temperature       = 0.0F;
    options.execution.sampling.presence_penalty  = 0.0F;
    options.execution.sampling.frequency_penalty = 0.0F;
    options.stop.include_model_defaults          = false;
    return options;
}

ninfer::PromptInput user_text(std::string text) {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back({ninfer::MessagePartKind::Text, std::move(text), {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    return input;
}

// A long, content-addressed shared prefix: reused by every request in the leg.
ninfer::PromptInput shared_prefix_input() {
    std::string text = "Review these Python functions. Then implement a bounded FIFO queue "
                       "with push, pop, and unit tests. Explain empty and full behavior.\n\n";
    for (int i = 0; i < 120; ++i) {
        text += "def add_" + std::to_string(i) + "(value):\n    return value + " +
                std::to_string(i) + "\n";
    }
    return user_text(std::move(text));
}

// The owner's long suffix tokens: taken from a disjoint prepared prompt, so they are valid token
// ids and span several prefill chunks.
Tokens suffix_tokens(ninfer::Engine& engine) {
    std::string text =
        "Analyze the asymptotic cost of each routine below, then rewrite them with explicit "
        "error handling, docstrings, and a short example for every public entry point. Keep the "
        "public signatures unchanged and preserve the original return types exactly.\n\n";
    for (int i = 0; i < 80; ++i) {
        text += "def helper_" + std::to_string(i) + "(value):\n    return value * " +
                std::to_string(i + 1) + "\n\n";
    }
    Tokens tokens = engine.prepare(user_text(std::move(text))).debug_token_ids();
    require(tokens.size() >= kSuffixTokens, "filler prompt is too short for the owner suffix");
    tokens.resize(kSuffixTokens);
    return tokens;
}

void run(const char* artifact) {
    // Token ids are model-deterministic, so compute the shared prefix and the owner's suffix once
    // and reuse them in both engines.
    const ninfer::PromptInput prefix_input = shared_prefix_input();
    ninfer::PromptInput decoder_input =
        user_text("Name the largest planet in our solar system in one word.");

    Tokens prefix;
    Tokens owner_tokens;
    {
        ninfer::Engine scratch(engine_options(artifact));
        require(scratch.load_summary().model_id == "qwen3.8-27b",
                "this gate requires the Qwen3.8-27B artifact at TP2");
        prefix = scratch.prepare(prefix_input).debug_token_ids();
        require(prefix.size() >= 1024, "shared prefix must be long enough to be reusable");
        owner_tokens = prefix;
        const Tokens extra = suffix_tokens(scratch);
        owner_tokens.insert(owner_tokens.end(), extra.begin(), extra.end());
    }

    // Control: a dedicated engine, seeded identically, running the owner single-lane with reuse.
    // A separate engine keeps the owner's own captured taps from changing the depth the subject
    // engine sees.
    Tokens control_tokens;
    std::uint32_t control_depth = 0;
    {
        ninfer::Engine engine(engine_options(artifact));
        auto seed = engine.generate(engine.prepare(prefix_input), request_options(false));
        require(seed.generated_token_ids.size() == kOutput, "seed did not generate its tokens");
        require(seed.reused_prompt_tokens == 0, "seed unexpectedly reused a prefix");
        const auto control =
            engine.generate(engine.prepare_tokens(owner_tokens, true), request_options(true));
        require(control.reused_prompt_tokens > 0,
                "control owner did not resume from the shared prefix");
        require(control.generated_token_ids.size() == kOutput,
                "control owner did not generate its tokens");
        control_tokens = control.generated_token_ids;
        control_depth  = control.reused_prompt_tokens;
    }

    // Subject: its own engine + seed. A short decoder is admitted first and becomes decode-ready;
    // the reused-prefix owner then prefills its long suffix while the decoder decodes, so the
    // suffix chunks are folded into mixed rounds with reuse_base > 0.
    ninfer::Engine engine(engine_options(artifact));
    {
        auto seed = engine.generate(engine.prepare(prefix_input), request_options(false));
        require(seed.generated_token_ids.size() == kOutput, "subject seed did not generate");
        require(seed.reused_prompt_tokens == 0, "subject seed unexpectedly reused a prefix");
    }
    auto decoder = engine.submit(engine.prepare(decoder_input), request_options(false));
    auto owner   = engine.submit(engine.prepare_tokens(owner_tokens, true), request_options(true));
    const auto decoder_result = decoder.wait();
    const auto owner_result   = owner.wait();

    require(decoder_result.generated_token_ids.size() == kOutput,
            "decoder did not generate its tokens");
    require(owner_result.generated_token_ids.size() == kOutput,
            "mixed owner did not generate its tokens");
    require(owner_result.reused_prompt_tokens == control_depth,
            "mixed owner resumed from a different prefix depth than the control");

    std::size_t agreed = 0;
    while (agreed < kOutput && owner_result.generated_token_ids[agreed] ==
                                   control_tokens[agreed]) {
        ++agreed;
    }
    std::cout << "mixed reused owner: reused=" << owner_result.reused_prompt_tokens
              << " agreement=" << agreed << "/" << kOutput << std::endl;
    require(agreed == kOutput,
            "mixed reused-prefix owner changed the control continuation");
    std::cout << "V100X2 mixed reused-prefix owner: identity holds" << std::endl;
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
        std::cout << "skip: the mixed reuse gate requires two CUDA devices\n";
        return 77;
    }
    try {
        run(artifact);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
