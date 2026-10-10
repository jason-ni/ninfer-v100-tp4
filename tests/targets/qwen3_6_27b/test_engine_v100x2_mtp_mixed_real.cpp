// M3 mixed MTP unit gate. Request A is admitted first and becomes decode-ready under MTP
// (speculative decode); request B is then admitted into the prefill lane while A decodes, so the
// worker folds each of B's NON-TAIL prefill chunks into A's speculative round as ONE shared flat
// pass: the speculative verify slab and the owner's chunk run through the same layer linears, the
// mixers dispatch per slab, and the MTP acceptance/alignment/proposal pipeline settles the
// speculative rows afterwards. Mixing changes batching, not math: each request's greedy
// continuation must equal its own isolated MTP run.
//
// B's final chunk (which seeds the MTP draft window) deliberately takes the standalone prefill
// step, so the owner tail is not part of this gate.
//
// NINFER_V100X2_ARTIFACT=/path/to/model.ninfer ctest -R v100x2_mtp_mixed
// Runs at TP2 by default; NINFER_V100X2_TP=4 uses four devices.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Tokens = std::vector<ninfer::TokenId>;

constexpr std::uint32_t kOutputTokens = 24;
constexpr std::uint32_t kPrefillChunk = 128;
constexpr std::uint32_t kDraftTokens  = 3;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    // The 27B artifacts do not fit one 16 GB V100, so the M3 gate runs at TP2 (or TP4 via
    // NINFER_V100X2_TP), the two acceptance shapes.
    std::uint32_t tp = 2;
    if (const char* value = std::getenv("NINFER_V100X2_TP")) {
        tp = static_cast<std::uint32_t>(std::stoul(value));
    }
    if (tp != 2 && tp != 4) { throw std::runtime_error("NINFER_V100X2_TP must be 2 or 4"); }
    options.tp                   = static_cast<int>(tp);
    options.devices              = tp == 2 ? std::vector<int>{0, 1} : std::vector<int>{0, 1, 2, 3};
    options.max_context          = 2048;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(2 * 2048);
    options.prefill_chunk        = kPrefillChunk;
    options.kv_cache             = ninfer::KvCacheStorage::Int8Group64;
    options.use_cuda_graph       = false;
    options.max_concurrency      = 2;
    options.max_pending_requests = 2;
    // MTP3 with the small draft head, matching the V100X2 acceptance draft window.
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = kDraftTokens;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    return options;
}

ninfer::RequestOptions request_options() {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens    = kOutputTokens;
    options.execution.allow_prefix_reuse         = false;
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

std::string repeated(std::string unit, int times) {
    std::string out;
    for (int i = 0; i < times; ++i) { out += unit; }
    return out;
}

Tokens isolated(ninfer::Engine& engine, const ninfer::PromptInput& prompt) {
    auto result = engine.generate(engine.prepare(prompt), request_options());
    require(result.generated_token_ids.size() == kOutputTokens,
            "isolated run produced short output");
    return result.generated_token_ids;
}

void run(const char* artifact, int b_repeat) {
    const ninfer::PromptInput a_prompt =
        user_text("State the capital of France in one word.");
    // `b_repeat` controls whether B is a single (finalize) chunk or spans several. Both must
    // reproduce B's isolated MTP run: the short case folds exactly the owner's tail (MTP alignment
    // + draft seeding inside the unit), the long case folds several chunks including the tail.
    const ninfer::PromptInput b_prompt = user_text(repeated(
        "Summarize the following note in one sentence. The build spends most of its time waiting "
        "on memory traffic, so the engineer measured bandwidth before touching the kernels. ",
        b_repeat));

    const ninfer::EngineOptions options = engine_options(artifact);
    ninfer::Engine engine(options);
    const auto summary = engine.load_summary();
    require(summary.tp == options.tp && summary.model_id == "qwen3.8-27b" &&
                summary.weights_id == "nvfp4",
            "this gate requires the Qwen3.8-27B NVFP4 (MTP-capable) artifact at the requested TP width");

    const Tokens a_isolated = isolated(engine, a_prompt);
    const Tokens b_isolated = isolated(engine, b_prompt);

    const ninfer::RuntimeStats before = engine.runtime_stats();
    auto handle_a = engine.submit(engine.prepare(a_prompt), request_options());
    auto handle_b = engine.submit(engine.prepare(b_prompt), request_options());
    const Tokens a_mixed = handle_a.wait().generated_token_ids;
    const Tokens b_mixed = handle_b.wait().generated_token_ids;
    const ninfer::RuntimeStats after = engine.runtime_stats();

    require(a_mixed.size() == kOutputTokens && b_mixed.size() == kOutputTokens,
            "concurrent run produced short output");
    std::cout << "mixed mtp: decode_rounds=" << (after.decode_rounds - before.decode_rounds)
              << " row_rounds=" << (after.decode_row_rounds - before.decode_row_rounds)
              << " prefill_tokens="
              << (after.computed_prefill_tokens - before.computed_prefill_tokens) << std::endl;

    if (a_mixed != a_isolated) {
        throw std::runtime_error("mixed MTP unit changed request A's greedy continuation");
    }
    if (b_mixed != b_isolated) {
        throw std::runtime_error("mixed MTP unit changed request B's greedy continuation");
    }
    std::cout << "V100X2 mixed MTP prefill+speculative (b_repeat=" << b_repeat
              << "): per-request identity holds" << std::endl;
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
        std::cout << "skip: the mixed MTP gate runs at TP2 and requires two CUDA devices\n";
        return 77;
    }
    try {
        run(artifact, 40);
        run(artifact, 2);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
