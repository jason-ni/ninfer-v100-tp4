#!/bin/bash

# Convenience launcher for the two-card Volta (V100X2) profile, defaulting to the IQ4_NL-G32 KV
# cache (--kv-dtype iq4nl) at tensor-parallel width 2 through tools/v100/ninfer-v100x2-serve.sh.
#
# IQ4_NL-G32 stores the growing KV cache in 32-value groups (one FP16 scale + 16 nibble bytes,
# 8 groups/row) instead of INT8-G64's 64-value groups, cutting KV payload ~45% for the same
# capacity. On two 16 GiB cards that is what lets the context budget reach the model's territory
# without spilling. Every value below is overridable via the environment; the serve script's
# --help lists them.
#
# Usage:  ./run-tp2.sh --messages req.json
#         ./run-tp2.sh --vision --messages with_image.json   # image/video input (sharded tower)

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
export GGML_CUDA_PDL=0
export NINFER_V100X2_SERVE_TP=2
export NINFER_V100X2_SERVE_DEVICES=${NINFER_V100X2_SERVE_DEVICES:-0,1}
export NINFER_V100X2_SERVE_KV_DTYPE=${NINFER_V100X2_SERVE_KV_DTYPE:-iq4nl}

# The registered 27B nvfp4 package whose Vision tower is already the native-FP16 image the current
# engine's target contract expects. The older BF16-vision package in the same directory is rejected
# at load with "tensor descriptor does not match target contract: vision/patch_embedding". Override
# with NINFER_V100X2_SERVE_ARTIFACT (e.g. the registered Q4_K_M .ninfer when present).
export NINFER_V100X2_SERVE_ARTIFACT=${NINFER_V100X2_SERVE_ARTIFACT:-/media/zt/models/orcarouter/conv_fp16vision/qwen3_8_27b_nvfp4.ninfer}

# Detect --vision (forwarded unchanged below). It loads the sharded Vision tower (~0.43 GiB more
# weights per rank) and reserves the Vision encode/transient workspace, so the text context budget
# has to drop; without it neither the Vision weights nor its workspace are present at all.
vision_requested=0
forwarded_args=()
for arg in "$@"; do
    case "${arg}" in
        --vision) vision_requested=1 ;;
    esac
    forwarded_args+=("${arg}")
done

# Context budget. The model's native ceiling is 262144. The binding cost is the context-scaled
# runtime reservation (KV pool + the text attention workspace, whose VoltaFlash gather/mask staging
# is sized by the whole visible key range) plus the weights; everything is provisioned for one full
# sequence, so a bigger --max-context needs a bigger reservation.
#
# The earlier "factory" failure at high context was NOT a runtime OOM: it was the capacity planner
# refusing to start. `--kv-capacity auto` mandates an extra 1 GiB headroom, and at --max-concurrency
# 1 auto has no room to raise the capacity anyway (min pages == max pages), so its only effect is
# that 1 GiB gate -- raise available_after_weights minus reservation past 1 GiB or it throws. Using
# an EXPLICIT capacity drops that gate and lets the budget reach the native ceiling.
#
# Measured on the heaviest available 27B package (nvfp4, ~10.5 GiB/GPU of text weights), TP2 +
# IQ4_NL-G32:
#
#   text,  auto:      200000 fits, 215000 fits, 225000 rejected (serve)  -> ~215000
#   text,  explicit:  262144 (native ceiling) fits -- free-after-startup 110 MiB, slack 310 MiB
#   vision, auto:     120000 fits, 140000 rejected                       -> ~120000
#
# Defaults: 262144 (text, native) and 120000 (--vision), with an explicit capacity so the auto
# headroom gate never applies. A lighter artifact fits more, a heavier one less; INT8
# (NINFER_V100X2_SERVE_KV_DTYPE=int8) lowers the ceiling because its KV pool is ~2x larger.
#if [[ "${vision_requested}" -eq 1 ]]; then
#    context_default=120000
#else
context_default=262144
#fi
export NINFER_V100X2_SERVE_MAX_CONTEXT=${NINFER_V100X2_SERVE_MAX_CONTEXT:-${context_default}}
# Explicit capacity by default (mirrors --max-context) to avoid --kv-capacity auto's 1 GiB gate.
export NINFER_V100X2_SERVE_KV_CAPACITY=${NINFER_V100X2_SERVE_KV_CAPACITY:-${NINFER_V100X2_SERVE_MAX_CONTEXT}}

# Prefer the freshly built V100X2 server, which carries the IQ4_NL-G32 engine support, over the
# manually deployed dist/ copy. Override with NINFER_V100X2_SERVE_EXECUTABLE.
if [[ -z "${NINFER_V100X2_SERVE_EXECUTABLE:-}" && -x "${script_dir}/build-v100/apps/ninfer-serve" ]]; then
    export NINFER_V100X2_SERVE_EXECUTABLE="${script_dir}/build-v100/apps/ninfer-serve"
fi

# Pinned host-RAM tier for the hybrid prefix cache, matching the TP4 launcher's default: evicted KV
# blocks/snapshots are demoted to pinned host RAM instead of dropped, so an evicted conversation can
# be restored without recomputation. Without this the prefix pool is device-only. Each snapshot slab
# is ~147 MiB, so 28192 MiB holds several conversations' taps. Override with
# NINFER_V100X2_SERVE_HOST_CACHE_MIB (0 keeps the cache device-only).
host_cache_mib=${NINFER_V100X2_SERVE_HOST_CACHE_MIB:-28192}

exec "${script_dir}/tools/v100/ninfer-v100x2-serve.sh" --prefix-cache \
    --host-cache-mib "${host_cache_mib}" \
    ${forwarded_args[@]+"${forwarded_args[@]}"}
