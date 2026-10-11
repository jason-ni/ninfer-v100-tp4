#!/bin/bash

# Convenience launcher for the four-card Volta (V100X4) profile: pins the four Tesla V100s and
# starts the HTTP server at tensor-parallel width 4 through tools/v100/ninfer-v100x2-serve.sh.
# See docs/maintainer/tp4-design.md.
#
# --raw-log enables the engine's temporary raw-output capture (NINFER_RAW_OUTPUT_LOG) and is
# consumed here rather than forwarded to ninfer-serve. Optional explicit destination:
#   ./run-tp4.sh --raw-log              # -> profiles/raw/raw-output.jsonl
#   ./run-tp4.sh --raw-log=/tmp/raw.jsonl

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# PCI_BUS_ID makes CUDA indices match nvidia-smi, so 1,2,3,4 select the four Tesla V100s (sm_70)
# rather than the default FASTEST_FIRST order, where index 4 is the Quadro K620 (sm_50).
export NINFER_V100X2_SERVE_PORT=8080
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3,4}
export GGML_CUDA_PDL=0
export NINFER_V100X2_SERVE_TP=4
export NINFER_V100X2_SERVE_DEVICES=${NINFER_V100X2_SERVE_DEVICES:-0,1,2,3}
# Four-way weight sharding frees ~5 GiB/GPU, so default to the model's full native 262144-token
# ceiling (the shared serve default of 140000 is the conservative two-card budget).
export NINFER_V100X2_SERVE_MAX_CONTEXT=${NINFER_V100X2_SERVE_MAX_CONTEXT:-262144}
# Four concurrent request lanes (bounded FIFO ingress, no preemption). With --kv-capacity auto the
# KV pool is provisioned for max-concurrency full sequences, so four lanes share the fixed VRAM
# budget: auto resolves the largest per-sequence capacity that fits all four.
export NINFER_V100X2_SERVE_MAX_CONCURRENCY=${NINFER_V100X2_SERVE_MAX_CONCURRENCY:-4}
# Round-trip the model's reasoning in assistant history so prefix reuse resumes at the exact
# generated frontier (append_frontier) instead of the conservative turn-closure checkpoint.
export NINFER_V100X2_SERVE_CHAT_TEMPLATE_KWARGS=${NINFER_V100X2_SERVE_CHAT_TEMPLATE_KWARGS:-'{"preserve_thinking": true, "reasoning_effort":"medium"}'}

# Enable the hybrid prefix cache with the pinned host tier (M2): KV blocks and snapshot images that
# no longer fit on device are demoted to pinned host RAM instead of dropped, so an evicted
# conversation can still be restored without recomputing. Each snapshot slab is ~147 MiB, so the
# pool is sized to hold several conversations' taps; override with NINFER_V100X2_SERVE_HOST_CACHE_MIB.
host_cache_mib=${NINFER_V100X2_SERVE_HOST_CACHE_MIB:-28192}

# The Engine's registered Qwen3.8-27B thinking preset keeps presence penalty at 0 (its general-task
# default), so an agent harness that sends only temperature/top-p/top-k runs unpenalized and long
# reasoning can degenerate into repetition. Pin the coding/long-reasoning value used by the llama.cpp
# baseline and docs/performance.md; a request's own presence_penalty still overrides it. Override the
# pin with NINFER_V100X2_SERVE_PRESENCE_PENALTY.
presence_penalty=${NINFER_V100X2_SERVE_PRESENCE_PENALTY:-1.0}

# Split the launcher-only --raw-log flag from the arguments forwarded to ninfer-serve. A bare
# --raw-log uses the default destination; --raw-log=PATH overrides it. NINFER_RAW_OUTPUT_LOG set in
# the environment is preserved unless the flag is given.
vision_requested=0
raw_log_path=${NINFER_RAW_OUTPUT_LOG:-}
forwarded_args=()
for arg in "$@"; do
    case "${arg}" in
        --vision)
            vision_requested=1
            forwarded_args+=("${arg}")
            ;;
        --raw-log)
            raw_log_path="${NINFER_V100X2_RAW_LOG:-${script_dir}/profiles/raw/raw-output.jsonl}"
            ;;
        --raw-log=*)
            raw_log_path="${arg#--raw-log=}"
            if [[ -z "${raw_log_path}" ]]; then
                raw_log_path="${NINFER_V100X2_RAW_LOG:-${script_dir}/profiles/raw/raw-output.jsonl}"
            fi
            ;;
        *)
            forwarded_args+=("${arg}")
            ;;
    esac
done
if [[ -n "${raw_log_path}" ]]; then
    mkdir -p -- "$(dirname -- "${raw_log_path}")"
    export NINFER_RAW_OUTPUT_LOG="${raw_log_path}"
    echo "raw output log: ${raw_log_path}" >&2
fi

if [[ "${vision_requested}" -eq 1 && -z "${NINFER_V100X2_SERVE_ARTIFACT:-}" ]]; then
    #vision_artifact_default=/media/zt/models/orcarouter/conv_fp16vision/qwen3_8_27b_nvfp4.ninfer
    vision_artifact_default=/media/zt/models/orcarouter/coder390-basemtp.ninfer
    export NINFER_V100X2_SERVE_ARTIFACT="${NINFER_V100X2_SERVE_VISION_ARTIFACT:-${vision_artifact_default}}"
fi

exec "${script_dir}/tools/v100/ninfer-v100x2-serve.sh" \
    --prefix-cache --host-cache-mib "${host_cache_mib}" \
    --presence-penalty "${presence_penalty}" \
    ${forwarded_args[@]+"${forwarded_args[@]}"}
