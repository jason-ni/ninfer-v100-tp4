#!/usr/bin/env bash
set -euo pipefail

# HTTP-serving launcher for the two-card Volta (V100X2) profile. Runs ninfer-serve so one resident
# model is exposed over the OpenAI Responses/Chat Completions and Anthropic Messages endpoints, as
# opposed to tools/v100/ninfer-v100x2.sh which drives the one-shot CLI. Runtime defaults live here
# (not in the Engine) so the public CLI stays target-agnostic.
#
# Physical GPU selection is left to the caller via CUDA_VISIBLE_DEVICES: on this box point it at two
# free V100s (e.g. `CUDA_VISIBLE_DEVICES=2,3`) so the app sees them as devices 0,1.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/../.." && pwd)

executable=${NINFER_V100X2_SERVE_EXECUTABLE:-"${repo_dir}/dist/ninfer-serve"}
artifact=${NINFER_V100X2_SERVE_ARTIFACT:-/media/zt/models/orcarouter/qwen3_8_27b_nvfp4.ninfer}
host=${NINFER_V100X2_SERVE_HOST:-0.0.0.0}
port=${NINFER_V100X2_SERVE_PORT:-8080}
tp=${NINFER_V100X2_SERVE_TP:-2}
presence_penalty=${PRESENCE_PENALTY:-1.0}
devices=${NINFER_V100X2_SERVE_DEVICES:-0,1}
max_context=${NINFER_V100X2_SERVE_MAX_CONTEXT:-140000}
kv_capacity=${NINFER_V100X2_SERVE_KV_CAPACITY:-auto}
prefill_chunk=${NINFER_V100X2_SERVE_PREFILL_CHUNK:-4096}
kv_dtype=${NINFER_V100X2_SERVE_KV_DTYPE:-int8}
draft_tokens=${NINFER_V100X2_SERVE_DRAFT_TOKENS:-3}
max_concurrency=${NINFER_V100X2_SERVE_MAX_CONCURRENCY:-1}
# --chat-template-kwargs defaults to the V100X2 reasoning profile. Override with a JSON object via
# NINFER_V100X2_SERVE_CHAT_TEMPLATE_KWARGS, or set it to an empty string to disable the flag.
if [[ -n "${NINFER_V100X2_SERVE_CHAT_TEMPLATE_KWARGS+set}" ]]; then
    chat_template_kwargs=${NINFER_V100X2_SERVE_CHAT_TEMPLATE_KWARGS}
else
    chat_template_kwargs='{"preserve_thinking": true, "reasoning_effort":"low"}'
fi
runtime_lib_dir=${NINFER_V100X2_RUNTIME_LIBDIR:-"${repo_dir}/build/_deps/install/lib"}
cuda_lib_dir=${NINFER_V100X2_CUDA_LIBDIR:-/usr/local/cuda-12.8/lib64}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<EOF
usage: CUDA_VISIBLE_DEVICES=<${tp} V100s> ${BASH_SOURCE[0]} [extra ninfer-serve options]

Starts the TP${tp} HTTP server (OpenAI + Anthropic endpoints) for the V100X2 profile.

Defaults: artifact=${artifact}
          host=${host} port=${port} tp=${tp} devices=${devices}
          max-context=${max_context} kv-capacity=${kv_capacity} prefill-chunk=${prefill_chunk}
           kv-dtype=${kv_dtype} spec=mtp draft-tokens=${draft_tokens} lm-head-draft
           max-concurrency=${max_concurrency}
           chat-template-kwargs=${chat_template_kwargs}

Environment overrides: NINFER_V100X2_SERVE_EXECUTABLE, NINFER_V100X2_SERVE_ARTIFACT,
NINFER_V100X2_SERVE_HOST, NINFER_V100X2_SERVE_PORT, NINFER_V100X2_SERVE_TP,
NINFER_V100X2_SERVE_DEVICES,
NINFER_V100X2_SERVE_MAX_CONTEXT, NINFER_V100X2_SERVE_KV_CAPACITY,
NINFER_V100X2_SERVE_PREFILL_CHUNK, NINFER_V100X2_SERVE_KV_DTYPE,
NINFER_V100X2_SERVE_DRAFT_TOKENS, NINFER_V100X2_SERVE_MAX_CONCURRENCY,
NINFER_V100X2_SERVE_CHAT_TEMPLATE_KWARGS, NINFER_V100X2_RUNTIME_LIBDIR,
NINFER_V100X2_CUDA_LIBDIR.
EOF
    exit 0
fi

if [[ ! -x "${executable}" ]]; then
    echo "ninfer-serve executable is missing: ${executable}" >&2
    echo "build it with tools/v100/build_dependencies.sh and the README's sm_70 CMake command" >&2
    exit 1
fi
if [[ ! -f "${artifact}" ]]; then
    echo "V100X2 artifact is missing: ${artifact}" >&2
    echo "set NINFER_V100X2_SERVE_ARTIFACT to a qwen3_8_27b .ninfer path" >&2
    exit 1
fi

# A source build keeps FFmpeg and CUDA beside the build tree rather than installing them system
# wide. Make the launcher self-contained for the normal build-v100 layout while preserving any
# caller-provided library path. (Same rationale as tools/v100/ninfer-v100x2.sh.)
runtime_ld_parts=()
if [[ -d "${runtime_lib_dir}" ]]; then runtime_ld_parts+=("${runtime_lib_dir}"); fi
if [[ -d "${cuda_lib_dir}" ]]; then runtime_ld_parts+=("${cuda_lib_dir}"); fi
if [[ ${#runtime_ld_parts[@]} -gt 0 ]]; then
    runtime_ld_path=$(IFS=:; echo "${runtime_ld_parts[*]}")
    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        export LD_LIBRARY_PATH="${runtime_ld_path}:${LD_LIBRARY_PATH}"
    else
        export LD_LIBRARY_PATH="${runtime_ld_path}"
    fi
fi

# --chat-template-kwargs is passed only when non-empty so an empty override disables it.
serve_args=()
if [[ -n "${chat_template_kwargs}" ]]; then
    serve_args+=(--chat-template-kwargs "${chat_template_kwargs}")
fi

exec "${executable}" "${artifact}" \
    --host "${host}" --port "${port}" \
    --tp "${tp}" --devices "${devices}" \
    --max-context "${max_context}" \
    --kv-capacity "${kv_capacity}" \
    --prefill-chunk "${prefill_chunk}" \
    --kv-dtype "${kv_dtype}" \
    --max-concurrency "${max_concurrency}" \
    --pending-timeout-ms "300000" \
    --presence-penalty "${presence_penalty}" \
    --spec mtp --draft-tokens "${draft_tokens}" \
    --lm-head-draft \
    ${serve_args[@]+"${serve_args[@]}"} "$@"
