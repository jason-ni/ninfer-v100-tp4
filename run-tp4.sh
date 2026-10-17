#!/bin/bash

# Convenience launcher for the four-card Volta (V100X4) profile: pins the four Tesla V100s and
# starts the HTTP server at tensor-parallel width 4 through tools/v100/ninfer-v100x2-serve.sh.
# See docs/maintainer/tp4-design.md.
#
# --kv-dtype bf16|int8|iq4nl selects the KV cache storage type and is consumed here (exported as
# NINFER_V100X2_SERVE_KV_DTYPE); it overrides the environment value.
#
# --raw-log enables the engine's temporary raw-output capture (NINFER_RAW_OUTPUT_LOG) and is
# consumed here rather than forwarded to ninfer-serve. Optional explicit destination:
#   ./run-tp4.sh --raw-log              # -> profiles/raw/raw-output.jsonl
#   ./run-tp4.sh --raw-log=/tmp/raw.jsonl
#
# --remote-req-log ensures the Python request-persistence receiver is running (spawning it
# detached when its port is not already listening), stores the SQLite DB under
# /media/optane/reqlogs, and forwards --persist-endpoint to ninfer-serve. Optional explicit
# endpoint:
#   ./run-tp4.sh --remote-req-log                    # -> 127.0.0.1:9100
#   ./run-tp4.sh --remote-req-log=127.0.0.1:9101

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
# KV cache storage type: int8 (group-64, the acceptance profile, default) or iq4nl (low-bit growing
# cache for higher capacity). Select with the launcher --kv-dtype option (see below) or the
# NINFER_V100X2_SERVE_KV_DTYPE environment variable; the option overrides the environment. The
# concurrent-lane count is derived from this value below (see NINFER_V100X2_SERVE_MAX_CONCURRENCY).
kv_dtype=${NINFER_V100X2_SERVE_KV_DTYPE:-int8}
# Round-trip the model's reasoning in assistant history so prefix reuse resumes at the exact
# generated frontier (append_frontier) instead of the conservative turn-closure checkpoint.
export NINFER_V100X2_SERVE_CHAT_TEMPLATE_KWARGS=${NINFER_V100X2_SERVE_CHAT_TEMPLATE_KWARGS:-'{"preserve_thinking": true, "reasoning_effort":"medium"}'}

# Optional public HTTP model alias (the OpenAI/Responses `model` field). Empty advertises the
# loaded artifact's identity.model_id. Override with NINFER_V100X2_SERVE_MODEL_ID, or pass
# --model-id <alias> as a free argument (which wins over the environment value).
export NINFER_V100X2_SERVE_MODEL_ID=${NINFER_V100X2_SERVE_MODEL_ID:-}

# Enable the hybrid prefix cache with the pinned host tier (M2): KV blocks and snapshot images that
# no longer fit on device are demoted to pinned host RAM instead of dropped, so an evicted
# conversation can still be restored without recomputing. Each snapshot slab is ~147 MiB, so the
# pool is sized to hold several conversations' taps; override with NINFER_V100X2_SERVE_HOST_CACHE_MIB.
host_cache_mib=${NINFER_V100X2_SERVE_HOST_CACHE_MIB:-16384}

# Bounded, crash-durable disk tier below the host tier (M3a): host slabs evicted under pressure are
# spilled to a fixed-slot slab file, and admission restores them without recomputation. The index
# and KV bytes survive a crash (SIGKILL) because writes go through at runtime, not only on shutdown.
# A directory path stores <dir>/prefix-cache.bin (index) + <dir>/prefix-cache.slabs. Override with
# NINFER_V100X2_SERVE_PREFIX_CACHE_FILE; an empty value disables the disk tier.
prefix_cache_file=${NINFER_V100X2_SERVE_PREFIX_CACHE_FILE:-/media/optane/kvcache}
# Disk-tier size budget. Must fit the volume holding prefix_cache_file (the slab file is sparse,
# so it only consumes space for slots actually written, but a budget larger than the volume would
# eventually hit ENOSPC under fill). /media/optane is ~53G with ~49G free, so default to 49152 MiB
# (48 GiB); override with NINFER_V100X2_SERVE_DISK_CACHE_MIB.
disk_cache_mib=${NINFER_V100X2_SERVE_DISK_CACHE_MIB:-49152}

# The Engine's registered Qwen3.8-27B thinking preset keeps presence penalty at 0 (its general-task
# default), so an agent harness that sends only temperature/top-p/top-k runs unpenalized and long
# reasoning can degenerate into repetition. Pin the coding/long-reasoning value used by the llama.cpp
# baseline and docs/performance.md; a request's own presence_penalty still overrides it. Override the
# pin with NINFER_V100X2_SERVE_PRESENCE_PENALTY.
presence_penalty=${NINFER_V100X2_SERVE_PRESENCE_PENALTY:-1.0}

# Returns 0 when something is accepting connections on the given port.
port_listening() {
    local host=$1 port=$2
    if command -v ss >/dev/null 2>&1; then
        [[ -n "$(ss -ltnH "sport = :${port}" 2>/dev/null)" ]]
    else
        (exec 3<>"/dev/tcp/${host}/${port}") 2>/dev/null
    fi
}

# Split the launcher-only flags (--kv-dtype, --raw-log, --vision) from the arguments forwarded to
# ninfer-serve. --kv-dtype overrides NINFER_V100X2_SERVE_KV_DTYPE; a bare --raw-log uses the default
# destination and --raw-log=PATH overrides it; NINFER_RAW_OUTPUT_LOG set in the environment is
# preserved unless --raw-log is given.
print_help() {
    cat <<EOF
usage: ./run-tp4.sh [launcher options] [ninfer-serve options...]

V100X4 profile: TP4 (devices 0,1,2,3), max-context 262144, hybrid prefix cache with the pinned host
tier, and MTP3. max-concurrency follows the KV dtype (5 for iq4nl, 4 otherwise; override with
NINFER_V100X2_SERVE_MAX_CONCURRENCY). Every other argument is forwarded to ninfer-serve.

Launcher options:
  --vision                 enable media/Vision and default the artifact to the Coder390 package
  --kv-dtype bf16|int8|iq4nl
                           KV cache storage type (default int8; overrides the environment)
  --raw-log[=PATH]         capture the engine's raw output (default profiles/raw/raw-output.jsonl)
  --remote-req-log[=HOST:PORT]
                           persist each request body to a SQLite DB under /media/optane/reqlogs
                           via the Python receiver (default endpoint 127.0.0.1:9100); spawns the
                           receiver detached when its port is not already listening
  -h, --help               show this help

Environment: NINFER_V100X2_SERVE_KV_DTYPE, NINFER_V100X2_SERVE_MODEL_ID,
NINFER_V100X2_SERVE_MAX_CONTEXT, NINFER_V100X2_SERVE_MAX_CONCURRENCY,
NINFER_V100X2_SERVE_HOST_CACHE_MIB,
NINFER_V100X2_SERVE_PRESENCE_PENALTY, CUDA_VISIBLE_DEVICES, NINFER_V100X2_SERVE_ARTIFACT,
NINFER_REMOTE_REQ_LOG_ENDPOINT, NINFER_REMOTE_REQ_LOG_HTTP_HOST, NINFER_REMOTE_REQ_LOG_HTTP_PORT,
NINFER_REMOTE_REQ_LOG_MAX_BODY_KB, NINFER_REMOTE_REQ_LOG_DIR, NINFER_REMOTE_REQ_LOG_DB,
NINFER_REMOTE_REQ_LOG_PYTHON.
Run tools/v100/ninfer-v100x2-serve.sh --help for the full ninfer-serve option set.
EOF
}

vision_requested=0
max_context=262144
raw_log_path=${NINFER_RAW_OUTPUT_LOG:-}
remote_req_log=0
remote_req_log_endpoint=""
forwarded_args=()
while [[ $# -gt 0 ]]; do
    arg=$1
    case "${arg}" in
        --vision)
            vision_requested=1
            forwarded_args+=("${arg}")
            shift
            ;;
        --max-context)
            if [[ $# -lt 2 ]]; then
                echo "run-tp4.sh: --max-context requires a value" >&2
                exit 2
            fi
            max_context=$2   
            shift 2
            ;;
        --max-context=*)
            max_context=${arg#--max-context=}
            if [[ -z "${max_context}" ]]; then
                echo "run-tp4.sh: --max-context requires a value" >&2
                exit 2
            fi
            shift
            ;;
        --kv-dtype)
            if [[ $# -lt 2 ]]; then
                echo "run-tp4.sh: --kv-dtype requires a value (bf16|int8|iq4nl)" >&2
                exit 2
            fi
            kv_dtype=$2
            shift 2
            ;;
        --kv-dtype=*)
            kv_dtype=${arg#--kv-dtype=}
            if [[ -z "${kv_dtype}" ]]; then
                echo "run-tp4.sh: --kv-dtype requires a value (bf16|int8|iq4nl)" >&2
                exit 2
            fi
            shift
            ;;
        --raw-log)
            raw_log_path="${NINFER_V100X2_RAW_LOG:-${script_dir}/profiles/raw/raw-output.jsonl}"
            shift
            ;;
        --raw-log=*)
            raw_log_path="${arg#--raw-log=}"
            if [[ -z "${raw_log_path}" ]]; then
                raw_log_path="${NINFER_V100X2_RAW_LOG:-${script_dir}/profiles/raw/raw-output.jsonl}"
            fi
            shift
            ;;
        --remote-req-log)
            remote_req_log=1
            shift
            ;;
        --remote-req-log=*)
            remote_req_log=1
            remote_req_log_endpoint=${arg#--remote-req-log=}
            if [[ -z "${remote_req_log_endpoint}" ]]; then
                echo "run-tp4.sh: --remote-req-log= requires HOST:PORT" >&2
                exit 2
            fi
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            forwarded_args+=("${arg}")
            shift
            ;;
    esac
done
export NINFER_V100X2_SERVE_MAX_CONTEXT=${NINFER_V100X2_SERVE_MAX_CONTEXT:-${max_context}}
export NINFER_V100X2_SERVE_KV_DTYPE="${kv_dtype}"

# Bounded FIFO ingress with no preemption; the lane count is derived from the KV storage type here.
# With --kv-capacity auto the shared Main KV pool is provisioned at the curve ceiling
# max_concurrency * ceil(max_context / 64) pages, but that ceiling is in PAGES, so a lower-bit dtype
# stores each page in fewer bytes and leaves VRAM unused at four lanes. iq4nl (Int4NLGroup32: 4-bit
# codes + one FP16 scale per 32 values, ~288 KiB/page) fits a fifth lane inside that slack (auto then
# grows into it, leaving only the default 1 GiB automatic headroom); int8 (~2x) and bf16 (~4x) pages
# are too large to add a lane, so four is already the budget. An explicit
# NINFER_V100X2_SERVE_MAX_CONCURRENCY always wins.
if [[ -n "${NINFER_V100X2_SERVE_MAX_CONCURRENCY:-}" ]]; then
    max_concurrency=${NINFER_V100X2_SERVE_MAX_CONCURRENCY}
else
    case "${kv_dtype}" in
        iq4nl) max_concurrency=5 ;;
        *)     max_concurrency=4 ;;
    esac
fi
export NINFER_V100X2_SERVE_MAX_CONCURRENCY="${max_concurrency}"
if [[ -n "${raw_log_path}" ]]; then
    mkdir -p -- "$(dirname -- "${raw_log_path}")"
    export NINFER_RAW_OUTPUT_LOG="${raw_log_path}"
    echo "raw output log: ${raw_log_path}" >&2
fi

if [[ "${vision_requested}" -eq 1 && -z "${NINFER_V100X2_SERVE_ARTIFACT:-}" ]]; then
    vision_artifact_default=/media/zt/models/orcarouter/conv_fp16vision/qwen3_8_27b_nvfp4.ninfer
    #vision_artifact_default=/media/zt/models/orcarouter/coder390-basemtp.ninfer
    #vision_artifact_default=/media/zt/models/orcarouter/Qwen3.8-27B-Coder390-EfficientThink-W4A4-W8A8-MTP-DFlash2.ninfer
    export NINFER_V100X2_SERVE_ARTIFACT="${NINFER_V100X2_SERVE_VISION_ARTIFACT:-${vision_artifact_default}}"
fi

# --remote-req-log: ensure the Python request-persistence receiver is up, then point ninfer-serve
# at it. The receiver is a long-lived shared service, so it is spawned detached and left running
# across serve restarts; if its port is already listening we reuse that instance.
if [[ "${remote_req_log}" -eq 1 ]]; then
    req_endpoint=${remote_req_log_endpoint:-${NINFER_REMOTE_REQ_LOG_ENDPOINT:-127.0.0.1:9100}}
    req_host=${req_endpoint%:*}
    req_port=${req_endpoint##*:}
    req_host=${req_host#[}
    req_host=${req_host%]}
    if [[ -z "${req_host}" || -z "${req_port}" || "${req_port}" == "${req_endpoint}" ]]; then
        echo "run-tp4.sh: --remote-req-log endpoint must be HOST:PORT, got: ${req_endpoint}" >&2
        exit 2
    fi
    req_db_dir=${NINFER_REMOTE_REQ_LOG_DIR:-/media/optane/reqlogs}
    req_db=${NINFER_REMOTE_REQ_LOG_DB:-${req_db_dir}/request_persist.sqlite}
    req_log_file=${NINFER_REMOTE_REQ_LOG_FILE:-${req_db_dir}/request_persist_server.log}
    req_http_port=${NINFER_REMOTE_REQ_LOG_HTTP_PORT:-9101}
    # The REST admin + web UI bind address; 0.0.0.0 so it is reachable from other hosts. The TCP
    # ingest listener stays bound to the endpoint host (loopback by default) and is not exposed.
    req_http_host=${NINFER_REMOTE_REQ_LOG_HTTP_HOST:-0.0.0.0}
    req_max_body_kb=${NINFER_REMOTE_REQ_LOG_MAX_BODY_KB:-1024}
    req_python=${NINFER_REMOTE_REQ_LOG_PYTHON:-python3}
    req_server=${script_dir}/tools/request_persist/request_persist_server.py

    if [[ ! -f "${req_server}" ]]; then
        echo "run-tp4.sh: request-persistence receiver is missing: ${req_server}" >&2
        exit 2
    fi
    mkdir -p -- "${req_db_dir}"

    if port_listening "${req_host}" "${req_port}"; then
        echo "remote req log: reusing receiver on ${req_host}:${req_port} (rest ${req_http_host}:${req_http_port})" >&2
    else
        echo "remote req log: starting receiver on ${req_host}:${req_port} (rest ${req_http_host}:${req_http_port}) -> ${req_db}" >&2
        nohup "${req_python}" "${req_server}" \
            --host "${req_host}" --port "${req_port}" --db "${req_db}" \
            --http-host "${req_http_host}" --http-port "${req_http_port}" \
            --max-body-kb "${req_max_body_kb}" \
            >>"${req_log_file}" 2>&1 &
        disown 2>/dev/null || true
        # Wait briefly for the listener; the persistor also reconnects, so this only avoids an
        # obviously broken spawn. Warn (do not abort) since persistence must never block serving.
        for _ in $(seq 1 30); do
            port_listening "${req_host}" "${req_port}" && break
            sleep 0.1
        done
        if ! port_listening "${req_host}" "${req_port}"; then
            echo "run-tp4.sh: receiver did not come up on ${req_host}:${req_port}; see ${req_log_file}" >&2
        fi
    fi
    forwarded_args+=(--persist-endpoint "${req_endpoint}")
fi

prefix_cache_args=()
if [[ -n "${prefix_cache_file}" ]]; then
    prefix_cache_args+=(--prefix-cache-file "${prefix_cache_file}" --disk-cache-mib "${disk_cache_mib}")
fi

exec "${script_dir}/tools/v100/ninfer-v100x2-serve.sh" \
    --prefix-cache --host-cache-mib "${host_cache_mib}" \
    ${prefix_cache_args[@]+"${prefix_cache_args[@]}"} \
    --presence-penalty "${presence_penalty}" \
    ${forwarded_args[@]+"${forwarded_args[@]}"}
