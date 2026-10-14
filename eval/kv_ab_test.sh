#!/usr/bin/env bash
# A/B KV dtype comparison: int8 vs iq4nl on the same model, same hardware.
# Stops the running service, runs both variants, restores the original.
set -euo pipefail

REPO=/media/d500/build/repos/ninfer-V100X2
ARTIFACT=/media/zt/models/orcarouter/Qwen3.8-27B-Coder390-EfficientThink-W4A4-W8A8-MTP-DFlash2.ninfer
BIN="${REPO}/dist/ninfer-serve"
PORT=8080
PROBE="${REPO}/eval/kv_dtype_probe.py"
OUTDIR=/tmp/kvprobe_ab
mkdir -p "${OUTDIR}"

COMMON_ARGS=(
    "${ARTIFACT}"
    --host 0.0.0.0 --port "${PORT}"
    --tp 4 --devices 0,1,2,3
    --max-context 262144
    --kv-capacity auto
    --prefill-chunk 4096
    --max-concurrency 4
    --pending-timeout-ms 300000
    --presence-penalty 1.0
    --spec mtp --draft-tokens 3 --lm-head-draft
    --chat-template-kwargs '{"preserve_thinking": true, "reasoning_effort":"medium"}'
    --prefix-cache --host-cache-mib 28192
    --vision
    --model-id qwen3.8-27b
)

CTX="8000,32000,64000,128000"
DEPTHS="10,30,50,70,90"

# --- find and kill the existing server ---------------------------------------
EXISTING_PID=$(pgrep -f "ninfer-serve.*${ARTIFACT}" || true)
if [[ -n "${EXISTING_PID}" ]]; then
    echo "Stopping existing server PID ${EXISTING_PID}..."
    kill -TERM "${EXISTING_PID}"
    for i in $(seq 1 30); do
        kill -0 "${EXISTING_PID}" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "${EXISTING_PID}" 2>/dev/null; then
        echo "Force killing..."
        kill -KILL "${EXISTING_PID}"
        sleep 2
    fi
    echo "GPU memory after stop:"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
fi

wait_gpu_free() {
    echo "Waiting for GPU memory to drop below 2000 MiB per card..."
    for i in $(seq 1 60); do
        local max_mem
        max_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
        if (( max_mem < 2000 )); then
            echo "GPUs free (max ${max_mem} MiB)."
            return 0
        fi
        sleep 2
    done
    echo "WARNING: GPUs still occupied after 120s, proceeding anyway."
}

start_server() {
    local kv_dtype="$1"
    local log="${OUTDIR}/server_${kv_dtype}.log"
    echo "Starting server with --kv-dtype ${kv_dtype} ..."
    "${BIN}" "${COMMON_ARGS[@]}" --kv-dtype "${kv_dtype}" >"${log}" 2>&1 &
    SERVER_PID=$!
    echo "Server PID: ${SERVER_PID}"

    for i in $(seq 1 180); do
        if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            echo "Server ready after ${i}s."
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "Server died during startup. Last 30 lines of log:"
            tail -30 "${log}"
            exit 1
        fi
        sleep 1
    done
    echo "Server did not become ready in 180s. Last 30 lines:"
    tail -30 "${log}"
    exit 1
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Stopping server PID ${SERVER_PID}..."
        kill -TERM "${SERVER_PID}"
        for i in $(seq 1 30); do
            kill -0 "${SERVER_PID}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${SERVER_PID}" 2>/dev/null; then
            kill -KILL "${SERVER_PID}"
            sleep 2
        fi
        SERVER_PID=""
    fi
}

run_probe() {
    local label="$1"
    local out="${OUTDIR}/probe_${label}.jsonl"
    echo ""
    echo "=== Running probe: ${label} ==="
    python3 "${PROBE}" \
        --url "http://127.0.0.1:${PORT}/v1" \
        --model qwen3.8-27b \
        --out "${out}" \
        --context-lengths "${CTX}" \
        --depths "${DEPTHS}" \
        --seed 42
    echo ""
}

cleanup() {
    stop_server
    # Restore the original iq4nl service
    echo ""
    echo "Restoring original iq4nl service..."
    start_server "iq4nl"
    disown "${SERVER_PID}" 2>/dev/null || true
    echo "Original service restored on port ${PORT} (PID ${SERVER_PID})."
}
trap cleanup EXIT

wait_gpu_free

# --- Run A: int8 ---------------------------------------------------------------
start_server "int8"
run_probe "int8"
stop_server
wait_gpu_free

# --- Run B: iq4nl --------------------------------------------------------------
start_server "iq4nl"
run_probe "iq4nl"
stop_server

# --- Compare -------------------------------------------------------------------
echo ""
echo "==================== COMPARISON ===================="
python3 - "${OUTDIR}/probe_int8.jsonl" "${OUTDIR}/probe_iq4nl.jsonl" <<'PYEOF'
import json, sys

def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

a = load(sys.argv[1])  # int8
b = load(sys.argv[2])  # iq4nl

print(f"{'ctx':>8} {'depth':>5}  {'int8':>5}  {'iq4nl':>5}  {'match':>5}")
print("-" * 40)
for ra, rb in zip(a, b):
    key = (ra["context_tokens"], ra["depth_pct"])
    ia = "HIT" if ra["retrieved"] else ("ERR" if ra.get("error") else "MISS")
    ib = "HIT" if rb["retrieved"] else ("ERR" if rb.get("error") else "MISS")
    match = "same" if ia == ib else "DIFF"
    print(f"{key[0]:>8} {key[1]:>4}%  {ia:>5}  {ib:>5}  {match:>5}")

ha = sum(1 for r in a if r["retrieved"])
hb = sum(1 for r in b if r["retrieved"])
print("-" * 40)
print(f"Total: int8 {ha}/{len(a)}  iq4nl {hb}/{len(b)}")

diffs = [(ra, rb) for ra, rb in zip(a, b) if ra["retrieved"] != rb["retrieved"]]
if diffs:
    print(f"\n{len(diffs)} divergent cases:")
    for ra, rb in diffs:
        print(f"  ctx={ra['context_tokens']} depth={ra['depth_pct']}%")
        print(f"    int8:  retrieved={ra['retrieved']}  resp={ra['response_content'][:120]}")
        print(f"    iq4nl: retrieved={rb['retrieved']}  resp={rb['response_content'][:120]}")
else:
    print("\nNo divergent cases between int8 and iq4nl.")
PYEOF