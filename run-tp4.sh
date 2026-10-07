#!/bin/bash

# Convenience launcher for the four-card Volta (V100X4) profile: pins the four Tesla V100s and
# starts the HTTP server at tensor-parallel width 4 through tools/v100/ninfer-v100x2-serve.sh.
# See docs/maintainer/tp4-design.md.

# PCI_BUS_ID makes CUDA indices match nvidia-smi, so 1,2,3,4 select the four Tesla V100s (sm_70)
# rather than the default FASTEST_FIRST order, where index 4 is the Quadro K620 (sm_50).
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

# Vision needs the native-FP16 Vision build: the sharded binder binds the tower matrices as FP16 and
# rejects the original quantized-Vision artifact at load. When --vision is requested, default the
# artifact to the FP16-vision file so the caller does not have to set NINFER_V100X2_SERVE_ARTIFACT
# by hand. An explicit NINFER_V100X2_SERVE_ARTIFACT still wins; override the default with
# NINFER_V100X2_SERVE_VISION_ARTIFACT.
vision_requested=0
for arg in "$@"; do
    if [[ "${arg}" == "--vision" ]]; then
        vision_requested=1
    fi
done
if [[ "${vision_requested}" -eq 1 && -z "${NINFER_V100X2_SERVE_ARTIFACT:-}" ]]; then
    vision_artifact_default=/media/zt/models/orcarouter/conv_fp16vision/qwen3_8_27b_nvfp4.ninfer
    export NINFER_V100X2_SERVE_ARTIFACT="${NINFER_V100X2_SERVE_VISION_ARTIFACT:-${vision_artifact_default}}"
fi

exec "$(dirname -- "${BASH_SOURCE[0]}")/tools/v100/ninfer-v100x2-serve.sh" \
    --prefix-cache --host-cache-mib "${host_cache_mib}" "$@"
