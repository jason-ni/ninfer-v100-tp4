# Qwen3.6-27B Vision parity tool

This tool compares the independent `.ninfer` Python reference with the source BF16 Vision tower at
matching semantic boundaries.

Compare quantized artifact Vision activations with the source BF16 tower:

```bash
python -m tools.parity.qwen3_6_27b.vision \
  --weights out/qwen3_6_27b.ninfer \
  --model-dir /path/to/Qwen3.6-27B/base-hf-bf16 \
  --messages messages.json
```

The diagnostic reports numerical differences directly; it does not materialize an activation dump
or require exact generated-token equality between independent implementations.

Compare the **C++ Vision-encoder activations** of two artifacts on the same media (the companion
that exercises the real production schedule, including sharding, collectives, and padding):

```bash
python -m tools.parity.qwen3_6_27b.vision_tap \
  --artifact-a /path/conv_fp16vision/qwen3_8_27b_nvfp4.ninfer \
  --artifact-b /path/Qwen3.8-27B-Coder390-...-DFlash2.ninfer \
  --messages examples/cli/messages/image_chart.json --tp 2 --devices 0,1
```

The engine writes `patch_embed`, `block_00`/`block_13`/`block_26`, and `merger` to
`NINFER_VISION_TAP_DIR` as raw BF16; the tool reports per-stage RMSE / relative RMSE / cosine.
An artifact compared with itself must report `rmse = 0` and `cosine = 1`. Run both sides with the
same hardware and tensor-parallel width. Measured FP16-vs-quantized (tp2, `image_chart.json`):
cosine `0.9999` (patch_embed) → `0.9906` (block_26), i.e. the expected weight-quantization
profile rather than an execution defect.

The tool is also an assertion: every stage must reach `--min-cosine` (default `0.98`), and it exits
non-zero otherwise (a missing stage or a shape mismatch also fails). Pass `--min-cosine 0` to use it
as a pure diagnostic. To re-analyze dumps already captured under a `--workdir` (no target GPUs), add
`--from-dumps <workdir>`.
