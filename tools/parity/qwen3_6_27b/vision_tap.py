"""Compare the C++ Vision-encoder activations of two artifacts on the same media.

The engine writes one raw little-endian BF16 file per tapped stage when `NINFER_VISION_TAP_DIR`
is set (see `src/targets/qwen3_6/impl/runtime/vision_context_impl.h`): `patch_embed`, the layers
`block_00`/`block_13`/`block_26`, and `merger`. This tool runs the real production schedule once
per artifact (so sharding, collectives and padding are included) and reports per-stage
RMSE / relative RMSE / cosine. It is the C++-side numerical alignment companion to `vision.py`,
which validates the artifact weights against the source BF16 tower from Python.

Typical use: align a quantized (Q4/Q5/Q6/W8) Vision tower with the FP16 tower it replaces:

    python -m tools.parity.qwen3_6_27b.vision_tap \
      --artifact-a /path/conv_fp16vision/qwen3_8_27b_nvfp4.ninfer \
      --artifact-b /path/Qwen3.8-27B-Coder390-...-DFlash2.ninfer \
      --messages examples/cli/messages/image_chart.json --tp 2 --devices 0,1

Both encodes must run on the same hardware and tensor-parallel width for the comparison to be
meaningful; set CUDA_VISIBLE_DEVICES so the requested `--devices` ids are available.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

STAGES = ("patch_embed", "block_00", "block_13", "block_26", "merger")


def run_tap(binary: str, artifact: str, messages: str, tap_dir: Path, tp: int, devices: str,
            log_path: Path) -> None:
    env = dict(os.environ)
    env["NINFER_VISION_TAP_DIR"] = str(tap_dir)
    command = [binary, artifact, "--vision", "--greedy", "--max-new", "1", "--messages", messages]
    if tp != 1:
        command += ["--tp", str(tp), "--devices", devices]
    with log_path.open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=env)


def load_stage(tap_dir: Path, stage: str) -> np.ndarray:
    # BF16 is the high 16 bits of an FP32; widening back is an exact bit reinterpretation.
    raw = np.fromfile(tap_dir / f"{stage}.bf16", dtype="<u2")
    return (raw.astype(np.uint32) << 16).view(np.float32)


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    # Accumulate in float64: the taps are BF16, and a float32 reduction of millions of terms
    # would drift the reported norms/cosine by ~1e-5.
    wide_actual, wide_expected = actual.astype(np.float64), expected.astype(np.float64)
    diff = wide_actual - wide_expected
    rmse = float(np.sqrt(np.mean(diff**2)))
    reference_rms = float(np.sqrt(np.mean(wide_expected**2)))
    norm_a = float(np.linalg.norm(wide_actual))
    norm_b = float(np.linalg.norm(wide_expected))
    cosine = float(np.dot(wide_actual, wide_expected) / (norm_a * norm_b)) if norm_a and norm_b else None
    return {
        "count": int(actual.size),
        "rmse": rmse,
        "relative_rmse": rmse / reference_rms if reference_rms != 0.0 else None,
        "cosine": cosine,
        "a_norm": norm_a,
        "b_norm": norm_b,
        "max_abs": float(np.max(np.abs(diff))),
    }


def compare(tap_a: Path, tap_b: Path, min_cosine: float) -> dict[str, object]:
    comparisons: dict[str, object] = {}
    for stage in STAGES:
        file_a, file_b = tap_a / f"{stage}.bf16", tap_b / f"{stage}.bf16"
        if not file_a.exists() or not file_b.exists():
            comparisons[stage] = {"error": "stage missing from one run", "aligned": False}
            continue
        tensor_a, tensor_b = load_stage(tap_a, stage), load_stage(tap_b, stage)
        if tensor_a.size != tensor_b.size:
            comparisons[stage] = {
                "error": "stage shape mismatch",
                "a_count": int(tensor_a.size),
                "b_count": int(tensor_b.size),
                "aligned": False,
            }
            continue
        stage_metrics = metrics(tensor_a, tensor_b)
        cosine = stage_metrics["cosine"]
        stage_metrics["aligned"] = cosine is not None and cosine >= min_cosine
        comparisons[stage] = stage_metrics
    return comparisons


def aligned_overall(comparisons: dict[str, object]) -> bool:
    return all(isinstance(value, dict) and value.get("aligned") is True
               for value in comparisons.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-a", help="reference artifact (e.g. FP16 tower)")
    parser.add_argument("--artifact-b", help="candidate artifact (e.g. quantized)")
    parser.add_argument("--messages", help="message JSON carrying the media")
    parser.add_argument("--from-dumps",
                        help="compare existing <dir>/a and <dir>/b dumps instead of running the "
                             "binary (for CI / re-analysis without the target GPUs)")
    parser.add_argument("--binary", default="build-v100/apps/ninfer")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--workdir", help="keep dumps under this directory instead of a temp dir")
    parser.add_argument("--output", help="write the JSON report here as well")
    parser.add_argument("--min-cosine", type=float, default=0.98,
                        help="per-stage cosine floor for the alignment assertion; the tool exits "
                             "non-zero when any stage is below it (default: 0.98)")
    args = parser.parse_args()

    def build_report(tap_a: Path, tap_b: Path, artifact_a: str, artifact_b: str,
                     messages: str) -> dict[str, object]:
        comparisons = compare(tap_a, tap_b, args.min_cosine)
        return {
            "format": "ninfer_vision_tap_comparison_v1",
            "artifact_a": artifact_a,
            "artifact_b": artifact_b,
            "messages": messages,
            "tp": args.tp,
            "min_cosine": args.min_cosine,
            "aligned": aligned_overall(comparisons),
            "comparisons": comparisons,
        }

    if args.from_dumps:
        tap_a, tap_b = Path(args.from_dumps) / "a", Path(args.from_dumps) / "b"
        report = build_report(tap_a, tap_b, args.artifact_a or str(tap_a),
                              args.artifact_b or str(tap_b), args.messages or "")
    else:
        if not (args.artifact_a and args.artifact_b and args.messages):
            parser.error("--artifact-a, --artifact-b and --messages are required without --from-dumps")
        workdir = (Path(args.workdir) if args.workdir
                   else Path(tempfile.mkdtemp(prefix="vision-tap-")))
        workdir.mkdir(parents=True, exist_ok=True)
        tap_a, tap_b = workdir / "a", workdir / "b"
        try:
            for directory in (tap_a, tap_b):
                if directory.exists():
                    shutil.rmtree(directory)
                directory.mkdir()
            run_tap(args.binary, args.artifact_a, args.messages, tap_a, args.tp, args.devices,
                    workdir / "a.log")
            run_tap(args.binary, args.artifact_b, args.messages, tap_b, args.tp, args.devices,
                    workdir / "b.log")
            report = build_report(tap_a, tap_b, str(Path(args.artifact_a).resolve()),
                                  str(Path(args.artifact_b).resolve()),
                                  str(Path(args.messages).resolve()))
        finally:
            if args.workdir is None:
                shutil.rmtree(workdir, ignore_errors=True)

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["aligned"] else 1)


if __name__ == "__main__":
    main()
