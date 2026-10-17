#!/usr/bin/env python3
"""Pretty viewer for the ninfer-serve ``/metrics`` endpoint.

Prints device VRAM, the two paged KV pools, the hybrid prefix cache (device snapshot slots, the
pinned host tier, and the bounded disk tier), and the scheduler counters, with human-readable byte
sizes and usage bars.

    ./tools/v100/ninfer-metrics.py                      # http://127.0.0.1:8080/metrics
    ./tools/v100/ninfer-metrics.py --port 8081
    ./tools/v100/ninfer-metrics.py --watch 1            # live refresh with token rates
    ./tools/v100/ninfer-metrics.py --json               # raw JSON instead
    ./tools/v100/ninfer-metrics.py --prometheus         # raw Prometheus text instead

Stdlib only: no third-party dependencies.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
CYAN = "\033[36m"

BLOCK_FULL = "\u2588"
BLOCK_EMPTY = "\u2591"


def human(n: float) -> str:
    """Human-readable byte count with a binary unit."""
    n = float(n)
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    i = 0
    while n >= 1024.0 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    if i == 0:
        return f"{int(n)}B"
    return f"{n:.1f}{units[i]}"


def count(n: float) -> str:
    """Compact integer with thousands separators."""
    return f"{int(n):,}"


def paint(s: str, code: str, color: bool) -> str:
    return f"{code}{s}{RESET}" if color else s


def bar(frac: float, width: int, color: bool) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    body = BLOCK_FULL * filled + BLOCK_EMPTY * (width - filled)
    if not color:
        return f"[{body}]"
    shade = GREEN if frac < 0.60 else (YELLOW if frac < 0.85 else RED)
    return f"[{shade}{body}{RESET}]"


def section(title: str, color: bool) -> str:
    return "\n" + paint(title, BOLD, color)


def kv_line(pool: str, kv: dict, color: bool) -> str:
    capacity = kv.get("capacity_pages", 0)
    mapped = kv.get("mapped_pages", 0)
    cached = kv.get("cached_pages", 0)
    free = kv.get("free_pages", 0)
    page = kv.get("page_bytes", 0)
    capacity_bytes = capacity * page
    used_bytes = (mapped + cached) * page
    frac = used_bytes / capacity_bytes if capacity_bytes else 0.0
    pct = frac * 100.0
    return (
        f"  {paint(pool, CYAN, color):<20}"
        f"capacity {count(capacity):>7} pg {human(capacity_bytes):>9}   "
        f"mapped {count(mapped):>6}   cached {count(cached):>6}   "
        f"free {count(free):>7}   page {human(page):>8}\n"
        f"  {'':<20}"
        f"used {human(used_bytes):>9} / {human(capacity_bytes):<9} "
        f"{bar(frac, 24, color)} {pct:5.1f}%"
    )


def render(metrics: dict, color: bool, rates: dict | None) -> str:
    out: list[str] = []
    emit = out.append

    emit(paint("NInfer metrics", BOLD, color) +
         paint("   " + time.strftime("%Y-%m-%d %H:%M:%S"), DIM, color))

    devices = metrics.get("devices", [])
    if devices:
        emit(section("Device memory (cudaMemGetInfo)", color))
        emit(f"  {'GPU':<4}{'Total':>10}{'Used':>10}{'Free':>10}   {'Usage':<28}{'%':>5}")
        for dev in devices:
            total = dev.get("total_bytes", 0)
            used = dev.get("used_bytes", 0)
            free = dev.get("free_bytes", 0)
            frac = used / total if total else 0.0
            emit(
                f"  {str(dev.get('device', '?')):<4}"
                f"{human(total):>10}{human(used):>10}{human(free):>10}   "
                f"{bar(frac, 24, color)} {frac * 100:4.1f}%"
            )

    emit(section("KV pools (paged)", color))
    emit(kv_line("main", metrics.get("main_kv", {}), color))
    backend = metrics.get("backend_kv", {})
    if backend.get("capacity_pages"):
        emit(kv_line("backend", backend, color))

    pc = metrics.get("prefix_cache", {})
    emit(section("Prefix cache", color))
    enabled = "yes" if pc.get("enabled") else "no"
    emit(f"  enabled            {enabled}")
    emit(
        f"  device snapshots   "
        f"{count(pc.get('device_snapshot_slots_used', 0))} / "
        f"{count(pc.get('device_snapshot_slots', 0))} used"
    )
    host = pc.get("host", {})
    pinned = host.get("pinned_bytes", 0)
    used = host.get("used_bytes", 0)
    host_state = paint("active", GREEN, color) if host.get("enabled") else paint("off", RED, color)
    frac = used / pinned if pinned else 0.0
    emit(
        f"  host tier          {host_state}   "
        f"allocated {human(pinned):>9}   used {human(used):>9} "
        f"{bar(frac, 20, color)} {frac * 100:5.1f}%"
    )
    emit(
        f"  host slabs         {count(host.get('block_slab_count', 0))} block, "
        f"{count(host.get('snapshot_slab_count', 0))} snapshot"
    )
    disk = pc.get("disk", {})
    budget = disk.get("budget_bytes", 0)
    dused = disk.get("used_bytes", 0)
    disk_state = paint("active", GREEN, color) if disk.get("enabled") else paint("off", RED, color)
    frac = dused / budget if budget else 0.0
    emit(
        f"  disk tier          {disk_state}   "
        f"budget {human(budget):>9}   used {human(dused):>9} "
        f"{bar(frac, 20, color)} {frac * 100:5.1f}%"
    )
    emit(
        f"  disk slabs         {count(disk.get('block_slab_count', 0))} block, "
        f"{count(disk.get('snapshot_slab_count', 0))} snapshot"
    )

    sched = metrics.get("scheduler", {})
    emit(section("Scheduler", color))
    emit(
        f"  requests   running {sched.get('running_requests', 0)} · "
        f"prefilling {sched.get('prefilling_requests', 0)} · "
        f"decode-ready {sched.get('decode_ready_requests', 0)} · "
        f"waiting {sched.get('waiting_requests', 0)}"
    )
    emit(
        f"  totals     prefill {count(sched.get('computed_prefill_tokens', 0))} tok · "
        f"decode {count(sched.get('committed_decode_tokens', 0))} tok · "
        f"{count(sched.get('decode_rounds', 0))} rounds · "
        f"{count(sched.get('decode_row_rounds', 0))} rows"
    )
    if rates:
        emit(
            f"  rate       prefill {rates['prefill']:6.1f} tok/s · "
            f"decode {rates['decode']:6.1f} tok/s · "
            f"{rates['rounds']:6.1f} rounds/s"
        )
    emit("")
    return "\n".join(out)


def fetch(url: str, accept_json: bool, timeout: float) -> str:
    headers = {"Accept": "application/json"} if accept_json else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def rates_between(prev: dict, curr: dict, dt: float) -> dict:
    if dt <= 0:
        return {"prefill": 0.0, "decode": 0.0, "rounds": 0.0}
    ps, cs = prev["scheduler"], curr["scheduler"]
    return {
        "prefill": (cs["computed_prefill_tokens"] - ps["computed_prefill_tokens"]) / dt,
        "decode": (cs["committed_decode_tokens"] - ps["committed_decode_tokens"]) / dt,
        "rounds": (cs["decode_rounds"] - ps["decode_rounds"]) / dt,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pretty viewer for ninfer-serve /metrics.")
    parser.add_argument("--url", default=None, help="full metrics URL (overrides --host/--port)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--timeout", type=float, default=30.0, help="request timeout seconds")
    parser.add_argument("--watch", nargs="?", type=float, const=1.0, default=None,
                        metavar="SECONDS", help="refresh continuously (default interval 1s)")
    parser.add_argument("--json", action="store_true", help="print the raw JSON snapshot")
    parser.add_argument("--prometheus", action="store_true", help="print the raw Prometheus text")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    args = parser.parse_args(argv)

    url = args.url or f"http://{args.host}:{args.port}/metrics"
    color = sys.stdout.isatty() and not args.no_color

    if args.json and args.prometheus:
        parser.error("--json and --prometheus are mutually exclusive")

    prev: dict | None = None
    prev_time = 0.0
    watch = args.watch is not None

    def report_error(exc: Exception) -> None:
        reason = getattr(exc, "reason", exc)
        msg = f"{paint('error:', RED, color)} request to {url} failed: {reason}"
        if watch:
            if sys.stdout.isatty():
                sys.stdout.write("\033[H\033[J")
            print(msg + paint(f"   retrying in {args.watch:g}s", DIM, color))
            sys.stdout.flush()
        else:
            print(msg, file=sys.stderr)

    while True:
        try:
            if args.prometheus:
                sys.stdout.write(fetch(url, accept_json=False, timeout=args.timeout))
                sys.stdout.write("\n")
                return 0
            metrics = json.loads(fetch(url, accept_json=True, timeout=args.timeout))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            # ValueError covers a non-JSON body; TimeoutError is a read timeout that urllib does
            # not wrap in URLError. Watch mode retries; a one-shot run reports and exits.
            report_error(exc)
            if not watch:
                return 1
            try:
                time.sleep(max(0.1, args.watch))
            except KeyboardInterrupt:
                return 0
            continue

        if args.json:
            print(json.dumps(metrics, indent=2, sort_keys=True))
            return 0

        now = time.monotonic()
        rates = None
        if watch and prev is not None:
            rates = rates_between(prev, metrics, now - prev_time)
        prev, prev_time = metrics, now

        if not watch:
            print(render(metrics, color, rates))
            return 0

        if sys.stdout.isatty():
            sys.stdout.write("\033[H\033[J")
        print(render(metrics, color, rates), end="")
        sys.stdout.flush()
        try:
            time.sleep(max(0.1, args.watch))
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
