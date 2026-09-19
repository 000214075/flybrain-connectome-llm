"""Sample real GPU utilisation on Windows via performance counters.

Claiming "the GPU is fully utilised" requires a measurement, not an inference from
throughput. Windows exposes per-engine GPU busy time through the `GPU Engine`
performance counter set, which is what this reads while a workload runs, so
training runs can be reported with the achieved utilisation next to tokens/s.

    python scripts/gpu_util.py --seconds 20            # watch for 20s
    python scripts/gpu_util.py -- python -m flybrain.train --config ...
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import subprocess
import sys
import threading
import time


def sample_engines() -> dict[str, float]:
    """One reading of every GPU engine's busy percentage, summed per engine type."""
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -ErrorAction Stop | "
        "Select-Object -ExpandProperty CounterSamples | "
        "ForEach-Object { \"$($_.InstanceName)`t$($_.CookedValue)\" }",
    ]
    try:
        raw = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    except subprocess.TimeoutExpired:
        return {}
    totals: dict[str, float] = collections.defaultdict(float)
    for line in raw.stdout.splitlines():
        if "\t" not in line:
            continue
        instance, _, value = line.partition("\t")
        try:
            busy = float(value)
        except ValueError:
            continue
        # Instance names look like "pid_1234_luid_0x00000000_0x0000FFFF_phys_0_eng_0_engtype_Compute"
        engine = instance.rsplit("engtype_", 1)[-1] if "engtype_" in instance else "Unknown"
        totals[engine] += busy
    return dict(totals)


def pick_gpu_engines(readings: list[dict[str, float]]) -> dict[str, list[float]]:
    """Collect the engine types that were ever meaningfully busy."""
    series: dict[str, list[float]] = collections.defaultdict(list)
    for reading in readings:
        for engine, busy in reading.items():
            series[engine].append(busy)
    return {engine: values for engine, values in series.items() if max(values) > 0.5}


def watch(seconds: float, interval: float, label: str) -> dict:
    readings: list[dict[str, float]] = []
    deadline = time.time() + seconds
    while time.time() < deadline:
        readings.append(sample_engines())
        time.sleep(interval)
    series = pick_gpu_engines(readings)
    summary = {
        "samples": len(readings),
        "engines": {},
    }
    for engine, values in sorted(series.items()):
        summary["engines"][engine] = {
            "mean_pct": round(statistics.fmean(values), 2),
            "max_pct": round(max(values), 2),
            "p50_pct": round(statistics.median(values), 2),
        }
    print(f"=== GPU utilisation while {label} ({len(readings)} samples) ===")
    if not summary["engines"]:
        print("  no GPU engine reported activity")
    for engine, stats in summary["engines"].items():
        print(
            f"  {engine:<14} mean {stats['mean_pct']:6.2f}%  median {stats['p50_pct']:6.2f}%  "
            f"peak {stats['max_pct']:6.2f}%"
        )
    return summary


def run_with_sampling(command: list[str], interval: float, label: str) -> dict:
    """Run a child process while sampling GPU utilisation in a background thread."""
    readings: list[dict[str, float]] = []
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            readings.append(sample_engines())
            stop.wait(interval)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    started = time.time()
    try:
        process = subprocess.run(command, check=False)
    finally:
        stop.set()
        thread.join(timeout=70)
    elapsed = time.time() - started

    series = pick_gpu_engines(readings)
    summary = {
        "label": label,
        "command": command,
        "elapsed_seconds": round(elapsed, 1),
        "samples": len(readings),
        "engines": {
            engine: {
                "mean_pct": round(statistics.fmean(values), 2),
                "max_pct": round(max(values), 2),
                "p50_pct": round(statistics.median(values), 2),
            }
            for engine, values in series.items()
        },
        "exit_code": process.returncode,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--out", default="")
    parser.add_argument("--label", default="the workload")
    parser.add_argument("command", nargs="*", help="optional command to run while sampling")
    args = parser.parse_args()

    if args.command:
        summary = run_with_sampling(args.command, args.interval, args.label)
    else:
        summary = watch(args.seconds, args.interval, args.label)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
