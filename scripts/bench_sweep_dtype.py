"""Can the connectome sweep run in half-width floats?

The sweep is bandwidth-bound and its dominant traffic is the gather of
pre-synaptic activity: nnz x streams values. bf16 sparse matmul is not
implemented in this ROCm build, but fp16 may be. Halving the operand width would
halve that traffic, and the values involved (0/1 spikes, O(1) currents) are well
inside fp16 range.

    python scripts/bench_sweep_dtype.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.device import tune_rocm  # noqa: E402
from flybrain.wholebrain import load_whole_brain  # noqa: E402

PEAK_BANDWIDTH_GBS = 960.0


def bench(fn, *, warmup: int = 3, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / iters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/sweep_dtype.json")
    args = parser.parse_args()

    tune_rocm()
    circuit, meta, _ = load_whole_brain("data/wholebrain.npz", device="cuda", impl="csr")
    nnz, n_neurons = circuit.pre.numel(), circuit.n_neurons
    circuit._ensure_csr()

    results: dict = {}
    print(f"{'values':>8} {'activity':>10} {'streams':>8} {'ms':>9} {'GB/s':>8} {'% peak':>8}")

    for value_dtype in (torch.float32, torch.float16):
        for activity_dtype in (torch.float32, torch.float16):
            # fp16 sparse with fp32 dense is not a valid hipSPARSE combination.
            if value_dtype != activity_dtype:
                continue
            for streams in (16, 32):
                values = circuit.prepare_synapses().to(value_dtype)
                activity = torch.rand(streams, n_neurons, device="cuda", dtype=activity_dtype)
                sparse = torch.sparse_csr_tensor(
                    circuit.csr_crow,
                    circuit.csr_col,
                    values,
                    size=(n_neurons, n_neurons),
                    check_invariants=False,
                )
                label = str(value_dtype).replace("torch.", "")
                try:
                    with torch.no_grad():
                        ms = bench(lambda: torch.sparse.mm(sparse, activity.t().contiguous())) * 1e3
                except Exception as exc:
                    print(f"{label:>8} {label:>10} {streams:>8}  FAILED {type(exc).__name__}: {str(exc)[:80]}")
                    results[f"{label}_{streams}"] = {"error": str(exc)[:200]}
                    continue
                element = 2 if activity_dtype == torch.float16 else 4
                gb = (nnz * element + nnz * streams * element + n_neurons * streams * element) / 1024**3
                gbs = gb / (ms / 1e3)
                results[f"{label}_{streams}"] = {
                    "ms": round(ms, 3),
                    "gb_per_s": round(gbs, 1),
                    "pct_of_peak": round(100 * gbs / PEAK_BANDWIDTH_GBS, 1),
                }
                print(
                    f"{label:>8} {label:>10} {streams:>8} {ms:9.3f} {gbs:8.1f} "
                    f"{100 * gbs / PEAK_BANDWIDTH_GBS:7.1f}%"
                )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
