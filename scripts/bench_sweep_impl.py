"""Compare the two connectome sweep implementations on the real connectome.

Decides which one the model uses, and at what cost, from measurement.

    python scripts/bench_sweep_impl.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.device import tune_rocm  # noqa: E402
from flybrain.wholebrain import load_whole_brain  # noqa: E402

PEAK_BANDWIDTH_GBS = 960.0


def bench(fn, *, warmup: int = 3, iters: int = 15) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / iters


def main() -> int:
    tune_rocm()
    circuit, meta, _ = load_whole_brain("data/wholebrain.npz", device="cuda")
    nnz, n_neurons = circuit.pre.numel(), circuit.n_neurons

    with torch.no_grad():
        synapses = circuit.effective_synapses()

    results: dict = {}
    print(f"{'impl':>10} {'dtype':>8} {'streams':>8} {'ms':>9} {'GB/s':>8} {'% peak':>8}")
    for dtype in (torch.float32, torch.bfloat16):
        for impl in ("index_add", "csr"):
            circuit.impl = impl
            for streams in (8, 16, 32):
                activity = torch.rand(streams, n_neurons, device="cuda", dtype=dtype)
                try:
                    with torch.no_grad():
                        syn = circuit.prepare_synapses().to(dtype)
                        ms = bench(
                            lambda: circuit.synaptic_current(activity, syn, impl=impl)
                        ) * 1e3
                except Exception as exc:
                    print(f"{impl:>10} {str(dtype)[6:]:>8} {streams:>8}  FAILED {type(exc).__name__}: {exc}")
                    results[f"{impl}_{dtype}_{streams}"] = {"error": str(exc)}
                    continue
                # Gather + synapse read + scatter write + scatter read + output RMW.
                gb = (nnz * streams * 2 + nnz * 2 + nnz * streams * 4 + n_neurons * streams * 8) / 1024**3
                gbs = gb / (ms / 1e3)
                results[f"{impl}_{str(dtype)[6:]}_{streams}"] = {
                    "ms": round(ms, 3),
                    "gb_per_s": round(gbs, 1),
                    "pct_of_peak": round(100 * gbs / PEAK_BANDWIDTH_GBS, 1),
                }
                print(
                    f"{impl:>10} {str(dtype)[6:]:>8} {streams:>8} {ms:9.3f} {gbs:8.1f} "
                    f"{100 * gbs / PEAK_BANDWIDTH_GBS:7.1f}%"
                )

    # What a full training step costs with each implementation.
    print("\n--- full neuron step (forward and forward+backward, fp32, streams=16) ---")
    for impl in ("index_add", "csr"):
        state = torch.zeros(16, n_neurons, device="cuda", requires_grad=True)
        drive = torch.zeros(16, n_neurons, device="cuda")

        def fwd():
            with torch.no_grad():
                syn = circuit.effective_synapses()
                return circuit.synaptic_current(state.detach(), syn, impl=impl)

        try:
            ms_fwd = bench(fwd) * 1e3
        except Exception as exc:
            print(f"  {impl}: forward failed: {exc}")
            continue

        state_g = torch.zeros(16, n_neurons, device="cuda", requires_grad=True)

        def fwd_bwd():
            syn = circuit.effective_synapses()
            out = circuit.synaptic_current(state_g, syn, impl=impl)
            out.sum().backward()
            circuit.zero_grad(set_to_none=True)

        try:
            ms_fb = bench(fwd_bwd, warmup=2, iters=8) * 1e3
        except Exception as exc:
            print(f"  {impl}: backward failed: {exc}")
            continue
        results[f"step_{impl}"] = {"forward_ms": round(ms_fwd, 3), "forward_backward_ms": round(ms_fb, 3)}
        print(f"  {impl:<10} forward {ms_fwd:7.3f} ms   forward+backward {ms_fb:7.3f} ms")

    os.makedirs("reports", exist_ok=True)
    with open("reports/sweep_impl.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("\nwritten to reports/sweep_impl.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
