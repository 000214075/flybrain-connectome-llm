"""Is the degree-preserving shuffle's 3.3x training slowdown in the sweep itself?

Both graphs hold the same 25,563,096 synapses over the same 164,587 neurons with
the same in- and out-degree for every cell, so they cost the same number of edge
operations. The training logs disagree anyway: the real connectome ran 492 steps
in 40 minutes and the shuffled one 150. A re-sorted copy of the shuffled edge list
(both edge orders are the same graph) ran at the same speed, so it is not file
layout. This isolates whether it is the sweep.

    python scripts/probe_edge_order_cost.py
"""

from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.device import tune_rocm  # noqa: E402
from flybrain.wholebrain import load_whole_brain  # noqa: E402


def bench(fn, warmup: int = 2, iters: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    tune_rocm()
    paths = [
        ("real", "data/wholebrain.npz"),
        ("shuffled", "data/wholebrain_shuffled.npz"),
        ("shuffled-resorted", "data/wholebrain_shuffled_sorted.npz"),
    ]
    streams = 32
    out: dict = {}
    for name, path in paths:
        for impl in ("index_add", "csr"):
            circuit, meta, _ = load_whole_brain(path, device="cuda", impl=impl)
            circuit.edge_chunk = 4_000_000
            state = torch.zeros(streams, meta.n_neurons, device="cuda", dtype=torch.float32)
            synapses = circuit.effective_synapses()
            with torch.no_grad():
                forward_ms = bench(lambda: circuit.synaptic_current(state, synapses))
            grad_state = torch.zeros(
                streams, meta.n_neurons, device="cuda", dtype=torch.float32, requires_grad=True
            )
            drive = torch.zeros(streams, meta.n_neurons, device="cuda", dtype=torch.float32)

            def step():
                out_ = circuit.step(grad_state, drive, circuit.effective_synapses())
                out_.sum().backward()
                circuit.zero_grad(set_to_none=True)

            forward_backward_ms = bench(step, warmup=2, iters=4)
            out[f"{name}.{impl}"] = {
                "forward_ms": round(forward_ms, 2),
                "forward_backward_ms": round(forward_backward_ms, 2),
            }
            print(f"{name:>18} {impl:>10}  forward {forward_ms:8.2f} ms  fwd+bwd {forward_backward_ms:8.2f} ms")
            del circuit, state, grad_state
            torch.cuda.empty_cache()
    with open("reports/edge_order_cost.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
