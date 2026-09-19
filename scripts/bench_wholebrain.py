"""Measure how fast the whole connectome can be swept on this GPU.

Everything about the brain module's configuration is decided from these numbers:
how many streams, how large an edge chunk, how many sweeps per forward pass, and
whether index_add or a sparse matmul is faster. The achieved bandwidth is printed
next to the theoretical limit of the card so it is obvious how much headroom is
left.

    python scripts/bench_wholebrain.py
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

# RX 7900 XTX: 24 GB GDDR6 at 20 Gbps on a 384-bit bus.
PEAK_BANDWIDTH_GBS = 960.0
PEAK_BF16_TFLOPS = 123.0


def bench(fn, *, warmup: int = 3, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / iters


def traffic_gb(nnz: int, streams: int, n_neurons: int, bytes_per_value: int = 4) -> float:
    """Bytes moved by one sweep: gather, synapse read, scatter, plus the output."""
    return (
        nnz * streams * bytes_per_value  # gather pre-synaptic activity
        + nnz * bytes_per_value  # synapse strengths
        + nnz * streams * bytes_per_value  # message written to the temporary
        + nnz * streams * bytes_per_value  # temporary read by the scatter
        + n_neurons * streams * bytes_per_value * 2  # output read-modify-write
    ) / 1024**3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default="data/wholebrain.npz")
    parser.add_argument("--out", default="reports/wholebrain_bench.json")
    args = parser.parse_args()

    tune_rocm()
    circuit, meta, extras = load_whole_brain(args.path, device="cuda")
    nnz = circuit.pre.numel()
    n_neurons = circuit.n_neurons
    print(json.dumps(meta.__dict__, indent=2))
    print(f"resident buffers: {(nnz * 12 + n_neurons * 8) / 1024**2:.1f} MiB")

    results: dict = {"meta": meta.__dict__, "sweep": {}, "step": {}, "train_step": {}, "sparse_mm": {}}

    # 1. The raw connectome sweep at several stream counts.
    print(f"\n{'streams':>8} {'chunk':>10} {'ms':>9} {'GB/s':>9} {'% of peak':>10}")
    for streams in (4, 8, 16, 32, 64):
        for edge_chunk in (2_000_000, 8_000_000):
            circuit.edge_chunk = edge_chunk
            state = torch.zeros(streams, n_neurons, device="cuda", dtype=torch.float32)
            with torch.no_grad():
                synapses = circuit.effective_synapses()
                ms = bench(lambda: circuit.synaptic_current(state, synapses)) * 1e3
            gb = traffic_gb(nnz, streams, n_neurons)
            gbs = gb / (ms / 1e3)
            results["sweep"][f"streams{streams}_chunk{edge_chunk}"] = {
                "ms": round(ms, 3),
                "gb_per_s": round(gbs, 1),
                "pct_of_peak_bandwidth": round(100 * gbs / PEAK_BANDWIDTH_GBS, 1),
            }
            print(
                f"{streams:>8} {edge_chunk:>10,} {ms:9.3f} {gbs:9.1f} "
                f"{100 * gbs / PEAK_BANDWIDTH_GBS:9.1f}%"
            )

    # 2. Full neuron step (sweep + LIF) including the gradient path.
    print(f"\n{'streams':>8} {'phase':>10} {'ms':>9}")
    for streams in (8, 16, 32):
        circuit.edge_chunk = 8_000_000
        state = torch.zeros(streams, n_neurons, device="cuda", dtype=torch.float32, requires_grad=True)
        drive = torch.zeros(streams, n_neurons, device="cuda", dtype=torch.float32)

        def step_no_grad():
            with torch.no_grad():
                synapses = circuit.effective_synapses()
                return circuit.step(state.detach(), drive, synapses)

        ms_fwd = bench(step_no_grad) * 1e3

        state_g = torch.zeros(streams, n_neurons, device="cuda", dtype=torch.float32, requires_grad=True)

        def step_with_grad():
            synapses = circuit.effective_synapses()
            out = circuit.step(state_g, drive, synapses)
            out.sum().backward()
            circuit.zero_grad(set_to_none=True)
            return out

        ms_fb = bench(step_with_grad, warmup=2, iters=8) * 1e3
        results["step"][f"streams{streams}"] = {"forward_ms": round(ms_fwd, 3), "forward_backward_ms": round(ms_fb, 3)}
        print(f"{streams:>8} {'forward':>10} {ms_fwd:9.3f}")
        print(f"{streams:>8} {'fwd+bwd':>10} {ms_fb:9.3f}")

    # 3. Sparse matmul alternative, for comparison against gather/scatter.
    print("\n--- torch.sparse.mm (CSR) ---")
    try:
        indices = torch.stack([circuit.post.long(), circuit.pre.long()])
        sp = torch.sparse_coo_tensor(indices, circuit.weight, (n_neurons, n_neurons)).to("cuda")
        sp = sp.to_sparse_csr()
        for streams in (16,):
            dense = torch.zeros(n_neurons, streams, device="cuda")
            ms = bench(lambda: torch.sparse.mm(sp, dense), warmup=2, iters=5) * 1e3
            results["sparse_mm"][f"streams{streams}"] = {"ms": round(ms, 3)}
            print(f"  streams={streams}: {ms:.3f} ms")
    except Exception as exc:
        results["sparse_mm"]["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  sparse mm unavailable or failed: {type(exc).__name__}: {exc}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
