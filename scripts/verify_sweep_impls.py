"""Do the two connectome sweeps compute the same thing, and which is faster?

The circuit has two implementations of one sweep across all 25,563,096 synapses:
a sparse CSR matmul (`torch.sparse.mm`) and a gather/scatter (`index_add`). They
exist because the CSR path was once seen to hang inside a training loop (~step 60 of
the hybrid model) and was never reproduced in isolation, so training moved to
`index_add` and stayed there.

That decision costs throughput. A separate single-sweep benchmark measured CSR at
67.98 ms forward+backward against 132.63 ms for `index_add` -- roughly 2x -- and the
connectome-only model sweeps *once per token*, so this is the largest single lever on
its per-token cost. Before switching on the strength of a timing number, this checks
that the two paths agree on the arithmetic, including the gradients, because a fast
kernel that computes something else is not a speedup.

Usage:
    python scripts/verify_sweep_impls.py --sweeps 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.train import TrainConfig, load_brain  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_connectome_only.json")
    parser.add_argument("--streams", type=int, default=32)
    parser.add_argument("--sweeps", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = TrainConfig.from_json(args.config)
    device = torch.device(args.device)
    brain = load_brain(cfg)
    assert brain is not None
    brain = brain.to(device)
    circuit = brain.circuit

    # A fixed state and drive, so any difference between the two paths is the kernel
    # and not the input.
    generator = torch.Generator(device="cpu").manual_seed(0)
    state = torch.rand(
        args.streams, circuit.n_neurons, generator=generator, dtype=torch.float32
    ).to(device)
    drive = torch.randn(
        args.streams, circuit.n_neurons, generator=generator, dtype=torch.float32
    ).to(device)
    # Something to differentiate against; its only job is to make the backward
    # non-trivial and identical for both runs.
    readout = torch.randn(
        args.streams, circuit.n_neurons, generator=generator, dtype=torch.float32
    ).to(device)

    results: dict[str, dict] = {}
    for impl in ("index_add", "csr"):
        circuit.impl = impl
        circuit.zero_grad(set_to_none=True)
        # `prepare_synapses`, never `effective_synapses`: the CSR structure is sorted by
        # postsynaptic neuron, so its values must be permuted into that same edge order.
        # Feeding it the un-permuted values -- this script's first version did -- applies
        # every edge's strength, sign and release gain to a different edge, and the two
        # paths then disagree on 99.9% of the output. That is an API misuse, not a
        # kernel bug, and the mistake is easy enough to make that it deserves a test.
        synapses = circuit.prepare_synapses()
        started = time.perf_counter()
        out = circuit.step(state, drive, synapses, spiking=False)
        (out * readout).sum().backward()
        torch.cuda.synchronize()
        first = time.perf_counter() - started
        grad = circuit.raw_gain.grad.detach().clone()

        # Timed sweeps, forward+backward, same weights.
        circuit.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(args.sweeps):
            synapses = circuit.prepare_synapses()
            out = circuit.step(state, drive, synapses, spiking=False)
            (out * readout).sum().backward()
            circuit.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        per_sweep = (time.perf_counter() - started) / args.sweeps

        results[impl] = {
            "out": out.detach().float().cpu(),
            "grad": grad.float().cpu(),
            "first_sweep_s": round(first, 4),
            "ms_per_sweep_fwd_bwd": round(1000 * per_sweep, 2),
        }
        print(f"{impl:>10}: {1000 * per_sweep:8.2f} ms per sweep (forward+backward)")

    a, b = results["index_add"], results["csr"]
    print()
    report: dict = {
        "streams": args.streams,
        "neurons": circuit.n_neurons,
        "synapses": int(circuit.pre.numel()),
        "ms_per_sweep_fwd_bwd_index_add": a["ms_per_sweep_fwd_bwd"],
        "ms_per_sweep_fwd_bwd_csr": b["ms_per_sweep_fwd_bwd"],
        "speedup_csr_vs_index_add": round(
            a["ms_per_sweep_fwd_bwd"] / max(b["ms_per_sweep_fwd_bwd"], 1e-9), 3
        ),
    }
    for name in ("out", "grad"):
        x, y = a[name], b[name]
        max_abs = float((x - y).abs().max())
        scale = float(x.abs().max())
        rel = max_abs / scale if scale > 0 else 0.0
        mismatched = int((x != y).sum())
        report[f"{name}_max_abs_diff"] = max_abs
        report[f"{name}_rel_to_scale"] = round(rel, 10)
        report[f"{name}_elements_differing"] = mismatched
        report[f"{name}_elements"] = int(x.numel())
        report[f"{name}_scale"] = scale
        print(
            f"{name:>4}: max|a-b| {max_abs:.6g}  (scale {scale:.6g}, rel {rel:.3g}, "
            f"{mismatched:,}/{x.numel():,} elements differ)"
        )

    report["agree"] = (
        report["out_rel_to_scale"] < 1e-5 and report["grad_rel_to_scale"] < 1e-4
    )
    print(f"\nagree for training: {report['agree']}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"wrote {args.out}")
    return 0 if report["agree"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
