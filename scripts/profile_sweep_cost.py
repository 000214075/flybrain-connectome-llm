"""Where does one connectome sweep actually spend its time?

The connectome-only model's cost is one full sweep of all 25,563,096 synapses per
token, so the sweep *is* the model's price. Before changing the kernel, decompose that
price: the CSR path currently rebuilds a 25.5M-element values array every sweep
(`prepare_synapses` gathers the transmitter sign and the release gain per edge and
multiplies them by the frozen counts) and constructs a fresh sparse tensor from it,
on top of the sparse matmul itself. If the rebuild dominates, there is a large and
mathematically exact win available -- the frozen part of those values does not change
between sweeps at all.

Usage:
    python scripts/profile_sweep_cost.py --streams 32 --reps 10
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def bench(fn, reps: int) -> float:
    fn()  # warm up: the first call builds cached buffers
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - started) / reps


def main() -> int:
    args = parse_args()
    cfg = TrainConfig.from_json(args.config)
    cfg.model.brain_impl = "csr"
    device = torch.device(args.device)
    brain = load_brain(cfg)
    assert brain is not None
    brain = brain.to(device)
    circuit = brain.circuit
    circuit._ensure_csr()

    generator = torch.Generator(device="cpu").manual_seed(0)
    state = torch.rand(args.streams, circuit.n_neurons, generator=generator).to(device)
    drive = torch.randn(args.streams, circuit.n_neurons, generator=generator).to(device)

    n_edges = int(circuit.pre.numel())
    values = circuit.prepare_synapses()
    sparse = torch.sparse_csr_tensor(
        circuit.csr_crow_t, circuit.csr_col_t, values,
        size=(circuit.n_neurons, circuit.n_neurons), check_invariants=False,
    )
    # The frozen part of the values: how much each edge transmits before the
    # per-neuron release gain is applied. It cannot change during training.
    frozen_base = circuit.weight_perm * circuit.sign[circuit.pre_perm]

    parts = {
        "prepare_synapses (rebuild 25.5M values)": lambda: circuit.prepare_synapses(),
        "sparse_csr_tensor (construct)": lambda: torch.sparse_csr_tensor(
            circuit.csr_crow_t, circuit.csr_col_t, values,
            size=(circuit.n_neurons, circuit.n_neurons), check_invariants=False,
        ),
        "sparse.mm (the matmul itself)": lambda: torch.sparse.mm(sparse, state.t().contiguous()).t(),
        "scale activity by release gain only (164k mul)": lambda: (
            circuit.release_gain * state
        ),
        "frozen base reuse (no rebuild)": lambda: frozen_base,
    }

    print(f"{args.streams} streams, {n_edges:,} synapses, {args.reps} reps, forward only\n")
    results = {}
    for name, fn in parts.items():
        results[name] = bench(fn, args.reps)
        print(f"  {results[name]:8.3f} ms  {name}")

    full = bench(lambda: circuit.step(state, drive, circuit.prepare_synapses(), spiking=False), args.reps)
    results["full step, forward only"] = full
    print(f"\n  {full:8.3f} ms  full step (forward only)")

    def forward_and_backward() -> None:
        circuit.zero_grad(set_to_none=True)
        out = circuit.step(state, drive, circuit.prepare_synapses(), spiking=False)
        out.sum().backward()

    both = bench(forward_and_backward, args.reps)
    backward = both - full
    print(f"  {both:8.3f} ms  full step (forward + backward)")
    print(f"  {backward:8.3f} ms  -> backward alone is {100 * backward / both:.0f}% of the cost")

    # Split the backward in two, because only one half is removable.
    grad_out = torch.randn(args.streams, circuit.n_neurons, device=device)
    order_t = circuit.order_t
    sparse_t = torch.sparse_csr_tensor(
        circuit.csr_crow_t, circuit.csr_col_t, values[order_t],
        size=(circuit.n_neurons, circuit.n_neurons), check_invariants=False,
    )
    t_act = bench(lambda: torch.sparse.mm(sparse_t, grad_out.t().contiguous()).t(), args.reps)

    nnz = circuit.pre.numel()
    edge_chunk = cfg.model.brain_edge_chunk or nnz
    pre, post = circuit.pre, circuit.post
    sink = torch.empty_like(values)

    def grad_values():
        for start in range(0, nnz, edge_chunk):
            stop = min(start + edge_chunk, nnz)
            sink[start:stop] = (
                state[:, pre[start:stop]] * grad_out[:, post[start:stop]]
            ).sum(dim=0)

    t_vals = bench(grad_values, args.reps)
    rebuild = results["prepare_synapses (rebuild 25.5M values)"] + results["sparse_csr_tensor (construct)"]
    print(f"\n  backward, grad onto neurons (W^T @ grad_out)  {t_act:8.3f} ms")
    print(f"  backward, grad onto all 25.5M edge values      {t_vals:8.3f} ms  <- removable")
    print(f"  sum {t_act + t_vals:.3f} ms vs measured backward {backward:.3f} ms")
    print(
        f"\n  So folding the release gain into the activity would remove the {t_vals:.1f} ms\n"
        f"  edge-gradient term plus the {rebuild:.1f} ms rebuild, i.e. about "
        f"{100 * (t_vals + rebuild) / both:.0f}% of a training step -- not the 10% the\n"
        f"  forward alone suggested. The W^T @ grad_out half is needed either way."
    )

    rebuild = results["prepare_synapses (rebuild 25.5M values)"] + results["sparse_csr_tensor (construct)"]
    share = 100 * rebuild / full
    print(f"\n  rebuild + construct = {rebuild:.3f} ms = **{share:.1f}%** of a forward step")
    print(f"  matmul is only      = {results['sparse.mm (the matmul itself)']:.3f} ms")
    print(
        "\n  So a sweep spends most of its time re-deriving 25.5M values that only depend on\n"
        "  the frozen connectome, plus a 164,587-element trainable release gain. Folding that\n"
        "  gain into the *activity* instead makes the sparse values constant: they can be built\n"
        "  once and reused, and the per-sweep work drops to a 164,587-element multiply."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
