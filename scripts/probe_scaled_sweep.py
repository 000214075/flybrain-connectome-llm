"""Can plain autograd drive the sweep once the sparse values are constant?

The connectome's sparse matrices depend only on data that never trains: the synapse
counts and the per-neuron transmitter signs, both frozen buffers. The only trainable
factor per edge is the *pre-synaptic* neuron's release gain, and because it is indexed
by the source neuron it can be applied to the activity instead of to the edges:

    out[target, s] = sum_e base[e] * (gain[pre[e]] * activity[pre[e], s])

That is the same sum of the same products, only the multiply order changes, so it is
exact rather than an approximation. The prize is that the sparse matrix becomes a
constant: no 25.5M-element values rebuild per sweep, and -- measured at 32 streams --
the per-edge gradient term that costs 32.3 ms of a 55.7 ms step simply disappears.

This checks the two things that decide the implementation: whether PyTorch's own
autograd can carry the backward through `torch.sparse.mm` when only the dense operand
trains, and what that costs compared with the current custom Function.

Usage:
    python scripts/probe_scaled_sweep.py --streams 32 --reps 10
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
    return parser.parse_args()


def bench(fn, reps: int) -> float:
    fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - started) / reps


class _ScaledSweep(torch.autograd.Function):
    """The scaled route with a hand-written backward instead of autograd's.

    Autograd's sparse-CSR backward does not use the transposed CSR the circuit
    already caches, so on its own it gives back only part of the win. Here the
    forward keeps the constant matrix and the backward keeps the cached transpose.
    """

    @staticmethod
    def forward(ctx, activity, gain, crow, col, crow_t, col_t, base, base_t):  # type: ignore[override]
        n_neurons = crow.numel() - 1
        scaled = gain * activity
        sparse = torch.sparse_csr_tensor(
            crow, col, base, size=(n_neurons, n_neurons), check_invariants=False
        )
        out = torch.sparse.mm(sparse, scaled.t().contiguous()).t()
        ctx.save_for_backward(activity, gain, scaled)
        ctx.sweep = (crow_t, col_t, base_t, n_neurons)
        return out

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        activity, gain, scaled = ctx.saved_tensors
        crow_t, col_t, base_t, n_neurons = ctx.sweep
        sparse_t = torch.sparse_csr_tensor(
            crow_t, col_t, base_t, size=(n_neurons, n_neurons), check_invariants=False
        )
        grad_scaled = torch.sparse.mm(sparse_t, grad_out.t().contiguous()).t()
        grad_activity = grad_scaled * gain
        grad_gain = (grad_scaled * activity).sum(dim=0)
        return grad_activity, grad_gain, None, None, None, None, None, None


def main() -> int:
    args = parse_args()
    cfg = TrainConfig.from_json(args.config)
    device = torch.device(args.device)
    brain = load_brain(cfg)
    assert brain is not None
    brain = brain.to(device)
    circuit = brain.circuit
    circuit._ensure_csr()

    generator = torch.Generator(device="cpu").manual_seed(0)
    state = torch.rand(args.streams, circuit.n_neurons, generator=generator).to(device)
    grad_out = torch.randn(args.streams, circuit.n_neurons, generator=generator).to(device)

    # The constant part: how much each edge transmits before the release gain.
    base = (circuit.weight_perm * circuit.sign[circuit.pre_perm]).detach()
    base_t = base[circuit.order_t].contiguous()
    edge_chunk = cfg.model.brain_edge_chunk or int(circuit.pre.numel())

    print("=== 1. does PyTorch autograd carry the backward with constant values? ===")
    raw_gain = torch.nn.Parameter(torch.zeros(circuit.n_neurons, device=device))
    sparse_const = torch.sparse_csr_tensor(
        circuit.csr_crow, circuit.csr_col, base,
        size=(circuit.n_neurons, circuit.n_neurons), check_invariants=False,
    )
    sparse_const_t = torch.sparse_csr_tensor(
        circuit.csr_crow_t, circuit.csr_col_t, base_t,
        size=(circuit.n_neurons, circuit.n_neurons), check_invariants=False,
    )
    try:
        gain = torch.nn.functional.softplus(raw_gain) + 0.5
        activity = state.clone().requires_grad_(True)
        scaled = gain * activity
        out = torch.sparse.mm(sparse_const, scaled.t().contiguous()).t()
        (out * grad_out).sum().backward()
        print("  autograd backward: OK")
        print(f"  raw_gain.grad norm {float(raw_gain.grad.norm()):.6g}, "
              f"activity.grad norm {float(activity.grad.norm()):.6g}")
    except Exception as exc:  # noqa: BLE001
        print(f"  autograd backward FAILED: {type(exc).__name__}: {exc}")
        return 1

    print("\n=== 2. does it compute the same thing as the current kernel? ===")
    circuit.zero_grad(set_to_none=True)
    reference = circuit.step(state, torch.zeros_like(state), circuit.prepare_synapses(), spiking=False)
    # Same sweep via the scaled route, on the same state and the same gain.
    with torch.no_grad():
        scaled_route = (circuit.release_gain * state)
        got = torch.sparse.mm(sparse_const, scaled_route.t().contiguous()).t()
    reference_out = torch.sparse.mm(
        torch.sparse_csr_tensor(
            circuit.csr_crow, circuit.csr_col, circuit.prepare_synapses(),
            size=(circuit.n_neurons, circuit.n_neurons), check_invariants=False,
        ),
        state.t().contiguous(),
    ).t()
    diff = float((reference_out - got).abs().max())
    scale = float(reference_out.abs().max())
    print(f"  max |difference| {diff:.6g}  (scale {scale:.6g}, relative {diff / scale:.3g})")
    print(f"  exact to fp32 rounding: {diff / scale < 1e-4}")

    print("\n=== 3. what does each route cost (forward + backward)? ===")

    # Kernel against kernel: both sides end at the raw synaptic current. Comparing
    # against `circuit.step` would charge the sweep for the LIF step, the read-out
    # sigmoid and the dtype casts, and overstate the speedup.
    def current_route() -> None:
        circuit.zero_grad(set_to_none=True)
        out = circuit.synaptic_current_csr(state, circuit.prepare_synapses())
        (out * grad_out).sum().backward()

    optimizer = torch.optim.SGD([raw_gain], lr=0.0)

    def scaled_route_run() -> None:
        optimizer.zero_grad(set_to_none=True)
        g = torch.nn.functional.softplus(raw_gain) + 0.5
        act = state
        s = g * act
        out = torch.sparse.mm(sparse_const, s.t().contiguous()).t()
        (out * grad_out).sum().backward()

    def scaled_kernel_run() -> None:
        raw_gain.grad = None
        out = _ScaledSweep.apply(
            state, torch.nn.functional.softplus(raw_gain) + 0.5,
            circuit.csr_crow, circuit.csr_col, circuit.csr_crow_t, circuit.csr_col_t,
            base, base_t,
        )
        (out * grad_out).sum().backward()

    t_current = bench(current_route, args.reps)
    t_scaled = bench(scaled_route_run, args.reps)
    t_kernel = bench(scaled_kernel_run, args.reps)
    print(f"  current (trainable edge values)  {t_current:8.3f} ms")
    print(f"  scaled  (constant values)        {t_scaled:8.3f} ms   {t_current / t_scaled:.2f}x")
    print(f"  scaled + hand-written backward   {t_kernel:8.3f} ms   {t_current / t_kernel:.2f}x")

    print("\n=== 4. does the hand-written backward match the current gradients? ===")
    circuit.zero_grad(set_to_none=True)
    out_a = circuit.synaptic_current_csr(state, circuit.prepare_synapses())
    (out_a * grad_out).sum().backward()
    grad_a = circuit.raw_gain.grad.clone()
    circuit.zero_grad(set_to_none=True)
    out_b = _ScaledSweep.apply(
        state, circuit.release_gain, circuit.csr_crow, circuit.csr_col,
        circuit.csr_crow_t, circuit.csr_col_t, base, base_t,
    )
    (out_b * grad_out).sum().backward()
    grad_b = circuit.raw_gain.grad.clone()
    out_diff = float((out_a - out_b).abs().max())
    grad_diff = float((grad_a - grad_b).abs().max())
    grad_scale = float(grad_a.abs().max())
    print(f"  forward max |difference| {out_diff:.6g} (scale {scale:.6g})")
    print(f"  raw_gain.grad max |difference| {grad_diff:.6g} (scale {grad_scale:.6g}, "
          f"relative {grad_diff / grad_scale:.3g})")
    best = t_current / min(t_scaled, t_kernel)
    if best < 1.15:
        print("\n  Not worth it: below 1.15x there is no point carrying the change.")
    else:
        print(f"\n  Worth carrying: {best:.2f}x on the sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
