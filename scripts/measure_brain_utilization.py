"""Measure how much of the whole fly brain the language model actually uses.

"Running the whole brain" and "using the whole brain" are different claims, and
only the second one means anything. This script measures both, on the real
connectome, from a real forward/backward pass:

  * participation   -- what share of the 164,587 neurons carry a non-zero state
  * transmission    -- what share of the 25,563,096 synapses carried a spike
  * bandwidth       -- the rank of the read-out and of the token drive
  * depth coverage  -- the share of transformer blocks the brain read-out reaches
  * token coverage  -- the share of token positions the brain read-out reaches
  * gradient cover  -- the share of neurons the loss can actually reach
  * sweeps          -- how many times the connectome is iterated per pass
  * compute share   -- the share of forward-pass time spent in the connectome

Usage:
    python scripts/measure_brain_utilization.py --config configs/train_full.json
    python scripts/measure_brain_utilization.py --config configs/train_full.json --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.model import FlyBrainLM  # noqa: E402
from flybrain.train import TrainConfig, load_brain, load_pathway  # noqa: E402
from flybrain.wholebrain import effective_rank, expand_state_dict_rank  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None, help="cpu | cuda (default: cuda when available)")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--checkpoint", default="", help="optional checkpoint to measure instead of random init")
    parser.add_argument("--out", default="", help="write the metrics as JSON here")
    parser.add_argument("--std-scale", type=float, default=0.1, help="rank-widening scale, for checkpoints")
    return parser.parse_args()


def measure(args: argparse.Namespace) -> dict:
    cfg = TrainConfig.from_json(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)

    model = FlyBrainLM(cfg.model, load_pathway(cfg), load_brain(cfg)).to(device)
    brain = model.brain
    if brain is None:
        raise SystemExit("this config has no whole-brain circuit; nothing to measure")
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        expanded = expand_state_dict_rank(state["model"], model, std_scale=args.std_scale)
        result = model.load_state_dict(expanded, strict=False)
        print(f"loaded {args.checkpoint}: missing={list(result.missing_keys)}")

    circuit = brain.circuit
    n_neurons = circuit.n_neurons
    nnz = int(circuit.pre.numel())
    dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float32

    # A real batch of text, so the measurement is not made on noise.
    tokens = torch.randint(0, cfg.model.vocab_size, (args.batch, args.tokens), device=device)

    def forward_backward() -> Tensor:
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            _, loss = model(tokens, tokens)
        loss.backward()
        return loss

    # --- gradient coverage -------------------------------------------------
    # `up` and `read_down` both have one row/column per neuron, so counting the
    # non-zero ones counts the neurons the loss can actually reach.
    model.train()
    forward_backward()
    up_grad = brain.up.weight.grad
    read_grad = brain.read_down.weight.grad
    drive_cover = (up_grad.abs().sum(dim=1) > 0).float().mean().item() if up_grad is not None else 0.0
    read_cover = (read_grad.abs().sum(dim=0) > 0).float().mean().item() if read_grad is not None else 0.0
    activity = float(brain.mean_activity.detach()) if brain.mean_activity is not None else 0.0
    synapse_use = brain.synapse_use(brain.last_state) if brain.last_state is not None else 0.0
    sweeps = int(getattr(brain, "sweeps", 0))

    # --- bandwidth ---------------------------------------------------------
    # The composed maps say what the model can carry; the factor ranks say whether
    # training kept the bandwidth it was given. A rank-64 projection whose learned
    # factors collapsed to rank 1 is a rank-1 interface with 64 parameters of
    # decoration.
    with torch.no_grad():
        read_rank = effective_rank(brain.read_up.weight @ brain.read_down.weight)
        drive_rank = effective_rank(brain.up.weight @ brain.down.weight)
        drive_down_rank = effective_rank(brain.down.weight)
        drive_up_rank = effective_rank(brain.up.weight.t())
        read_down_rank = effective_rank(brain.read_down.weight)
        read_up_rank = effective_rank(brain.read_up.weight)
        rest_active = float((brain.resting_state > 0.01).float().mean())

    # --- how differently the batch's tokens reach and leave the brain ------
    # Factor ranks can look healthy while the composed map still hands every token
    # the same vector, so this is measured on real data through both interfaces.
    with torch.no_grad():
        model(tokens)
    drive_pr = brain.drive_participation()
    read_pr = brain.readout_participation()
    drive_pr = float(drive_pr) if drive_pr is not None else 0.0
    read_pr = float(read_pr) if read_pr is not None else 0.0

    # --- depth coverage ----------------------------------------------------
    # A block is covered when the brain read-out changes what it computes, so
    # watch for the read-out actually being handed to the block.
    seen: dict[int, torch.Tensor | None] = {}
    handles = [
        block.register_forward_pre_hook(
            (lambda index: lambda module, a, kw: seen.__setitem__(index, kw.get("bias")))(i),
            with_kwargs=True,
        )
        for i, block in enumerate(model.blocks)
    ]
    try:
        with torch.no_grad():
            model(tokens)
    finally:
        for handle in handles:
            handle.remove()
    covered = [i for i, bias in seen.items() if bias is not None and float(bias.abs().sum()) > 0]

    # --- token coverage ----------------------------------------------------
    # Perturb the read-out and count the positions whose logits move.
    with torch.no_grad():
        base = model(tokens).float()
        backup = brain.read_up.weight.detach().clone()
        brain.read_up.weight.mul_(0.5)
        shifted = model(tokens).float()
        brain.read_up.weight.copy_(backup)
    moved = (base - shifted).abs().amax(dim=-1)[0] > 0
    token_coverage = float(moved.float().mean())

    # --- where the time goes ----------------------------------------------
    def timed(fn, repeat: int = 3) -> float:
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeat):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - start) / repeat

    model.train()
    step_seconds = timed(forward_backward)
    # The honest share of the step: the same pass with the brain detached. Timing a
    # standalone sweep would miss the drive, the read-out and the backward through
    # the connectome, which is most of the connectome's real cost.
    keep = model.brain
    model.brain = None  # type: ignore[assignment]
    try:
        plain_seconds = timed(forward_backward)
    finally:
        model.brain = keep  # type: ignore[assignment]
    model.zero_grad(set_to_none=True)
    return {
        "config": os.path.basename(args.config),
        "device": f"{device} ({cfg.dtype})",
        "neurons": n_neurons,
        "synapses": nnz,
        "sweeps_per_pass": sweeps,
        "neuron_active_fraction": round(activity, 4),
        "synapse_transmission_fraction": round(synapse_use, 4),
        "resting_neurons_fraction": round(rest_active, 4),
        "readout_rank": int(brain.read_down.weight.shape[0]),
        "readout_effective_rank": read_rank,
        "drive_effective_rank": drive_rank,
        "drive_factor_ranks": [drive_down_rank, drive_up_rank],
        "readout_factor_ranks": [read_down_rank, read_up_rank],
        "batch": int(args.batch),
        "drive_participation": round(drive_pr, 2),
        "readout_participation": round(read_pr, 2),
        "blocks_total": len(model.blocks),
        "blocks_modulated_by_brain": len(covered),
        "depth_coverage": round(len(covered) / max(1, len(model.blocks)), 4),
        "token_coverage": round(token_coverage, 4),
        "neurons_with_drive_gradient": round(drive_cover, 4),
        "neurons_with_readout_gradient": round(read_cover, 4),
        "step_seconds": round(step_seconds, 4),
        "step_seconds_without_brain": round(plain_seconds, 4),
        # The connectome's own cost, in context: the drive, the sweeps, the read-out
        # and the backward through all of them, divided by the sweeps it ran.
        "seconds_per_sweep": round(
            (step_seconds - plain_seconds) / max(1, sweeps), 4
        ),
        "brain_time_share": round(1.0 - plain_seconds / max(step_seconds, 1e-9), 4),
    }


def report(metrics: dict) -> None:
    print()
    print(f"whole-brain utilisation -- {metrics['config']} on {metrics['device']}")
    print("-" * 66)
    rows = [
        ("connectome", f"{metrics['neurons']:,} neurons, {metrics['synapses']:,} synapses"),
        ("sweeps per forward pass", f"{metrics['sweeps_per_pass']}"),
        ("neurons active per sweep", f"{100 * metrics['neuron_active_fraction']:.1f}%"),
        ("synapses transmitting", f"{100 * metrics['synapse_transmission_fraction']:.1f}%"),
        ("neurons at rest above 1%", f"{100 * metrics['resting_neurons_fraction']:.1f}%"),
        (
            "read-out bandwidth",
            f"{metrics['readout_rank']} dims, effective rank {metrics['readout_effective_rank']} "
            f"(factors {metrics['readout_factor_ranks'][0]}/{metrics['readout_factor_ranks'][1]})",
        ),
        (
            "drive bandwidth",
            f"effective rank {metrics['drive_effective_rank']} "
            f"(factors {metrics['drive_factor_ranks'][0]}/{metrics['drive_factor_ranks'][1]})",
        ),
        (
            "batch participation",
            f"drive {metrics['drive_participation']}/{metrics['batch']}, "
            f"read-out {metrics['readout_participation']}/{metrics['batch']}",
        ),
        (
            "depth coverage",
            f"{metrics['blocks_modulated_by_brain']}/{metrics['blocks_total']} blocks "
            f"({100 * metrics['depth_coverage']:.0f}%)",
        ),
        ("token coverage", f"{100 * metrics['token_coverage']:.1f}% of positions"),
        ("neurons reached by the loss", f"{100 * metrics['neurons_with_readout_gradient']:.1f}% read-out"),
        ("neurons reached by the loss", f"{100 * metrics['neurons_with_drive_gradient']:.1f}% drive"),
        (
            "step time",
            f"{metrics['step_seconds']:.3f} s (without the brain {metrics['step_seconds_without_brain']:.3f} s)",
        ),
        ("per connectome sweep", f"{metrics['seconds_per_sweep']:.3f} s"),
        ("connectome share of the step", f"{100 * metrics['brain_time_share']:.1f}%"),
    ]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"  {name:<{width}}  {value}")


def main() -> int:
    args = parse_args()
    metrics = measure(args)
    report(metrics)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
