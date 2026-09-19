"""Does the real connectome keep different inputs distinct, or collapse them?

A purely recurrent model is only a language model if its state still depends on the
past after a few steps. The hybrid model does not care -- attention re-reads the
whole prefix -- but a connectome-only model has nothing else, so this measures the
one property it cannot do without.

For several distinct drives, it tracks how far apart the population states stay as
the sweeps accumulate, and how much of the initial difference survives after N steps.

Usage:
    python scripts/probe_connectome_memory.py --steps 16 --drives 8
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.train import TrainConfig, load_brain  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_domain_v3.json")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--drives", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--spiking", choices=("on", "off", "both"), default="both")
    parser.add_argument(
        "--gain-scales",
        default="1.0",
        help="comma-separated multipliers on the recurrent release gain; the collapse is "
        "a property of the recurrence, so this is the knob that decides whether the state "
        "follows the input or its own wiring",
    )
    return parser.parse_args()


def trace(brain, drives: torch.Tensor, spiking: bool, steps: int) -> list[torch.Tensor]:
    """Hold N distinct drives constant and watch whether the states stay distinct.

    `drives` is (streams, neurons), one fixed drive per stream, so a state that
    converges means the connectome collapsed different inputs onto one trajectory.
    """
    brain.spiking = spiking
    brain.reset_stats()
    brain.start_sequence()
    state = brain.initial_state(drives.shape[0], drives.device, drives.dtype)
    out = []
    with torch.no_grad():
        for _ in range(steps):
            state = brain.advance_drive(state, drives)
            out.append(state.clone())
    return out


def main() -> int:
    args = parse_args()
    cfg = TrainConfig.from_json(args.config)
    device = torch.device(args.device)
    brain = load_brain(cfg)
    assert brain is not None
    brain = brain.to(device)

    torch.manual_seed(0)
    n_neurons = brain.circuit.n_neurons
    modes = (True, False) if args.spiking == "both" else (args.spiking == "on",)
    baseline_gain = brain.circuit.release_gain.detach().clone()
    gains = [float(g) for g in args.gain_scales.split(",")]
    for gain_scale in gains:
        with torch.no_grad():
            brain.circuit.release_gain.copy_(baseline_gain * gain_scale)
        for spiking in modes:
            # Distinct drives, each the same magnitude, as `ConnectomeLM` produces them.
            raw = torch.randn(args.drives, n_neurons, device=device)
            raw = raw - raw.mean(dim=1, keepdim=True)
            raw = raw / raw.norm(dim=1, keepdim=True) * (n_neurons**0.5)
            drives = raw * brain.drive_scale
            states = trace(brain, drives, spiking, args.steps)
            # Two statistics that cannot be misread. `differing` is the share of
            # neurons whose state differs from the first stream's; `similarity` is the
            # mean pairwise distance as a fraction of the distance two *independent*
            # patterns would have. An earlier version of this probe reported only the
            # max elementwise difference, which for a binary state is always 0 or 1 and
            # says nothing -- reading it as a magnitude produced a wrong conclusion
            # that the connectome collapses its inputs.
            last = states[-1]
            differ = float((last != last[0:1]).float().mean())
            spread = float((last - last[0:1]).abs().mean())
            independent = float(last.std() * (2**0.5))
            pairwise = float(torch.cdist(last.float(), last.float()).mean())
            ratio = pairwise / independent if independent > 0 else 0.0
            print(
                f"gain x{gain_scale:<5} spiking={str(spiking):5s}: "
                f"neurons differing from stream 0 = {100 * differ:5.1f}%, "
                f"mean pairwise distance = {ratio:.3f} of independent patterns, "
                f"mean state = {float(last.mean()):.4f}, spread {spread:.4g}"
            )
    with torch.no_grad():
        brain.circuit.release_gain.copy_(baseline_gain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
