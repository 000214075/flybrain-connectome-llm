"""Probe when the whole brain starts answering differently for different tokens.

The connectome can be running perfectly and still carry nothing to the language
model: if the population settles into the same state whatever the input, the
read-out is a constant and every token is handed the same vector. That is exactly
what the batch-participation metric measures, and this script finds out what has to
change for it to move.

For a grid of drive/recurrence scales it reports, on real text:

  * firing fraction and the share of synapses that transmit
  * drive participation   -- do the tokens reach the brain differently
  * read-out participation -- does the brain answer them differently (1.0 = identical)

Usage:
    python scripts/probe_brain_information.py --config configs/train_domain_ft.json \
        --checkpoint checkpoints/flybrain-full/best.pt
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.model import FlyBrainLM  # noqa: E402
from flybrain.train import TrainConfig, load_brain, load_pathway  # noqa: E402
from flybrain.wholebrain import BrainPathway  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "seed the model build. Needed when comparing two configs that differ only in "
            "their wiring: without it the interfaces get different random weights and the "
            "comparison measures the init, not the connectome."
        ),
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--data", default="data/tokenized_domain/val.bin")
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--levels",
        action="store_true",
        help="measure participation at each stage: drive -> state -> read-out projection",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = TrainConfig.from_json(args.config)
    device = torch.device(args.device)

    model = FlyBrainLM(cfg.model, load_pathway(cfg), load_brain(cfg)).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=False)
    brain = model.brain
    assert brain is not None

    # Real, different text per stream: the whole point is that the streams differ.
    import numpy as np

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    span = args.tokens
    tokens = torch.stack(
        [
            torch.from_numpy(np.asarray(data[i * span : (i + 1) * span]).astype(np.int64))
            for i in range(args.batch)
        ]
    ).to(device)

    base_drive = float(brain.drive_scale)
    base_release = brain.circuit.release_gain.detach().clone()
    gain_floor = float(brain.circuit.gain_floor)
    drive_floor = float(brain.drive_floor)

    def inverse_softplus(target: float) -> float:
        """The raw value whose softplus gives `target`, so the probe can set a scale."""
        return float(torch.log(torch.expm1(torch.tensor(max(target, 1e-3)))))

    rows = []
    model.eval()
    if args.levels:
        # Where does the batch's information die? Participation measured after each
        # stage of the interface, on the same real batch.
        stages: dict[str, float] = {}
        def hook(_module, _inputs, output):
            stages["state"] = float(BrainPathway.participation_ratio(output.detach()))
        handle = brain.circuit.register_forward_hook(hook)
        with torch.no_grad():
            model(tokens)
        handle.remove()
        state = brain.last_state

        def stage(vectors) -> float | None:
            """Participation of one optional stage; None when that stage never ran.

            `_last_drive` / `_last_readout` are only filled by the paths that use them,
            so a forward that skips a stage (or a model built without training) leaves
            them None -- report that instead of crashing on `.detach()`.
            """
            if vectors is None:
                return None
            return float(BrainPathway.participation_ratio(vectors.detach()))

        with torch.no_grad():
            stages["drive"] = stage(brain._last_drive)
            stages["read_state"] = float(
                BrainPathway.participation_ratio(brain.read_state(state).detach())
            )
            stages["read_down"] = float(BrainPathway.participation_ratio(brain.read_down(state)))
            stages["readout"] = stage(brain._last_readout)
            stages["analog_weight"] = float(brain.analog_read)
            stages["state_mean"] = float(state.mean())
            stages["state_rows_identical"] = float((state - state[0:1]).abs().max())
        print(f"batch {args.batch}, tokens {args.tokens}")
        for name, value in stages.items():
            print(f"  {name:22s} {'-' if value is None else f'{value:.3f}'}", flush=True)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump({"batch": args.batch, **stages}, fh, indent=2, ensure_ascii=False)
            print(f"wrote {args.out}")
        return 0

    grid = (0.5, 1.0, 2.0, 4.0, 8.0)
    gains = (0.25, 0.5, 1.0, 2.0)
    with torch.no_grad():
        for drive_mult, gain_mult in itertools.product(grid, gains):
            brain.raw_drive_scale.fill_(inverse_softplus(base_drive * drive_mult - drive_floor))
            brain.circuit.raw_gain.copy_(
                torch.log(torch.expm1((base_release * gain_mult - gain_floor).clamp_min(1e-3)))
            )
            model(tokens)
            row = {
                "drive_mult": drive_mult,
                "gain_mult": gain_mult,
                "drive_scale": round(float(brain.drive_scale), 4),
                "release_gain": round(float(brain.circuit.release_gain.mean()), 4),
                "firing": round(float(brain.mean_activity), 4),
                "synapse_use": round(brain.synapse_use(brain.last_state), 4),
                "drive_participation": round(float(brain.drive_participation()), 2),
                "readout_participation": round(float(brain.readout_participation()), 2),
                "batch": args.batch,
            }
            rows.append(row)
            print(
                f"drive x{drive_mult:<4} gain x{gain_mult:<4} | "
                f"firing {100 * row['firing']:5.1f}%  syn {100 * row['synapse_use']:5.1f}% | "
                f"participation drive {row['drive_participation']:.2f}/{args.batch} "
                f"read-out {row['readout_participation']:.2f}/{args.batch}",
                flush=True,
            )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, ensure_ascii=False)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
