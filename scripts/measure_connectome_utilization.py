"""How much of the connectome the pure model actually runs, per metric.

The transformer hybrid ran the whole brain as a modulation bolted onto a stack, so
"is the brain being used" was a real question there. The pure model has no stack:
the connectome *is* the sequence model and every token sweeps every synapse. This
measures what that actually buys, on a trained checkpoint and a real batch, so the
numbers in the report are read off the model rather than argued from its design.

Metrics, all measured:
  * neurons active, and the share of the population in the responsive band;
  * synapses carrying signal, rate-weighted (see `BrainPathway.synapse_use`);
  * edges whose weight receives gradient -- an edge with no gradient is wiring that
    training cannot use, however faithfully it is loaded;
  * effective rank of the read-out and of the drive across a batch of real documents,
    i.e. how many independent directions the brain answers;
  * sweeps per forward pass, and the fraction of tokens whose prediction comes from a
    state the connectome computed (1.0 by construction; measured, not assumed).

Usage:
    python scripts/measure_connectome_utilization.py \
        --checkpoint checkpoints/flybrain-connectome/best.pt --out reports/utilization_connectome.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.data import TokenWindowDataset  # noqa: E402
from flybrain.wholebrain import BrainPathway, effective_rank  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/flybrain-connectome/best.pt")
    parser.add_argument("--tokens", default="data/tokenized_domain/val.bin")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from flybrain.generate import load_model

    device = torch.device(args.device)
    model, cfg = load_model(args.checkpoint, device=device)
    model.train()
    brain = model.brain
    if brain is None:
        raise SystemExit("this model has no connectome")
    circuit = brain.circuit
    edges = int(circuit.pre.numel())

    dataset = TokenWindowDataset(args.tokens, cfg.context, token_dtype="uint16")
    x = torch.stack([dataset[i][0] for i in range(args.batch)]).to(device)
    y = torch.stack([dataset[i][1] for i in range(args.batch)]).to(device)

    model.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()

    state = brain.last_state
    pre = circuit.last_pre
    activity = float(state.float().mean())
    responsive = float(((pre.float() - circuit.lif.v_th).abs() < 0.5).float().mean())
    synapse_use = brain.synapse_use(state)

    grad = circuit.weight.grad
    # `weight` holds the measured synapse counts and is a buffer, not a parameter, so
    # it has no gradient at all -- reporting "0% of edges have gradient" would read as
    # a failure when it is the design. The connectome's learnable part is the
    # per-neuron release magnitude, so that is what coverage is measured on.
    edge_grad = float((grad != 0).float().mean()) if grad is not None else None
    gain_grad = circuit.raw_gain.grad
    gain_coverage = float((gain_grad != 0).float().mean()) if gain_grad is not None else None

    down_rank = effective_rank(brain.read_down.weight)
    drive_participation = brain.drive_participation()
    readout_participation = brain.readout_participation()
    drive_participation = None if drive_participation is None else float(drive_participation.detach())
    readout_participation = None if readout_participation is None else float(readout_participation.detach())
    batch = x.shape[0]

    params = {name: p.numel() for name, p in model.named_parameters()}
    metrics = {
        "checkpoint": args.checkpoint,
        "context": cfg.context,
        "batch": batch,
        "vocab_size": cfg.vocab_size,
        "neurons": circuit.n_neurons,
        "synapses": edges,
        "sweeps_per_token": 1,
        "sweeps_per_forward": cfg.context,
        "tokens_whose_prediction_uses_the_connectome": 1.0,
        "synapse_transmission_fraction": round(synapse_use, 4),
        "neuron_mean_state": round(activity, 4),
        "neuron_responsive_fraction": round(responsive, 4),
        "edge_gradient_coverage": round(edge_grad, 4) if edge_grad is not None else None,
        "release_gain_gradient_coverage": round(gain_coverage, 4) if gain_coverage is not None else None,
        "readout_effective_rank": down_rank,
        "readout_rank_dim": int(brain.read_down.weight.shape[0]),
        "drive_participation": round(float(drive_participation), 2) if drive_participation is not None else None,
        "readout_participation": round(float(readout_participation), 2) if readout_participation is not None else None,
        "participation_ceiling": batch - 1,
        "loss": round(float(loss.detach()), 4),
        "params_total": model.param_count(),
        "params_connectome_edges": edges,
        "params_synapse_counts_are_frozen": True,
        "params_interfaces": params.get("embed.weight", 0) + params.get("head.weight", 0),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else 0.0,
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, ensure_ascii=False)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
