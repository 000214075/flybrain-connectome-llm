"""How much of the input survives one pass through the connectome, per read mode?

A connectome-only language model has no attention and no skip connections: the only
thing a token can do to the prediction of a later token is change the population
state. So the question that decides whether it can learn at all is how much of the
difference between two inputs is still present in what the read-out sees.

This measures that directly, for the three read modes of `BrainPathway.read_state`,
on the real 164,587-neuron circuit with the model's own token drives:

* how far apart the states of four different token streams are, in neurons;
* how far apart the read-out latents are, as a fraction of their own scale -- this
  is the signal the vocabulary projection can turn into different predictions;
* the loss at initialisation, and the gradient reaching the token path
  (`sensory.weight`, `embed.weight`) relative to the gradient reaching the output
  head. A path with no gradient cannot learn, however the loss moves.

Usage:
    python scripts/diagnose_readout.py --config configs/train_connectome_only.json
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.connectome_lm import build_lm  # noqa: E402
from flybrain.data import TokenWindowDataset  # noqa: E402
from flybrain.train import TrainConfig, load_brain  # noqa: E402

MODES = ("spike", "rate", "current")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_connectome_only.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--positions", type=int, default=8)
    parser.add_argument("--grad-batches", type=int, default=2)
    return parser.parse_args()


def pairwise_mean(x: torch.Tensor) -> float:
    """Mean L1 distance between the rows of x, in the rows' own units."""
    d = torch.cdist(x.float(), x.float(), p=1)
    n = x.shape[0]
    return float(d.sum() / (n * (n - 1))) if n > 1 else 0.0


def main() -> int:
    args = parse_args()
    cfg = TrainConfig.from_json(args.config)
    device = torch.device(args.device)
    torch.manual_seed(cfg.seed)

    brain = load_brain(cfg)
    assert brain is not None, "this diagnostic needs the connectome"
    brain = brain.to(device)
    model = build_lm(cfg.model, None, brain).to(device)
    if not hasattr(model, "brain"):
        raise SystemExit("this diagnostic is about the connectome-only model")
    print(
        f"{model.param_count():,} params, context {cfg.model.context}, "
        f"{brain.circuit.n_neurons:,} neurons, {brain.circuit.pre.numel():,} synapses, "
        f"spiking={brain.spiking}"
    )

    # -- 1. does the input survive the sweep? ------------------------------
    idx = torch.randint(
        0, cfg.model.vocab_size, (args.streams, args.positions), device=device
    )
    model.eval()
    states: list[torch.Tensor] = []
    pres: list[torch.Tensor] = []
    analogs: list[torch.Tensor] = []
    with torch.no_grad():
        brain.reset_stats()
        brain.start_sequence()
        drives = model._drives(idx)
        state = brain.initial_state(args.streams, device, drives.dtype)
        for position in range(args.positions):
            state = brain.advance_drive(state, drives[:, position])
            states.append(state.detach().clone())
            pres.append(brain.circuit.last_pre.detach().clone())
            analogs.append(brain.circuit.last_analog.detach().clone())

    last_state = states[-1]
    last_pre = pres[-1]
    responsive = float(((last_pre - brain.circuit.lif.v_th).abs() < 0.5).float().mean())
    print(
        f"\nafter {args.positions} sweeps: {args.streams} distinct token streams, "
        f"state mean {float(last_state.mean()):.4f}, "
        f"pre-activation std {float(last_pre.std()):.3f} (v_th {float(brain.circuit.lif.v_th):.2f}), "
        f"{100 * responsive:.2f}% of neurons within half a threshold of firing"
    )
    print(
        f"neurons whose state differs from stream 0: "
        f"{float((last_state != last_state[0:1]).float().sum()):.0f} of "
        f"{brain.circuit.n_neurons:,}"
    )

    # The read-out latent is what the vocabulary projection turns into predictions,
    # so its spread across streams -- relative to its own spread across dimensions --
    # is the signal a learned read-out has to work with.
    print(f"\n{'mode':8s} {'latent spread':>14s} {'latent scale':>13s} {'signal/scale':>13s}")
    for mode in MODES:
        brain.analog_mode = mode
        with torch.no_grad():
            latents = brain.read_latent(last_state)
            scale = float(latents.std())
            spread = pairwise_mean(latents)
        print(f"{mode:8s} {spread:14.4f} {scale:13.4f} {spread / max(scale, 1e-9):13.4f}")

    # -- 2. does the token path get a gradient? ----------------------------
    dataset = TokenWindowDataset(
        os.path.join(cfg.data_dir, "val.bin"), cfg.model.context, token_dtype=cfg.token_dtype
    )
    batch = min(4, len(dataset))
    x = torch.stack([dataset[i][0] for i in range(batch)]).to(device)
    y = torch.stack([dataset[i][1] for i in range(batch)]).to(device)
    print(f"\nloss and gradient on {args.grad_batches} real batches of {x.shape[1]} tokens")
    print(f"{'mode':8s} {'loss':>8s} {'sensory grad':>13s} {'embed grad':>12s} {'head grad':>11s} {'token/head':>11s}")
    for mode in MODES:
        brain.analog_mode = mode
        model.zero_grad(set_to_none=True)
        losses = []
        for _ in range(args.grad_batches):
            out = model(x, y, use_cache=False)
            loss = out[1] if isinstance(out, tuple) else out
            (loss / args.grad_batches).backward()
            losses.append(float(loss))
        norms = {
            name: float(p.grad.norm()) if p.grad is not None else 0.0
            for name, p in (
                ("sensory", model.brain.up.weight),
                ("embed", model.embed.weight),
                ("head", model.head.weight),
            )
        }
        token = (norms["sensory"] + norms["embed"]) / 2
        print(
            f"{mode:8s} {sum(losses) / len(losses):8.4f} {norms['sensory']:13.4e} "
            f"{norms['embed']:12.4e} {norms['head']:11.4e} {token / max(norms['head'], 1e-30):11.4f}"
        )
    model.zero_grad(set_to_none=True)
    print(f"\nuniform-guess loss for reference: {torch.log(torch.tensor(float(cfg.model.vocab_size))):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
