"""Which learning rate and weight decay actually make the connectome-only model learn?

The first two attempts at training this model never beat a uniform guess: the batch
loss climbed from ln(16384) = 9.70 to 10.36, and validation settled at 9.87.

A first version of this script could not see the problem, and the reason is worth
recording: it trained on *one* fixed batch for the whole run. With 135M parameters
and 64 tokens that is memorisation, and the loss fell to 0.30 while the read-out
standard deviation grew 5x -- the model was inflating its output scale, not learning
anything. On fresh data the inflated scale makes the loss *worse* than uniform.

So this version draws a new batch every step from a pool of windows and reports the
loss on data the model has not just seen, plus the read-out scale that drives the
pathology. Short context and small batch keep it affordable: the point is the
direction of the curve, not the quality of the model.

Usage:
    python scripts/tune_connectome_lr.py --lrs 1e-4,5e-5 --wd 0.1 --steps 100
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.connectome_lm import ConnectomeLM  # noqa: E402
from flybrain.data import TokenWindowDataset  # noqa: E402
from flybrain.train import TrainConfig, load_brain  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_connectome_only.json")
    parser.add_argument("--lrs", default="2e-4,1e-4,5e-5")
    parser.add_argument("--wd", default="0.0", help="comma-separated weight decays to try")
    parser.add_argument("--clip", type=float, default=1.0, help="0 disables gradient clipping")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--pool", type=int, default=512, help="windows held out for fresh batches")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = TrainConfig.from_json(args.config)
    cfg.model.context = args.context
    device = torch.device(args.device)

    brain = load_brain(cfg)
    assert brain is not None
    brain = brain.to(device)
    model = ConnectomeLM(cfg.model, brain).to(device)

    # One snapshot of the untrained model, restored before every setting, so each
    # learning rate starts from exactly the same weights.
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}

    dataset = TokenWindowDataset(
        os.path.join(cfg.data_dir, "train.bin"), args.context, token_dtype=cfg.token_dtype
    )
    pool_x = torch.stack([dataset[i][0] for i in range(args.pool)])
    pool_y = torch.stack([dataset[i][1] for i in range(args.pool)])
    generator = torch.Generator().manual_seed(0)

    print(f"uniform-guess loss = {torch.log(torch.tensor(float(cfg.model.vocab_size))):.4f}")
    for wd in (float(v) for v in args.wd.split(",")):
        for lr in (float(v) for v in args.lrs.split(",")):
            model.load_state_dict(initial)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=wd
            )
            model.train()
            print(f"\nlr {lr:g}  wd {wd:g}  clip {args.clip or 'off'}")
            for step in range(1, args.steps + 1):
                pick = torch.randint(0, args.pool, (args.batch,), generator=generator)
                x, y = pool_x[pick].to(device), pool_y[pick].to(device)
                logits, loss = model(x, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                raw = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) if args.clip else 0.0
                optimizer.step()
                if step % 20 == 0 or step == 1:
                    with torch.no_grad():
                        latent = model.brain.read_latent(model.brain.last_state)
                    print(
                        f"  step {step:>4} loss {float(loss.detach()):7.4f} "
                        f"logit_std {float(logits.detach().std()):7.4f} "
                        f"readout_std {float(latent.std()):7.4f} "
                        f"state_mean {float(model.brain.last_state.mean()):6.4f} "
                        f"gnorm {float(raw):6.2f}"
                    )
    model.zero_grad(set_to_none=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
