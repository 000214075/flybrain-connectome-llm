"""What does one training step of the connectome-only model actually cost?

The pure model sweeps all 25,563,096 synapses once per token, in sequence, so the
step cost is `context` sweeps and the batch only sets how much work each sweep
carries. This times forward and backward over a few shapes so a bounded training
budget can be chosen from measurements rather than guesses.

Usage:
    python scripts/bench_connectome_lm.py --config configs/train_connectome_only.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.connectome_lm import build_lm  # noqa: E402
from flybrain.train import TrainConfig, load_brain  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_connectome_only.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shapes", default="8x32,32x32,64x32,32x64")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--edge-chunk",
        type=int,
        default=0,
        help="override model.brain_edge_chunk; the scatter trades memory for launches",
    )
    parser.add_argument(
        "--impl",
        default="",
        choices=("", "csr", "csr-scaled", "index_add"),
        help="override model.brain_impl; csr-scaled keeps the sparse matrix constant",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = TrainConfig.from_json(args.config)
    if args.edge_chunk:
        cfg.model.brain_edge_chunk = args.edge_chunk
    if args.impl:
        cfg.model.brain_impl = args.impl
    device = torch.device(args.device)
    brain = load_brain(cfg)
    assert brain is not None
    brain = brain.to(device)
    model = build_lm(cfg.model, None, brain).to(device)
    print(f"{model.param_count():,} params on {torch.cuda.get_device_name(0)}")

    print(f"\n{'batch':>6s} {'context':>8s} {'s/step':>9s} {'ms/token':>9s} {'peak GiB':>9s}")
    for shape in args.shapes.split(","):
        batch, context = (int(v) for v in shape.split("x"))
        x = torch.randint(0, cfg.model.vocab_size, (batch, context), device=device)
        y = torch.randint(0, cfg.model.vocab_size, (batch, context), device=device)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(args.steps):
            logits, loss = model(x, y)
            loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / args.steps
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(
            f"{batch:6d} {context:8d} {elapsed:9.3f} {1000 * elapsed / (batch * context):9.2f} {peak:9.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
