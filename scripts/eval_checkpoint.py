"""Score a checkpoint on one fixed token file, so two models can be compared.

Validation loss is only comparable between runs that scored the same tokens. This
project rebuilt its fine-tuning corpus twice, and each rebuild replaced the
validation split -- so `val_loss` in one log and `val_loss` in another measure
different data and must not be put in the same table. This script removes the
ambiguity: point it at any checkpoint and one token file.

Usage:
    python scripts/eval_checkpoint.py --checkpoint checkpoints/flybrain-full/best.pt \
        --tokens data/tokenized_domain/val.bin --mask data/tokenized_domain/meta.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.data import TokenWindowDataset  # noqa: E402
from flybrain.generate import load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokens", default="data/tokenized_domain/val.bin")
    parser.add_argument("--mask", default="data/tokenized_domain/meta.json")
    parser.add_argument("--context", type=int, default=0, help="defaults to the model's")
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def checkpoint_step(path: str) -> int | None:
    """The training step a checkpoint was saved at, or `None` if it is not recorded.

    Reported alongside the score because a wall-clock-capped run stops short of
    `max_steps`, so "3.63 at 17,8xx steps" and "3.63 at 18,000 steps" are different
    claims. `mmap=True` keeps this from copying the whole checkpoint (model +
    optimiser state, ~2.4 GB here) into memory; any failure returns `None` rather
    than breaking the measurement.
    """
    try:
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        return None
    step = state.get("step") if isinstance(state, dict) else None
    return int(step) if isinstance(step, int) else None


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    model, model_cfg = load_model(args.checkpoint, device=device)
    context = args.context or model_cfg.context

    data = np.memmap(args.tokens, dtype=np.uint16, mode="r")
    windows = (len(data) - 1) // context
    if windows < args.batch:
        raise SystemExit(f"{args.tokens} is too small for even one batch of {args.batch}")

    # The project's own dataset, so the windows and the next-token shift are exactly
    # the ones training scored. An earlier version of this script fed the model
    # `model(x, x)` -- input as its own target -- and dutifully reported val_loss
    # 10.7 (worse than the 9.7 of a uniform guess) for a model the trainer had just
    # measured at 1.60.
    dataset = TokenWindowDataset(args.tokens, context)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False, drop_last=True)

    losses: list[float] = []
    started = time.perf_counter()
    with torch.no_grad():
        for step, (x, y) in enumerate(loader):
            if step >= args.batches:
                break
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, loss = model(x, y)
            losses.append(float(loss))
    mean = sum(losses) / len(losses)
    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint_step(args.checkpoint),
        "tokens": args.tokens,
        "context": context,
        "batches": args.batches,
        "batch": args.batch,
        "val_loss": round(mean, 4),
        "val_ppl": round(math.exp(min(mean, 20)), 3),
        "seconds": round(time.perf_counter() - started, 1),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
