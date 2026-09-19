"""Hash the first N training batches under different `num_workers` settings.

This is the CPU-only half of the worker-equivalence experiment (report 7.32): it
answers whether `num_workers` changes the *data order* at all, without touching the
GPU. The batch contents are hashed rather than the losses, so a difference here
means a different batch sequence and a difference in the loss alone does not.

Each configuration is built exactly like `flybrain.train.Trainer` does it:
`torch.manual_seed(seed)` first, then `DataLoader(..., shuffle=True, drop_last=True,
persistent_workers=num_workers > 0)`. Pin memory is left off on purpose so the probe
can run alongside a training arm without competing for the accelerator.

Usage:
    .venv\\Scripts\\python.exe scripts\\check_worker_data_order.py [--batches 70]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flybrain.data import TokenWindowDataset  # noqa: E402

DATA = ROOT / "data" / "tokenized_domain" / "train.bin"
CONTEXT = 32
BATCH = 32
SEED = 1337


def batch_hashes(num_workers: int, batches: int) -> list[str]:
    torch.manual_seed(SEED)
    dataset = TokenWindowDataset(str(DATA), CONTEXT, token_dtype="uint16", seed=SEED)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )
    out: list[str] = []
    for i, (x, _y) in enumerate(loader):
        if i >= batches:
            break
        out.append(hashlib.sha256(x.numpy().tobytes()).hexdigest()[:16])
    return out


def first_difference(a: list[str], b: list[str]) -> int | None:
    return next((i for i, (p, q) in enumerate(zip(a, b)) if p != q), None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=70)
    args = parser.parse_args()

    two_a = batch_hashes(2, args.batches)
    zero = batch_hashes(0, args.batches)
    two_b = batch_hashes(2, args.batches)

    print(f"batches compared: {len(two_a)}")
    print(f"nw=2 vs nw=0 first differing batch: {first_difference(two_a, zero)}")
    print(f"nw=2 vs nw=2 first differing batch: {first_difference(two_a, two_b)}")
    print(f"first three hashes (nw=2): {two_a[:3]}")
    print(f"first three hashes (nw=0): {zero[:3]}")
    identical = first_difference(two_a, zero) is None
    print("VERDICT: data order is " + ("identical" if identical else "different"))
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
