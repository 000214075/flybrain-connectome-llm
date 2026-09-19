"""Compare two checkpoints tensor by tensor, bitwise.

Two uses in this project:

1. Verifying that a promotion really moved the artifact (report 7.40): the new
   `checkpoints/flybrain-connectome/best.pt` must be bitwise equal to the
   `E:\\flybrain-connectome\\rank512-long\\best.pt` it was copied from.
2. Answering whether a configuration change perturbs the weights after a single
   training step (the worker-equivalence question in report 7.32/7.35/7.39): run
   the same config with `num_workers` 2 and 0 for one step, then compare the two
   `last.pt` files. Bitwise-equal weights after one step put the divergence later
   than step 1; differing weights put it in the data-delivery path itself.

`mmap=True` keeps this from copying either file (2.4 GB here) into memory: the
tensors are read straight off disk. Equality is exact (`torch.equal`), never a
tolerance -- the question is determinism, not agreement.

Usage:
    .venv\\Scripts\\python.exe scripts\\compare_checkpoint_state.py A.pt B.pt [--section model]
"""

from __future__ import annotations

import argparse
import sys

import torch


def describe(a: object, b: object, key: str) -> str | None:
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return None if type(a) is type(b) and a == b else f"{key}: {type(a).__name__} vs {type(b).__name__}"
    if a.shape != b.shape:
        return f"{key}: shape {tuple(a.shape)} vs {tuple(b.shape)}"
    if torch.equal(a, b):
        return None
    diff = (a.float() - b.float()).abs()
    return f"{key}: max|diff| {float(diff.max()):.3e}, mean|diff| {float(diff.mean()):.3e}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--section", default="model")
    parser.add_argument("--show", type=int, default=8, help="how many differing keys to print")
    args = parser.parse_args()

    a = torch.load(args.first, map_location="cpu", weights_only=False, mmap=True)
    b = torch.load(args.second, map_location="cpu", weights_only=False, mmap=True)
    for field in ("step", "best_val"):
        print(f"{field}: {a.get(field)} vs {b.get(field)}")
    if args.section not in a or args.section not in b:
        print(f"no {args.section!r} in both; keys are {sorted(set(a) & set(b))}")
        return 1

    left, right = a[args.section], b[args.section]
    keys = sorted(set(left) | set(right))
    equal = differing = 0
    shown = 0
    for key in keys:
        if key not in left or key not in right:
            print(f"{key}: present only in one")
            differing += 1
            continue
        line = describe(left[key], right[key], key)
        if line is None:
            equal += 1
        else:
            differing += 1
            if shown < args.show:
                print(line)
                shown += 1
    print(f"{args.section}: tensors compared {len(keys)}, bitwise equal {equal}, differing {differing}")
    return 0 if differing == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
