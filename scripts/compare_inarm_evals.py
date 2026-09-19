"""Matched-step table of two runs' own `eval` records.

Report 7.36 uses this to compare the rank512 arm against the rank256 canonical
*before* the arm is finished. Both runs evaluate with the same in-arm instrument
(`eval_iters` batches from `val.bin`, `shuffle=False`), so their in-arm numbers are
comparable with each other at the same step -- but they are NOT comparable to the
fixed 24-batch `scripts/eval_checkpoint.py` command (systematic offset +0.17..+0.22,
report 7.25/7.35). The two columns are therefore labelled `in-arm`, and any
projection to the fixed command is printed as an extrapolation, not a result.

Usage:
    .venv\\Scripts\\python.exe scripts\\compare_inarm_evals.py <A/metrics.jsonl> <B/metrics.jsonl>
"""

from __future__ import annotations

import argparse
import json
import sys


def evals(path: str) -> dict[int, float]:
    out: dict[int, float] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("event") == "eval":
                out[int(record["step"])] = float(record["val_loss"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first")
    parser.add_argument("second")
    args = parser.parse_args()

    a = evals(args.first)
    b = evals(args.second)
    common = sorted(set(a) & set(b))
    print(f"step   {args.first}   {args.second}   diff(second-first)")
    for step in common:
        print(f"{step:>6}   {a[step]:>10.4f}   {b[step]:>10.4f}   {b[step] - a[step]:+8.4f}")
    better = sum(1 for step in common if b[step] < a[step])
    if common:
        mean_gap = sum(b[step] - a[step] for step in common) / len(common)
        print(f"second better on {better}/{len(common)} matched steps; mean diff {mean_gap:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
