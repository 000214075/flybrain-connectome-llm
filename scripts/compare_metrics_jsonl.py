"""Compare two training `metrics.jsonl` files field by field.

Report 7.35 recomputes the worker-equivalence result (7.32) from the files rather
than from any task/watchdog status field, so the comparison itself is a script
anyone can re-run:

    .venv\\Scripts\\python.exe scripts\\compare_metrics_jsonl.py A\\metrics.jsonl B\\metrics.jsonl [event]

`event` defaults to `train`; pass `eval` for the evaluation records. Fields that
differ are printed per step; `wall` and `event` are skipped because the first is a
timestamp and the second is the filter itself. `tokens_per_second` differs in
almost every step by construction (it is a wall-clock rate, not a model result),
so a run whose only differences are timing fields is bit-identical in every
quantity the training actually computes.
"""

from __future__ import annotations

import argparse
import json
import sys


def load(path: str, event: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("event") == event:
                out[int(record["step"])] = record
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("event", nargs="?", default="train")
    args = parser.parse_args()

    a = load(args.first, args.event)
    b = load(args.second, args.event)
    common = sorted(set(a) & set(b))
    print(f"{args.event} steps: {args.first} n={len(a)}, {args.second} n={len(b)}, common={len(common)}")
    differing = 0
    for step in common:
        diff = {
            key: (a[step].get(key), b[step].get(key))
            for key in sorted(set(a[step]) | set(b[step]))
            if key not in ("wall", "event") and a[step].get(key) != b[step].get(key)
        }
        print(step, diff if diff else "-")
        if any(key not in ("tokens_per_second", "vram_gb") for key in diff):
            differing += 1
    print(f"steps differing in a model quantity: {differing}/{len(common)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
