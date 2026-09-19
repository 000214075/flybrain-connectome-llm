"""Measure the spread of a set of runs that were supposed to differ in one thing only.

Every conclusion this project has drawn from a training curve rests on a handful of
runs, so the runs have to be matched: same config, same schedule, same number of
steps, differing only in the field under test. Two uses:

* **a noise floor.** The same config at several seeds, which is how large a
  difference has to be before it means anything. Measured at 150 steps: 0.0395 on a
  fixed set of windows (report 7.7).
* **a sweep.** The same config at several learning rates, where the differences by
  design are named on the command line and everything else must still match.

The script refuses to print anything if the runs disagree on a field nobody asked
them to vary -- an unmatched comparison silently answers a different question, which
is how this project once concluded that swapping the sweep kernel changed what the
model learns (it did not; the two arms had different cosine tables).

`val_loss` read from the training log is comparable across runs that share a seed,
because the validation windows are drawn from the seed. Runs at different seeds
evaluate on different windows, which is why the spread across seeds is larger than
the spread between the models themselves; `eval_checkpoint.py` scores saved
checkpoints on one fixed set of windows when the finer number is needed.

Usage:
    python scripts/seed_variance.py checkpoints/flybrain-connectome-variance/seed1337 \
        checkpoints/flybrain-connectome-variance/seed2024 \
        checkpoints/flybrain-connectome-variance/seed4242 --gap kernel=0.2667 --gap wiring=0.0016
    python scripts/seed_variance.py checkpoints/flybrain-connectome-lr/lr3e-05 \
        checkpoints/flybrain-connectome-lr/lr1e-04 --allow lr
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

# Where a run writes and what it is called never affect the arithmetic, so they are
# always allowed to differ. A seed varying is a legitimate sample of the same
# process. Anything else has to be named with --allow or it is a broken comparison.
ALLOWED_DIFFERENCES = {"seed", "out_dir", "run_name"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run directories, all of the same config")
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="KEY",
        help="a config field the runs were meant to vary (repeatable), e.g. lr",
    )
    parser.add_argument(
        "--gap",
        action="append",
        default=[],
        metavar="NAME=LOSS",
        help="a difference measured elsewhere, to express as a multiple of the spread",
    )
    return parser.parse_args()


def flatten(node: dict, prefix: str = "") -> dict:
    flat: dict = {}
    for key, value in node.items():
        if isinstance(value, dict):
            flat.update(flatten(value, prefix + key + "."))
        else:
            flat[prefix + key] = value
    return flat


def read_config(run_dir: str) -> dict:
    path = os.path.join(run_dir, "train_config.json")
    with open(path, encoding="utf-8") as handle:
        return flatten(json.load(handle))


def read_evals(run_dir: str) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    path = os.path.join(run_dir, "metrics.jsonl")
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") == "eval":
                rows[record["step"]] = record
    return rows


def check_same_config(run_dirs: list[str], allowed: set[str]) -> tuple[list[str], list[str]]:
    """Split the field differences into the ones asked for and the ones that break it."""
    reference = run_dirs[0]
    base = read_config(reference)
    problems: list[str] = []
    by_design: list[str] = []
    for other in run_dirs[1:]:
        config = read_config(other)
        for key in sorted(set(base) | set(config)):
            if base.get(key) == config.get(key):
                continue
            message = f"{key}: {base.get(key)!r} -> {config.get(key)!r}"
            if key in allowed:
                by_design.append(message)
            elif key not in ALLOWED_DIFFERENCES:
                problems.append(f"{message} ({reference} vs {other})")
    return problems, by_design


def column_label(run_dir: str, allowed: set[str]) -> str:
    """Name each column after what was varied: the seed, or the swept field."""
    config = read_config(run_dir)
    if len(allowed) == 1:
        key = next(iter(allowed))
        return f"{key} {config.get(key)}"
    return f"seed {config.get('seed')}"


def main() -> int:
    args = parse_args()
    allowed = set(args.allow)
    try:
        labels = [column_label(run, allowed) for run in args.runs]
    except OSError as exc:
        print(f"cannot read {args.runs[0]}: {exc}")
        return 1

    print(f"{len(args.runs)} runs of one config: " + ", ".join(labels))
    problems, by_design = check_same_config(args.runs, allowed)
    for item in by_design:
        print(f"  differs by design: {item}")
    if problems:
        print("\nNOT THE SAME CONFIG -- the spread below would not mean anything:")
        for item in problems:
            print(f"  {item}")
        return 1
    if not allowed:
        print("config check: only seed / out_dir / run_name differ")

    per_run = [read_evals(run) for run in args.runs]
    shared = sorted(set.intersection(*(set(rows) for rows in per_run)))
    if len(shared) < 2:
        print(f"\nonly {len(shared)} evaluation step(s) reached by every run; need at least 2")
        return 1

    header = "  step" + "".join(f"{label:>14}" for label in labels)
    header += f"{'spread':>10}{'std':>9}"
    print(f"\nvalidation loss, in-training\n{header}")
    spreads: dict[int, tuple[float, float]] = {}
    for step in shared:
        values = [rows[step]["val_loss"] for rows in per_run]
        spread = max(values) - min(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        spreads[step] = (spread, std)
        cells = "".join(f"{value:>14.4f}" for value in values)
        print(f"{step:>6}{cells}{spread:>10.4f}{std:>9.4f}")

    last = shared[-1]
    spread, std = spreads[last]
    base_loss = per_run[0][last]["val_loss"]
    what = "across the varied settings" if allowed else "over these seeds"
    print(
        f"\nspread at step {last}: {spread:.4f} {what} "
        f"({std:.4f} std, {100 * spread / base_loss:.3f}% of the loss)"
    )

    if args.gap:
        print("\ndifferences measured elsewhere, against that spread:")
        for item in args.gap:
            name, _, value = item.partition("=")
            try:
                gap = abs(float(value))
            except ValueError:
                print(f"  {name}: cannot parse {value!r} as a loss difference")
                return 1
            verdict = "above the spread" if gap > spread else "inside the spread"
            print(f"  {name}: {gap:.4f} = {gap / spread:.2f}x the spread -- {verdict}")

    if allowed:
        print(
            "\nnote: with one run per setting this bounds the size of the effect, not\n"
            "      whether it is real -- a second seed per setting would be needed for that."
        )
    else:
        print(
            "\nnote: this is a spread, not a confidence interval -- with this many seeds it\n"
            "      bounds the noise order of magnitude and nothing finer."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
