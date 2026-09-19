"""Compare two training runs' `metrics.jsonl` step by step.

Written for the `num_workers` equivalence check (experiment D). The claim to test is
that dropping the dataloader workers leaves the data order untouched -- and the test
for that is that every step's training loss is *identical*, not that the two curves
look alike. "Looks similar" would accept a run whose sample order drifted slightly,
which is exactly the difference that matters when the recipe change is supposed to be
free, and it is the difference that would silently make a later long run
incomparable with the canonical one.

Throughput is deliberately not part of the verdict: dropping workers is expected to
change speed, and reporting that separately is the point of the change. Only the
compared fields decide the exit code.

Usage:
    .venv\\Scripts\\python.exe scripts\\compare_metrics.py --a <runA/metrics.jsonl> --b <runB/metrics.jsonl>
    .venv\\Scripts\\python.exe scripts\\compare_metrics.py --a A --b B --fields loss lr --tol 1e-12 --json

Exit code 0 when every compared step matches, 1 otherwise (including when the two
files do not cover the same steps).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TRAIN_EVENT = "train"
# Reported, never compared: a recipe change made to save memory or time is allowed to
# move these, and folding them into the verdict would hide the thing being measured.
REPORT_ONLY = ("tokens_per_second", "wall", "vram_gb")


def load_steps(path: Path, fields: tuple[str, ...]) -> tuple[dict[int, dict], dict]:
    """Training records keyed by step, plus the mean of the report-only fields."""
    steps: dict[int, dict] = {}
    totals: dict[str, list[float]] = {name: [] for name in REPORT_ONLY}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") != TRAIN_EVENT:
            continue
        step = record.get("step")
        if step is None:
            continue
        steps[step] = {name: record.get(name) for name in fields}
        for name in REPORT_ONLY:
            if isinstance(record.get(name), (int, float)):
                totals[name].append(float(record[name]))
    means = {name: (sum(values) / len(values) if values else None) for name, values in totals.items()}
    return steps, means


def compare(path_a: Path, path_b: Path, fields: tuple[str, ...], tol: float) -> dict:
    a, a_means = load_steps(path_a, fields)
    b, b_means = load_steps(path_b, fields)
    shared = sorted(set(a) & set(b))
    result: dict = {
        "steps_a": len(a),
        "steps_b": len(b),
        "steps_compared": len(shared),
        "only_a": sorted(set(a) - set(b))[:5],
        "only_b": sorted(set(b) - set(a))[:5],
        "differences": [],
        "within_tolerance": [],
        "speed": {
            name: {"a": a_means.get(name), "b": b_means.get(name)}
            for name in REPORT_ONLY
            if a_means.get(name) is not None or b_means.get(name) is not None
        },
    }
    for step in shared:
        for name in fields:
            left, right = a[step].get(name), b[step].get(name)
            if left == right:
                continue
            entry = {"step": step, "field": name, "a": left, "b": right}
            if (isinstance(left, (int, float)) and isinstance(right, (int, float))
                    and abs(left - right) <= tol):
                result["within_tolerance"].append(entry)
            else:
                result["differences"].append(entry)
    return result


def render(result: dict, tol: float) -> str:
    lines = [
        f"steps: a={result['steps_a']} b={result['steps_b']} compared={result['steps_compared']}",
    ]
    if result["only_a"]:
        lines.append(f"steps only in a: {result['only_a']}")
    if result["only_b"]:
        lines.append(f"steps only in b: {result['only_b']}")
    for name, values in result["speed"].items():
        left, right = values["a"], values["b"]
        if left and right:
            lines.append(f"  {name}: a={left:.1f} b={right:.1f} ({100 * (right / left - 1):+.1f}%)")
        else:
            lines.append(f"  {name}: a={left} b={right}")
    if result["within_tolerance"]:
        lines.append(f"{len(result['within_tolerance'])} step(s) differ by <= {tol} (not counted):")
        for entry in result["within_tolerance"][:5]:
            lines.append(f"  step {entry['step']} {entry['field']}: {entry['a']} vs {entry['b']}")
    if result["differences"]:
        lines.append(f"DIFFER at {len(result['differences'])} place(s); first:")
        for entry in result["differences"][:5]:
            lines.append(f"  step {entry['step']} {entry['field']}: {entry['a']} vs {entry['b']}")
    else:
        lines.append("IDENTICAL on every compared step")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--fields", nargs="+", default=["loss"])
    parser.add_argument("--tol", type=float, default=0.0,
                        help="treat differences this small as equal (default 0: exact)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    path_a, path_b = Path(args.a), Path(args.b)
    for path in (path_a, path_b):
        if not path.is_file():
            print(f"missing file: {path}", file=sys.stderr)
            return 1

    result = compare(path_a, path_b, tuple(args.fields), args.tol)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render(result, args.tol))

    same_steps = result["steps_a"] == result["steps_b"] == result["steps_compared"]
    return 0 if (same_steps and not result["differences"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
