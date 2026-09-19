"""Generate the three `brain_rank` sweep configs from the canonical scaled recipe.

Experiment B (`research/next_experiments.md`) asks whether the read-out dimension
`model.brain_rank` is already saturated: the arms differ in that one model field,
and in nothing else that affects the model. `run_name`, `out_dir` and the step /
minute budget are run identity and budget, not model settings, and the seed is
written out explicitly so "same seed" is visible in the configs rather than implied
by the default.

Usage:
    .venv\\Scripts\\python.exe scripts\\make_rank_configs.py
    .venv\\Scripts\\python.exe scripts\\make_rank_configs.py --base configs/train_connectome_scaled.json --ranks 128 256 512
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="configs/train_connectome_scaled.json")
    parser.add_argument("--ranks", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--max-minutes", type=float, default=20.0)
    parser.add_argument(
        "--out-root", default="checkpoints",
        help="directory for the arm's out_dir; point it at another drive when C: is full",
    )
    args = parser.parse_args(argv)

    base = json.loads((ROOT / args.base).read_text(encoding="utf-8"))
    if base.get("seed", args.seed) != args.seed:
        print(f"warning: base seed {base.get('seed')} overridden to {args.seed}")

    for rank in args.ranks:
        config = json.loads(json.dumps(base))
        config["run_name"] = f"flybrain-connectome-rank{rank}"
        config["out_dir"] = f"{args.out_root.rstrip('/')}/flybrain-connectome-rank{rank}"
        config["seed"] = args.seed
        config["max_steps"] = args.max_steps
        config["max_minutes"] = args.max_minutes
        config["model"]["brain_rank"] = rank
        target = ROOT / "configs" / f"train_connectome_rank{rank}.json"
        target.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{target.name}: brain_rank={rank} out_dir={config['out_dir']} "
              f"steps={config['max_steps']} minutes={config['max_minutes']} seed={config['seed']}")

    # Every other field must be identical, so that a difference between arms can
    # only come from brain_rank.
    variants = [json.loads((ROOT / "configs" / f"train_connectome_rank{r}.json")
                           .read_text(encoding="utf-8")) for r in args.ranks]
    for left, right, left_rank, right_rank in zip(
            variants, variants[1:], args.ranks, args.ranks[1:]):
        # `model` differs too, by design; it is checked field-by-field below.
        differing = [key for key in left if key != "model" and left[key] != right[key]]
        assert differing == ["run_name", "out_dir"], differing
        model_diff = [key for key in left["model"] if left["model"][key] != right["model"][key]]
        assert model_diff == ["brain_rank"], model_diff
        print(f"rank{left_rank} vs rank{right_rank}: differs in {differing + ['model.' + model_diff[0]]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
