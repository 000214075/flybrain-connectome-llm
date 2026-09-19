"""Generate the stage-S1 "biological stimulation" configs (see `research/biological_training.md`).

The deliverable model currently runs with `brain_spiking: false` and drives *every*
one of the 164,587 neurons through a learned dense projection. S1 makes three
changes that the literature and this project's own diagnostics point at, and this
script keeps them honest:

  * spiking ON (it is a config flag that the canonical recipe turns off),
  * stimulation restricted to the real sensory neurons (`data/ports_sensory.npz`),
  * read-out restricted to the real output neurons (`data/ports_output.npz`).

All three arms turn spiking on, so the arms differ *only* in the ports. The
equal-count **random** port control is mandatory: experiment A established that
without it a port result cannot be attributed to anatomy rather than to shrinking
the interface to N neurons. The reference point for the spiking change itself is
the existing 600-step `flybrain-connectome-rank256` arm (spiking off, no masks),
which used the same base recipe and budget.

Usage:
    .venv\\Scripts\\python.exe scripts\\make_bio_configs.py
    .venv\\Scripts\\python.exe scripts\\make_bio_configs.py --out-root I:/flybrain-bio
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# name -> (input mask, output mask). "" means the field is absent from the model.
ARMS: dict[str, tuple[str, str]] = {
    "biospike-nomask": ("", ""),
    "biospike-sensory": ("data/ports_sensory.npz", "data/ports_output.npz"),
    "biospike-random": ("data/ports_random_sensory.npz", "data/ports_random_output.npz"),
    # The two arms above change the input mask *and* the output mask at once, so
    # `nomask` versus either of them cannot say which restriction caused the loss.
    # These two isolate each side: input-only and output-only.
    "biospike-inonly": ("data/ports_sensory.npz", ""),
    "biospike-outonly": ("", "data/ports_output.npz"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="configs/train_connectome_scaled.json")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--max-minutes", type=float, default=20.0)
    parser.add_argument(
        "--out-root", default="I:/flybrain-bio",
        help="C: sits near the 15 GB gate, so the arms write to I: by default",
    )
    args = parser.parse_args(argv)

    base = json.loads((ROOT / args.base).read_text(encoding="utf-8"))
    if base.get("seed", args.seed) != args.seed:
        print(f"warning: base seed {base.get('seed')} overridden to {args.seed}")

    # Guard: the arms are only interpretable if the base is the *analog* recipe
    # and every arm is the one that turns spiking on.
    assert base["model"].get("brain_spiking") is False, "base recipe is not the analog one"

    written: list[Path] = []
    for name, (input_mask, output_mask) in ARMS.items():
        config = json.loads(json.dumps(base))
        config["run_name"] = name
        config["out_dir"] = f"{args.out_root.rstrip('/')}/{name}"
        config["seed"] = args.seed
        config["max_steps"] = args.max_steps
        config["max_minutes"] = args.max_minutes
        config["model"]["brain_spiking"] = True
        config["model"]["brain_input_mask"] = input_mask
        config["model"]["brain_output_mask"] = output_mask
        target = ROOT / "configs" / f"train_{name}.json"
        target.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(target)
        print(f"{target.name}: input={input_mask or '(none)'} output={output_mask or '(none)'} "
              f"out_dir={config['out_dir']} steps={config['max_steps']} seed={config['seed']}")

    # The whole design rests on "only the ports differ", so check it rather than
    # trust it: any other difference silently turns this into a different experiment.
    variants = [json.loads(p.read_text(encoding="utf-8")) for p in written]
    mask_fields = ["brain_input_mask", "brain_output_mask"]
    for left, right, (lname, _), (rname, _) in zip(
            variants, variants[1:], list(ARMS.items()), list(ARMS.items())[1:]):
        differing = [k for k in left if k != "model" and left[k] != right[k]]
        assert differing == ["run_name", "out_dir"], differing
        model_diff = sorted(k for k in left["model"] if left["model"][k] != right["model"][k])
        # Any *combination* of port masks is a legitimate arm, but nothing else may
        # differ, or the comparison silently stops being about the ports.
        assert model_diff, "arms must differ in at least one port mask"
        assert set(model_diff) <= set(mask_fields), model_diff
        assert left["model"]["brain_spiking"] is True
        print(f"{lname} vs {rname}: differs in {differing + ['model.' + m for m in model_diff]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
