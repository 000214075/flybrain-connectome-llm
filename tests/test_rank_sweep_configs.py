"""Guard the validity of the `brain_rank` sweep (experiment B).

The experiment only means anything if the three arms differ in `brain_rank` and in
nothing else that affects the model: a stray edit to one arm's budget, seed, kernel
or learning rate would turn a rank comparison into a comparison of something else,
and nothing in the training log would show it. `scripts/make_rank_configs.py`
asserts this when it generates the configs; this checks the configs that are
actually on disk, so a later hand edit cannot quietly invalidate the sweep.

Run with: .venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RANKS = (128, 256, 512, 1024)

# Everything an arm is allowed to change besides `model.brain_rank`.
RUN_IDENTITY = ("run_name", "out_dir")


def load_arm(rank: int) -> dict:
    path = ROOT / "configs" / f"train_connectome_rank{rank}.json"
    if not path.is_file():
        pytest.skip(f"{path.name} not generated; run scripts/make_rank_configs.py")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_arms_differ_only_in_run_identity_and_brain_rank():
    arms = {rank: load_arm(rank) for rank in RANKS}
    reference = arms[RANKS[0]]

    for rank, arm in arms.items():
        assert arm["model"]["brain_rank"] == rank
        assert arm["seed"] == reference["seed"]
        assert arm["max_steps"] == reference["max_steps"]
        assert arm["max_minutes"] == reference["max_minutes"]

        # Everything that is not run identity or the rank must be identical.
        for key in reference:
            if key in RUN_IDENTITY or key == "model":
                continue
            assert arm[key] == reference[key], f"rank{rank}: {key} differs from rank{RANKS[0]}"
        for key in reference["model"]:
            if key == "brain_rank":
                continue
            assert arm["model"][key] == reference["model"][key], (
                f"rank{rank}: model.{key} differs from rank{RANKS[0]}"
            )


def test_the_arms_use_distinct_output_directories():
    dirs = {load_arm(rank)["out_dir"] for rank in RANKS}

    assert len(dirs) == len(RANKS), f"arms would overwrite each other: {dirs}"


def test_the_arms_are_scored_on_one_kernel_and_one_recipe():
    """A different kernel or port mask would confound the rank comparison."""
    arms = [load_arm(rank) for rank in RANKS]

    assert {arm["model"]["brain_impl"] for arm in arms} == {"csr-scaled"}
    assert {arm["lr"] for arm in arms} == {arms[0]["lr"]}
    assert {arm["batch_size"] for arm in arms} == {arms[0]["batch_size"]}
