"""Tests for the anatomical input/output port masks.

These guard the properties that make the experiment interpretable. A mask that is
misaligned with the connectome's neuron order, or a control that does not match the
count of the set it stands in for, still produces numbers -- they are just numbers
that cannot be attributed to the anatomy, which is the whole point of the exercise.

The masks are built by `scripts/build_port_masks.py`; if they have not been built the
tests skip rather than fail, so a fresh clone is not blocked on a data download.

Run with: .venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import os

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")

MASKS = {
    "sensory": "ports_sensory.npz",
    "output": "ports_output.npz",
    "random_sensory": "ports_random_sensory.npz",
    "random_output": "ports_random_output.npz",
}


def load(name: str) -> dict:
    path = os.path.join(DATA, MASKS[name])
    if not os.path.isfile(path):
        pytest.skip(f"{path} not built; run scripts/build_port_masks.py")
    with np.load(path) as handle:
        return {"mask": handle["mask"], "neuron_ids": handle["neuron_ids"]}


def test_every_mask_is_aligned_to_the_same_neuron_order():
    """A mask is only usable if it indexes the connectome the model actually loads."""
    loaded = {name: load(name) for name in MASKS}
    reference = loaded["sensory"]["neuron_ids"]
    for name, item in loaded.items():
        assert item["mask"].dtype == np.bool_, f"{name} mask is not boolean"
        assert item["neuron_ids"].shape == reference.shape
        assert np.array_equal(item["neuron_ids"], reference), (
            f"{name} is aligned to a different neuron order than the other masks"
        )


def test_the_random_control_matches_the_count_of_the_real_set():
    """Otherwise a difference could be "fewer ports" rather than "these ports"."""
    assert load("random_sensory")["mask"].sum() == load("sensory")["mask"].sum()
    assert load("random_output")["mask"].sum() == load("output")["mask"].sum()


def test_the_anatomical_sets_are_not_empty_nor_the_whole_brain():
    """An empty or universal mask would make the experiment a silent no-op."""
    n_neurons = len(load("sensory")["mask"])
    for name in ("sensory", "output"):
        count = int(load(name)["mask"].sum())
        assert 0 < count < n_neurons, f"{name} mask covers {count} of {n_neurons} neurons"


def test_no_neuron_is_both_an_input_and_an_output_port():
    """Sensory in, motor/descending out: the roles are disjoint in the annotation.

    If they overlapped, an "input vs output" comparison would be partly measuring the
    same neurons doing both jobs.
    """
    overlap = load("sensory")["mask"] & load("output")["mask"]
    assert overlap.sum() == 0, f"{int(overlap.sum())} neurons are in both port sets"


def test_the_random_control_is_not_accidentally_a_contiguous_block():
    """A sorted or range-based selection would be a different experiment entirely."""
    mask = load("random_sensory")["mask"]
    assert not np.array_equal(np.flatnonzero(mask), np.arange(int(mask.sum())))
