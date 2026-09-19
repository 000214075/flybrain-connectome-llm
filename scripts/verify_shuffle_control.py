"""Re-verify that a shuffled connectome is a valid rewiring control, from the files.

The wiring controls are load-bearing: every "is it the fly's wiring?" claim compares a
model on `data/wholebrain.npz` against one on `data/wholebrain_shuffled.npz`. That
comparison only means something if the control keeps the graph's statistics and moves
nothing else, so this checks the two files against each other rather than trusting the
meta written by `scripts/shuffle_wholebrain.py`:

    same neuron count and edge count
    identical in-degree sequence and out-degree sequence (cell for cell)
    identical sign per neuron, identical weight multiset
    `post` is a permutation of the original `post`
    no self-loops, no duplicate edges

    .venv\\Scripts\\python.exe scripts\\verify_shuffle_control.py \\
        --real data/wholebrain.npz --control data/wholebrain_shuffled.npz
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def load(path: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "pre": data["pre"].astype(np.int64),
            "post": data["post"].astype(np.int64),
            "weight": data["weight"].astype(np.float64),
            "sign": data["sign"].astype(np.float64),
            "neuron_ids": data["neuron_ids"].astype(np.int64),
            "meta": json.loads(str(data["meta"])),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", default="data/wholebrain.npz")
    parser.add_argument("--control", default="data/wholebrain_shuffled.npz")
    args = parser.parse_args()

    real, control = load(args.real), load(args.control)
    n_real = int(real["meta"]["n_neurons"])
    n_ctrl = int(control["meta"]["n_neurons"])
    checks: dict[str, object] = {}

    checks["neurons_equal"] = n_real == n_ctrl
    checks["edges_equal"] = real["pre"].size == control["pre"].size
    checks["neuron_ids_identical"] = bool(np.array_equal(real["neuron_ids"], control["neuron_ids"]))
    checks["sign_identical"] = bool(np.array_equal(real["sign"], control["sign"]))
    checks["in_degree_identical"] = bool(
        np.array_equal(np.bincount(real["post"], minlength=n_real), np.bincount(control["post"], minlength=n_ctrl))
    )
    checks["out_degree_identical"] = bool(
        np.array_equal(np.bincount(real["pre"], minlength=n_real), np.bincount(control["pre"], minlength=n_ctrl))
    )
    checks["pre_untouched"] = bool(np.array_equal(real["pre"], control["pre"]))
    checks["post_is_permutation"] = bool(np.array_equal(np.sort(real["post"]), np.sort(control["post"])))
    checks["weight_sum_identical"] = bool(np.isclose(real["weight"].sum(), control["weight"].sum(), rtol=1e-9))
    checks["weight_multiset_identical"] = bool(
        np.allclose(np.sort(real["weight"]), np.sort(control["weight"]), rtol=1e-6, atol=0.0)
    )
    flat = control["pre"] * n_ctrl + control["post"]
    checks["control_self_loops"] = int((control["pre"] == control["post"]).sum())
    checks["control_duplicate_edges"] = int(flat.size - np.unique(flat).size)
    checks["edges_repointed_fraction"] = round(float((control["post"] != real["post"]).mean()), 4)

    print(json.dumps(checks, indent=2))
    failed = [k for k, v in checks.items() if k.startswith(("neurons", "edges_", "neuron_ids", "sign", "in_degree", "out_degree", "pre_", "post_", "weight_")) and not v]
    failed += [k for k in ("control_self_loops", "control_duplicate_edges") if checks[k]]
    print("VERDICT:", "valid control" if not failed else f"INVALID: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
