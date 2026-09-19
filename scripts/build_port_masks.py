"""Build anatomical input/output "port" masks for the connectome model.

Why this exists
---------------
The model currently drives *every* one of the 164,587 neurons with the token
embedding and reads the whole population back out. Neither choice is anatomical.
FlyToLLM (`ArtyomITA/flytollm`), working on the same MaleCNS v1.0 dataset, ran
pre-registered controls on exactly this question and found the entry point mattered
*more than the wiring itself*: with real wiring, injecting through random ports scored
3.689 while injecting through the real sensory neurons scored 3.959 -- the sensory
neurons are peripheral and sit behind local inhibition, so they are a handicap. It
also read out from the descending/motor/efferent neurons.

That makes "where does the text enter and leave" a cheap, biologically grounded lever:
both are linear layers, so changing them costs nothing in throughput. This script
produces the masks so the model change is a wiring problem rather than an anatomy one.

Three masks per role:

    sensory     the real sensory superclasses (the anatomically correct entry point)
    output      descending + motor + efferent (the anatomically correct exit point)
    random      an equal-count random subset, the control FlyToLLM used

`random` is not decoration. Without it, a difference between `all` and `sensory`
cannot be attributed to the anatomy: the sensory set is also simply smaller.

Writes one .npz per mask into data/, aligned to the neuron order of
data/wholebrain.npz, so the model can consume it without knowing about annotations.

Usage:
    python scripts/build_port_masks.py
    python scripts/build_port_masks.py --annotation <path> --brain data/wholebrain.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Grouped by the role they play, not alphabetically, so the set is auditable against
# the MaleCNS `superclass` vocabulary rather than being a bag of strings.
SENSORY_SUPERCLASSES = (
    "vnc_sensory",
    "cb_sensory",
    "ol_sensory",
    "sensory_ascending",
    "sensory_descending",
    "vnc_sensory_tbc",
    "cb_sensory_tbc",
    "sensory_ascending_tbc",
)
OUTPUT_SUPERCLASSES = (
    "descending_neuron",
    "vnc_motor",
    "cb_motor",
    "vnc_efferent",
    "cb_efferent",
    "efferent_ascending",
    "efferent_descending",
)


def build(annotation_path: str, brain_path: str, out_dir: str, seed: int) -> int:
    brain = np.load(brain_path, allow_pickle=True)
    neuron_ids = pd.Index(brain["neuron_ids"])
    n = len(neuron_ids)

    annotations = pd.read_feather(annotation_path)
    if "bodyId" not in annotations.columns or "superclass" not in annotations.columns:
        raise SystemExit(f"{annotation_path} has no bodyId/superclass columns")
    # Several rows can share a bodyId (different sources); keep the first, which is
    # the one already used elsewhere in this project.
    classes = annotations.drop_duplicates("bodyId").set_index("bodyId")["superclass"]
    superclass = classes.reindex(neuron_ids)

    matched = int(superclass.notna().sum())
    print(f"neurons: {n}   with a superclass: {matched}   unmatched: {n - matched}")

    sensory = superclass.isin(SENSORY_SUPERCLASSES).to_numpy()
    output = superclass.isin(OUTPUT_SUPERCLASSES).to_numpy()
    print(f"  sensory (input)  {int(sensory.sum())}")
    print(f"  output (exit)    {int(output.sum())}")

    # The control has to match the *count* of the set it stands in for, or a
    # difference could just be "fewer ports is better" rather than "these ports".
    rng = np.random.default_rng(seed)
    random_of_sensory = np.zeros(n, dtype=bool)
    random_of_sensory[rng.choice(n, size=int(sensory.sum()), replace=False)] = True
    random_of_output = np.zeros(n, dtype=bool)
    random_of_output[rng.choice(n, size=int(output.sum()), replace=False)] = True

    # A port set that is empty or that covers everything would make the experiment a
    # no-op, and both are easy to produce by a typo in a superclass name above.
    for name, mask in (("sensory", sensory), ("output", output)):
        if mask.sum() == 0:
            raise SystemExit(f"{name} mask is empty -- check the superclass names")
        if mask.all():
            raise SystemExit(f"{name} mask covers every neuron -- check the superclass names")

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, mask in (
        ("ports_sensory", sensory),
        ("ports_output", output),
        ("ports_random_sensory", random_of_sensory),
        ("ports_random_output", random_of_output),
    ):
        path = os.path.join(out_dir, f"{name}.npz")
        np.savez_compressed(
            path,
            mask=mask,
            neuron_ids=brain["neuron_ids"],
            source=np.array([annotation_path]),
        )
        written.append((path, int(mask.sum())))
    print()
    for path, count in written:
        print(f"  wrote {path}  ({count} neurons)")
    return 0


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation",
        default=os.path.join(
            root, "data", "connectome", "malecns",
            "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        ),
    )
    parser.add_argument("--brain", default=os.path.join(root, "data", "wholebrain.npz"))
    parser.add_argument("--out-dir", default=os.path.join(root, "data"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for path in (args.annotation, args.brain):
        if not os.path.isfile(path):
            raise SystemExit(f"missing input: {path}")
    return build(args.annotation, args.brain, args.out_dir, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
