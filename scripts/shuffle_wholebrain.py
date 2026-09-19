"""Write a degree-preserving rewiring of the whole connectome.

The control arm for "is it the fly's wiring that does the work?". Same neurons,
same synapse count, same in-degree and out-degree for every cell, same sign per
neuron, same normalised weights -- only the partners are drawn at random.

    python scripts/shuffle_wholebrain.py --in data/wholebrain.npz \
        --out data/wholebrain_shuffled.npz --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.wholebrain import shuffled_edges  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src", default="data/wholebrain.npz")
    parser.add_argument("--out", default="data/wholebrain_shuffled.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--proposals", type=int, default=2_000_000)
    args = parser.parse_args()

    with np.load(args.src, allow_pickle=False) as data:
        pre = data["pre"].astype(np.int64)
        post = data["post"].astype(np.int64)
        weight = data["weight"].astype(np.float32)
        sign = data["sign"].astype(np.float32)
        neuron_ids = data["neuron_ids"].astype(np.int64)
        meta = json.loads(str(data["meta"]))
    n_neurons = int(meta["n_neurons"])
    print(f"{pre.size:,} edges over {n_neurons:,} neurons")

    started = time.time()
    new_post = shuffled_edges(
        pre, post, n_neurons, seed=args.seed, rounds=args.rounds, proposals=args.proposals
    )
    print(f"rewired in {time.time() - started:.1f} s")

    # The checks that make this a control rather than a different graph. `pre` is
    # untouched, so out-degrees and neuron signs follow for free.
    flat = pre * n_neurons + new_post
    moved = int((new_post != post).sum())
    checks = {
        "in_degree_identical": bool(
            np.array_equal(np.bincount(post, minlength=n_neurons), np.bincount(new_post, minlength=n_neurons))
        ),
        "post_is_a_permutation": bool(np.array_equal(np.sort(post), np.sort(new_post))),
        "edges_repointed": moved,
        "edges_repointed_fraction": round(moved / flat.size, 4),
        "self_loops": int((pre == new_post).sum()),
        "duplicate_edges": int(flat.size - np.unique(flat).size),
    }
    print(json.dumps(checks, indent=2))
    failed = [k for k, v in checks.items() if k.startswith(("in_degree", "post_is")) and not v]
    bad = [k for k in ("self_loops", "duplicate_edges") if checks[k]]
    if failed or bad:
        print(f"refusing to write: degree sequence changed ({failed}) or invalid edges ({bad})")
        return 1

    meta = {
        **meta,
        "source": meta.get("source", "malecns") + "+degree-preserving-shuffle",
        "control": "degree-preserving-shuffle",
        "shuffled_from": os.path.basename(args.src),
        "seed": args.seed,
        **checks,
    }
    np.savez_compressed(
        args.out, pre=pre, post=new_post, weight=weight, sign=sign, neuron_ids=neuron_ids,
        meta=json.dumps(meta),
    )
    print(f"\nwrote {args.out} ({os.path.getsize(args.out) / 1024**2:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
