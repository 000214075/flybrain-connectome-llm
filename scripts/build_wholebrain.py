"""Export the whole connectome as a GPU-executable circuit.

Produces a single npz holding every neuron and every synapse of the MaleCNS
dataset, plus a sign per neuron derived from the predicted neurotransmitter.
The sign matters: with it, GABAergic and glutamatergic cells subtract from their
targets, so inhibition in the model is real rather than an assumption.

Edge weights are normalised per post-synaptic neuron, so each neuron receives a
weighted average of its inputs scaled by the learned gain. Without that, a neuron
with 2,000 incoming synapses would saturate instantly while one with 3 would
never fire.

    python scripts/build_wholebrain.py --data data/connectome/malecns \
        --out data/wholebrain.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.connectome import _pick_column, _read_table  # noqa: E402

# Transmitter -> sign in the fast synaptic sum. In Drosophila acetylcholine is the
# main excitatory transmitter; GABA, glutamate and histamine are inhibitory. The
# amines (dopamine, serotonin, octopamine, tyramine) are neuromodulators rather
# than fast transmitters, so they contribute nothing to the instantaneous sum.
TRANSMITTER_SIGNS: dict[str, float] = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
    "dopamine": 0.0,
    "serotonin": 0.0,
    "octopamine": 0.0,
    "tyramine": 0.0,
    "unclear": 1.0,
}
DEFAULT_SIGN = 1.0  # unknown transmitter: most fly neurons are cholinergic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/connectome/malecns")
    parser.add_argument("--out", default="data/wholebrain.npz")
    parser.add_argument("--min-weight", type=float, default=1.0)
    args = parser.parse_args()

    weights_path = os.path.join(
        args.data, "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"
    )
    nt_path = os.path.join(args.data, "body-neurotransmitters-male-cns-v1.0.feather")

    print("reading edges...")
    edges = _read_table(weights_path)
    pre_col = _pick_column(edges.columns, ("body_pre", "pre"), "pre")
    post_col = _pick_column(edges.columns, ("body_post", "post"), "post")
    weight_col = _pick_column(edges.columns, ("weight", "count"), "weight")
    pre_ids = edges[pre_col].to_numpy(dtype=np.int64)
    post_ids = edges[post_col].to_numpy(dtype=np.int64)
    weight = edges[weight_col].to_numpy(dtype=np.float32)
    del edges
    print(f"  {pre_ids.size:,} typed edges")

    keep = (weight >= args.min_weight) & (pre_ids != post_ids)
    pre_ids, post_ids, weight = pre_ids[keep], post_ids[keep], weight[keep]
    print(f"  {pre_ids.size:,} edges after filtering self-loops and weak synapses")

    ids = np.unique(np.concatenate([pre_ids, post_ids]))
    n_neurons = ids.size
    index = {int(v): i for i, v in enumerate(ids)}
    pre = np.fromiter((index[int(v)] for v in pre_ids), dtype=np.int64, count=pre_ids.size)
    post = np.fromiter((index[int(v)] for v in post_ids), dtype=np.int64, count=post_ids.size)
    del pre_ids, post_ids

    # Collapse the per-cell-type rows onto one weight per neuron pair.
    print("aggregating parallel synapses...")
    flat = pre.astype(np.int64) * n_neurons + post.astype(np.int64)
    order = np.argsort(flat, kind="stable")
    flat_sorted = flat[order]
    unique_flat, first = np.unique(flat_sorted, return_index=True)
    sums = np.add.reduceat(weight[order], first).astype(np.float32)
    pre = (unique_flat // n_neurons).astype(np.int32)
    post = (unique_flat % n_neurons).astype(np.int32)
    weight = sums
    del flat, flat_sorted, order, sums
    nnz = weight.size
    print(f"  {nnz:,} unique neuron-to-neuron connections")

    print("reading neurotransmitters...")
    nt = _read_table(nt_path)
    nt_id_col = _pick_column(nt.columns, ("body", "bodyId"), "transmitter body id")
    nt_col = "consensus_nt" if "consensus_nt" in nt.columns else "predicted_nt"
    nt_ids = nt[nt_id_col].to_numpy(dtype=np.int64)
    nt_names = nt[nt_col].astype(str).str.strip().str.lower().to_numpy()
    nt_map = dict(zip(nt_ids.tolist(), nt_names.tolist()))
    del nt

    sign = np.full(n_neurons, DEFAULT_SIGN, dtype=np.float32)
    counted: dict[str, int] = {}
    for slot, body_id in enumerate(ids):
        name = nt_map.get(int(body_id), "")
        value = TRANSMITTER_SIGNS.get(name, DEFAULT_SIGN)
        sign[slot] = value
        counted[name or "(unknown)"] = counted.get(name or "(unknown)", 0) + 1
    print("  transmitter counts:", json.dumps(dict(sorted(counted.items(), key=lambda kv: -kv[1]))))

    # Normalise per post-synaptic neuron: each target sees a weighted average of
    # its inputs, so fan-in does not set the firing rate.
    print("normalising by post-synaptic in-degree...")
    incoming = np.zeros(n_neurons, dtype=np.float32)
    np.add.at(incoming, post, np.abs(weight))
    scale = np.where(incoming > 0, incoming, 1.0)
    weight = (weight / scale[post]).astype(np.float32)

    meta = {
        "source": "malecns-v1.0",
        "n_neurons": int(n_neurons),
        "nnz": int(nnz),
        "min_weight": args.min_weight,
        "excitatory": int((sign > 0).sum()),
        "inhibitory": int((sign < 0).sum()),
        "modulatory": int((sign == 0).sum()),
        "mean_in_degree": round(float(np.bincount(post, minlength=n_neurons).mean()), 3),
        "mean_out_degree": round(float(np.bincount(pre, minlength=n_neurons).mean()), 3),
        "max_in_degree": int(np.bincount(post, minlength=n_neurons).max()),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        pre=pre,
        post=post,
        weight=weight,
        sign=sign,
        neuron_ids=ids,
        meta=json.dumps(meta),
    )
    size_mb = os.path.getsize(args.out) / 1024**2
    print(f"\nwrote {args.out} ({size_mb:.1f} MiB)")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
