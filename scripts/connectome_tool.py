"""Inspect the downloaded connectome tables and export the mushroom body pathway.

    python scripts/connectome_tool.py inspect --data data/connectome/malecns
    python scripts/connectome_tool.py build --data data/connectome/malecns \
        --out data/pathway_malecns.npz

`inspect` prints the real column names so the loader can be checked against the
release rather than against assumptions about it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.connectome import (  # noqa: E402
    ConnectomeSchemaError,
    build_pathway,
    build_synthetic_pathway,
    load_flywire,
    load_malecns,
)


def inspect(data_dir: str, limit: int = 12) -> None:
    import pandas as pd

    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if not os.path.isfile(path):
            continue
        size_mb = os.path.getsize(path) / 1024**2
        print(f"\n=== {name}  ({size_mb:.1f} MiB) ===")
        try:
            if name.endswith(".feather"):
                df = pd.read_feather(path)
            elif name.endswith(".parquet"):
                df = pd.read_parquet(path)
            elif name.endswith((".tsv", ".txt")):
                df = pd.read_csv(path, sep="\t", nrows=2000, low_memory=False)
                print("(showing the first 2000 rows only)")
            else:
                print("skipped")
                continue
        except Exception as exc:
            print(f"could not read: {type(exc).__name__}: {exc}")
            continue
        print(f"rows={len(df):,} columns={list(df.columns)}")
        print(df.head(limit).to_string(max_colwidth=28))


def build(source: str, data_dir: str, out: str, *, max_pn, max_kc, max_mbon, min_weight: float) -> dict:
    loader = load_malecns if source == "malecns" else load_flywire
    connectome = loader(data_dir)
    summary = connectome.describe()
    print(json.dumps(summary, indent=2))

    pathway = build_pathway(
        connectome,
        min_weight=min_weight,
        max_pn=max_pn,
        max_kc=max_kc,
        max_mbon=max_mbon,
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pathway.save(out)
    print(f"\nwrote {out}")
    print(json.dumps(pathway.meta, indent=2))
    print(
        f"shapes: PN->KC {pathway.pn_to_kc.shape} nnz={np.count_nonzero(pathway.pn_to_kc)}, "
        f"KC->MBON {pathway.kc_to_mbon.shape} nnz={np.count_nonzero(pathway.kc_to_mbon)}"
    )
    return {"summary": summary, "meta": pathway.meta}


def synthetic(out: str, n_pn: int, n_kc: int, n_mbon: int, seed: int) -> None:
    pathway = build_synthetic_pathway(n_pn, n_kc, n_mbon, seed=seed)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pathway.save(out)
    print(f"wrote synthetic pathway to {out}")
    print(json.dumps(pathway.meta, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="print table schemas")
    p_inspect.add_argument("--data", required=True)

    p_build = sub.add_parser("build", help="export a mushroom body pathway")
    p_build.add_argument("--source", choices=("malecns", "flywire"), default="malecns")
    p_build.add_argument("--data", required=True)
    p_build.add_argument("--out", required=True)
    p_build.add_argument("--max-pn", type=int, default=None)
    p_build.add_argument("--max-kc", type=int, default=None)
    p_build.add_argument("--max-mbon", type=int, default=None)
    p_build.add_argument("--min-weight", type=float, default=1.0)

    p_syn = sub.add_parser("synthetic", help="export a random control pathway")
    p_syn.add_argument("--out", required=True)
    p_syn.add_argument("--n-pn", type=int, default=240)
    p_syn.add_argument("--n-kc", type=int, default=2400)
    p_syn.add_argument("--n-mbon", type=int, default=34)
    p_syn.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if args.command == "inspect":
        inspect(args.data)
        return 0
    if args.command == "synthetic":
        synthetic(args.out, args.n_pn, args.n_kc, args.n_mbon, args.seed)
        return 0
    try:
        build(
            args.source,
            args.data,
            args.out,
            max_pn=args.max_pn,
            max_kc=args.max_kc,
            max_mbon=args.max_mbon,
            min_weight=args.min_weight,
        )
    except ConnectomeSchemaError as exc:
        print(f"schema error: {exc}", file=sys.stderr)
        print("run `inspect` and adjust the candidate column names in src/flybrain/connectome.py", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
