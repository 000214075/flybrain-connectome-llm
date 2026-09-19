"""Find how MaleCNS labels the mushroom body cell types.

The annotation vocabulary decides whether the projection-neuron -> Kenyon cell ->
MBON pathway can be extracted at all, so this inspects the real label columns
instead of assuming FlyWire-style names.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

DATA = "data/connectome/malecns"


def main() -> int:
    annot = pd.read_feather(os.path.join(DATA, "body-annotations-male-cns-v1.0-minconf-0.5.feather"))
    print(f"annotations: {len(annot):,} rows")

    for column in ("superclass", "class", "subclass"):
        if column not in annot.columns:
            continue
        counts = annot[column].dropna().value_counts()
        print(f"\n--- {column} ({counts.size} distinct) ---")
        print(counts.head(45).to_string())
        # Anything mushroom-body flavoured
        hits = [v for v in counts.index if "kenyon" in str(v).lower() or str(v).lower() == "kc"]
        if hits:
            print("  mushroom body candidates:", hits)

    print("\n--- cell types matching the pathway ---")
    types = annot["type"].dropna().astype(str)
    for needle in ("kc", "kenyon", "pn", "projection", "mbon", "apl", "dan", "orn"):
        matched = sorted({t for t in types.unique() if needle in t.lower()})
        print(f"  type contains {needle!r}: {len(matched)} -> {matched[:14]}")

    print("\n--- type values whose class/superclass mentions kenyon ---")
    mask = (
        annot["class"].astype(str).str.lower().str.contains("kenyon", na=False)
        | annot["superclass"].astype(str).str.lower().str.contains("kenyon", na=False)
        | annot["type"].astype(str).str.lower().str.contains("kenyon", na=False)
        | annot["subclass"].astype(str).str.lower().str.contains("kenyon", na=False)
    )
    print(f"  rows: {int(mask.sum()):,}")
    if mask.any():
        print(annot.loc[mask, ["bodyId", "type", "class", "superclass", "subclass", "instance"]].head(10).to_string())

    print("\n--- connectivity type pairs involving KC/APL/MBON ---")
    edges = pd.read_feather(
        os.path.join(DATA, "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"),
        columns=["body_pre", "body_post", "weight", "type_pre", "type_post"],
    )
    print(f"edges: {len(edges):,}")
    pre_types = edges["type_pre"].dropna().astype(str)
    kc_like = sorted({t for t in pre_types.unique() if t.lower().startswith("kc")})
    print(f"  type_pre starting with 'kc': {len(kc_like)} -> {kc_like[:20]}")
    apl_rows = edges[edges["type_post"].astype(str).str.fullmatch("APL", case=False, na=False)]
    print(f"  edges into APL: {len(apl_rows):,}")
    print(apl_rows["type_pre"].value_counts().head(10).to_string())
    mbon_rows = edges[edges["type_post"].astype(str).str.contains("MBON", case=False, na=False)]
    print(f"  edges into MBON: {len(mbon_rows):,}; distinct pre types:")
    print(mbon_rows["type_pre"].value_counts().head(12).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
