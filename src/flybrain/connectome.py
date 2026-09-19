"""Loading real Drosophila connectome data and turning it into sparse matrices.

Two datasets are supported:

* Janelia MaleCNS v1.0 ("male-cns:v1.0"): whole male central nervous system,
  166,700 neurons / ~6.2M connection pairs / 25.5M typed edges, plus cell-type
  annotations and predicted neurotransmitters, from a public Google Cloud bucket.
* FlyWire / FAFB v783: 139,255 neurons; the pre-baked connectivity parquet plus
  the FlyWire annotation table.

The connectome is used as *wiring*, not as weights: a sparse adjacency matrix
fixes the structure of a layer while a small number of parameters (projections in
and out, plus inhibitory gains) are learned. That is the approach validated in
Lappalainen et al., Nature 2024 (doi:10.1038/s41586-024-07939-3).

Cell classes are matched by explicit rules over every label field a neuron has,
because a single field is not enough: MaleCNS puts the mushroom body classes in
`class` ("Kenyon_Cell", "ALPN", "MBON") but the Kenyon cell subtypes only in
`type` ("KCg-m"), and a naive substring search for "pn" would also swallow the
SEZ projection neurons, which do not feed the mushroom body.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

# Label-field names worth collecting, in rough order of specificity.
LABEL_FIELDS = (
    "cell_type",
    "cellType",
    "class",
    "type",
    "subclass",
    "supertype",
    "superclass",
    "super_class",
    "primary_type",
    "hemibrainType",
    "flywireType",
    "instance",
)


def _normalise(value: object) -> str:
    text = str(value).strip().lower()
    return "" if text in ("nan", "none", "na", "") else text


def _is_kc(labels: Sequence[str]) -> bool:
    return any(lb == "kenyon_cell" or lb.startswith("kc") for lb in labels)


def _is_pn(labels: Sequence[str]) -> bool:
    """Antennal lobe projection neurons, the cells that feed Kenyon cells."""
    return any(
        lb == "alpn" or (lb.endswith("pn") and not lb.startswith("sez")) for lb in labels
    )


def _is_mbon(labels: Sequence[str]) -> bool:
    return any(lb.startswith("mbon") for lb in labels)


def _is_apl(labels: Sequence[str]) -> bool:
    return any(lb in ("apl", "anterior_paired_lateral") for lb in labels)


CELL_CLASS_RULES: dict[str, Callable[[Sequence[str]], bool]] = {
    "kc": _is_kc,
    "pn": _is_pn,
    "mbon": _is_mbon,
    "apl": _is_apl,
}


class ConnectomeSchemaError(RuntimeError):
    """Raised when a table's columns cannot be mapped to the expected fields."""


def _pick_column(columns: Sequence[str], candidates: Sequence[str], what: str) -> str:
    """Find the column matching a candidate name, preferring exact matches."""
    lowered = {_normalise(c): c for c in columns}
    for cand in candidates:
        if _normalise(cand) in lowered:
            return lowered[_normalise(cand)]
    for cand in candidates:
        for low, original in lowered.items():
            if _normalise(cand) and _normalise(cand) in low:
                return original
    raise ConnectomeSchemaError(
        f"could not find a column for {what}; looked for {list(candidates)} among {list(columns)}"
    )


@dataclass
class Connectome:
    """A directed, weighted neuron graph plus every label each neuron carries."""

    ids: np.ndarray  # (n_neurons,) int64 body/root ids, sorted
    pre: np.ndarray  # (n_edges,) int64 index into `ids`
    post: np.ndarray  # (n_edges,) int64 index into `ids`
    weight: np.ndarray  # (n_edges,) float32 synapse counts
    labels: dict[int, tuple[str, ...]] = field(default_factory=dict)
    neurotransmitter: dict[int, str] = field(default_factory=dict)
    name: str = "connectome"

    @property
    def n_neurons(self) -> int:
        return int(self.ids.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.pre.shape[0])

    def id_to_index(self) -> dict[int, int]:
        return {int(v): i for i, v in enumerate(self.ids)}

    def by_class(self, klass: str) -> np.ndarray:
        """Ids of neurons matching a mushroom-body cell class."""
        rule = CELL_CLASS_RULES[klass]
        keep = [int(bid) for bid in self.ids if rule(self.labels.get(int(bid), ()))]
        return np.asarray(keep, dtype=np.int64)

    def primary_label(self, body_id: int) -> str:
        labels = self.labels.get(body_id, ())
        return labels[0] if labels else ""

    def adjacency(
        self,
        pre_ids: np.ndarray,
        post_ids: np.ndarray,
        *,
        min_weight: float = 1.0,
        normalise: str | None = "pre",
    ) -> np.ndarray:
        """Dense adjacency restricted to `pre_ids` -> `post_ids`.

        Rows are pre-synaptic neurons, columns post-synaptic. Duplicate pre/post
        pairs are summed, which happens whenever the source table is already
        split by cell-type pair or neuropil.
        """
        pre_flat, post_flat, w = self.pre, self.post, self.weight

        def positions_of(target: np.ndarray) -> np.ndarray:
            """For every neuron in the full connectome, its index in `target` or -1."""
            if target.size == 0:
                return np.full(self.n_neurons, -1, dtype=np.int64)
            ordered = np.sort(np.asarray(target))
            slot = np.clip(np.searchsorted(ordered, self.ids), 0, ordered.size - 1)
            return np.where(ordered[slot] == self.ids, slot, -1)

        out = np.zeros((len(pre_ids), len(post_ids)), dtype=np.float32)
        mapped_pre = positions_of(pre_ids)[pre_flat]
        mapped_post = positions_of(post_ids)[post_flat]
        keep = (mapped_pre >= 0) & (mapped_post >= 0) & (w >= min_weight)
        if np.any(keep):
            np.add.at(out, (mapped_pre[keep], mapped_post[keep]), w[keep])

        if normalise == "pre":  # each pre-synaptic neuron contributes a fixed budget
            denom = out.sum(axis=1, keepdims=True)
            np.divide(out, np.where(denom > 0, denom, 1.0), out=out)
        elif normalise == "post":
            denom = out.sum(axis=0, keepdims=True)
            np.divide(out, np.where(denom > 0, denom, 1.0), out=out)
        elif normalise is not None:
            raise ValueError(f"unknown normalise mode: {normalise}")
        return out

    def describe(self) -> dict:
        return {
            "name": self.name,
            "neurons": self.n_neurons,
            "edges": self.n_edges,
            "class_counts": {k: int(self.by_class(k).size) for k in CELL_CLASS_RULES},
        }


def _read_table(path: str):
    """Read a .feather / .parquet / .csv table into a pandas DataFrame."""
    import pandas as pd

    ext = os.path.splitext(path)[1].lower()
    if ext == ".feather":
        return pd.read_feather(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t", low_memory=False)
    if ext == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"unsupported table format: {path}")


def _labels_from_frame(df, id_col: str) -> dict[int, tuple[str, ...]]:
    """Collect every non-empty label field for each neuron id."""
    columns = [c for c in LABEL_FIELDS if c in df.columns]
    if not columns:
        raise ConnectomeSchemaError(
            f"no recognised label column in {list(df.columns)}; expected one of {list(LABEL_FIELDS)}"
        )
    ids = df[id_col].to_numpy(dtype=np.int64)
    values = {c: df[c].astype(str).to_numpy(dtype=object) for c in columns}
    out: dict[int, tuple[str, ...]] = {}
    for row, body_id in enumerate(ids):
        collected: list[str] = []
        for column in columns:
            text = _normalise(values[column][row])
            if text and text not in collected:
                collected.append(text)
        out[int(body_id)] = tuple(collected)
    return out


def _load_neurotransmitters(path: str) -> dict[int, str]:
    """Predicted transmitter per neuron, from the arg-max of the probability columns."""
    df = _read_table(path)
    id_col = _pick_column(df.columns, ("body", "bodyId", "body_id", "id", "root_id"), "transmitter body id")
    for candidate in ("consensus_nt", "predicted_nt"):
        if candidate in df.columns:
            ids = df[id_col].to_numpy(dtype=np.int64)
            return {
                int(i): _normalise(v)
                for i, v in zip(ids, df[candidate].astype(str).to_numpy(dtype=object))
                if _normalise(v)
            }
    numeric = [c for c in df.columns if c != id_col and np.issubdtype(df[c].dtype, np.number)]
    if not numeric:
        return {}
    ids = df[id_col].to_numpy(dtype=np.int64)
    matrix = df[numeric].to_numpy(dtype=np.float32)
    return {
        int(i): _normalise(numeric[j]) for i, j in zip(ids, np.argmax(matrix, axis=1))
    }


def _first_existing(data_dir: str, names: Sequence[str]) -> str:
    for name in names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"none of {list(names)} found in {data_dir}")


def _edges_from_frame(df) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalise an edge table into (ids, pre, post, weight) with index-mapped endpoints."""
    pre_col = _pick_column(df.columns, ("body_pre", "pre", "pre_root_id", "bodyId_pre", "source"), "pre-synaptic id")
    post_col = _pick_column(df.columns, ("body_post", "post", "post_root_id", "bodyId_post", "target"), "post-synaptic id")
    weight_col = _pick_column(df.columns, ("weight", "count", "syn_count", "synapses"), "synapse count")

    pre_ids = df[pre_col].to_numpy(dtype=np.int64)
    post_ids = df[post_col].to_numpy(dtype=np.int64)
    weight = df[weight_col].to_numpy(dtype=np.float32)
    ids = np.unique(np.concatenate([pre_ids, post_ids]))
    index = {int(v): i for i, v in enumerate(ids)}
    pre = np.fromiter((index[int(v)] for v in pre_ids), dtype=np.int64, count=pre_ids.size)
    post = np.fromiter((index[int(v)] for v in post_ids), dtype=np.int64, count=post_ids.size)
    return ids, pre, post, weight


def load_malecns(data_dir: str) -> Connectome:
    """Load the Janelia MaleCNS v1.0 flat connectome (see configs/assets.json)."""
    weights_path = _first_existing(
        data_dir,
        (
            "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
            "connectome-weights-male-cns-v1.0.feather",
        ),
    )
    annot_path = _first_existing(
        data_dir,
        (
            "body-annotations-male-cns-v1.0-minconf-0.5.feather",
            "body-annotations-male-cns-v1.0.feather",
        ),
    )
    nt_path = _first_existing(data_dir, ("body-neurotransmitters-male-cns-v1.0.feather",))

    edges = _read_table(weights_path)
    ids, pre, post, weight = _edges_from_frame(edges)

    annotations = _read_table(annot_path)
    id_col = _pick_column(annotations.columns, ("bodyId", "body_id", "id", "root_id"), "annotation body id")
    labels = _labels_from_frame(annotations, id_col)

    transmitter = _load_neurotransmitters(nt_path)
    return Connectome(ids, pre, post, weight, labels, transmitter, name="malecns-v1.0")


def load_flywire(data_dir: str) -> Connectome:
    """Load FlyWire/FAFB v783 connectivity plus the FlyWire neuron annotations."""
    conn_path = _first_existing(
        data_dir, ("2025_Connectivity_783.parquet", "2023_Connectivity_630.parquet")
    )
    edges = _read_table(conn_path)
    ids, pre, post, weight = _edges_from_frame(edges)

    labels: dict[int, tuple[str, ...]] = {}
    annot = os.path.join(data_dir, "Supplemental_file1_neuron_annotations.tsv")
    if os.path.exists(annot):
        df = _read_table(annot)
        id_col = _pick_column(df.columns, ("root_id", "root_783", "bodyId", "id"), "annotation root id")
        labels = _labels_from_frame(df, id_col)
    return Connectome(ids, pre, post, weight, labels, {}, name="flywire-783")


# --------------------------------------------------------------------------
# Pathway extraction
# --------------------------------------------------------------------------


@dataclass
class Pathway:
    """The sparse matrices of a mushroom-body-style expansion circuit."""

    pn_to_kc: np.ndarray  # (n_pn, n_kc)
    kc_to_mbon: np.ndarray  # (n_kc, n_mbon)
    apl_to_kc: np.ndarray  # (n_kc,) APL inhibition weight onto each Kenyon cell
    n_pn: int
    n_kc: int
    n_mbon: int
    meta: dict

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path,
            pn_to_kc=self.pn_to_kc,
            kc_to_mbon=self.kc_to_mbon,
            apl_to_kc=self.apl_to_kc,
            meta=json.dumps(self.meta),
        )

    @classmethod
    def load(cls, path: str) -> "Pathway":
        with np.load(path, allow_pickle=False) as data:
            pn_to_kc = data["pn_to_kc"]
            kc_to_mbon = data["kc_to_mbon"]
            apl_to_kc = data["apl_to_kc"]
            meta = json.loads(str(data["meta"]))
        return cls(
            pn_to_kc=pn_to_kc,
            kc_to_mbon=kc_to_mbon,
            apl_to_kc=apl_to_kc,
            n_pn=int(pn_to_kc.shape[0]),
            n_kc=int(pn_to_kc.shape[1]),
            n_mbon=int(kc_to_mbon.shape[1]),
            meta=meta,
        )


def build_pathway(
    connectome: Connectome,
    *,
    min_weight: float = 1.0,
    max_pn: int | None = None,
    max_kc: int | None = None,
    max_mbon: int | None = None,
    seed: int = 0,
) -> Pathway:
    """Extract the projection-neuron -> Kenyon-cell -> MBON expansion circuit.

    The matrices keep the real synapses: an entry is the row-normalised number of
    synapses the connectome reports between those two cells. `max_*` caps keep
    only the most connected neurons, which is what bounds the layer size.
    """
    pn_ids = connectome.by_class("pn")
    kc_ids = connectome.by_class("kc")
    mbon_ids = connectome.by_class("mbon")
    apl_ids = connectome.by_class("apl")
    if kc_ids.size == 0 or pn_ids.size == 0:
        raise ConnectomeSchemaError(
            "no projection neurons or Kenyon cells found; "
            f"counts were {connectome.describe()['class_counts']}"
        )

    pn_ids = _cap(pn_ids, max_pn, connectome)
    kc_ids = _cap(kc_ids, max_kc, connectome)
    mbon_ids = _cap(mbon_ids, max_mbon, connectome)

    pn_to_kc = connectome.adjacency(pn_ids, kc_ids, min_weight=min_weight, normalise="pre")
    kc_to_mbon = connectome.adjacency(kc_ids, mbon_ids, min_weight=min_weight, normalise="pre")

    # APL is a single bilateral pair providing global feedback inhibition onto
    # Kenyon cells; its per-KC strength is the real synapse count, normalised.
    if apl_ids.size:
        apl_strength = connectome.adjacency(
            apl_ids, kc_ids, min_weight=min_weight, normalise=None
        ).sum(axis=0)
    else:
        apl_strength = np.zeros(kc_ids.size, dtype=np.float32)
    peak = float(apl_strength.max()) if apl_strength.size else 0.0
    apl_to_kc = (apl_strength / peak).astype(np.float32) if peak > 0 else apl_strength.astype(np.float32)

    meta = {
        "source": connectome.name,
        "min_weight": min_weight,
        "n_pn_available": int(connectome.by_class("pn").size),
        "n_kc_available": int(connectome.by_class("kc").size),
        "n_mbon_available": int(connectome.by_class("mbon").size),
        "n_apl_available": int(apl_ids.size),
        "pn_to_kc_nnz": int(np.count_nonzero(pn_to_kc)),
        "kc_to_mbon_nnz": int(np.count_nonzero(kc_to_mbon)),
        "kc_indegree_mean": round(float((pn_to_kc > 0).sum(axis=0).mean()), 3),
        "kc_indegree_max": int((pn_to_kc > 0).sum(axis=0).max()),
        "kc_outdegree_mean": round(float((kc_to_mbon > 0).sum(axis=1).mean()), 3),
        "kc_connected_fraction": round(float((pn_to_kc > 0).any(axis=0).mean()), 4),
    }
    return Pathway(pn_to_kc, kc_to_mbon, apl_to_kc, len(pn_ids), len(kc_ids), len(mbon_ids), meta)


def _cap(ids: np.ndarray, cap: int | None, connectome: Connectome) -> np.ndarray:
    """Keep at most `cap` neurons, preferring those with the most synapses."""
    if cap is None or ids.size <= cap:
        return ids
    index = connectome.id_to_index()
    positions = np.fromiter((index[int(v)] for v in ids), dtype=np.int64, count=ids.size)
    in_deg = np.zeros(connectome.n_neurons, dtype=np.int64)
    np.add.at(in_deg, connectome.post, 1)
    out_deg = np.zeros(connectome.n_neurons, dtype=np.int64)
    np.add.at(out_deg, connectome.pre, 1)
    degree = (in_deg + out_deg)[positions]
    chosen = np.argsort(-degree, kind="stable")[:cap]
    return np.sort(ids[chosen])


def build_synthetic_pathway(
    n_pn: int,
    n_kc: int,
    n_mbon: int,
    *,
    kc_fan_in: int = 6,
    mb_input: float = 0.1,
    seed: int = 0,
    mode: str = "random",
) -> Pathway:
    """A random expansion circuit of the same shape, for smoke tests.

    Used both as the no-connectome fallback when training without a pathway file
    and as the scaffold when rebuilding a checkpoint whose wiring lives in its
    buffer. The scientific control is `shuffle_pathway`, not this.
    """
    if mode not in ("random", "shuffled"):
        raise ValueError(f"unknown synthetic mode: {mode}")
    rng = np.random.default_rng(seed)
    pn_to_kc = np.zeros((n_pn, n_kc), dtype=np.float32)
    fan_in = min(kc_fan_in, n_pn)
    for k in range(n_kc):
        partners = rng.choice(n_pn, size=fan_in, replace=False)
        pn_to_kc[partners, k] = 1.0 / fan_in
    kc_to_mbon = np.zeros((n_kc, n_mbon), dtype=np.float32)
    n_active = max(1, int(round(n_kc * mb_input)))
    for m in range(n_mbon):
        partners = rng.choice(n_kc, size=n_active, replace=False)
        kc_to_mbon[partners, m] = 1.0 / n_active
    apl_to_kc = np.ones(n_kc, dtype=np.float32)
    meta = {
        "source": f"synthetic-{mode}",
        "kc_fan_in": fan_in,
        "mb_input_fraction": mb_input,
        "pn_to_kc_nnz": int(np.count_nonzero(pn_to_kc)),
        "kc_to_mbon_nnz": int(np.count_nonzero(kc_to_mbon)),
    }
    return Pathway(pn_to_kc, kc_to_mbon, apl_to_kc, n_pn, n_kc, n_mbon, meta)


def shuffle_pathway(pathway: Pathway, *, seed: int = 0, keep_degree: bool = True) -> Pathway:
    """Control: keep the degree sequence but destroy the specific partners.

    Configuration-model construction: each neuron keeps exactly the number of
    partners it had, but the partners are redrawn at random weighted by the target
    degree distribution. Row degrees (and therefore the synapse count) are
    preserved exactly and column degrees in expectation, which is what makes this
    a fair control rather than a smaller, sparser network.
    """
    rng = np.random.default_rng(seed)

    def _shuffle(mat: np.ndarray) -> np.ndarray:
        rows, cols = mat.shape
        if not keep_degree:
            return mat.ravel()[rng.permutation(mat.size)].reshape(mat.shape)
        in_degree = (mat > 0).sum(axis=0)
        if in_degree.sum() == 0:
            return mat.copy()
        probabilities = in_degree / in_degree.sum()
        out = np.zeros_like(mat)
        for r in range(rows):
            nz = np.flatnonzero(mat[r])
            if nz.size == 0:
                continue
            chosen = rng.choice(cols, size=nz.size, replace=False, p=probabilities)
            out[r, chosen] = mat[r, nz]
        return out

    return Pathway(
        _shuffle(pathway.pn_to_kc),
        _shuffle(pathway.kc_to_mbon),
        pathway.apl_to_kc.copy(),
        pathway.n_pn,
        pathway.n_kc,
        pathway.n_mbon,
        {**pathway.meta, "control": "degree-preserving-shuffle", "seed": seed},
    )
