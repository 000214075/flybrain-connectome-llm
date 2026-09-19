"""Strip optimiser state from checkpoints so they fit as GitHub Release assets.

A training ``best.pt`` carries the AdamW moments as well as the weights, which is
more than half of the file size and useless to anyone doing inference or
fine-tuning.  This writes a smaller ``<name>.pt`` that keeps exactly the keys
``generate.load_model`` needs, plus the original training metadata, and records a
SHA256 per file in ``manifest.json``.

    .venv\\Scripts\\python.exe scripts\\export_weights.py --out-dir I:\\flybrain-release
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

# release name -> the training run's best.pt
SOURCES: dict[str, str] = {
    "rank128": "checkpoints/flybrain-connectome-rank128/best.pt",
    "rank256": "E:/flybrain-connectome/rank256/best.pt",
    "rank512": "E:/flybrain-connectome/rank512/best.pt",
    "rank1024": "E:/flybrain-connectome/flybrain-connectome-rank1024/best.pt",
    "canonical-rank512": "checkpoints/flybrain-connectome/best.pt",
    "biospike-nomask": "checkpoints/bio-screening/biospike-nomask/best.pt",
    "biospike-sensory": "checkpoints/bio-screening/biospike-sensory/best.pt",
    "biospike-random": "checkpoints/bio-screening/biospike-random/best.pt",
    "biospike-inonly": "checkpoints/bio-screening/biospike-inonly/best.pt",
    "biospike-outonly": "checkpoints/bio-screening/biospike-outonly/best.pt",
    "ports-sensory": "checkpoints/ports_sensory/best.pt",
    "ports-random-sensory": "checkpoints/ports_random_sensory/best.pt",
    "ports-output": "checkpoints/ports_output/best.pt",
    "ports-all": "checkpoints/ports_all/best.pt",
}

# Everything else worth keeping before the local checkpoints were deleted: the arms
# behind the kernel, learning-rate, seed-variance and data-loader findings, plus the
# hybrid transformer model (the only one here with usable language ability).
EXTRA_SOURCES: dict[str, str] = {
    "wiring-shuffled-step150": "checkpoints/flybrain-connectome-shuffled/best.pt",
    "recipe-scaled": "checkpoints/flybrain-connectome-scaled/best.pt",
    "recipe-scaled-2021": "checkpoints/flybrain-connectome-scaled-2021/best.pt",
    "recipe-scaled-9000": "checkpoints/flybrain-connectome-scaled-9000/best.pt",
    "recipe-sched800": "checkpoints/flybrain-connectome-sched800/best.pt",
    "recipe-long2": "checkpoints/flybrain-connectome-long2/best.pt",
    "kernel-csr": "checkpoints/flybrain-connectome-csr/best.pt",
    "kernel-csr-full": "checkpoints/flybrain-connectome-csr-full/best.pt",
    "kernel-csr-lr6e-05-804": "checkpoints/flybrain-connectome-csr-lr6e-05-804/best.pt",
    "kernel-indexadd-492": "checkpoints/flybrain-connectome-indexadd-492/best.pt",
    "lr-1e-04": "checkpoints/flybrain-connectome-lr/lr1e-04/best.pt",
    "lr-2e-04": "checkpoints/flybrain-connectome-lr/lr2e-04/best.pt",
    "lr-3e-05": "checkpoints/flybrain-connectome-lr/lr3e-05/best.pt",
    "lr-6e-05": "checkpoints/flybrain-connectome-lr/lr6e-05/best.pt",
    "lr-3e-05-800": "checkpoints/flybrain-connectome-lr3e-05-800/best.pt",
    "seed-1337": "checkpoints/flybrain-connectome-variance/seed1337/best.pt",
    "seed-2024": "checkpoints/flybrain-connectome-variance/seed2024/best.pt",
    "seed-4242": "checkpoints/flybrain-connectome-variance/seed4242/best.pt",
    "hybrid-transformer": "checkpoints/flybrain-full/best.pt",
    "domain-domv2": "checkpoints/flybrain-domv2/best.pt",
    "workers-eq0": "E:/flybrain-workers/eq0/best.pt",
    "workers-eq2": "E:/flybrain-workers/eq2/best.pt",
    "workers-eq2b": "E:/flybrain-workers/eq2b/best.pt",
    "workers-nw0": "I:/flybrain-probe/nw0/best.pt",
    "workers-nw2": "I:/flybrain-probe/nw2/best.pt",
    "smoke-brain": "checkpoints/smoke-brain/best.pt",
    "probe": "checkpoints/probe/best.pt",
}

SETS = {"release": SOURCES, "extra": EXTRA_SOURCES}

KEEP = ("model", "model_config", "config", "step", "best_val")

# The wiring itself lives in the state dict as four frozen arrays.  They are not
# parameters (they are absent from named_parameters() and from the optimiser), and at
# 76,853,875 elements they are large enough that lumping them in would overstate the
# model by ~40%.
FROZEN = (
    "brain.circuit.pre",
    "brain.circuit.post",
    "brain.circuit.weight",
    "brain.circuit.sign",
)


def count_elements(model: dict) -> tuple[int, int, int]:
    """Return (parameters, frozen_connectome_elements, total) for a model state dict."""
    sizes = {name: int(t.numel()) for name, t in model.items() if torch.is_tensor(t)}
    frozen = sum(size for name, size in sizes.items() if name in FROZEN)
    total = sum(sizes.values())
    return total - frozen, frozen, total

# Deliberately not published: the degree-preserving rewiring control
# (checkpoints/flybrain-connectome-shuffled) stopped at step 150, and its matched
# real-wiring partner at step 492 was overwritten when the canonical checkpoint was
# promoted into the same directory.  The two files are not step-matched, so shipping
# them together would invite a comparison that the report explicitly rules out
# (reports/FINAL_REPORT.md section 7.6).  Rebuild the control from
# configs/train_connectome_shuffled.json instead.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(path: Path) -> dict:
    """Read one exported (or source) checkpoint and describe it."""
    blob = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    missing = [key for key in KEEP if key not in blob]
    if missing:
        raise KeyError(f"{path} lacks {missing}")
    params, frozen, total = count_elements(blob["model"])
    return {
        "step": int(blob["step"]),
        "best_val": float(blob["best_val"]),
        "parameters": params,
        "frozen_connectome_elements": frozen,
        "model_state_elements": total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--set", choices=sorted(SETS), default="release", help="which source list to use")
    parser.add_argument("--only", nargs="*", default=None, help="subset of names to export")
    parser.add_argument("--force", action="store_true", help="re-export files that already exist")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="do not copy weights; just rewrite manifest.json from the files already there",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = SETS[args.set]
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}

    for name in args.only or list(sources):
        src = Path(sources[name])
        dst = out_dir / f"{name}.pt"

        if args.manifest_only:
            if not dst.exists():
                print(f"[skip] {name}: not exported yet", flush=True)
                continue
            manifest[name] = {"source": src.as_posix(), **inspect(dst), "bytes": dst.stat().st_size,
                              "sha256": sha256(dst)}
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")
            print(f"[ok]   {name}: {manifest[name]['parameters']:,} parameters", flush=True)
            continue

        if not src.exists():
            print(f"[skip] {name}: source missing ({src})", flush=True)
            continue
        if dst.exists() and not args.force:
            print(f"[have] {name}: {dst.stat().st_size / 2**30:.2f} GiB", flush=True)
            continue

        blob = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
        missing = [key for key in KEEP if key not in blob]
        if missing:
            print(f"[fail] {name}: source lacks {missing}", flush=True)
            continue

        payload = {key: blob[key] for key in KEEP}
        step, best_val = int(payload["step"]), float(payload["best_val"])
        params, frozen, total = count_elements(payload["model"])
        del blob

        torch.save(payload, dst)
        del payload

        size = dst.stat().st_size
        manifest[name] = {
            "source": src.as_posix(),
            "step": step,
            "best_val": best_val,
            "parameters": params,
            "frozen_connectome_elements": frozen,
            "model_state_elements": total,
            "bytes": size,
            "sha256": sha256(dst),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")
        print(
            f"[ok]   {name}: step {step} best_val {best_val:.4f} "
            f"params {params:,} (+{frozen:,} frozen) {size / 2**30:.2f} GiB -> {dst}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
