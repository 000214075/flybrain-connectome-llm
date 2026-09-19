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
    "wiring-shuffled": "checkpoints/flybrain-connectome-shuffled/best.pt",
    "ports-sensory": "checkpoints/ports_sensory/best.pt",
    "ports-random-sensory": "checkpoints/ports_random_sensory/best.pt",
    "ports-output": "checkpoints/ports_output/best.pt",
    "ports-all": "checkpoints/ports_all/best.pt",
}

KEEP = ("model", "model_config", "config", "step", "best_val")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--only", nargs="*", default=None, help="subset of names to export")
    parser.add_argument("--force", action="store_true", help="re-export files that already exist")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}

    for name in args.only or list(SOURCES):
        src = Path(SOURCES[name])
        dst = out_dir / f"{name}.pt"
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
        n_params = sum(int(t.numel()) for t in payload["model"].values() if torch.is_tensor(t))
        del blob

        torch.save(payload, dst)
        del payload

        size = dst.stat().st_size
        manifest[name] = {
            "source": src.as_posix(),
            "step": step,
            "best_val": best_val,
            "params": n_params,
            "bytes": size,
            "sha256": sha256(dst),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")
        print(
            f"[ok]   {name}: step {step} best_val {best_val:.4f} "
            f"params {n_params:,} {size / 2**30:.2f} GiB -> {dst}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
