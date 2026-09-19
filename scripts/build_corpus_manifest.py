"""Write configs/corpus.json with exact remote sizes for the training corpus.

Sizes are queried from the HuggingFace mirror API so the downloader can verify
completeness and resume correctly, rather than trusting hard-coded numbers.
"""

from __future__ import annotations

import json
import os
import urllib.request

MIRROR = "https://hf-mirror.com"
# Chinese Wikipedia gives broad Chinese coverage; two shards is a good balance
# between corpus size and download time on this connection.
ZH_SHARDS = ["20231101.zh/train-00002-of-00006.parquet", "20231101.zh/train-00005-of-00006.parquet"]
# TinyStories only needs its leading slice: 120 MiB is plenty of simple English
# to teach fluency, and it downloads in minutes rather than hours.
TINYSTORIES_MAX_BYTES = 120 * 1024 * 1024
OUT = "configs/corpus.json"


def api(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read().decode())


def main() -> int:
    manifest: dict[str, dict] = {}

    zh_index = api(f"{MIRROR}/api/datasets/wikimedia/wikipedia/tree/main/20231101.zh")
    sizes = {entry["path"]: entry.get("size") for entry in zh_index}
    for shard in ZH_SHARDS:
        size = sizes.get(shard)
        if size is None:
            raise SystemExit(f"shard {shard} not found in the mirror index")
        manifest[f"corpus/wiki_zh/{os.path.basename(shard)}"] = {
            "url": f"{MIRROR}/datasets/wikimedia/wikipedia/resolve/main/{shard}",
            "size": int(size),
        }

    manifest["corpus/tinystories/TinyStories-train.txt"] = {
        "url": f"{MIRROR}/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
        "max_bytes": TINYSTORIES_MAX_BYTES,
    }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {OUT}")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
