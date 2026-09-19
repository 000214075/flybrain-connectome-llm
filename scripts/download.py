"""Resumable HTTP downloader for the connectome and corpus assets.

Usage: python scripts/download.py <manifest.json> [dest_root]
The manifest maps a relative destination path to a URL and expected size.
Files already present at the expected size are skipped, and partial files are
resumed with a Range request.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

CHUNK = 1 << 20  # 1 MiB


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


def download(url: str, dest: str, expected: int | None = None, max_bytes: int | None = None) -> None:
    """Fetch `url` to `dest`, resuming a partial file and optionally truncating.

    `max_bytes` takes only the leading slice of a large file, which is how the
    multi-gigabyte corpora are sampled without paying for the whole download.
    """
    if max_bytes is not None:
        expected = max_bytes
    if os.path.exists(dest) and expected and os.path.getsize(dest) == expected:
        print(f"[skip] {dest} already complete ({human(expected)})")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if expected and have >= expected:
        have = 0
    req = urllib.request.Request(url)
    if max_bytes is not None:
        req.add_header("Range", f"bytes={have}-{max_bytes - 1}")
    elif have:
        req.add_header("Range", f"bytes={have}-")
    mode = "ab" if have else "wb"
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and expected:  # already fully downloaded
            print(f"[skip] {dest} complete")
            return
        raise

    total = expected
    if resp.headers.get("Content-Length"):
        clen = int(resp.headers["Content-Length"])
        total = have + clen if max_bytes is None else max_bytes
    if have and resp.status != 206:
        have, mode = 0, "wb"  # server ignored Range; restart cleanly

    print(f"[get ] {dest}  {human(have)}/{human(total) if total else '?'}")
    done, t0, last = have, time.time(), 0.0
    with open(dest, mode) as fh:
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if time.time() - last > 5:
                rate = (done - have) / max(time.time() - t0, 1e-9)
                pct = f"{100 * done / total:.1f}%" if total else "?"
                print(f"       {pct} {human(done)} @ {human(rate)}/s", flush=True)
                last = time.time()
    size = os.path.getsize(dest)
    if expected and size != expected:
        raise RuntimeError(f"size mismatch for {dest}: got {size}, expected {expected}")
    print(f"[ok  ] {dest} {human(size)}")


def main() -> int:
    manifest_path = sys.argv[1]
    dest_root = sys.argv[2] if len(sys.argv) > 2 else "."
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    failures = []
    for rel, spec in manifest.items():
        dest = os.path.join(dest_root, rel)
        try:
            download(spec["url"], dest, spec.get("size"), spec.get("max_bytes"))
        except Exception as exc:  # keep going: one bad URL must not sink the rest
            print(f"[FAIL] {rel}: {type(exc).__name__}: {exc}")
            failures.append(rel)
    if failures:
        print("failed:", failures)
        return 1
    print("all downloads complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
