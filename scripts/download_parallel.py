"""Parallel chunked downloader for hosts that throttle a single connection.

Google Cloud Storage serves these connectome files at a few hundred KB/s on one
connection regardless of available bandwidth. Splitting the file into ranges and
fetching them concurrently restores full speed, and any bytes already on disk are
reused as the head of the file so an interrupted single-stream download is not
wasted.

    python scripts/download_parallel.py <url> <dest> [--size N] [--workers 8]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import threading
import time
import urllib.error
import urllib.request

CHUNK = 8 << 20  # 8 MiB per request
_progress_lock = threading.Lock()


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return int(resp.headers["Content-Length"])


def fetch_range(url: str, start: int, end: int, dest: str, *, retries: int = 4) -> int:
    """Download [start, end) into `dest`, retrying transient failures."""
    expected = end - start
    for attempt in range(retries):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if have == expected:
            return expected
        if have > expected:
            os.remove(dest)
            have = 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end - 1}"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "ab") as fh:
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    fh.write(block)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} for range {start}-{end}: {exc}", flush=True)
            time.sleep(2 * (attempt + 1))
    return os.path.getsize(dest) if os.path.exists(dest) else 0


def download(url: str, dest: str, *, size: int | None = None, workers: int = 8) -> None:
    total = size or remote_size(url)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)

    # Whatever is already on disk becomes the head, so a killed single-stream
    # download continues instead of starting over.
    head_path = dest + ".head"
    head = 0
    if os.path.exists(dest):
        if os.path.getsize(dest) >= total:
            print(f"[skip] {dest} already complete ({total / 1024**2:.1f} MiB)")
            return
        os.replace(dest, head_path)
        head = os.path.getsize(head_path)
        print(f"reusing {head / 1024**2:.1f} MiB already downloaded")

    ranges = [(start, min(start + CHUNK, total)) for start in range(head, total, CHUNK)]
    parts = [f"{dest}.part{i}" for i in range(len(ranges))]
    done = 0
    started = time.time()

    def report(extra: int) -> None:
        nonlocal done
        with _progress_lock:
            done += extra
            rate = done / max(time.time() - started, 1e-9)
            pct = 100 * (head + done) / total
            print(
                f"       {pct:.1f}% {((head + done) / 1024**2):.1f}/{(total / 1024**2):.1f} MiB "
                f"@ {rate / 1024**2:.1f} MiB/s",
                flush=True,
            )

    print(f"[get ] {dest}  {total / 1024**2:.1f} MiB in {len(ranges)} chunks on {workers} threads")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_range, url, start, end, part): (start, end, part)
            for (start, end), part in zip(ranges, parts)
        }
        for future in concurrent.futures.as_completed(futures):
            start, end, part = futures[future]
            try:
                written = future.result()
            except Exception as exc:
                raise SystemExit(f"chunk {start}-{end} failed: {type(exc).__name__}: {exc}") from exc
            if written != end - start:
                raise SystemExit(f"chunk {start}-{end} short: {written} of {end - start} bytes")
            report(written)

    with open(dest, "wb") as out:
        if head:
            with open(head_path, "rb") as fh:
                while True:
                    block = fh.read(1 << 22)
                    if not block:
                        break
                    out.write(block)
        for part in parts:
            with open(part, "rb") as fh:
                while True:
                    block = fh.read(1 << 22)
                    if not block:
                        break
                    out.write(block)

    written_size = os.path.getsize(dest)
    if written_size != total:
        raise SystemExit(f"assembled file is {written_size} bytes, expected {total}")
    if os.path.exists(head_path):
        os.remove(head_path)
    for part in parts:
        os.remove(part)
    elapsed = time.time() - started
    print(f"[ok  ] {dest} {written_size / 1024**2:.1f} MiB in {elapsed:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("dest")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    download(args.url, args.dest, size=args.size, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
