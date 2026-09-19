"""Turn the downloaded corpus into a trained tokenizer plus packed token shards.

    python scripts/prepare_data.py --corpus data/corpus --out data/tokenized \
        --vocab-size 16384 --tokenizer-sample-mb 120

Two stages: a byte-level BPE tokenizer is fit on a bounded sample of the corpus
(fitting on all of it costs far more time than it is worth), then the whole
corpus is tokenised into `train.bin` / `val.bin`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.data import corpus_files, tokenize_corpus  # noqa: E402
from flybrain.tokenizer import load_tokenizer, train_tokenizer  # noqa: E402


def sample_lines(files: list[str], max_bytes: int, max_line_chars: int = 4000) -> list[str]:
    """Read a bounded, evenly spread sample of lines for tokenizer training."""
    if not files:
        return []
    per_file = max(1, max_bytes // len(files))
    sample: list[str] = []
    for path in files:
        taken = 0
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    # Parquet is binary; the reader below handles that case.
                    if "\x00" in line:
                        break
                    line = line.rstrip("\n")
                    if not line.strip():
                        continue
                    sample.append(line[:max_line_chars])
                    taken += len(line.encode("utf-8"))
                    if taken >= per_file:
                        break
        except (OSError, UnicodeDecodeError):
            continue
    return sample


def parquet_sample(files: list[str], max_bytes: int) -> list[str]:
    """Sample text out of parquet corpora without materialising them."""
    import pyarrow.parquet as pq

    sample: list[str] = []
    per_file = max(1, max_bytes // max(1, len(files)))
    for path in files:
        pf = pq.ParquetFile(path)
        column = next(
            (f.name for f in pf.schema_arrow if f.name.lower() in ("text", "content", "document")), None
        )
        if column is None:
            continue
        taken, batch_index = 0, 0
        while taken < per_file and batch_index * 512 < pf.metadata.num_rows:
            batch = next(pf.iter_batches(batch_size=512, columns=[column], use_threads=False))
            for value in batch.column(0).to_pylist():
                if not value:
                    continue
                text = str(value)
                sample.append(text[:4000])
                taken += len(text.encode("utf-8"))
                if taken >= per_file:
                    break
            batch_index += 1
            if batch_index > 8:  # a few thousand documents is plenty for BPE
                break
    return sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--out", default="data/tokenized")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--tokenizer-sample-mb", type=int, default=120)
    parser.add_argument("--val-fraction", type=float, default=0.005)
    parser.add_argument("--token-dtype", default="uint16")
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument(
        "--tokenizer",
        default="",
        help=(
            "reuse this tokenizer instead of fitting a new one. Required when "
            "extending a corpus for an already trained model: the token ids must "
            "not move, or the checkpoint's embeddings stop meaning anything."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = corpus_files(args.corpus)
    if not files:
        print(f"no corpus files under {args.corpus}", file=sys.stderr)
        return 1
    print(f"corpus files: {len(files)}")
    for path in files:
        print(f"  {path}  {os.path.getsize(path) / 1024**2:.1f} MiB")

    tokenizer_path = args.tokenizer or os.path.join(args.out, "tokenizer.json")
    if args.skip_tokenizer and os.path.exists(tokenizer_path):
        tokenizer = load_tokenizer(tokenizer_path)
        print(f"reusing tokenizer at {tokenizer_path} (vocab {tokenizer.get_vocab_size()})")
        # Keep a copy next to the shards so the trained run can find it.
        local = os.path.join(args.out, "tokenizer.json")
        if os.path.abspath(local) != os.path.abspath(tokenizer_path):
            os.makedirs(args.out, exist_ok=True)
            with open(tokenizer_path, "rb") as src, open(local, "wb") as dst:
                dst.write(src.read())
            print(f"copied tokenizer to {local}")
    else:
        budget = args.tokenizer_sample_mb * 1024 * 1024
        parquet = [f for f in files if f.endswith(".parquet")]
        text = [f for f in files if not f.endswith(".parquet")]
        started = time.time()
        # Parquet corpora are far too large to stream in full for tokenizer
        # fitting, so they contribute a bounded sample; plain text files are read
        # directly by the trainer.
        sample = parquet_sample(parquet, budget) if parquet else []
        print(f"tokenizer sample: {len(sample):,} documents in {time.time() - started:.1f}s")
        tokenizer = train_tokenizer(
            text,
            tokenizer_path,
            vocab_size=args.vocab_size,
            sample_texts=sample or None,
        )
        print(f"trained tokenizer with {tokenizer.get_vocab_size()} tokens -> {tokenizer_path}")

    summary = tokenize_corpus(
        files,
        tokenizer,
        args.out,
        val_fraction=args.val_fraction,
        token_dtype=args.token_dtype,
    )
    summary["tokenizer"] = tokenizer_path
    print(json.dumps({k: v for k, v in summary.items() if k != "files"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
