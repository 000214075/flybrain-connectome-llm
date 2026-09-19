"""Build the domain fine-tuning corpus from the DeepSeek teacher output.

A few hundred teacher records are a rounding error next to 455 MB of base corpus:
dumping them into pretraining changes nothing measurable. Knowledge is transferred
by *repeating* the domain text so the model actually sees it many times, which is
what this script prepares, with a slice of base text mixed in so the model does
not forget how to write anything else.

    python scripts/build_domain_corpus.py --domain data/corpus/domain \
        --base data/corpus --out data/corpus_domain --repeat 60 --base-fraction 0.25
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.data import _iter_text, corpus_files, is_domain_text, iter_jsonl_text  # noqa: E402

# How much base text to keep per pulled passage. Bounded, so one long Wikipedia
# article cannot dominate the mix, and small enough that packing stays uniform.
PASSAGE_CHARS = 4000


def collect_domain_texts(domain_dir: str) -> list[str]:
    """Every teacher-written passage, in a stable order.

    Both forms of each explanation are kept. The plain passage teaches the content;
    the chat rendering of the same pair teaches the *interface* -- the base corpus
    is Wikipedia and TinyStories, so without these the `<user>`/`<assistant>` tokens
    the CLI and the web page generate with would never have been trained on
    anything, and the model would answer a question by continuing an encyclopedia.

    The teacher's own dialogue mode cannot supply them on this model: its reasoning
    pass consumes the entire token budget before the answer begins, and every call
    comes back empty. The pairs it did produce are re-rendered here instead.
    """
    texts: list[str] = []
    for path in sorted(corpus_files(domain_dir)):
        if path.endswith(".jsonl"):
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # The teacher may still be appending to this file; a half-written
                    # last line is not a reason to fail the build.
                    continue
                if isinstance(record.get("text"), str) and record["text"].strip():
                    text = record["text"].strip()
                    prompt = record.get("prompt")
                    if not is_domain_text(f"{prompt or ''} {text}"):
                        continue
                    texts.append(text)
                    if isinstance(prompt, str) and prompt.strip():
                        texts.append(render_chat_turns(prompt.strip(), text))
                elif isinstance(record.get("messages"), list):
                    turns = [
                        f"<{m.get('role')}>{m.get('content', '')}<eos>"
                        for m in record["messages"]
                        if isinstance(m, dict)
                    ]
                    body = " ".join(str(m.get("content", "")) for m in record["messages"] if isinstance(m, dict))
                    if turns and is_domain_text(body):
                        texts.append("<bos>" + "".join(turns))
    return texts


def render_chat_turns(question: str, answer: str) -> str:
    """The exact template `format_chat` produces, minus the generation prompt."""
    return f"<bos><user>{question}<eos><assistant>{answer}<eos>"


def collect_base_texts(base_dir: str, target_chars: int, *, seed: int = 0) -> list[str]:
    """Sample base-corpus passages totalling roughly `target_chars` characters.

    Round-robin over persistent per-file streams, so the sample is spread across
    every corpus. Restarting each file from its first row on every round -- which is
    what an earlier version did -- returns the same passage forever and never
    reaches the target, leaving the mix with no general text in it at all.
    """
    files = [f for f in corpus_files(base_dir) if "domain" not in f]
    if not files:
        return []
    streams = {path: _iter_text(path, PASSAGE_CHARS) for path in files}
    picked: list[str] = []
    total = 0
    while total < target_chars and streams:
        for path in list(streams):
            if total >= target_chars:
                break
            try:
                chunk = next(streams[path])
            except (StopIteration, OSError, ValueError):
                del streams[path]
                continue
            passage = chunk.strip()[:PASSAGE_CHARS]
            if len(passage) < 40:
                continue
            picked.append(passage)
            total += len(passage)
    rng = random.Random(seed)
    rng.shuffle(picked)
    return picked


def chunk_prose(text: str, size: int = PASSAGE_CHARS) -> list[str]:
    """Split prose into `size`-ish pieces, preferring to cut at a sentence end.

    Needed because the fetched pages are one long run of text: splitting only on
    blank lines kept one passage per file and threw away most of the material.
    """
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind("。"), window.rfind(". "), window.rfind("！"), window.rfind("？"))
            if cut > size // 2:
                end = start + cut + 1
        piece = text[start:end].strip()
        if len(piece) >= 120:
            pieces.append(piece)
        start = end
    return pieces


def collect_source_texts(source_dir: str) -> list[str]:
    """The downloaded primary sources, split into paragraph-sized passages.

    Real text from the releases and the literature, not generated. Repetition is
    controlled separately from the teacher records because these passages are
    already the ground truth the teacher was grounded on.
    """
    texts: list[str] = []
    if not source_dir or not os.path.isdir(source_dir):
        return texts
    for name in sorted(os.listdir(source_dir)):
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(source_dir, name), encoding="utf-8") as fh:
            raw = fh.read()
        body = "\n".join(line for line in raw.splitlines() if not line.startswith("# source:"))
        for paragraph in re.split(r"\n\s*\n", body):
            paragraph = " ".join(paragraph.split())
            if len(paragraph) < 120:
                continue
            texts.extend(piece for piece in chunk_prose(paragraph) if is_domain_text(piece))
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="data/corpus/domain")
    parser.add_argument("--base", default="data/corpus")
    parser.add_argument("--sources", default="data/corpus/source")
    parser.add_argument("--out", default="data/corpus_domain")
    parser.add_argument("--repeat", type=int, default=60)
    parser.add_argument("--source-repeat", type=int, default=8)
    parser.add_argument("--base-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    domain = collect_domain_texts(args.domain)
    sources = collect_source_texts(args.sources)
    if not domain and not sources:
        print(f"no teacher records or sources found", file=sys.stderr)
        return 1
    domain_chars = sum(len(t) for t in domain)
    source_chars = sum(len(t) for t in sources)
    print(f"teacher: {len(domain)} passages, {domain_chars:,} characters, repeating {args.repeat}x")
    print(f"source:  {len(sources)} passages, {source_chars:,} characters, repeating {args.source_repeat}x")
    mix_chars = domain_chars * args.repeat + source_chars * args.source_repeat
    print(f"domain side: {mix_chars:,} characters")

    target_base = int(mix_chars * args.base_fraction / max(1e-9, 1 - args.base_fraction))
    base = collect_base_texts(args.base, target_base, seed=args.seed)
    base_chars = sum(len(t) for t in base)
    print(f"base:    {len(base)} passages, {base_chars:,} characters")

    rng = random.Random(args.seed)
    rows: list[dict] = []
    for _ in range(args.repeat):
        for text in domain:
            rows.append({"text": text, "source": "deepseek-teacher"})
    for _ in range(args.source_repeat):
        for text in sources:
            rows.append({"text": text, "source": "open-source"})
    for text in base:
        rows.append({"text": text, "source": "base"})
    rng.shuffle(rows)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "domain_mix.jsonl")
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_chars = domain_chars * args.repeat + base_chars
    print(f"wrote {len(rows):,} records ({total_chars:,} characters) to {out_path}")
    print(
        "next: python scripts/prepare_data.py --corpus data/corpus_domain "
        "--out data/tokenized_domain --skip-tokenizer --tokenizer data/tokenized/tokenizer.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
