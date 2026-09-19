"""Command-line driver for the DeepSeek teacher corpus generator.

    python -m flybrain.teacher_cli --probe
    python -m flybrain.teacher_cli --out data/corpus/domain/flybrain.jsonl --records 300
    python -m flybrain.teacher_cli --out data/corpus/domain/dialogues.jsonl --records 200 --mode qa

The API generates text; it does not train anything. What it produces is stored as
a corpus and the local model is trained on it afterwards (distillation).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from flybrain.teacher import (
    generate_doc_corpus,
    TeacherConfig,
    TeacherError,
    api_key,
    generate_domain_corpus,
    generate_qa_corpus,
    probe,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/corpus/domain/flybrain.jsonl", help="output JSONL path")
    parser.add_argument("--records", type=int, default=200, help="how many records to generate")
    parser.add_argument(
        "--mode",
        choices=("explain", "qa", "docqa"),
        default="explain",
        help="explain/qa write from the teacher's own knowledge; docqa grounds every "
        "record in a passage from --sources, which is both more accurate and the only "
        "dialogue mode that fits in this model's token budget",
    )
    parser.add_argument(
        "--sources",
        default="data/corpus/source",
        help="directory of fetched open-source text that docqa grounds on",
    )
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="concurrent teacher requests; the calls are network-bound",
    )
    parser.add_argument("--no-resume", action="store_true", help="start over instead of appending")
    parser.add_argument("--probe", action="store_true", help="check the key and list available models")
    args = parser.parse_args()

    try:
        api_key()
    except TeacherError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "\nSet the key first, for example:\n"
            '  $env:DEEPSEEK_API_KEY = "sk-..."      # PowerShell\n'
            "  export DEEPSEEK_API_KEY=sk-...        # bash",
            file=sys.stderr,
        )
        return 2

    if args.probe:
        try:
            models = probe()
        except Exception as exc:
            print(f"probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(models, indent=2, ensure_ascii=False))
        return 0

    cfg = TeacherConfig(model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)
    generator = generate_domain_corpus if args.mode == "explain" else generate_qa_corpus
    extra: dict = {}
    if args.mode == "docqa":
        generator = generate_doc_corpus
        extra["source_dir"] = args.sources
    try:
        total = generator(
            args.out,
            n_records=args.records,
            cfg=cfg,
            seed=args.seed,
            resume=not args.no_resume,
            workers=args.workers,
            **extra,
        )
    except TeacherError as exc:
        print(f"generation aborted: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {total} records to {args.out}")
    print("next: re-run scripts/prepare_data.py to fold the new corpus into training data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
