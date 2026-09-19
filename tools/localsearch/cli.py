"""Ask the local search server from the shell, with its reliability verdict.

The MCP client is the normal way in; this exists because an agent working in a
shell needs the same answer, and because the verdict (`reliable`, `term_coverage`)
is the part worth printing even when nobody reads the JSON.

Usage:
    python tools/localsearch/cli.py "query" [--limit 8] [--json] [--full]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import server  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--engines", default="")
    parser.add_argument("--json", action="store_true", help="print the raw payload")
    parser.add_argument("--full", action="store_true", help="print every description")
    parser.add_argument("--fetch", action="store_true", help="fetch each result and print its text")
    args = parser.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()] or None
    payload = server.do_search({"query": args.query, "limit": args.limit, "engines": engines})

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"backend        {payload['backend']}")
    print(f"reliable       {payload['reliable']}   term coverage {payload.get('term_coverage')}")
    if payload.get("missing_terms"):
        print(f"missing terms  {payload['missing_terms']}")
    for note in payload.get("failed_backends") or []:
        print(f"backend failed {note}")
    if payload.get("warning"):
        print(f"WARNING        {payload['warning']}")
    print()
    for index, item in enumerate(payload["results"], 1):
        print(f"{index:2d}. {item['title']}")
        print(f"    {item['url']}   [{item['engine']}]")
        if args.full and item.get("description"):
            print(f"    {item['description']}")
        if args.fetch:
            try:
                text = server.do_fetch({"url": item["url"], "max_chars": 6000})["text"]
                print("    ---")
                for line in text.splitlines()[:60]:
                    print(f"    {line}")
            except Exception as exc:  # noqa: BLE001
                print(f"    fetch failed: {type(exc).__name__}: {exc}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
