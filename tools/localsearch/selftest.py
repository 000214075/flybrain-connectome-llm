"""Drive the local search MCP server the way a client does, and check it answers.

This is deliberately an out-of-process test over the real stdio protocol: the
failures that matter here (a stray print on stdout, the wrong text encoding on
Windows, a backend that reports success while returning unrelated pages) are all
invisible to an in-process unit test that calls the handler functions directly.

Usage:
    python tools/localsearch/selftest.py [query]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "server.py")
DEFAULT_QUERY = "Drosophila mushroom body connectome language model"


def send(proc: subprocess.Popen, message: dict) -> None:
    proc.stdin.write(json.dumps(message, ensure_ascii=False).encode("utf-8") + b"\n")
    proc.stdin.flush()


def receive(proc: subprocess.Popen) -> dict:
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(f"server closed the stream; stderr:\n{proc.stderr.read().decode('utf-8', 'replace')}")
        if line.strip():
            return json.loads(line.decode("utf-8"))


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    proc = subprocess.Popen(
        [sys.executable, "-u", SERVER],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    failures: list[str] = []
    try:
        send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        init = receive(proc)
        info = init.get("result", {}).get("serverInfo", {})
        print(f"initialize -> {info.get('name')} {info.get('version')}")
        if info.get("name") != "local-search":
            failures.append("initialize did not report the expected server")

        send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = [t["name"] for t in receive(proc)["result"]["tools"]]
        print(f"tools/list -> {tools}")
        if set(tools) != {"search", "fetch"}:
            failures.append(f"unexpected tool set: {tools}")

        # A CJK query is the encoding check: it only survives if both sides of the
        # Windows pipe are treated as UTF-8 rather than the ANSI code page.
        for probe in (query, "苍蝇大脑 连接组 语言模型"):
            send(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                        "params": {"name": "search", "arguments": {"query": probe, "limit": 5}}})
            message = receive(proc)
            payload = json.loads(message["result"]["content"][0]["text"])
            print(f"\nquery {probe!r}")
            print(f"  backend        {payload.get('backend')}")
            print(f"  reliable       {payload.get('reliable')}  coverage {payload.get('term_coverage')}")
            print(f"  missing terms  {payload.get('missing_terms')}")
            for item in payload.get("results", [])[:3]:
                print(f"    - {item['title'][:88]}")
                print(f"      {item['url']}")
            if payload.get("warning"):
                print(f"  warning: {payload['warning']}")
            if payload.get("failed_backends"):
                print(f"  failed backends: {payload['failed_backends']}")
            if not payload.get("results"):
                failures.append(f"no results for {probe!r}: {payload.get('failed_backends')}")
            elif not payload.get("reliable"):
                failures.append(f"results for {probe!r} do not mention the query")
    finally:
        proc.kill()

    print()
    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        return 1
    print("all local-search checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
