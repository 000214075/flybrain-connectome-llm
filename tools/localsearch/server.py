"""A search MCP server that runs entirely on this machine.

Why this exists
---------------
The search path this replaces scraped one engine over a plain HTTP request, and on
this network that intermittently returned generic pages for highly specific
queries -- while still reporting `"status": "ok"`. The failure was silent, which is
the worst property a research tool can have: it looks like a search that found
nothing relevant rather than a search that did not run.

This server differs in three ways:

* it asks a **local SearXNG** instance first (no API key, no account, no third
  party service in the loop), and falls back to driving the already-installed
  Chrome through playwright, which is the path that was measured to work here;
* it **scores every result** against the query and reports `reliable: false` with
  an explicit warning when nothing matches, so a bad search can never be mistaken
  for a good one;
* it is standard-library only, so it cannot break because a dependency moved.

Protocol: MCP over stdio, newline-delimited JSON-RPC 2.0. Nothing is written to
stdout except protocol messages -- diagnostics go to stderr, because a stray
print on stdout corrupts the stream and the client reports the server as dead.

Usage (normally started by the MCP client, not by hand):
    python tools/localsearch/server.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

SERVER_NAME = "local-search"
SERVER_VERSION = "1.0.0"
PROTOCOL_VERSION = "2024-11-05"

SEARXNG_URL = os.environ.get("LOCALSEARCH_SEARXNG_URL", "http://127.0.0.1:8888")
OPEN_WEBSEARCH_ENTRY = os.environ.get(
    "LOCALSEARCH_OPEN_WEBSEARCH_ENTRY",
    os.path.join(
        os.environ.get("APPDATA", ""), "npm", "node_modules", "open-websearch", "build", "index.js"
    ),
)
NODE_BIN = os.environ.get("LOCALSEARCH_NODE", "node")
SEARCH_TIMEOUT = float(os.environ.get("LOCALSEARCH_TIMEOUT", "45"))

# Words that carry no discriminating power, so a result matching only these is not
# evidence that the search worked. Kept short on purpose: this is a smoke test for
# "did the engine answer the question", not a ranking function.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "vs", "what", "which",
    "with", "de", "la", "el", "y", "en", "的", "了", "是", "在", "和", "与",
}

TOKEN_RE = re.compile(r"[a-z0-9\u4e00-\u9fff]+")

# What counts as "the engine answered this question". A first version only required
# a non-zero score, and a real ROCm query came back at 0.18 coverage yet was labelled
# reliable -- one incidental term match was enough to pass. Requiring both a real
# share of the terms and at least two matches (when the query has two) closes that.
RELIABLE_COVERAGE = 0.4


def log(message: str) -> None:
    print(f"[{SERVER_NAME}] {message}", file=sys.stderr, flush=True)


def query_terms(query: str) -> list[str]:
    """The distinctive terms of a query, for checking that results are on topic."""
    terms = [t for t in TOKEN_RE.findall(query.lower()) if t not in STOPWORDS]
    # Single Latin characters are noise; keep short CJK runs, which are meaningful.
    return [t for t in terms if len(t) > 1 or "\u4e00" <= t[0] <= "\u9fff"]


def score_results(query: str, results: list[dict]) -> tuple[float, list[str]]:
    """Fraction of the query's distinctive terms that appear in the result set.

    A high score does not prove the results are good. A score of zero, however,
    reliably means the engine answered a different question -- which is exactly the
    failure this server exists to make visible.
    """
    terms = query_terms(query)
    if not terms:
        return 1.0, []
    blob = " ".join(
        f"{r.get('title', '')} {r.get('description', '')} {r.get('url', '')}" for r in results
    ).lower()
    missing = [t for t in terms if t not in blob]
    return (len(terms) - len(missing)) / len(terms), missing


def _http_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "local-search-mcp/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def search_searxng(query: str, limit: int, engines: list[str] | None) -> list[dict]:
    """Ask the local SearXNG instance. Raises on any failure so the caller can fall back."""
    params = {"q": query, "format": "json", "safesearch": "0"}
    if engines:
        params["engines"] = ",".join(engines)
    url = f"{SEARXNG_URL.rstrip('/')}/search?{urllib.parse.urlencode(params)}"
    payload = _http_json(url, SEARCH_TIMEOUT)
    if not isinstance(payload, dict) or "results" not in payload:
        raise RuntimeError(f"SearXNG returned no results key: {str(payload)[:200]}")
    out = []
    for item in payload["results"][:limit]:
        out.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": item.get("content", ""),
                "engine": ",".join(item.get("engines", []) or []),
            }
        )
    return out


def search_open_websearch(query: str, limit: int, engines: list[str] | None) -> list[dict]:
    """Fall back to driving Chrome through playwright, which works on this network.

    `searchMode=playwright` is forced: the request-only path is the one that
    silently returns unrelated pages, so it is never used here.
    """
    if not os.path.isfile(OPEN_WEBSEARCH_ENTRY):
        raise RuntimeError(f"open-websearch entry not found at {OPEN_WEBSEARCH_ENTRY}")
    env = dict(os.environ)
    env["SEARCH_MODE"] = "playwright"
    env.setdefault("PLAYWRIGHT_MODULE_PATH", os.path.join(os.path.dirname(OPEN_WEBSEARCH_ENTRY), "..", "..", "playwright-core"))
    env.setdefault("PLAYWRIGHT_EXECUTABLE_PATH", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    cmd = [NODE_BIN, OPEN_WEBSEARCH_ENTRY, "search", query, "--limit", str(limit), "--json"]
    if engines:
        cmd += ["--engine", ",".join(engines)]
    completed = subprocess.run(
        cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=SEARCH_TIMEOUT * 2,
    )
    stdout = completed.stdout or ""
    start = stdout.find("{")
    if start < 0:
        raise RuntimeError(f"no JSON in output: {stdout[:300]}{completed.stderr[:300]}")
    payload = json.loads(stdout[start:])
    data = payload.get("data") or {}
    if payload.get("status") != "ok":
        raise RuntimeError(f"open-websearch status={payload.get('status')} error={payload.get('error')}")
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "description": r.get("description", ""),
            "engine": r.get("engine", ""),
        }
        for r in (data.get("results") or [])[:limit]
    ]


def do_search(args: dict) -> dict:
    query = (args.get("query") or "").strip()
    if not query:
        raise ValueError("query must not be empty")
    limit = int(args.get("limit") or 8)
    engines = args.get("engines")
    if isinstance(engines, str):
        engines = [e.strip() for e in engines.split(",") if e.strip()]

    attempts: list[str] = []
    for name, fn in (("searxng", search_searxng), ("open-websearch-playwright", search_open_websearch)):
        try:
            results = fn(query, limit, engines)
        except Exception as exc:  # noqa: BLE001 -- one backend failing must not lose the other
            attempts.append(f"{name}: {type(exc).__name__}: {exc}")
            log(f"{name} failed: {type(exc).__name__}: {exc}")
            continue
        score, missing = score_results(query, results)
        matched = len(query_terms(query)) - len(missing) if query_terms(query) else 0
        reliable = bool(results) and matched >= min(2, max(1, len(query_terms(query)))) \
            and score >= RELIABLE_COVERAGE
        return {
            "backend": name,
            "query": query,
            "count": len(results),
            "reliable": reliable,
            "term_coverage": round(score, 3),
            "missing_terms": missing,
            "warning": (
                None
                if reliable
                else "The results do not cover the query (coverage "
                     f"{score:.2f}; matched {matched} of {len(query_terms(query))} distinctive "
                     "terms). An engine that answers a different question still returns pages, "
                     "so this is not an empty result set -- treat the results below as unrelated "
                     "and try different wording."
            ),
            "results": results,
            "failed_backends": attempts,
        }
    return {
        "backend": None,
        "query": query,
        "count": 0,
        "reliable": False,
        "results": [],
        "failed_backends": attempts,
        "warning": "Every search backend failed. This is an error, not an empty result set.",
    }


def strip_html(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript)\b.*?</\1>", " ", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    for entity, char in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&#39;", "'"), ("&quot;", '"')):
        html = html.replace(entity, char)
    return re.sub(r"[ \t\r\f\v]+", " ", re.sub(r"\n\s*\n+", "\n\n", html)).strip()


def do_fetch(args: dict) -> dict:
    url = (args.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError("url must be an absolute http(s) URL")
    max_chars = int(args.get("max_chars") or 30000)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept-Language": "en,zh-CN;q=0.9",
        },
    )
    with urllib.request.urlopen(request, timeout=SEARCH_TIMEOUT) as response:
        raw = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
    text = strip_html(raw.decode(charset, "replace"))
    return {"url": url, "chars": len(text), "truncated": len(text) > max_chars, "text": text[:max_chars]}


TOOLS = [
    {
        "name": "search",
        "description": (
            "Search the live web through a local SearXNG instance, falling back to a local "
            "playwright-driven browser. No API key. Every response carries `reliable` and "
            "`term_coverage`: if `reliable` is false the engine answered a different question "
            "and the results must not be used."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "limit": {"type": "number", "minimum": 1, "maximum": 50, "default": 8},
                "engines": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch",
        "description": "Fetch a URL and return its text with the HTML stripped.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_chars": {"type": "number", "default": 30000},
            },
            "required": ["url"],
        },
    },
]


def handle(request: dict) -> dict | None:
    method = request.get("method")
    request_id = request.get("id")

    if method == "initialize":
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }
    if method in ("notifications/initialized", "initialized"):
        return None
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        params = request.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            if name == "search":
                payload = do_search(args)
            elif name == "fetch":
                payload = do_fetch(args)
            else:
                raise ValueError(f"unknown tool: {name}")
        except Exception as exc:  # noqa: BLE001 -- report to the model, do not kill the server
            return {
                "content": [{"type": "text", "text": json.dumps(
                    {"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, indent=2)}],
                "isError": True,
            }
        return {
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}],
            "isError": False,
        }
    if request_id is None:
        return None
    raise ValueError(f"unsupported method: {method}")


def main() -> int:
    # Binary streams with explicit UTF-8: on Windows the default text encoding is
    # the ANSI code page, which mangles CJK queries on the way in and out.
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    while True:
        line = stdin.readline()
        if not line:
            return 0
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            log(f"ignoring unparsable line: {exc}")
            continue
        try:
            result = handle(request)
        except Exception as exc:  # noqa: BLE001
            result = None
            error = {"code": -32603, "message": f"{type(exc).__name__}: {exc}"}
        else:
            error = None
        if request.get("id") is None and result is None and error is None:
            continue
        message: dict = {"jsonrpc": "2.0", "id": request.get("id")}
        if error is not None:
            message["error"] = error
        else:
            message["result"] = result
        stdout.write(json.dumps(message, ensure_ascii=False).encode("utf-8") + b"\n")
        stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
