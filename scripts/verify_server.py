"""Exercise the chat server the way a browser would, without a browser.

Checks the page, the health endpoint, and a full streamed chat turn: that the SSE
frames arrive incrementally, that the reply is non-empty, and that the server
returns updated conversation history so the next turn has context.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8011"


def get(path: str) -> tuple[int, str]:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=60) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")


def main() -> int:
    # Wait for the server to come up.
    for attempt in range(60):
        try:
            status, body = get("/api/health")
            break
        except Exception as exc:
            if attempt == 59:
                print(f"server never came up: {exc}")
                return 1
            time.sleep(2)
    print(f"GET /api/health -> {status}")
    health = json.loads(body)
    print(f"  {json.dumps(health, ensure_ascii=False)}")
    assert health["status"] == "ok"
    assert health["params"] > 0

    status, page = get("/")
    print(f"GET / -> {status}, {len(page)} bytes")
    assert "FlyBrain-LLM" in page
    assert "/api/chat" in page, "the page does not wire up the chat endpoint"

    payload = json.dumps(
        {"message": "苍蝇的大脑是什么？", "history": [], "temperature": 0.8, "max_new_tokens": 24}
    ).encode()
    request = urllib.request.Request(
        f"{BASE}/api/chat", data=payload, headers={"Content-Type": "application/json"}
    )

    print("POST /api/chat (streaming)")
    deltas: list[str] = []
    frames = 0
    history_after = None
    started = time.perf_counter()
    first_delta_at = None
    with urllib.request.urlopen(request, timeout=600) as resp:
        print(f"  status {resp.status}, content-type {resp.headers.get('Content-Type')}")
        assert resp.headers.get("Content-Type", "").startswith("text/event-stream")
        buffer = ""
        while True:
            block = resp.read(256)
            if not block:
                break
            buffer += block.decode("utf-8", "replace")
            while "\n\n" in buffer:
                frame, _, buffer = buffer.partition("\n\n")
                for line in frame.splitlines():
                    if not line.startswith("data: "):
                        continue
                    frames += 1
                    event = json.loads(line[6:])
                    if event.get("delta"):
                        if first_delta_at is None:
                            first_delta_at = time.perf_counter() - started
                        deltas.append(event["delta"])
                    if event.get("error"):
                        print(f"  server error: {event['error']}")
                        return 1
                    if event.get("done"):
                        history_after = event.get("history")

    reply = "".join(deltas)
    elapsed = time.perf_counter() - started
    print(f"  frames {frames}, first token after {first_delta_at:.2f}s, total {elapsed:.2f}s")
    print(f"  reply ({len(reply)} chars): {reply[:120]!r}")
    assert frames > 1, "the response was not streamed frame by frame"
    assert first_delta_at is not None, "no token deltas arrived"
    assert history_after is not None, "the server did not return the updated history"
    assert len(history_after) >= 2, f"history should hold the turn: {history_after}"

    # A second turn must carry the first one as context.
    payload = json.dumps(
        {
            "message": "刚才我问了什么？",
            "history": history_after,
            "temperature": 0.8,
            "max_new_tokens": 16,
        }
    ).encode()
    request = urllib.request.Request(
        f"{BASE}/api/chat", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as resp:
        second = resp.read().decode("utf-8", "replace")
    print(f"  second turn streamed {len(second)} bytes, history echoed back: "
          f"{'yes' if 'history' in second else 'no'}")
    assert "history" in second

    print("\nweb chat verified end to end (page, health, streaming, multi-turn history)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
