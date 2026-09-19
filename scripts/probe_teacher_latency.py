"""Time a single teacher call, so the corpus budget can be planned.

Reasoning models spend a lot of tokens thinking before answering, which sets the
wall-clock cost per record. Measuring it beats guessing.
"""

from __future__ import annotations

import os
import sys
import time

from flybrain.teacher import TeacherConfig, chat_detailed

MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")


def main() -> int:
    prompt = "请讲解果蝇蘑菇体的稀疏编码机制，要求给出具体的神经元数量与发放比例，约 250 字。"
    for max_tokens in (2048, 4096):
        cfg = TeacherConfig(model=MODEL, max_tokens=max_tokens)
        started = time.perf_counter()
        try:
            reply = chat_detailed([{"role": "user", "content": prompt}], cfg=cfg)
        except Exception as exc:
            print(f"max_tokens={max_tokens}: FAILED {type(exc).__name__}: {str(exc)[:200]}")
            continue
        elapsed = time.perf_counter() - started
        usage = reply["usage"]
        print(
            f"max_tokens={max_tokens}: {elapsed:6.1f}s  answer {len(reply['content'])} chars  "
            f"reasoning {len(reply['reasoning'])} chars  finish={reply['finish_reason']}  "
            f"usage={usage.get('completion_tokens')} completion "
            f"({usage.get('completion_tokens_details', {}).get('reasoning_tokens')} reasoning)"
        )
        print(f"  answer: {reply['content'][:150]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
