"""DeepSeek API as a teacher for building the domain corpus.

The API cannot train a model -- it only does inference -- so its role here is
the one that actually transfers knowledge: generating the fly-brain-domain text
that the local model is then trained on (knowledge distillation into a small
model's own weights). Everything the teacher writes is stored, versioned and
reusable, so a corpus can be extended without paying for it twice.

The key is read from the DEEPSEEK_API_KEY environment variable.
"""

from __future__ import annotations

import json
import os
import concurrent.futures
import random
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from flybrain.data import is_domain_text
from typing import Iterable, Iterator

API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
# `deepseek-flash` is a reasoning model: it emits a long `reasoning_content`
# before the answer, so the token budget has to cover both. A small budget makes
# it return an empty answer with finish_reason "length".
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")


class TeacherError(RuntimeError):
    pass


@dataclass
class TeacherConfig:
    model: str = DEFAULT_MODEL
    temperature: float = 1.0
    # 2048 is deliberate, not arbitrary. Measured on this model, raising the budget
    # from 2048 to 4096 tripled the latency (8.8s to 21.8s), tripled the reasoning
    # tokens (1214 to 3650) and produced a *shorter* answer. A larger budget makes
    # this model think longer without writing more.
    max_tokens: int = 2048
    retries: int = 3
    timeout: float = 300.0
    api_base: str = API_BASE


def api_key(explicit: str | None = None) -> str:
    key = explicit or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise TeacherError(
            "no DeepSeek API key: set DEEPSEEK_API_KEY in the environment "
            "(the API is only used to generate teacher text, never to train)"
        )
    return key


def chat(
    messages: list[dict],
    *,
    key: str | None = None,
    cfg: TeacherConfig | None = None,
    json_mode: bool = False,
) -> str:
    """One chat completion, with retries and exponential backoff.

    Returns the assistant's answer. Reasoning models also return a
    `reasoning_content` field; that is deliberately not part of the returned text,
    because the corpus should contain answers, not the teacher's private scratch
    work. `chat_detailed` exposes it when it is wanted.
    """
    return chat_detailed(messages, key=key, cfg=cfg, json_mode=json_mode)["content"]


def chat_detailed(
    messages: list[dict],
    *,
    key: str | None = None,
    cfg: TeacherConfig | None = None,
    json_mode: bool = False,
) -> dict:
    """One chat completion returning content, reasoning and finish reason."""
    cfg = cfg or TeacherConfig()
    last_error: Exception | None = None
    budget = cfg.max_tokens
    # Escalate at most once, and only to twice the configured budget. Compounding
    # the multiplier on every retry produced 55k-token requests that still came
    # back empty, which burns tokens without producing corpus text.
    escalation_cap = cfg.max_tokens * 2
    escalations = 0

    for attempt in range(cfg.retries):
        payload: dict = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": budget,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{cfg.api_base}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key(key)}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choice = data["choices"][0]
            message = choice["message"]
            content = (message.get("content") or "").strip()
            finish = choice.get("finish_reason")
            if not content and finish == "length":
                if escalations < 1 and budget < escalation_cap:
                    budget = escalation_cap
                    escalations += 1
                    last_error = TeacherError(
                        f"empty answer at max_tokens={budget // 2}; retrying once at {budget}"
                    )
                    time.sleep(1.0 + random.random())
                    continue
                raise TeacherError(
                    "empty answer with finish_reason=length even at "
                    f"max_tokens={budget}: the reasoning pass consumed the whole budget"
                )
            return {
                "content": content,
                "reasoning": (message.get("reasoning_content") or "").strip(),
                "finish_reason": finish,
                "usage": data.get("usage", {}),
                "model": cfg.model,
            }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            last_error = TeacherError(f"HTTP {exc.code}: {detail}")
            if exc.code in (400, 401, 403, 404):
                raise last_error from exc  # not worth retrying
        except Exception as exc:  # timeouts, connection resets, bad JSON
            last_error = exc
        time.sleep(min(2**attempt, 20) + random.random())
    raise TeacherError(f"teacher call failed after {cfg.retries} attempts: {last_error}")


def probe(key: str | None = None) -> dict:
    """Check the key works and report which models it can see."""
    req = urllib.request.Request(
        f"{API_BASE}/models", headers={"Authorization": f"Bearer {api_key(key)}"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


DOMAIN_SYSTEM = (
    "你是一位果蝇（Drosophila melanogaster）神经科学与连接组学专家，"
    "同时精通机器学习。你写作准确、具体，会给出数字与术语，但用清晰的中文解释。"
)

DOMAIN_TASKS: tuple[tuple[str, str], ...] = (
    (
        "概念讲解",
        "请讲解一个果蝇大脑的关键结构或概念：{topic}。"
        "要求：先给出定义，再说明它在神经环路中的位置与连接关系，"
        "给出具体的神经元数量或突触数量（如果已知），最后用一段话总结它的功能。",
    ),
    (
        "连接组科普",
        "请写一段科普文字，主题是：{topic}。面向有理工背景但非神经科学的读者，"
        "说明研究方法和已知结论，并指出哪些结论仍然不确定。",
    ),
    (
        "问答",
        "请针对主题「{topic}」提出一个研究生水平的问答对，"
        "回答需要有推导或数据支撑，约 200 字。",
    ),
    (
        "机器学习类比",
        "请讨论：果蝇的{topic}与人工神经网络中的哪个机制可以类比，"
        "两者的关键差别是什么？避免过度类比。",
    ),
)

DOMAIN_TOPICS: tuple[str, ...] = (
    "蘑菇体（mushroom body）与稀疏编码",
    "Kenyon 细胞的数量与稀疏发放",
    "投射神经元与触角叶的球状体",
    "APL 神经元提供的全局抑制",
    "MBON 与记忆读出",
    "多巴胺能神经元 PAM 与 PPL1 的奖惩信号",
    "FlyWire 全脑连接组数据集的构建过程",
    "Janelia 雄性中枢神经系统连接组（MaleCNS）",
    "全脑约 14 万个神经元的规模",
    "果蝇视觉系统的运动检测通路 T4/T5",
    "嗅觉受体与气味编码",
    "果蝇的昼夜节律神经网络",
    "连接组约束的神经网络模型",
    "脉冲神经网络与替代梯度",
    "稀疏扩张编码为何有利于模式分离",
    "苍蝇的逃逸反应与巨型纤维系统",
    "果蝇的学习与记忆实验范式",
    "连接组数据的图论分析方法",
    "神经元形态分类与 cell type 的判定",
    "果蝇大脑的神经递质类型与兴奋抑制平衡",
)



# The teacher calls are network-bound and each takes tens of seconds, so a corpus
# is built with a small pool of concurrent requests. Measured single-call latency
# for one explanation is ~9s; a serial build of a few hundred records would take
# hours of mostly waiting.
DEFAULT_WORKERS = 6


def _run_concurrent(
    out_path: str,
    *,
    n_records: int,
    build_task,
    build_record,
    cfg: TeacherConfig,
    key: str | None,
    seed: int,
    resume: bool,
    workers: int,
    label: str,
) -> int:
    """Drive teacher calls concurrently and append each record as it lands.

    Results are written under a lock and flushed immediately, so interrupting the
    run keeps every completed record and the next run resumes from there.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = 0
    if resume and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            done = sum(1 for line in fh if line.strip())
        print(f"resuming: {done} {label} already in {out_path}")
    if done >= n_records:
        print(f"{out_path} already holds {done} {label} (target {n_records})")
        return done

    lock = threading.Lock()
    written = 0
    failed = 0

    def work(index: int):
        # Seeded per index so the corpus is reproducible regardless of completion
        # order -- concurrency must not change what gets generated. The seed is a
        # string because Python's random module only accepts None/int/float/str/
        # bytes/bytearray, not the tuple this used to pass.
        rng = random.Random(f"{seed}:{index}")
        context = build_task(rng)
        reply = chat_detailed(
            [
                {"role": "system", "content": DOMAIN_SYSTEM},
                {"role": "user", "content": context["prompt"]},
            ],
            key=key,
            cfg=cfg,
        )
        return build_record(context, reply, cfg)

    remaining = n_records - done
    with open(out_path, "a", encoding="utf-8") as fh:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(work, index): index for index in range(done, n_records)}
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                try:
                    record = future.result()
                except Exception as exc:  # one bad record must not sink the corpus
                    failed += 1
                    print(f"  {label[:-1]} {index} failed: {type(exc).__name__}: {str(exc)[:150]}")
                    continue
                if record is None:
                    failed += 1
                    continue
                with lock:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fh.flush()
                    written += 1
                    if written % 10 == 0:
                        print(f"  {written}/{remaining} new {label}", flush=True)
    print(f"wrote {written} new {label}, {failed} failed")
    return done + written


def _domain_task(rng: random.Random) -> dict:
    topic = rng.choice(DOMAIN_TOPICS)
    label, template = rng.choice(DOMAIN_TASKS)
    return {"topic": topic, "task": label, "prompt": template.format(topic=topic)}


def _domain_record(context: dict, reply: dict, cfg: TeacherConfig) -> dict | None:
    if not reply["content"]:
        return None
    return {
        "text": reply["content"],
        "topic": context["topic"],
        "task": context["task"],
        "prompt": context["prompt"],
        "model": cfg.model,
        "teacher": "deepseek",
        "reasoning_chars": len(reply["reasoning"]),
        "usage": reply["usage"],
    }


def _qa_task(rng: random.Random) -> dict:
    topic = rng.choice(DOMAIN_TOPICS)
    question = rng.choice(
        (
            f"苍蝇大脑里的{topic}是什么？",
            f"请解释{topic}，说清楚它在环路里的作用。",
            f"关于{topic}，目前我们知道什么、还不知道什么？",
            f"如果要用机器学习复现{topic}的机制，应该怎么做？",
        )
    )
    return {
        "topic": topic,
        "question": question,
        "prompt": (
            f"请围绕「{topic}」写一段多轮对话，用户是学习神经科学的工程师，"
            "共 3 轮问答，回答准确、有具体数字。"
            "严格按照每行以 <user> 或 <assistant> 开头的格式输出，不要输出其它内容。"
        ),
    }


def _qa_record(context: dict, reply: dict, cfg: TeacherConfig) -> dict | None:
    text = reply["content"]
    if not text:
        return None
    turns: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        for role_tag, role in (("<user>", "user"), ("<assistant>", "assistant")):
            if line.startswith(role_tag):
                content = line[len(role_tag) :].strip()
                if content:
                    turns.append({"role": role, "content": content})
    if not turns:
        # The teacher ignored the requested format; keep the exchange usable
        # rather than discarding a paid-for completion.
        turns = [
            {"role": "user", "content": context["question"]},
            {"role": "assistant", "content": text},
        ]
    return {
        "messages": turns,
        "topic": context["topic"],
        "teacher": "deepseek",
        "model": cfg.model,
        "reasoning_chars": len(reply["reasoning"]),
    }


def generate_domain_corpus(
    out_path: str,
    *,
    n_records: int,
    key: str | None = None,
    cfg: TeacherConfig | None = None,
    seed: int = 0,
    resume: bool = True,
    workers: int = DEFAULT_WORKERS,
) -> int:
    """Teacher-written explanations of fly brain topics, as JSONL records."""
    return _run_concurrent(
        out_path,
        n_records=n_records,
        build_task=_domain_task,
        build_record=_domain_record,
        cfg=cfg or TeacherConfig(),
        key=key,
        seed=seed,
        resume=resume,
        workers=workers,
        label="records",
    )


def generate_qa_corpus(
    out_path: str,
    *,
    n_records: int,
    key: str | None = None,
    cfg: TeacherConfig | None = None,
    seed: int = 0,
    resume: bool = True,
    workers: int = DEFAULT_WORKERS,
) -> int:
    """Teacher-written dialogues in the chat format the local model is trained on.

    This is what makes the trained model usable in a conversational interface.
    """
    return _run_concurrent(
        out_path,
        n_records=n_records,
        build_task=_qa_task,
        build_record=_qa_record,
        cfg=cfg or TeacherConfig(),
        key=key,
        seed=seed + 991,
        resume=resume,
        workers=workers,
        label="dialogues",
    )


# --------------------------------------------------------------------------
# document-grounded generation
# --------------------------------------------------------------------------
# Asking the teacher to write about the fly brain from its own memory produces
# fluent text of uncertain accuracy, and the free-form dialogue mode cannot
# produce anything at all on this model: the reasoning pass consumes the whole
# token budget before the answer starts. Grounding every record in a passage from
# a public source fixes both -- the facts come from the source, and the answer
# only has to restate what is in front of it, which fits in the budget.
DOC_TASKS: list[tuple[str, str]] = [
    (
        "docqa",
        "下面是一段关于果蝇脑的公开资料。请根据它写 2 轮问答，提问者是学习神经科学的工程师。"
        "回答必须严格基于资料内容，可以用资料里的数字。"
        "只输出以 <user> 或 <assistant> 开头的行，每行一整句，不要输出其它内容：\n\n{passage}",
    ),
    (
        "docqa",
        "下面是一段果蝇脑研究资料。请据此写 2 轮问答：第一轮问概念，第二轮问机制或数字。"
        "只输出以 <user> 或 <assistant> 开头的行，不要输出其它内容：\n\n{passage}",
    ),
    (
        "chinese",
        "下面是一段英文的果蝇脑研究资料。请用中文完整地重述它的内容，"
        "术语保留英文原名并在括号里给出中文，数字必须与资料一致：\n\n{passage}",
    ),
    (
        "chinese",
        "把下面这段英文资料翻译成准确的中文，保留所有数字与专有名词，不要增删内容：\n\n{passage}",
    ),
    (
        "summary",
        "下面是一段关于果蝇脑与连接组的公开资料。请把它改写成一段准确、通顺的中文说明，"
        "保留其中的具体数字、脑区名称和神经元名称，不要添加资料里没有的信息：\n\n{passage}",
    ),
    (
        "mechanism",
        "下面是一段关于果蝇脑的公开资料。请解释资料中出现的神经环路或计算方法，"
        "说明它在整个系统里起什么作用，并引用资料给出的关键数字：\n\n{passage}",
    ),
]


def load_passages(
    source_dir: str, *, min_chars: int = 300, max_chars: int = 900
) -> list[dict]:
    """Chunks of the downloaded primary sources, each tagged with its origin.

    Sources are paragraph-chunked rather than split at a fixed width so a passage
    is a coherent piece of prose; the URL in the file header travels with the
    chunk so every generated record can be traced back to what it was grounded in.
    """
    passages: list[dict] = []
    for name in sorted(os.listdir(source_dir)):
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(source_dir, name), encoding="utf-8") as fh:
            raw = fh.read()
        origin = ""
        for line in raw.splitlines()[:3]:
            if line.startswith("# source:"):
                origin = line.split(":", 1)[1].strip()
                break
        body = "\n".join(line for line in raw.splitlines() if not line.startswith("# source:"))
        buffer = ""
        for paragraph in re.split(r"\n\s*\n|\n(?=[A-Z\u4e00-\u9fff])", body):
            paragraph = " ".join(paragraph.split())
            if len(paragraph) < 60:
                continue
            if len(buffer) + len(paragraph) + 1 <= max_chars:
                buffer = f"{buffer} {paragraph}".strip()
                continue
            if len(buffer) >= min_chars:
                passages.append({"source": name, "origin": origin, "text": buffer})
            buffer = paragraph
        if len(buffer) >= min_chars:
            passages.append({"source": name, "origin": origin, "text": buffer})
    kept = [p for p in passages if is_domain_text(p["text"])]
    dropped = len(passages) - len(kept)
    if dropped:
        print(f"dropped {dropped} passages that are not about the fly brain")
    return kept


def make_doc_task(passages: list[dict]):
    """A task builder that grounds each record in a different real passage."""

    def build(rng: random.Random) -> dict:
        passage = passages[rng.randrange(len(passages))]
        label, template = rng.choice(DOC_TASKS)
        return {
            "topic": passage["source"],
            "task": label,
            "source": passage["origin"],
            "passage": passage["text"],
            "prompt": template.format(passage=passage["text"]),
        }

    return build


def _doc_record(context: dict, reply: dict, cfg: TeacherConfig) -> dict | None:
    text = reply["content"]
    if not text:
        return None
    record = {
        "topic": context["topic"],
        "task": context["task"],
        "source": context["source"],
        "teacher": "deepseek",
        "model": cfg.model,
        "reasoning_chars": len(reply["reasoning"]),
    }
    if context["task"] == "docqa":
        turns: list[dict] = []
        for line in text.splitlines():
            line = line.strip()
            for role_tag, role in (("<user>", "user"), ("<assistant>", "assistant")):
                if line.startswith(role_tag):
                    content = line[len(role_tag) :].strip()
                    if content:
                        turns.append({"role": role, "content": content})
        if any(turn["role"] == "user" for turn in turns) and len(turns) >= 2:
            record["messages"] = turns
            return record
        # The teacher ignored the format; keep the exchange rather than dropping a
        # paid-for completion, as the free-form mode does.
    record["text"] = text
    record["prompt"] = context["prompt"].split("\n\n")[0]
    return record


def generate_doc_corpus(
    out_path: str,
    *,
    source_dir: str,
    n_records: int,
    key: str | None = None,
    cfg: TeacherConfig | None = None,
    seed: int = 0,
    resume: bool = True,
    workers: int = DEFAULT_WORKERS,
) -> int:
    """Teacher-written passages and dialogues grounded in the downloaded sources."""
    passages = load_passages(source_dir)
    if not passages:
        raise SystemExit(f"no source passages under {source_dir}; run scripts/fetch_sources.py")
    print(f"grounding on {len(passages)} passages from {source_dir}")
    return _run_concurrent(
        out_path,
        n_records=n_records,
        build_task=make_doc_task(passages),
        build_record=_doc_record,
        cfg=cfg or TeacherConfig(),
        key=key,
        seed=seed + 2027,
        resume=resume,
        workers=workers,
        label="grounded records",
    )
