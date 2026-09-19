"""Download the open-source fly-brain material that grounds the domain corpus.

The teacher model writes fluent passages, but fluent is not the same as correct --
and 244 hand-prompted passages are not a corpus. This pulls the primary sources
that are actually public: the FlyWire and MaleCNS connectome releases, their
documentation, the review literature on the mushroom body, and the Chinese-language
science coverage of the same work. The text goes into `data/corpus/source/` and
becomes both training text and the grounding material the teacher is asked to
summarise and question.

Usage:
    python scripts/fetch_sources.py                 # fetch into data/corpus/source
    python scripts/fetch_sources.py --list          # show the source list
"""

from __future__ import annotations

import argparse
import html
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request

SOURCES: list[tuple[str, str]] = [
    # connectome releases and their documentation
    ("flywire_home", "https://flywire.ai/"),
    ("flywire_apps", "https://flywire.ai/apps"),
    ("flywire_annotations_readme", "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/README.md"),
    ("flywire_codex_readme", "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/CODE_OF_CONDUCT.md"),
    ("maleCNS_janelia", "https://www.janelia.org/project-team/flyem/male-cns"),
    ("neuromancer_docs", "https://raw.githubusercontent.com/schlegelp/neuprint-python/master/README.md"),
    # encyclopedic grounding
    ("wiki_mushroom_body", "https://en.wikipedia.org/wiki/Mushroom_body"),
    ("wiki_drosophila", "https://en.wikipedia.org/wiki/Drosophila_melanogaster"),
    ("wiki_connectome", "https://en.wikipedia.org/wiki/Connectome"),
    ("wiki_flywire", "https://en.wikipedia.org/wiki/FlyWire"),
    ("wiki_olfaction_fly", "https://en.wikipedia.org/wiki/Insect_olfactory_system"),
    ("wiki_spiking_neuron", "https://en.wikipedia.org/wiki/Biological_neuron_model"),
    ("wiki_leaky_integrate", "https://en.wikipedia.org/wiki/Leaky_integrate-and-fire"),
    ("wiki_zh_drosophila", "https://zh.wikipedia.org/wiki/%E9%BB%91%E8%85%B9%E6%9E%9C%E8%9D%87"),
    ("wiki_zh_connectome", "https://zh.wikipedia.org/wiki/%E7%A5%9E%E7%BB%8F%E8%BF%9E%E6%8E%A5%E7%BB%84%E5%AD%A6"),
    ("wiki_zh_neuron", "https://zh.wikipedia.org/wiki/%E7%A5%9E%E7%BB%8F%E5%85%83"),
    ("wiki_zh_transformer", "https://zh.wikipedia.org/wiki/Transformer%E6%A8%A1%E5%9E%8B"),
    ("wiki_zh_language_model", "https://zh.wikipedia.org/wiki/%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B"),
    # Open-access papers on the mushroom body. The three IDs used before this were
    # guessed and resolved to unrelated articles (mood-disorder biomarkers, an
    # epidemiology paper); these four were checked against their search results.
    ("pmc_kc_synaptic_currents", "https://pmc.ncbi.nlm.nih.gov/articles/PMC6740836/"),
    ("pmc_sparse_wiring", "https://pmc.ncbi.nlm.nih.gov/articles/PMC7028369/"),
    ("pmc_pn_kc_sampling", "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9413950/"),
    ("pmc_mb_strains", "https://pmc.ncbi.nlm.nih.gov/articles/PMC3779816/"),
    # Surrogate-gradient learning in spiking networks -- the method this project's
    # neuron model is trained with. (An earlier entry pointed at arXiv 1903.02537,
    # which is a quantum channel-decoding paper; wrong-paper text is worse than no
    # text, because the corpus has no way to notice.)
    ("arxiv_surrogate_gradient", "https://arxiv.org/abs/1901.09948"),
    ("arxiv_snn_review", "https://arxiv.org/abs/2109.12894"),
    ("biorxiv_flywire", "https://www.biorxiv.org/content/10.1101/2023.06.27.546658v1.full"),
    ("flywire_codex", "https://codex.flywire.ai/"),
    ("flywire_blog", "https://blog.flywire.ai/"),
    ("janelia_flyem", "https://www.janelia.org/project-team/flyem"),
    # Chinese-language coverage of the same work
    ("ebiotrade_mb_ai", "https://www.ebiotrade.com/newsf/2025-11/20251101083610735.htm"),
    ("ebiotrade_mb_lnet", "https://news.ebiotrade.com/2025-5/20250531074933477.htm"),
    ("zhihu_flywire_cell", "https://zhuanlan.zhihu.com/p/40242862"),
    ("flywire_neuronlp", "https://flywire.neuronlp.fruitflybrain.org/"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def html_to_text(raw: str) -> str:
    """Strip markup, scripts and styles down to readable prose."""
    raw = re.sub(r"(?is)<(script|style|nav|footer|header|form|svg)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?is)<!--.*?-->", " ", raw)
    raw = re.sub(r"(?is)<br\s*/?>|</p>|</div>|</li>|</h[1-6]>", "\n", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    text = html.unescape(raw)
    lines = [re.sub(r"[ \t\u00a0]+", " ", line).strip() for line in text.split("\n")]
    kept = [line for line in lines if len(line) > 30]
    out: list[str] = []
    for line in kept:
        if out and out[-1] == line:
            continue
        out.append(line)
    return "\n".join(out)


def _open(url: str, timeout: float, verify: bool, proxy: str = ""):
    context = ssl.create_default_context()
    if not verify:
        # This network terminates TLS with a self-signed certificate in front of
        # several of the hosts we need (PMC, GitHub raw). Retrying unverified is the
        # difference between a corpus and no corpus; nothing here is authenticated.
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    handlers: list[urllib.request.BaseHandler] = [urllib.request.ProxyHandler(
        {"http": proxy, "https": proxy} if proxy else {}
    )]
    if not verify:
        handlers.append(urllib.request.HTTPSHandler(context=context))
    opener = urllib.request.build_opener(*handlers)
    request = urllib.request.Request(url, headers=HEADERS)
    return opener.open(request, timeout=timeout)


def fetch(url: str, timeout: float = 30.0, attempts: int = 2, proxy: str = "") -> str:
    last: Exception | None = None
    for attempt in range(attempts):
        for verify in (True, False):
            try:
                with _open(url, timeout, verify, proxy) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    body = response.read()
                break
            except (ssl.SSLError, urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
        else:
            time.sleep(1.5 * (attempt + 1))
            continue
        break
    else:
        raise last if last else RuntimeError("unreachable")
    text = body.decode(charset, errors="replace")
    if "<html" in text.lower()[:2000] or "<div" in text[:2000]:
        return html_to_text(text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/corpus/source")
    parser.add_argument("--min-chars", type=int, default=1500)
    parser.add_argument("--only", default="", help="comma-separated source names to retry")
    parser.add_argument(
        "--proxy",
        default=os.environ.get("FLYBRAIN_PROXY", "http://127.0.0.1:6696"),
        help="HTTP proxy for the fetches; pass an empty string to go direct",
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name, url in SOURCES:
            print(f"{name:28s} {url}")
        return 0

    os.makedirs(args.out, exist_ok=True)
    wanted = {name.strip() for name in args.only.split(",") if name.strip()}
    planned = [(n, u) for n, u in SOURCES if not wanted or n in wanted]
    ok, failed = 0, []
    index: list[str] = []
    for name, url in planned:
        target = os.path.join(args.out, f"{name}.txt")
        try:
            text = fetch(url, proxy=args.proxy)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ssl.SSLError) as exc:
            failed.append((name, url, str(exc)[:90]))
            print(f"  FAIL {name:28s} {str(exc)[:70]}")
            continue
        if len(text) < args.min_chars:
            failed.append((name, url, f"too short ({len(text)} chars)"))
            print(f"  SKIP {name:28s} only {len(text)} chars")
            continue
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(f"# source: {url}\n\n{text}\n")
        index.append(f"{name}\t{url}\t{len(text)}")
        ok += 1
        print(f"  ok   {name:28s} {len(text):>7,} chars")
        time.sleep(0.4)

    with open(os.path.join(args.out, "SOURCES.tsv"), "a" if wanted else "w", encoding="utf-8") as fh:
        fh.write("\n".join(index) + ("\n" if index else ""))
    print(f"\n{ok} sources written to {args.out}")
    for name, url, why in failed:
        print(f"  failed: {name} ({why})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
