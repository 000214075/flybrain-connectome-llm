"""Tests for the local search server's reliability verdict.

The point of that server is that a search which answered a *different* question must
never look like a search that found nothing, so the scoring is the part worth
pinning down. These call the module directly and touch no network, so they belong in
the normal suite; the end-to-end check over the real stdio protocol is a separate
script (`tools/localsearch/selftest.py`) because it needs a browser and a live
network.

Run with: .venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import importlib.util
import os

import pytest

MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "localsearch", "server.py",
)


def load_server():
    spec = importlib.util.spec_from_file_location("local_search_server", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


server = load_server()


def test_results_that_do_not_mention_the_query_are_flagged_unreliable():
    """This is the failure the server exists to catch.

    The real incident was a highly specific query answered with generic pages about
    the organism in it, reported as success. A user cannot tell that from "nothing
    relevant exists", so the verdict has to be computed rather than assumed.
    """
    query = "Drosophila mushroom body connectome language model"
    generic = [
        {"title": "Drosophila - Wikipedia", "url": "https://en.wikipedia.org/wiki/Drosophila",
         "description": "Drosophila is a genus of flies."},
        {"title": "FlyBase Homepage", "url": "https://flybase.org/", "description": "A database."},
    ]
    score, missing = server.score_results(query, generic)
    assert score < 0.5
    assert "connectome" in missing and "mushroom" in missing


def test_results_that_match_the_query_are_not_flagged():
    query = "Drosophila mushroom body connectome language model"
    # A partial match is expected and allowed: half of these terms are enough for the
    # verdict to be `reliable`, because the check is "did the engine answer this
    # question", not "is every result perfect".
    good = [
        {"title": "Wiring the Drosophila mushroom body connectome into a language model",
         "url": "https://example.invalid/paper",
         "description": "we use the mushroom body connectome as the substrate of a language model"},
    ]
    score, missing = server.score_results(query, good)
    assert score > 0.8, missing
    assert missing == []


def test_stopwords_alone_do_not_make_a_result_relevant():
    """A stopword match must not be able to certify a result.

    "the" and "of" appear in almost every page; counting them would let an unrelated
    result pass the check and reintroduce the silent failure.
    """
    terms = server.query_terms("the connectome of the fly")
    assert "the" not in terms and "of" not in terms
    assert set(terms) == {"connectome", "fly"}


def test_a_cjk_query_is_tokenised_rather_than_dropped():
    """Chinese characters are single tokens, so a length filter written for Latin
    text would silently erase the whole query and score every result as relevant."""
    terms = server.query_terms("苍蝇大脑 连接组 语言模型")
    assert terms, "a CJK query must produce at least one term"
    score, _ = server.score_results("苍蝇大脑 连接组 语言模型", [
        {"title": "果蝇全脑连接组接入大语言模型", "url": "u", "description": "把全脑连接组接入语言模型"},
    ])
    assert score > 0


def test_a_low_but_nonzero_coverage_is_still_unreliable(monkeypatch):
    """One incidental term match must not certify a search.

    Found in real use: a ROCm query about driver crashes returned AMD's generic
    download pages, covered 0.18 of the terms, and was reported `reliable` because
    the first version of the check only asked for a non-zero score.
    """
    query = "ROCm Windows HIP training crash device lost driver recovery large model"
    results = [
        {"title": "AMD ROCm Software", "url": "https://www.amd.com/rocm",
         "description": "ROCm is AMD's open-source GPU computing platform"},
        {"title": "ROCm 百度百科", "url": "https://baike.baidu.com/item/ROCm",
         "description": "ROCm software ecosystem"},
    ]
    score, missing = server.score_results(query, results)
    matched = len(server.query_terms(query)) - len(missing)
    assert matched >= 1, "the sample must match something, or this tests the wrong thing"
    assert score < server.RELIABLE_COVERAGE

    def backend(query, limit, engines):
        return results

    monkeypatch.setattr(server, "search_searxng", backend)
    monkeypatch.setattr(server, "search_open_websearch", backend)
    payload = server.do_search({"query": query, "limit": 5})
    assert payload["count"] == 2, "the results are still returned..."
    assert payload["reliable"] is False, "...but they are not certified"
    assert "coverage" in payload["warning"]


def test_an_empty_result_set_is_not_reported_as_reliable():
    """Zero results must read as a failure, never as "the search worked"."""
    score, missing = server.score_results("connectome language model", [])
    assert score == 0.0
    assert missing


def test_a_search_with_no_backend_available_reports_the_reason(monkeypatch):
    """Every backend failing is an error with a reason, not an empty result set."""
    def explode(*_args, **_kwargs):
        raise RuntimeError("backend deliberately unavailable")

    monkeypatch.setattr(server, "search_searxng", explode)
    monkeypatch.setattr(server, "search_open_websearch", explode)
    payload = server.do_search({"query": "connectome language model", "limit": 3})
    assert payload["reliable"] is False
    assert payload["count"] == 0
    assert len(payload["failed_backends"]) == 2
    assert "not an empty result set" in payload["warning"]


def test_the_first_working_backend_wins_and_later_ones_are_not_tried(monkeypatch):
    """SearXNG is preferred; the browser path is a fallback, not a second query."""
    calls: list[str] = []

    def searxng(query, limit, engines):
        calls.append("searxng")
        return [{"title": "connectome language model fly brain", "url": "u", "description": "connectome"}]

    def never(query, limit, engines):
        calls.append("open-websearch")

    monkeypatch.setattr(server, "search_searxng", searxng)
    monkeypatch.setattr(server, "search_open_websearch", never)
    payload = server.do_search({"query": "connectome language model", "limit": 3})
    assert calls == ["searxng"]
    assert payload["backend"] == "searxng"
    assert payload["reliable"] is True


@pytest.mark.parametrize("query", ["", "   "])
def test_an_empty_query_is_rejected_rather_than_searched(query):
    with pytest.raises(ValueError):
        server.do_search({"query": query})
