"""Tests for the subagent-model consistency checker.

The checker exists because a `[subagents.models]` pin in config.toml made every
general-purpose subagent run on a different model from the session that spawned it,
and nothing in the session's normal output showed it. Two properties decide whether
its verdict can be trusted, so those are what these tests pin down: that a config
alias and the underlying model name it points at compare equal (a session records
`deepseek-flash` while a subagent can record `deepseek-v4-flash`), and that an
unreadable record is surfaced rather than silently passed.

Run with: .venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import importlib.util
import json
import os

MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "check_subagent_models.py",
)


def load_checker():
    spec = importlib.util.spec_from_file_location("check_subagent_models", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


checker = load_checker()

ALIAS_CONFIG = """
[models]
default = "deepseek-flash"

[model.deepseek-flash]
model = "deepseek-v4-flash"

[model."mimo-v2.5"]
model = "mimo-v2.5"
"""


def build_home(tmp_path, *, session_model, subagent_models, config=ALIAS_CONFIG,
               updated="2026-09-19T01:00:00Z"):
    """A synthetic grok home with one session and one meta.json per subagent."""
    home = tmp_path / "grokhome"
    session = home / "sessions" / "proj" / "01a0aaaa-0000-7000-8000-000000000001"
    (session / "subagents").mkdir(parents=True)
    if config is not None:
        (home / "config.toml").write_text(config, encoding="utf-8")
    (session / "summary.json").write_text(json.dumps({
        "info": {"id": session.name, "cwd": "C:\\work"},
        "current_model_id": session_model,
        "updated_at": updated,
    }), encoding="utf-8")
    for index, model in enumerate(subagent_models):
        record_dir = session / "subagents" / f"01a0bbbb-0000-7000-8000-{index:012d}"
        record_dir.mkdir()
        if model is None:
            (record_dir / "meta.json").write_text("{ truncated", encoding="utf-8")
            continue
        (record_dir / "meta.json").write_text(json.dumps({
            "subagent_id": record_dir.name,
            "subagent_type": "general-purpose",
            "status": "completed",
            "description": f"job{index}",
            "started_at": "2026-09-19T01:00:00Z",
            "effective_model_id": model,
        }), encoding="utf-8")
    return home, session


def test_an_alias_and_its_underlying_model_are_the_same_model(tmp_path):
    """The session records the alias, the subagent records the API model name."""
    home, session = build_home(
        tmp_path, session_model="deepseek-v4-flash", subagent_models=["deepseek-flash"],
    )

    result = checker.check_session(session, checker.load_alias_map(home))

    assert result["session_model_resolved"] == "deepseek-v4-flash"
    assert result["mismatched"] == []
    assert result["matched_count"] == 1
    assert "DIFFER" not in checker.render(result)


def test_a_subagent_on_another_model_is_a_mismatch(tmp_path):
    """This is the failure the checker exists to catch."""
    home, session = build_home(
        tmp_path,
        session_model="deepseek-v4-flash",
        subagent_models=["deepseek-flash", "mimo-v2.5"],
    )

    result = checker.check_session(session, checker.load_alias_map(home))

    assert [record["model"] for record in result["mismatched"]] == ["mimo-v2.5"]
    assert "mimo-v2.5" in checker.render(result)


def test_a_self_referencing_alias_does_not_loop(tmp_path):
    """`[model."mimo-v2.5"] model = "mimo-v2.5"` is how the real config spells it."""
    home, _ = build_home(tmp_path, session_model="deepseek-flash", subagent_models=[])
    alias_map = checker.load_alias_map(home)

    assert checker.normalize("mimo-v2.5", alias_map) == "mimo-v2.5"
    assert checker.normalize("deepseek-flash", alias_map) == "deepseek-v4-flash"
    assert checker.normalize(None, alias_map) is None


def test_a_chained_alias_resolves_to_the_end_of_the_chain(tmp_path):
    home, _ = build_home(
        tmp_path, session_model="a", subagent_models=[],
        config='[model.a]\nmodel = "b"\n\n[model.b]\nmodel = "c"\n',
    )

    assert checker.normalize("a", checker.load_alias_map(home)) == "c"


def test_an_unreadable_record_fails_closed(tmp_path, capsys):
    """A record that cannot be read is unverified, so it must not pass quietly."""
    home, session = build_home(
        tmp_path, session_model="deepseek-v4-flash", subagent_models=["deepseek-flash", None],
    )

    result = checker.check_session(session, checker.load_alias_map(home))
    code = checker.main(["--session-dir", str(session), "--grok-home", str(home)])

    assert result["unreadable"] and not result["mismatched"]
    assert code == 1
    assert "UNREAD" in capsys.readouterr().out


def test_a_session_without_subagents_passes(tmp_path):
    home, session = build_home(tmp_path, session_model="deepseek-flash", subagent_models=[])

    assert checker.main(["--session-dir", str(session), "--grok-home", str(home)]) == 0


def test_a_matching_session_exits_zero(tmp_path):
    home, session = build_home(
        tmp_path, session_model="deepseek-flash", subagent_models=["deepseek-flash"],
    )

    assert checker.main(["--session-dir", str(session), "--grok-home", str(home)]) == 0


def test_the_newest_session_for_a_cwd_is_selected(tmp_path):
    home, older = build_home(
        tmp_path, session_model="deepseek-flash", subagent_models=[],
        updated="2026-09-18T01:00:00Z",
    )
    newer = home / "sessions" / "proj" / "01a0aaaa-0000-7000-8000-000000000002"
    (newer / "subagents").mkdir(parents=True)
    (newer / "summary.json").write_text(json.dumps({
        "info": {"id": newer.name, "cwd": "C:\\work"},
        "current_model_id": "deepseek-flash",
        "updated_at": "2026-09-19T02:00:00Z",
    }), encoding="utf-8")

    found = checker.find_sessions(home, "C:\\work")

    assert [path.name for path in found] == [newer.name, older.name]
    assert checker.find_sessions(home, "C:\\somewhere-else") == []
