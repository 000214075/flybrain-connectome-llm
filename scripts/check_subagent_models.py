"""Check that a session's subagents ran on the session's own model.

A `[subagents.models]` entry in config.toml overrides model resolution for every
subagent of that type. A session whose subagents ran on some other model therefore
looks identical to a correct one until you read the recorded model ids, which is how
this project ran many subagents on `mimo-v2.5` while the session itself ran
`deepseek-v4-flash` without anyone noticing.

Config is read once, at process start, so an edit does not reach a session that is
already running -- only a restart picks it up. That distinction is the point of this
script: a mismatch reported here means the running process predates the config edit,
not necessarily that the config is wrong. Check the session process start time
against `config.toml`'s last write time before concluding either way.

Model ids are compared after resolving config aliases, because a session records the
alias it was started with (`deepseek-flash`) while a subagent may record the
underlying model name (`deepseek-v4-flash`); those are the same model.

Usage:
    .venv\\Scripts\\python.exe scripts\\check_subagent_models.py
    .venv\\Scripts\\python.exe scripts\\check_subagent_models.py --session-dir <dir>
    .venv\\Scripts\\python.exe scripts\\check_subagent_models.py --grok-home <dir> --json

Exit code 0 when every readable subagent record resolves to the session's model,
1 when any differs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path


def default_grok_home() -> Path:
    env = os.environ.get("GROK_HOME")
    if env:
        return Path(env)
    return Path.home() / ".grok"


def load_alias_map(grok_home: Path) -> dict[str, str]:
    """Map each `[model.<alias>]` table to the API model name it points at."""
    config = grok_home / "config.toml"
    if not config.is_file():
        return {}
    with config.open("rb") as handle:
        data = tomllib.load(handle)
    alias_map: dict[str, str] = {}
    for alias, table in (data.get("model") or {}).items():
        if isinstance(table, dict) and isinstance(table.get("model"), str):
            alias_map[alias] = table["model"]
    return alias_map


def normalize(model_id: str | None, alias_map: dict[str, str]) -> str | None:
    """Resolve a model id through config aliases, tolerating self-references."""
    seen: set[str] = set()
    current = model_id
    while current and current in alias_map and current not in seen:
        seen.add(current)
        current = alias_map[current]
    return current


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def find_sessions(grok_home: Path, cwd: str | None) -> list[Path]:
    """Every session directory under the grok home, newest first."""
    root = grok_home / "sessions"
    if not root.is_dir():
        return []
    found: list[tuple[float, str, Path]] = []
    for cwd_dir in root.iterdir():
        if not cwd_dir.is_dir():
            continue
        for session_dir in cwd_dir.iterdir():
            summary = session_dir / "summary.json"
            if not summary.is_file():
                continue
            try:
                data = read_json(summary)
            except (OSError, json.JSONDecodeError):
                continue
            if cwd:
                recorded = (data.get("info") or {}).get("cwd") or data.get("cwd")
                if not recorded or os.path.normcase(recorded) != os.path.normcase(cwd):
                    continue
            stamp = data.get("updated_at") or ""
            found.append((summary.stat().st_mtime, stamp, session_dir))
    found.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return [item[2] for item in found]


def collect_subagents(session_dir: Path) -> list[dict]:
    """One record per subagent, including records whose meta.json cannot be read."""
    subagent_root = session_dir / "subagents"
    if not subagent_root.is_dir():
        return []
    records: list[dict] = []
    for entry in sorted(subagent_root.iterdir()):
        meta = entry / "meta.json"
        if not meta.is_file():
            continue
        record: dict = {"subagent_id": entry.name}
        try:
            data = read_json(meta)
        except (OSError, json.JSONDecodeError) as exc:
            record["unreadable"] = True
            record["error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            continue
        record.update(
            unreadable=False,
            description=data.get("description") or "",
            subagent_type=data.get("subagent_type") or "",
            status=data.get("status") or "",
            model=data.get("effective_model_id"),
            started_at=data.get("started_at") or "",
        )
        records.append(record)
    return records


def check_session(session_dir: Path, alias_map: dict[str, str]) -> dict:
    data = read_json(session_dir / "summary.json")
    session_model = data.get("current_model_id")
    wanted = normalize(session_model, alias_map)
    subagents = collect_subagents(session_dir)
    for record in subagents:
        if record.get("unreadable"):
            continue
        record["normalized"] = normalize(record.get("model"), alias_map)
        record["matches"] = record["normalized"] == wanted
    readable = [r for r in subagents if not r.get("unreadable")]
    matched = [r for r in readable if r["matches"]]
    return {
        "session_dir": str(session_dir),
        "session_id": (data.get("info") or {}).get("id") or session_dir.name,
        "cwd": (data.get("info") or {}).get("cwd") or data.get("cwd"),
        "session_model": session_model,
        "session_model_resolved": wanted,
        "subagent_count": len(subagents),
        "matched": matched,
        "mismatched": [r for r in readable if not r["matches"]],
        "unreadable": [r for r in subagents if r.get("unreadable")],
        "matched_count": len(matched),
    }


def render(result: dict) -> str:
    lines = [
        f"session  : {result['session_id']}",
        f"cwd      : {result['cwd']}",
        f"model    : {result['session_model']} -> {result['session_model_resolved']}",
        (
            f"subagents: {result['subagent_count']} recorded, "
            f"{result['matched_count']} match, {len(result['mismatched'])} differ, "
            f"{len(result['unreadable'])} unreadable"
        ),
    ]
    for record in result["mismatched"]:
        label = record["description"][:48] or record["subagent_type"]
        lines.append(
            f"  DIFFER  {record['subagent_id'][:8]}  {record['subagent_type']:<16}"
            f"{str(record.get('model')):<20}{label}"
        )
    for record in result["unreadable"]:
        lines.append(f"  UNREAD  {record['subagent_id'][:8]}  {record.get('error')}")
    if result["matched"]:
        counts: dict[str, int] = {}
        for record in result["matched"]:
            counts[str(record.get("model"))] = counts.get(str(record.get("model")), 0) + 1
        summary = ", ".join(f"{model} x{count}" for model, count in sorted(counts.items()))
        lines.append(f"  matched models: {summary}")
    if result["mismatched"] or result["unreadable"]:
        lines += [
            "",
            "These subagents are not confirmed to have used the session's model.",
            "Config is read once at process start, so if the session process started",
            "before config.toml's last write, it is still running the old config:",
            "restart the session to apply the edit. Compare the process start time",
            "with config.toml's LastWriteTime before changing anything.",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session-dir", help="session directory holding summary.json")
    parser.add_argument("--grok-home", help="grok home (default: $GROK_HOME or ~/.grok)")
    parser.add_argument("--cwd", help="pick the newest session for this directory")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    grok_home = Path(args.grok_home) if args.grok_home else default_grok_home()
    if args.session_dir:
        session_dir = Path(args.session_dir)
        if not (session_dir / "summary.json").is_file():
            print(f"no summary.json under {session_dir}", file=sys.stderr)
            return 1
    else:
        cwd = args.cwd or os.getcwd()
        sessions = find_sessions(grok_home, cwd)
        if not sessions:
            print(f"no session for {cwd} under {grok_home / 'sessions'}", file=sys.stderr)
            return 1
        session_dir = sessions[0]

    result = check_session(session_dir, load_alias_map(grok_home))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render(result))
    return 1 if (result["mismatched"] or result["unreadable"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
