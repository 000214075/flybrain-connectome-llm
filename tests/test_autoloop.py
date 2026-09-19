"""The Stop hook that keeps this project's request running must fail safe.

The hook decides, at the end of every turn in this project, whether to feed the
standing request back to the agent as a new user message. Getting that wrong in one
direction is a runaway loop; getting it wrong in the other is a hook that silently
does nothing. Neither is visible from the transcript, so every guard is tested here
against the real script, run the way grok runs it: JSON on stdin, a decision on
stdout.

These tests drive powershell.exe, so they are Windows-only. The script is pointed at
a temporary project and state directory through `AUTOLOOP_PROJECT` / `AUTOLOOP_ROOT`
so a test run can never touch the live round counter or the live kill switch.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="the hook is a PowerShell script")

REPO = Path(__file__).resolve().parent.parent
HOOK = REPO / "scripts" / "autoloop_stop_hook.ps1"
PROJECT = r"C:\Users\heyiy\Desktop\cangying"


def run_hook(payload: object | str, project: Path, root: Path) -> tuple[int, bytes]:
    """Run the hook exactly as grok does: envelope on stdin, decision on stdout."""
    stdin = payload if isinstance(payload, str) else json.dumps(payload)
    env = {
        **os.environ,
        "AUTOLOOP_PROJECT": str(project),
        "AUTOLOOP_ROOT": str(root),
    }
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(HOOK),
        ],
        input=stdin.encode("utf-8"),
        capture_output=True,
        env=env,
        timeout=120,
    )
    return completed.returncode, completed.stdout


@pytest.fixture()
def loopdir(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    root = project / "autoloop"
    root.mkdir(parents=True)
    (root / "prompt.txt").write_text("第 {{ROUND}} 轮：继续推翻传统 LLM。", encoding="utf-8")
    return project, root


def envelope(project: Path, **overrides) -> dict:
    base = {
        "hookEventName": "stop",
        "sessionId": "s-1",
        "promptId": "p-1",
        "reason": "end_turn",
        "cwd": str(project),
        "workspaceRoot": str(project),
        "timestamp": "2026-09-17T00:00:00Z",
        "backgroundTasks": [],
        "sessionCrons": [],
    }
    base.update(overrides)
    return base


def test_a_completed_turn_is_kept_working_with_the_prompt_intact(loopdir):
    """The happy path: block the stop and hand the whole prompt back as the reason."""
    project, root = loopdir
    code, raw = run_hook(envelope(project), project, root)
    assert code == 0
    # This host's console code page is not UTF-8 and a raw stream write put a BOM in
    # front of the decision, which made grok fail to parse it -- a hook that silently
    # never fires. The wire format is therefore pure ASCII with \uXXXX escapes.
    assert not raw.startswith(b"\xef\xbb\xbf"), "the decision carries a UTF-8 BOM"
    raw.decode("ascii")
    decision = json.loads(raw.decode("utf-8"))
    assert decision["decision"] == "block"
    # The prompt is Chinese; the escapes must still decode back to the real text.
    assert "第 1 轮" in decision["reason"], decision["reason"][:200]
    assert "推翻传统 LLM" in decision["reason"]
    # Read as utf-8-sig: the file must be valid regardless of whether a writer adds a
    # BOM. A BOM here is exactly the bug this test caught (Set-Content -Encoding UTF8
    # on Windows PowerShell 5.1 writes one, and plain-UTF-8 readers then fail).
    state = json.loads((root / "state.json").read_text(encoding="utf-8-sig"))
    assert state["rounds"] == 1
    assert not (root / "state.json").read_bytes().startswith(b"\xef\xbb\xbf"), "state.json has a BOM"


def test_the_round_counter_advances_so_the_agent_knows_which_pass_this_is(loopdir):
    project, root = loopdir
    run_hook(envelope(project), project, root)
    code, raw = run_hook(envelope(project), project, root)
    assert code == 0
    assert "第 2 轮" in json.loads(raw.decode("utf-8"))["reason"]


def test_session_end_does_not_start_another_round(loopdir):
    """A shutdown fire has no turn left to continue; blocking there would be lost work."""
    project, root = loopdir
    code, raw = run_hook(envelope(project, reason="channel_closed"), project, root)
    assert code == 0 and raw.strip() == b""
    assert not (root / "state.json").exists()


def test_another_project_is_left_alone(loopdir, tmp_path):
    """The hook is global (this repo is untrusted), so it must scope itself."""
    project, root = loopdir
    code, raw = run_hook(envelope(project, workspaceRoot=str(tmp_path / "elsewhere")), project, root)
    assert code == 0 and raw.strip() == b""


def test_work_already_in_flight_is_not_stacked_on(loopdir):
    """A running training job will wake the session on completion; adding a round
    on top of it would put two jobs on one GPU."""
    project, root = loopdir
    payload = envelope(project, backgroundTasks=[{"id": "t1", "type": "shell", "status": "running"}])
    code, raw = run_hook(payload, project, root)
    assert code == 0 and raw.strip() == b""


def test_the_kill_switch_stops_the_loop(loopdir):
    project, root = loopdir
    (root / "DISABLE").write_text("stopped by the user\n", encoding="utf-8")
    code, raw = run_hook(envelope(project), project, root)
    assert code == 0 and raw.strip() == b""


def test_a_missing_prompt_allows_the_stop(loopdir):
    project, root = loopdir
    (root / "prompt.txt").unlink()
    code, raw = run_hook(envelope(project), project, root)
    assert code == 0 and raw.strip() == b""


@pytest.mark.parametrize("stdin", ["", "   ", "not json at all", "null"])
def test_junk_on_stdin_allows_the_stop(loopdir, stdin):
    project, root = loopdir
    code, raw = run_hook(stdin, project, root)
    assert code == 0 and raw.strip() == b""


def test_the_guards_are_recorded_so_a_silent_hook_is_diagnosable(loopdir):
    """Every allow path logs why, so "it did not fire" is answerable from the log."""
    project, root = loopdir
    run_hook(envelope(project, reason="shutdown"), project, root)
    run_hook(envelope(project), project, root)
    log = (root / "inject.log").read_text(encoding="utf-8", errors="replace")
    assert "allow: reason is 'shutdown'" in log
    assert "inject: round 1" in log


def test_the_workspace_is_matched_regardless_of_separator_style(loopdir):
    """A forward-slash spelling of the workspace must still match.

    The hook's scope gate is a string compare against the project path, and the
    envelope's path style is not ours to choose -- the documented examples use
    POSIX-style paths. A compare that only accepts backslashes would make the hook
    silently never fire on a host or client that normalises to `C:/...`, while the
    status output still said the hook was installed. That is the exact failure this
    project has been bitten by repeatedly, so both spellings are asserted.
    """
    project, root = loopdir
    for spelling in (str(project), str(project).replace("\\", "/"), str(project) + "\\"):
        code, raw = run_hook(envelope(project, workspaceRoot=spelling, cwd=spelling), project, root)
        assert code == 0
        assert raw.strip(), f"the hook stood down on workspace spelling {spelling!r}"
        assert json.loads(raw.decode("utf-8"))["decision"] == "block"


def test_a_recent_file_write_does_not_stop_the_loop_but_a_running_job_does(loopdir):
    """The hook must ignore the busy check's recency signal and honour its job signal.

    Two ways to get this wrong, both silent. Honour the recency signal and the hook
    stands down after *every* round -- because the round that just ended wrote those
    files -- so the fast engine never fires at all. Ignore the job signal and a
    hand-started training run gets a round stacked on top of it, which is how a
    contaminated run nearly became a published "the shuffled wiring is 3.3x slower"
    (report section 7.6). The hook therefore passes `-QuietMinutes 0`.
    """
    project, root = loopdir
    scripts = project / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)

    # A stand-in check whose two signals can be driven independently of the host.
    (scripts / "autoloop_busy_check.ps1").write_text(
        "param([int]$QuietMinutes = 5, [string]$AlsoMatch = 'flybrain')\n"
        'if ($QuietMinutes -gt 0) { Write-Output "BUSY: recency"; exit 3 }\n'
        'Write-Output "FREE: no job"; exit 0\n',
        encoding="utf-8",
    )

    # A file written a moment ago is exactly what every round leaves behind.
    (project / "reports").mkdir(exist_ok=True)
    (project / "reports" / "just-written.md").write_text("x", encoding="utf-8")

    code, raw = run_hook(envelope(project), project, root)
    assert code == 0
    assert raw.strip(), (
        "the hook stood down because files had just been written; with the default "
        "recency window it would never inject after any round"
    )

    # Now a project job really is running.
    (scripts / "autoloop_busy_check.ps1").write_text(
        "param([int]$QuietMinutes = 5, [string]$AlsoMatch = 'flybrain')\n"
        'Write-Output "BUSY: 1 project job(s) running"; exit 3\n',
        encoding="utf-8",
    )
    code, raw = run_hook(envelope(project), project, root)
    assert code == 0 and raw.strip() == b""
    log = (root / "inject.log").read_text(encoding="utf-8", errors="replace")
    assert "a project job is running" in log


def test_a_broken_busy_check_fails_free_rather_than_wedging_the_loop(loopdir):
    """A check that always says busy would stall the loop forever, so errors allow."""
    project, root = loopdir
    scripts = project / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "autoloop_busy_check.ps1").write_text(
        "throw 'deliberately broken'\n", encoding="utf-8"
    )
    code, raw = run_hook(envelope(project), project, root)
    assert code == 0
    assert raw.strip(), "a broken busy check must not stop the loop"


# --------------------------------------------------------------------------
# the scheduler half's in-flight guard
#
# The hook refuses to inject while background tasks are in flight, so it never puts
# two rounds on one GPU. The scheduler fires on wall clock and had no such guard,
# which is how a round that takes longer than the interval gets a second round
# started on top of it -- and on a single GPU that corrupts the first round's
# timings rather than merely slowing it down.
# --------------------------------------------------------------------------

BUSY_CHECK = REPO / "scripts" / "autoloop_busy_check.ps1"


def run_busy_check(project: Path, *extra: str) -> tuple[int, str]:
    """Run the check the way the scheduler would, with the host's own jobs ignored.

    `-AlsoMatch` is the check's second way of recognising this project's jobs, used
    when a venv launcher re-execs the base interpreter and the command line names no
    project path. Tests point it at something that cannot match, so their verdict
    depends only on the temporary project and not on what the developer is running.
    """
    env = {**os.environ, "AUTOLOOP_PROJECT": str(project)}
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(BUSY_CHECK),
            "-AlsoMatch",
            "no-such-module-on-this-host",
            *extra,
        ],
        capture_output=True,
        env=env,
        timeout=120,
    )
    return completed.returncode, completed.stdout.decode("utf-8", errors="replace")


@pytest.fixture()
def idle_project(tmp_path: Path) -> Path:
    """A project directory that no round is working in."""
    project = tmp_path / "idle"
    (project / "reports").mkdir(parents=True)
    (project / "configs").mkdir()
    (project / "scripts").mkdir()
    return project


def test_an_idle_project_is_free(idle_project):
    """No job running and nothing written recently: the backup engine may fire."""
    code, text = run_busy_check(idle_project)
    assert code == 0, text
    assert text.startswith("FREE")


def test_a_file_written_seconds_ago_means_a_round_is_in_progress(idle_project):
    """Every round writes into reports/ within its first minute."""
    (idle_project / "reports" / "train_something.log").write_text("step 1\n", encoding="utf-8")
    code, text = run_busy_check(idle_project)
    assert code == 3, text
    assert text.startswith("BUSY")


def test_old_writes_do_not_count(idle_project):
    """The recency signal has to decay, or a finished round would wedge the loop."""
    stale = idle_project / "reports" / "train_old.log"
    stale.write_text("step 900\n", encoding="utf-8")
    old = time.time() - 3600
    os.utime(stale, (old, old))
    code, text = run_busy_check(idle_project)
    assert code == 0, text


def test_a_missing_project_is_free(tmp_path):
    """Fail-safe direction: a check that wrongly says busy would stall the loop forever."""
    code, text = run_busy_check(tmp_path / "does-not-exist")
    assert code == 0, text
    assert text.startswith("FREE")


def test_a_long_lived_project_helper_does_not_make_the_project_look_busy(idle_project):
    """A process that never exits must not be mistaken for work in progress.

    The local search MCP server lives in the project, is started by the client at
    session start, and runs until the session ends -- so its command line names the
    project and it looks exactly like a job. Counting it meant the gate reported BUSY
    on every single firing and the loop never got any work done, which is the
    "always busy" failure this check was written to avoid.
    """
    helper_dir = idle_project / "tools" / "localsearch"
    helper_dir.mkdir(parents=True)
    helper = helper_dir / "server.py"
    helper.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    child = subprocess.Popen(
        [sys.executable, str(helper)], cwd=str(idle_project),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(3)  # let the process appear with its command line
        code, text = run_busy_check(idle_project, "-QuietMinutes", "0")
        assert code == 0, f"the helper was counted as work: {text}"
        assert text.startswith("FREE")
    finally:
        child.terminate()
        child.wait(timeout=30)


def test_a_running_project_job_is_work_even_if_nothing_was_written(idle_project):
    """A long benchmark writes nothing for minutes, but it still owns the GPU.

    This is the case the hook's own guard covers for its session; the scheduler sees
    other sessions, so it has to look at the process list.
    """
    sleeper = idle_project / "scripts" / "probe_sleep.py"
    sleeper.write_text("import time\ntime.sleep(45)\n", encoding="utf-8")
    child = subprocess.Popen(
        [sys.executable, str(sleeper)], cwd=str(idle_project),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(3)  # let the process appear with its command line
        code, text = run_busy_check(idle_project, "-QuietMinutes", "0")
        assert code == 3, text
        assert text.startswith("BUSY")
        assert "probe_sleep.py" in text
    finally:
        child.terminate()
        child.wait(timeout=30)
