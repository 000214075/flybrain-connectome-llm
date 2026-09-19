"""Tests for the metrics comparison used by the `num_workers` equivalence check.

The script decides whether a recipe change is free, so its verdict has to be exact:
a comparison that tolerates a slightly different loss would pass a run whose data
order drifted, and that run would then be silently incomparable with the ones it is
measured against. The two properties worth pinning down are therefore that an exact
match passes, and that a difference at one step fails while naming that step.

Run with: .venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import importlib.util
import json
import os

MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "compare_metrics.py",
)


def load_module():
    spec = importlib.util.spec_from_file_location("compare_metrics", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


comparer = load_module()


def write_metrics(path, losses, *, event="train", extra=None):
    lines = []
    for step, loss in enumerate(losses, start=1):
        record = {
            "step": step,
            "event": event,
            "loss": loss,
            "ppl": 2.0 ** loss,
            "tokens_per_second": 900.0 + step,
            "vram_gb": 8.1,
        }
        record.update(extra or {})
        lines.append(json.dumps(record))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_identical_runs_pass(tmp_path, capsys):
    a = write_metrics(tmp_path / "a.jsonl", [9.0, 8.5, 8.0])
    b = write_metrics(tmp_path / "b.jsonl", [9.0, 8.5, 8.0])

    code = comparer.main(["--a", str(a), "--b", str(b)])

    assert code == 0
    assert "IDENTICAL" in capsys.readouterr().out


def test_a_single_step_difference_fails_and_names_the_step(tmp_path, capsys):
    a = write_metrics(tmp_path / "a.jsonl", [9.0, 8.5, 8.0])
    b = write_metrics(tmp_path / "b.jsonl", [9.0, 8.4, 8.0])

    code = comparer.main(["--a", str(a), "--b", str(b)])

    assert code == 1
    out = capsys.readouterr().out
    assert "step 2 loss" in out
    assert "8.5 vs 8.4" in out


def test_throughput_is_reported_but_does_not_decide(tmp_path, capsys):
    """Dropping workers is expected to change speed; that must not read as a failure."""
    a = write_metrics(tmp_path / "a.jsonl", [9.0, 8.5], extra={"tokens_per_second": 900.0})
    b = write_metrics(tmp_path / "b.jsonl", [9.0, 8.5], extra={"tokens_per_second": 500.0})

    result = comparer.compare(a, b, ("loss",), 0.0)

    assert result["differences"] == []
    assert result["speed"]["tokens_per_second"]["b"] == 500.0
    assert comparer.main(["--a", str(a), "--b", str(b)]) == 0
    assert "tokens_per_second" in capsys.readouterr().out


def test_different_step_counts_fail(tmp_path):
    a = write_metrics(tmp_path / "a.jsonl", [9.0, 8.5, 8.0])
    b = write_metrics(tmp_path / "b.jsonl", [9.0, 8.5])

    assert comparer.main(["--a", str(a), "--b", str(b)]) == 1


def test_non_train_records_are_ignored(tmp_path):
    """Eval lines carry a different meaning and a different key set."""
    a = write_metrics(tmp_path / "a.jsonl", [9.0, 8.5])
    b = write_metrics(tmp_path / "b.jsonl", [9.0, 8.5])
    with (tmp_path / "b.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 1, "event": "eval", "val_loss": 7.0}) + "\n")

    assert comparer.main(["--a", str(a), "--b", str(b)]) == 0


def test_a_tolerance_can_be_allowed_but_is_reported_separately(tmp_path, capsys):
    a = write_metrics(tmp_path / "a.jsonl", [9.0, 8.5])
    b = write_metrics(tmp_path / "b.jsonl", [9.0, 8.5000001])

    strict = comparer.main(["--a", str(a), "--b", str(b)])
    strict_out = capsys.readouterr().out
    loose = comparer.main(["--a", str(a), "--b", str(b), "--tol", "1e-6"])
    loose_out = capsys.readouterr().out

    assert strict == 1
    assert "step 2 loss" in strict_out
    assert loose == 0
    # Relaxing the threshold must still surface the difference rather than hide it.
    assert "not counted" in loose_out
    assert "step 2 loss" in loose_out
