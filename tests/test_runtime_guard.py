"""Synthetic operational contracts; never a claim of MSSM scan efficiency."""
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import pytest

from mssm_mls_modern.runner import run_scan
from mssm_mls_modern.settings import ModernSettings

from mssm_mls_modern.operational_recovery import GuardedReplayBackend, OperationalSafetyStop

IDENTITY = {"config": "c", "adapter": "a", "binaries": {"omega": "o"}}


def small_settings():
    return replace(ModernSettings(), initial=10, budget=18, exploit=2,
                   uncertain=1, random_extra=1, pool_size=32, epochs=2,
                   patience=1, hidden_layers=1, hidden_width=8, min_physical=2,
                   gp_steps=3, gp_max_anchors=16, max_consecutive_tool_errors=19)


def synthetic_answer(parameters, eval_id):
    """Smooth invented truth, containing no MSSM measurements."""
    mass = 10. + parameters["M1"]
    return {"status": "ok", "observables": {"mh": mass, "mH": 125.,
            "chi2_hs": 90., "omega": .12}, "reason": "fixture",
            "diagnostics": {"hb_result": int(parameters["M1"] > 5),
                            "hb_obsratio": 0.5 if parameters["M1"] > 5 else 1.5},
            "provenance": {}}


class FakeBackend:
    def __init__(self, answer=synthetic_answer):
        self.answer = answer
        self.calls = []

    def evaluate(self, parameters, eval_id):
        self.calls.append((dict(parameters), eval_id))
        return self.answer(parameters, eval_id)


def original_runner():
    path = Path(__file__).resolve().parent / "_snapshots/v4_core/runner.py"
    spec = spec_from_file_location("mssm_mls_modern._frozen_v4_runner", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_scan


@pytest.mark.parametrize("method", ["modern", "random"])
def test_original_none_and_guard_observer_have_identical_scientific_trajectory(tmp_path, method):
    settings = small_settings()
    old_backend, none_backend, observer_backend = FakeBackend(), FakeBackend(), FakeBackend()
    original = original_runner()(settings, old_backend, tmp_path / "original", seed=7,
                                 method=method, physics_identity=IDENTITY)
    without = run_scan(settings, none_backend, tmp_path / "none", seed=7,
                       method=method, physics_identity=IDENTITY)
    out = tmp_path / "observed"
    guard = GuardedReplayBackend(observer_backend, out, IDENTITY)
    events = []

    def observe(event):
        events.append(dict(event))
        if event["event"] == "point":
            # The completed row and progress are readable before the hook runs.
            rows = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
            assert rows[-1]["eval_id"] == event["eval_id"]
            assert json.loads((out / "progress.json").read_text())["attempts"] == event["eval_id"]
        guard.observe(event)
        event["status"] = "observer-local-mutation"  # No live scan object is exposed.

    observed = run_scan(settings, guard, out, seed=7, method=method,
                        physics_identity=IDENTITY, observer=observe)
    assert original["records"] == without["records"] == observed["records"]
    assert original["history"] == without["history"] == observed["history"]
    assert old_backend.calls == none_backend.calls == observer_backend.calls
    assert events[0] == {"event": "started"}
    assert events[-1] == {"event": "finished", "status": "completed"}
    assert [e["eval_id"] for e in events if e["event"] == "point"] == list(range(1, 19))
    assert guard.report()["new_evaluator_requests"] == 18


def test_fifth_unknown_error_is_journaled_before_safety_stop(tmp_path):
    out = tmp_path / "unknown"
    backend = FakeBackend(lambda p, i: {"status": "tool_error", "reason": "unknown fixture"})
    guard = GuardedReplayBackend(backend, out, IDENTITY)
    events = []

    def observe(event):
        events.append(dict(event))
        guard.observe(event)

    with pytest.raises(OperationalSafetyStop, match="Five consecutive"):
        run_scan(small_settings(), guard, out, method="random", observer=observe,
                 physics_identity=IDENTITY)
    rows = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
    assert [r["eval_id"] for r in rows] == [1, 2, 3, 4, 5]
    assert all(r["status"] == "tool_error" and not r["target"] for r in rows)
    assert [i for p, i in backend.calls] == [1, 2, 3, 4, 5]
    summary = json.loads((out / "summary.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert summary["attempts"] == summary["tool_error"] == 5
    assert summary["status"] == manifest["status"] == "aborted"
    assert manifest["error"].startswith("OperationalSafetyStop:")
    assert events[-1] == {"event": "finished", "status": "aborted"}
    assert len((out / "operational_guard.jsonl").read_text().splitlines()) == 5
    assert json.loads((out / "operational_recovery.json").read_text())["unidentified_failure_streak"] == 5
    assert (out / "training_history.json").exists()


def known_numerical_error(parameters, eval_id):
    return {"status": "tool_error", "reason": "numerical fixture", "observables": {},
            "provenance": {"config_sha256": "c", "adapter_sha256": "a",
                "binary_sha256": IDENTITY["binaries"], "paper_version_match": True,
                "higgs": {"status": "ok"},
                "commands": [{"stage": "higgs", "returncode": 0},
                             {"stage": "omega", "returncode": 70}],
                "micromegas": {"status": "technical_failure", "reason": "darkOmega_failed",
                               "source_version": "4.3.2"}}, "diagnostics": {}}


def test_recognized_numerical_errors_count_once_and_are_not_relabelled(tmp_path):
    out = tmp_path / "known"
    backend = FakeBackend(known_numerical_error)
    guard = GuardedReplayBackend(backend, out, IDENTITY)
    result = run_scan(small_settings(), guard, out, method="random", observer=guard.observe,
                      physics_identity=IDENTITY)
    assert result["summary"]["tool_error"] == result["summary"]["attempts"] == 18
    assert result["summary"]["targets"] == 0
    assert result["summary"]["status"] == "completed"
    assert [i for p, i in backend.calls] == list(range(1, 19))
    assert guard.report()["recognized_failure_returns_this_process"] == 18
    assert guard.report()["unidentified_failure_streak"] == 0


def test_resume_event_and_exact_suffix_replay_preserve_record_and_proposal(tmp_path):
    settings = small_settings()
    full = run_scan(settings, FakeBackend(), tmp_path / "full", seed=9,
                    method="random", physics_identity=IDENTITY)

    def interrupt(parameters, eval_id):
        if eval_id == 17:
            raise KeyboardInterrupt()
        return synthetic_answer(parameters, eval_id)

    with pytest.raises(KeyboardInterrupt):
        run_scan(settings, FakeBackend(interrupt), tmp_path / "partial", seed=9,
                 method="random", physics_identity=IDENTITY)
    checkpoint = tmp_path / "partial/checkpoint.json"
    assert len(json.loads(checkpoint.read_text())["payload"]["records"]) == 14
    backend = FakeBackend()
    out = tmp_path / "resumed"
    guard = GuardedReplayBackend(backend, out, IDENTITY,
                                source_run=tmp_path / "partial", checkpoint_count=14, prefix_count=16)
    events = []

    def observe(event):
        events.append(dict(event))
        guard.observe(event)

    resumed = run_scan(settings, guard, out, seed=9, method="random", physics_identity=IDENTITY,
                       resume=checkpoint, observer=observe)
    assert resumed["records"] == full["records"]
    assert resumed["history"] == full["history"]
    assert events[0] == {"event": "resumed"}
    assert [i for p, i in backend.calls] == [17, 18]
    assert guard.report()["prefix_verified"]
    assert guard.report()["replayed_attempts"] == 2


def test_safety_stop_from_backend_is_not_converted_to_tool_error(tmp_path):
    def stop(parameters, eval_id):
        raise OperationalSafetyStop("pre-call fixture stop")
    out = tmp_path / "stopped"
    events = []
    with pytest.raises(OperationalSafetyStop, match="pre-call"):
        run_scan(small_settings(), FakeBackend(stop), out, method="random", observer=events.append)
    assert (out / "records.jsonl").read_text() == ""
    assert json.loads((out / "summary.json").read_text())["attempts"] == 0
    assert events[-1] == {"event": "finished", "status": "aborted"}
