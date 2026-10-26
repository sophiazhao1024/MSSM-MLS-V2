"""Control-flow fixtures, not MSSM physical observations."""
from dataclasses import replace
import json
import numpy as np
import pytest
from mssm_mls_modern.settings import ModernSettings
from mssm_mls_modern.runner import run_scan, mark_validation


def settings():
    return replace(ModernSettings(), initial=10, budget=22, exploit=2, uncertain=1,
                   random_extra=1, pool_size=32, ensemble_size=2, epochs=2,
                   patience=1, hidden_layers=1, hidden_width=8, min_physical=2)


def truth(parameters, eval_id):
    return {"status": "ok", "observables": {"mh": 10. + parameters["M1"],
            "mH": 125., "chi2_hs": 90., "omega": .12},
            "diagnostics": {"hb_result": int(parameters["M1"] > 5)}}


@pytest.mark.parametrize("method", ["modern", "random"])
def test_resume_preserves_all_pointwise_results_and_training(tmp_path, method):
    s = settings()
    full = run_scan(s, truth, tmp_path / "full", seed=3, method=method, physics_identity={"fixture": 1})
    def interrupted(parameters, eval_id):
        if eval_id == 17:
            raise KeyboardInterrupt()
        return truth(parameters, eval_id)
    with pytest.raises(KeyboardInterrupt):
        run_scan(s, interrupted, tmp_path / "partial", seed=3, method=method, physics_identity={"fixture": 1})
    checkpoint = tmp_path / "partial/checkpoint.json"
    assert len(json.loads(checkpoint.read_text())["payload"]["records"]) == 14
    resumed = run_scan(s, truth, tmp_path / "resumed", seed=3, method=method,
                       physics_identity={"fixture": 1}, resume=checkpoint)
    assert resumed["records"] == full["records"]
    assert resumed["history"] == full["history"]
    assert resumed["summary"]["attempts"] == 22
    assert len({tuple(r["x"]) for r in full["records"]}) == 22
    assert sum(r["validation"] for r in full["records"]) == 2
    assert not any(r["validation"] for r in full["records"][10:])


def test_resume_refuses_different_identity(tmp_path):
    s = settings()
    run_scan(s, truth, tmp_path / "first", seed=3, method="random", physics_identity={"fixture": 1})
    with pytest.raises(ValueError, match="identity"):
        run_scan(s, truth, tmp_path / "bad", seed=4, method="random", physics_identity={"fixture": 1},
                 resume=tmp_path / "first/checkpoint.json")


def test_errors_are_counted_but_not_targets(tmp_path):
    s = replace(settings(), max_consecutive_tool_errors=2)
    with pytest.raises(RuntimeError, match="technical"):
        run_scan(s, lambda p, i: {"status": "tool_error", "reason": "fixture failure"},
                 tmp_path / "errors", method="random")
    summary = json.loads((tmp_path / "errors/summary.json").read_text())
    assert summary["attempts"] == 2 and summary["tool_error"] == 2
    assert summary["targets"] == 0 and summary["status"] == "aborted"
    assert not (tmp_path / "errors/checkpoint.json").exists()


def test_saved_probe_keeps_separate_budget_and_truth_policy(tmp_path):
    s = settings()
    original = run_scan(replace(s, budget=10), truth, tmp_path / "initial", method="random")
    result = run_scan(s, truth, tmp_path / "probe", method="random", initial_records=original["records"])
    assert result["summary"]["kind"] == "saved_initial_probe"
    assert result["summary"]["inherited_initial_count"] == 10
    assert result["summary"]["post_initial_probe_attempts"] == 12
    assert result["summary"]["new_attempts_this_process"] == 12
    assert result["summary"]["attempts"] == 22
    duplicate = original["records"][:]
    duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError, match="duplicate"):
        mark_validation(duplicate, s, seed=0)
