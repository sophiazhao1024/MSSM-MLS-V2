"""Public CLI integration using invented answers, never external physics tools."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from mssm_mls_modern import cli
from mssm_mls_modern.settings import ModernSettings
from mssm_mls_modern.reference.baseline import ScanSettings as ReferenceSettings

IDENTITY = {"config": "c", "adapter": "a", "binaries": {"fh_driver": "f", "omega_driver": "o"}}
METHODS = ("modern", "random", "strong-reference", "historical")


def config_for(tmp_path, method, budget=6):
    cls = ModernSettings if method in ("modern", "random") else ReferenceSettings
    values = asdict(cls())
    values.update(initial=6, budget=budget, epochs=1)
    # A deliberately low obsolete coarse guard must not override the shared
    # typed guard or truncate recognized numerical failures in the public CLI.
    values["max_consecutive_tool_errors"] = 1
    path = tmp_path / f"{method}.json"
    path.write_text(json.dumps(values))
    return path


def install_fake_backend(monkeypatch, answer):
    calls = []

    class Backend:
        def __init__(self, path, work):
            self.config = {}

        def doctor(self):
            return {"ready": True, "config_sha256": "c", "adapter_sha256": "a",
                    "checks": {name: {"sha256": digest} for name, digest in IDENTITY["binaries"].items()}}

        def evaluate(self, parameters, eval_id):
            calls.append((dict(parameters), eval_id))
            return answer(parameters, eval_id)

    monkeypatch.setattr(cli, "PhysicsBackend", Backend)
    return calls


def valid(parameters, eval_id):
    return {"status": "ok", "observables": {"mh": 10., "mH": 125., "chi2_hs": 90., "omega": .12},
            "diagnostics": {"hb_result": 1, "hb_obsratio": .5}, "provenance": {}, "reason": "fixture"}


def known_failure(parameters, eval_id):
    return {"status": "tool_error", "reason": "numerical fixture", "observables": {}, "diagnostics": {},
            "provenance": {"config_sha256": "c", "adapter_sha256": "a",
                "binary_sha256": IDENTITY["binaries"], "paper_version_match": True,
                "higgs": {"status": "ok"},
                "commands": [{"stage": "higgs", "returncode": 0}, {"stage": "omega", "returncode": 70}],
                "micromegas": {"status": "technical_failure", "reason": "darkOmega_failed", "source_version": "4.3.2"}}}


@pytest.mark.parametrize("method", METHODS)
def test_public_cli_counts_recognized_failures_with_same_guard(tmp_path, monkeypatch, method):
    calls = install_fake_backend(monkeypatch, known_failure)
    settings = config_for(tmp_path, method)
    out = tmp_path / "run"
    cli.main(["scan", "--method", method, "--settings", str(settings), "--output", str(out)])
    summary = json.loads((out / "summary.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert summary["attempts"] == summary["tool_error"] == 6
    assert summary["targets"] == 0 and summary["status"] == "completed"
    assert manifest["settings"]["max_consecutive_tool_errors"] == 7
    assert [i for p, i in calls] == list(range(1, 7))
    guard = json.loads((out / "operational_recovery.json").read_text())
    assert guard["recognized_failure_returns_this_process"] == 6
    assert guard["unidentified_failure_streak"] == 0
    identity = json.loads((out / "method_identity.json").read_text())
    assert identity["public_method"] == method
    assert identity["strong_benchmark_reference"] is (method == "strong-reference")
    assert identity["runtime_source_sha256"]["operational_recovery.py"] == hashlib.sha256(
        (Path(cli.__file__).parent / "operational_recovery.py").read_bytes()).hexdigest()
    assert "reference/learning.py" in identity["runtime_source_sha256"]


@pytest.mark.parametrize("method", METHODS)
def test_public_cli_preserves_fifth_unknown_failure_then_exits(tmp_path, monkeypatch, method):
    calls = install_fake_backend(monkeypatch, lambda p, i: {"status": "tool_error", "reason": "unknown fixture"})
    settings = config_for(tmp_path, method, budget=8)
    out = tmp_path / "run"
    with pytest.raises(SystemExit) as stopped:
        cli.main(["scan", "--method", method, "--settings", str(settings), "--output", str(out)])
    assert stopped.value.code == 3
    rows = [json.loads(s) for s in (out / "records.jsonl").read_text().splitlines()]
    assert [r["eval_id"] for r in rows] == [1, 2, 3, 4, 5]
    assert all(r["status"] == "tool_error" for r in rows)
    assert [i for p, i in calls] == [1, 2, 3, 4, 5]
    assert json.loads((out / "summary.json").read_text())["status"] == ("aborted" if method in ("modern", "random") else "interrupted")
    assert (out / "method_identity.json").exists()
    assert (out / "physics_identity.json").exists()
    assert (out / "operational_recovery.json").exists()


@pytest.mark.parametrize("method,scaling", [("strong-reference", "per-output mean/std"),
                                            ("historical", "no per-output")])
def test_public_cli_distinguishes_reference_preprocessing(tmp_path, monkeypatch, method, scaling):
    install_fake_backend(monkeypatch, valid)
    out = tmp_path / "run"
    settings = config_for(tmp_path, method)
    cli.main(["scan", "--method", method, "--settings", str(settings), "--output", str(out)])
    manifest = json.loads((out / "manifest.json").read_text())
    text = manifest["reconstruction_assumptions"]["output_scaling"]
    if method == "strong-reference":
        assert scaling in text
    else:
        assert "standard" not in text.lower() or "no" in text.lower()
        assert text != "log omega, then per-output mean/std from all current ok rows"


@pytest.mark.parametrize("case", ["resume", "probe", "nonempty_output"])
def test_unsafe_continuation_rejected_before_backend_or_journal_changes(tmp_path, monkeypatch, case):
    calls = []
    def forbidden_backend(*a, **kw):
        calls.append(True)
        raise AssertionError("No backend construction before unsafe continuation refusal")
    monkeypatch.setattr(cli, "PhysicsBackend", forbidden_backend)
    out = tmp_path / "output"
    argv = ["scan", "--output", str(out)]
    if case == "resume":
        argv += ["--resume", str(tmp_path / "checkpoint.json")]
    elif case == "probe":
        argv[0] = "probe"
    else:
        out.mkdir()
        (out / "records.jsonl").write_text('original partial evidence\n')
    with pytest.raises(SystemExit) as stopped:
        cli.main(argv)
    assert stopped.value.code == 2 and calls == []
    if case == "nonempty_output":
        assert (out / "records.jsonl").read_text() == 'original partial evidence\n'
    else:
        assert not out.exists()


def test_packaged_guard_and_strong_reference_are_exact_copies():
    package = Path(cli.__file__).parent
    expected = Path(__file__).resolve().parent / "_snapshots/expected_core"
    assert (package / "operational_recovery.py").read_bytes() == (expected / "operational_recovery.py").read_bytes()
    for name in ("baseline.py", "learning.py", "domain.py"):
        assert (package / "reference" / name).read_bytes() == (expected / name).read_bytes()


def test_default_strong_reference_settings_match_frozen_200m_comparator(tmp_path, monkeypatch):
    project = Path(cli.__file__).resolve().parents[2]
    snapshots = Path(__file__).resolve().parent / "_snapshots/expected_core"
    monkeypatch.chdir(project)
    args = cli.parser().parse_args(["scan", "--method", "strong-reference", "--output", str(tmp_path / "run")])
    actual = asdict(cli.load_settings(args))
    expected = json.loads((snapshots / "reference-typed-guard.json").read_text())
    original = json.loads((snapshots / "baseline-cap200m.json").read_text())
    assert actual == expected == {**original, "max_consecutive_tool_errors": original["budget"] + 1}
    assert actual["max_proposals"] == 200_000_000
    assert (project / "configs/strong-reference.json").read_bytes() == (snapshots / "reference-typed-guard.json").read_bytes()
