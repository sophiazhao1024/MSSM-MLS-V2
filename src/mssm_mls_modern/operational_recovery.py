"""Outcome-blind, budget-preserving operational guard and exact partial replay.

This module changes neither a physics answer nor a proposal. Expected numerical
failure remains tool_error. The scanner's coarse guard is budget+1; this layer
stops after five consecutive unidentified operational failures AFTER journaling.
It is shared by all methods. No failed point is automatically retried/cached.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

POLICY = "typed-darkomega-guard-v1"
PARAMETERS = ("M1", "mu", "tan_beta")


class OperationalSafetyStop(BaseException):
    """Not swallowed as scientific tool_error or ignored observer exception."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def recognized_numerical_failure(answer, physics_identity):
    """Narrow observed signature only; no physics output/target classification."""
    p = answer.get("provenance", {})
    mo = p.get("micromegas", {})
    commands = p.get("commands", [])
    return (answer.get("status") == "tool_error"
            and {"config": p.get("config_sha256"), "adapter": p.get("adapter_sha256"),
                 "binaries": p.get("binary_sha256")} == physics_identity
            and p.get("paper_version_match") is True
            and p.get("higgs", {}).get("status") == "ok"
            and [(c.get("stage"), c.get("returncode")) for c in commands] == [("higgs", 0), ("omega", 70)]
            and mo.get("status") == "technical_failure"
            and mo.get("reason") == "darkOmega_failed"
            and mo.get("source_version") == "4.3.2")


def advance_streak(streak, answer, identity):
    known = recognized_numerical_failure(answer, identity)
    return (streak + 1 if answer.get("status") == "tool_error" and not known else 0), known


def canonical_prefix(path, count):
    h = hashlib.sha256()
    with Path(path).open() as stream:
        for index in range(1, count + 1):
            line = stream.readline()
            if not line:
                raise OperationalSafetyStop("Scientific prefix is shorter than its recorded boundary")
            row = json.loads(line)
            if row.get("eval_id") != index:
                raise OperationalSafetyStop("Scientific prefix has unexpected evaluation IDs")
            h.update(canonical(row) + b"\n")
    return h.hexdigest()


class GuardedReplayBackend:
    """Evaluator + after-journal callback; call observe(event) for every event.

    source_run/checkpoint_count are optional. In a fresh run it is just the
    typed guard. In a resumed run it replays only the interrupted own-run suffix
    and requires EXACT request and full scientific-record equality. Replayed
    failures never reach the shared cache. Every original attempt counts once.
    """
    def __init__(self, evaluator, output, physics_identity, *, source_run=None,
                 checkpoint_count=0, prefix_count=0, unknown_limit=5):
        self.evaluator, self.output = evaluator, Path(output)
        self.identity, self.unknown_limit = deepcopy(physics_identity), unknown_limit
        self.source_run = None if source_run is None else Path(source_run)
        self.checkpoint_count, self.prefix_count = checkpoint_count, prefix_count
        self.streak = self.replayed = self.new_requests = self.known_failures = 0
        self.pending = None
        self.cursor = None
        self.expected = []
        self.prefix_verified = self.source_run is None
        self.next_id = checkpoint_count + 1
        self.report_path = self.output / "operational_recovery.json"
        if self.source_run is not None:
            if prefix_count < checkpoint_count:
                raise ValueError("Replay prefix cannot precede its checkpoint")
            with (self.source_run / "records.jsonl").open() as stream:
                for index, line in enumerate(stream, 1):
                    row = json.loads(line)
                    if row.get("eval_id") != index:
                        raise ValueError("Replay source has invalid evaluation sequence")
                    if index <= checkpoint_count:
                        self.streak, _ = advance_streak(self.streak, row, self.identity)
                    else:
                        self.expected.append(row)
            if len(self.expected) != prefix_count - checkpoint_count:
                raise ValueError("Replay source length differs from frozen prefix count")
            self.source_sha256 = file_hash(self.source_run / "records.jsonl")
            self.source_prefix_hash = canonical_prefix(self.source_run / "records.jsonl", prefix_count)
            self.prefix_verified = prefix_count == checkpoint_count
            if self.streak >= unknown_limit:
                raise ValueError("Checkpoint already contains an unidentified-failure stop")
        else:
            if checkpoint_count or prefix_count:
                raise ValueError("Fresh guard cannot silently inherit attempts")
            self.source_sha256 = self.source_prefix_hash = None

    def evaluate(self, parameters, eval_id):
        if self.pending is not None or eval_id != self.next_id:
            raise OperationalSafetyStop("Unexpected evaluation order or missing after-journal callback")
        if eval_id <= self.prefix_count:
            row = self.expected[eval_id - self.checkpoint_count - 1]
            if any(float(parameters[k]).hex() != float(row["parameters"][k]).hex() for k in PARAMETERS):
                raise OperationalSafetyStop("Replayed proposal differs from the original exact input; no backend call made")
            answer = {k: deepcopy(row[k]) for k in ("status", "reason", "provenance", "diagnostics")}
            answer["observables"] = deepcopy(row["observables"] or {})
            self.pending = (eval_id, answer, row)
            return answer
        if not self.prefix_verified:
            raise OperationalSafetyStop("New physics forbidden before complete interrupted-prefix validation")
        self.new_requests += 1
        try:
            answer = self.evaluator.evaluate(parameters, eval_id)
        except Exception as exc:
            answer = {"status": "tool_error", "reason": f"{type(exc).__name__}: {exc}"}
        self.pending = (eval_id, answer, None)
        return answer

    def observe(self, event):
        try:
            return self._observe(event)
        except Exception as exc:
            raise OperationalSafetyStop(f"Operational observer verification failed: {type(exc).__name__}") from exc

    def _observe(self, event):
        if event["event"] == "finished":
            if self.cursor is not None:
                self.cursor.close()
                self.cursor = None
            self.flush()
            return
        if event["event"] in {"started", "resumed"}:
            journal = self.output / "records.jsonl"
            if self.source_run is not None and canonical_prefix(journal, self.checkpoint_count) != canonical_prefix(
                    self.source_run / "records.jsonl", self.checkpoint_count):
                raise OperationalSafetyStop("Inherited checkpoint records differ from original journal")
            self.cursor = journal.open() if journal.exists() else None
            if self.cursor is not None:
                self.cursor.seek(0, 2)
            self.flush()
            return
        if event["event"] != "point":
            return
        if self.pending is None or self.pending[0] != event["eval_id"]:
            raise OperationalSafetyStop("Journal callback did not match a completed evaluator return")
        eval_id, answer, expected = self.pending
        if self.cursor is None:
            self.cursor = (self.output / "records.jsonl").open()
        actual = json.loads(self.cursor.readline())
        if actual.get("eval_id") != eval_id or actual.get("status") != answer.get("status"):
            raise OperationalSafetyStop("Journal changed the completed answer or evaluation sequence")
        if expected is not None:
            if canonical(actual) != canonical(expected):
                raise OperationalSafetyStop("Replayed scientific record differs from original, including predictions/provenance")
            self.replayed += 1
            if eval_id == self.prefix_count:
                if canonical_prefix(self.output / "records.jsonl", self.prefix_count) != self.source_prefix_hash:
                    raise OperationalSafetyStop("Full interrupted-prefix verification failed")
                self.prefix_verified = True
        self.streak, known = advance_streak(self.streak, answer, self.identity)
        self.known_failures += int(known)
        entry = {"eval_id": eval_id, "status": answer.get("status"),
                 "recognized_numerical_failure": known, "unidentified_failure_streak": self.streak,
                 "own_prefix_replay": expected is not None,
                 "new_backend_requests": 0 if expected is not None else 1,
                 "record_sha256": hashlib.sha256(canonical(actual)).hexdigest()}
        with (self.output / "operational_guard.jsonl").open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        self.pending = None
        self.next_id += 1
        self.flush()
        if self.streak >= self.unknown_limit:
            raise OperationalSafetyStop("Five consecutive unidentified technical failures; completed answers retained")

    def report(self):
        return {"policy": POLICY, "unknown_failure_limit": self.unknown_limit,
                "source_run": None if self.source_run is None else str(self.source_run.resolve()),
                "checkpoint_attempts": self.checkpoint_count, "source_prefix_attempts": self.prefix_count,
                "source_journal_sha256": self.source_sha256,
                "source_prefix_canonical_sha256": self.source_prefix_hash,
                "replayed_attempts": self.replayed, "replayed_new_external_calls": 0,
                "new_evaluator_requests": self.new_requests,
                "prefix_verified": self.prefix_verified, "next_eval_id": self.next_id,
                "recognized_failure_returns_this_process": self.known_failures,
                "unidentified_failure_streak": self.streak,
                "budget_rule": "Inherited boundary and interrupted-prefix attempts count once; no retry, relabeling or failure cache insertion.",
                "cost_rule": "Predecessor execution costs and isolated diagnostics are separate overhead, never erase failures or sum logical budgets."}

    def flush(self):
        save_json(self.report_path, self.report())
