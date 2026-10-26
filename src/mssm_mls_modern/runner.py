"""Auditable modern scans with fixed validation, truth journals and checkpoints.

The sampler proposes parameters. Only the external backend determines truth.
Checkpoints are written at complete batch boundaries; partial work remains in
the journal/cache and is never presented as a completed batch.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import time
import numpy as np
import torch
from .domain import (sample, parameter_dict, validate_observables, score,
                     OBSERVABLE_NAMES, BOUNDS, FIXED_PARAMETERS)
from .historical import target_decision, target_summary, write_json
from .settings import ModernSettings
from .learning import EnsembleSurrogates
from .acquisition import propose_batch


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def code_identity():
    root = Path(__file__).parent
    names = ["runner.py", "settings.py", "learning.py", "acquisition.py", "domain.py",
             "historical.py", "backend.py", "cards.py", "execution.py", "gp_learning.py", "cli.py",
             "operational_recovery.py", "reference/__init__.py", "reference/baseline.py",
             "reference/learning.py", "reference/domain.py"]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def initial_split(settings, seed):
    """Choose validation indices before observing outcomes, from a separate RNG."""
    streams = np.random.SeedSequence(seed).spawn(4)
    rngs = [np.random.default_rng(s) for s in streams]
    count = int(settings.initial * settings.validation_fraction)
    indices = set(rngs[1].choice(settings.initial, count, replace=False).tolist())
    return indices, rngs


def mark_validation(records, settings, seed):
    """Copy an initial dataset, assigning the fixed preselected holdout flags."""
    if len(records) != settings.initial:
        raise ValueError("Expected exactly settings.initial saved records")
    indices, _ = initial_split(settings, seed)
    clean = deepcopy(records)
    seen = set()
    for i, row in enumerate(clean):
        x = list(parameter_dict(row["x"]).values())
        if tuple(x) in seen:
            raise ValueError("Saved initial records contain duplicate parameters")
        seen.add(tuple(x))
        if row["status"] not in {"ok", "unphysical", "tool_error"}:
            raise ValueError("Unknown saved truth status")
        hit = False
        if row["status"] == "ok":
            values = validate_observables(row["observables"])
            hit = bool(score(values.reshape(1, 4))[1][0])
        row.update(target_decision(row, hit, settings.target_policy))
        row.update(eval_id=i + 1, x=x, parameters=parameter_dict(x),
                   validation=i in indices, origin="saved_initial", iteration=0)
    return clean


def coverage(records, bins=10):
    """Descriptive fixed-grid occupancy, NOT a claim of full target coverage."""
    occupied, targets = set(), set()
    for row in records:
        index = tuple(np.minimum(bins - 1, np.floor(
            (np.array(row["x"]) - BOUNDS[:, 0]) / np.ptp(BOUNDS, axis=1) * bins
        ).astype(int)))
        occupied.add(index)
        if row["target"]:
            targets.add(index)
    return {"bins_per_axis": bins, "total_cells": bins**3,
            "visited_cells": len(occupied), "target_cells": len(targets),
            "interpretation": "Descriptive occupancy at this grid resolution; true target volume is unknown"}


def run_scan(settings, evaluator, output, *, seed=0, method="modern",
             physics_identity=None, resume=None, initial_records=None,
             models=None, observer=None):
    """Run new truth evaluations, optionally resuming a complete boundary.

    ``initial_records`` is reserved for explicitly labelled saved-initial probes,
    not used by the ordinary scan CLI. Probe summaries separate inherited truth
    from newly attempted evaluations. No fake evaluator is exposed by the CLI.

    ``observer(event)`` receives only new dictionaries of scalar operational
    metadata. Point events occur after journal and progress writes; callbacks
    must not mutate external sampler state. Safety stops inheriting BaseException
    propagate through the evidence-preserving finally block.
    """
    if type(seed) is not int or seed < 0 or method not in {"modern", "random"}:
        raise ValueError("Invalid seed or method")
    if resume is not None and initial_records is not None:
        raise ValueError("Cannot combine resume with a saved-initial probe")
    if observer is not None and not callable(observer):
        raise TypeError("observer must be callable")
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new or empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    evaluate = evaluator.evaluate if hasattr(evaluator, "evaluate") else evaluator
    if not callable(evaluate):
        raise TypeError("Backend must provide evaluate(parameters, eval_id)")
    validation_indices, rngs = initial_split(settings, seed)
    initial_rng, _, pool_rng, explore_rng = rngs
    identity = {"settings": asdict(settings), "seed": seed, "method": method,
                "physics": physics_identity, "code": code_identity(),
                "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                            "torch": str(torch.__version__), "system": platform.system(),
                            "machine": platform.machine(), "torch_device": "cpu", "torch_threads": 1}}
    records, history, iteration, inherited, consecutive = [], [], 0, 0, 0
    kind = "saved_initial_probe" if initial_records is not None else "fresh_scan"
    if models is not None:
        model = models
    elif method == 'modern':
        if settings.surrogate_kind == 'gp':
            from .gp_learning import GPSurrogates
            model = GPSurrogates(settings, seed)
        else:
            model = EnsembleSurrogates(settings, seed)
    else:
        model = None
    if resume is not None:
        saved = json.loads(Path(resume).read_text())
        payload = saved["payload"]
        if saved["sha256"] != checksum(payload):
            raise ValueError("Checkpoint checksum mismatch")
        if payload["identity"] != identity:
            raise ValueError("Checkpoint settings/code/physics identity differs")
        records, history = payload["records"], payload["history"]
        iteration, consecutive = payload["iteration"], payload["consecutive_tool_errors"]
        kind = payload["kind"]
        inherited = payload["inherited_initial_count"]
        pool_rng.bit_generator.state = payload["pool_rng"]
        explore_rng.bit_generator.state = payload["explore_rng"]
        if model is not None:
            model.fit_count = payload["fit_count"]
    elif initial_records is not None:
        records = mark_validation(initial_records, settings, seed)
        inherited = len(records)
    entry_count = len(records)
    manifest = {"algorithm": "modern_mls" if method == "modern" else "uniform_random",
                "method": method, "kind": kind, "status": "running", **identity,
                "bounds": BOUNDS.tolist(), "fixed_parameters": FIXED_PARAMETERS,
                "inherited_initial_count": inherited, "records_at_process_start": entry_count,
                "validation_eval_ids": sorted(i + 1 for i in validation_indices),
                "validation_scope": "Initial uniform subset reused for early stopping; not an independent final test set",
                "uncertainty_scope": "Conditional GP latent spread and approximate feature propagation; not calibrated coverage" if settings.surrogate_kind == 'gp' else "Ensemble disagreement heuristic, not calibrated coverage or posterior uncertainty",
                "numpy": np.__version__, "torch": str(torch.__version__),
                "new_physics_constraints": False}
    started = time.perf_counter()
    write_json(out / "manifest.json", manifest)
    with (out / "records.jsonl").open("w") as f:
        for row in records:
            f.write(json.dumps(row, allow_nan=False) + "\n")

    def notify(event, **details):
        if observer is not None:
            observer({"event": event, **details})

    def save_boundary():
        payload = {"identity": identity, "kind": kind, "records": records,
                   "history": history, "iteration": iteration,
                   "inherited_initial_count": inherited,
                   "consecutive_tool_errors": consecutive,
                   "pool_rng": pool_rng.bit_generator.state,
                   "explore_rng": explore_rng.bit_generator.state,
                   "fit_count": 0 if model is None else model.fit_count}
        write_json(out / "checkpoint.json", {"payload": payload, "sha256": checksum(payload)})
        write_json(out / "training_history.json", history)

    def evaluate_batch(batch, current_iteration):
        nonlocal consecutive
        for item in batch:
            parameters = parameter_dict(item["x"])
            eval_id = len(records) + 1
            vector, observations, log_weight, hit = None, None, None, False
            try:
                answer = evaluate(parameters, eval_id)
                if not isinstance(answer, dict) or answer.get("status") not in {"ok", "unphysical", "tool_error"}:
                    raise ValueError("Backend returned an invalid status")
                if answer["status"] == "ok":
                    vector = validate_observables(answer.get("observables"))
                    observations = dict(zip(OBSERVABLE_NAMES, map(float, vector)))
                    weights, hits = score(vector.reshape(1, 4))
                    log_weight = float(weights[0]) if np.isfinite(weights[0]) else None
                    hit = bool(hits[0])
            except Exception as exc:
                answer = {"status": "tool_error", "reason": f"{type(exc).__name__}: {exc}"}
                vector, observations, log_weight, hit = None, None, None, False
            row = {"eval_id": eval_id, "x": list(parameters.values()), "parameters": parameters,
                   "status": answer["status"], "observables": observations,
                   "y": None if vector is None else vector.tolist(), "log_score": log_weight,
                   **target_decision(answer, hit, settings.target_policy),
                   "origin": item["origin"], "iteration": current_iteration,
                   "predicted": item.get("predicted"),
                   "validation": current_iteration == 0 and eval_id - 1 in validation_indices,
                   "diagnostics": answer.get("diagnostics", {}),
                   "provenance": answer.get("provenance", {}), "reason": answer.get("reason", "")}
            for key in ("uncertainty", "acquisition", "physical_probability", "hb_probability", "solver_probability", "candidate_source", "raw_acquisition", "penalized_acquisition", "min_distance", "raw_utility_gap", "log_hb_ratio_mean", "log_hb_ratio_std"):
                if key in item:
                    row[key] = item[key]
            records.append(row)
            with (out / "records.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            consecutive = consecutive + 1 if row["status"] == "tool_error" else 0
            write_json(out / "progress.json", {"attempts": len(records), "iteration": current_iteration,
                                                "last_status": row["status"], "elapsed_seconds": time.perf_counter() - started})
            notify("point", eval_id=eval_id, status=row["status"])
            if consecutive >= settings.max_consecutive_tool_errors:
                raise RuntimeError("Repeated technical failures; inspect preserved physics logs")

    try:
        notify("resumed" if resume is not None else "started")
        if not records:
            evaluate_batch([{"x": x, "origin": "initial", "predicted": None}
                            for x in sample(settings.initial, initial_rng)], 0)
            save_boundary()
        elif resume is None:
            save_boundary()
        while len(records) < settings.budget:
            iteration += 1
            size = min(settings.exploit + settings.uncertain + settings.random_extra,
                       settings.budget - len(records))
            info = {"iteration": iteration, "trained_through": len(records), "batch_size": size}
            history.append(info)
            if method == "random":
                batch = []
                seen = {tuple(r["x"]) for r in records}
                while len(batch) < size:
                    x = sample(1, explore_rng)[0]
                    if tuple(x) not in seen:
                        batch.append({"x": x, "origin": "random", "predicted": None})
                        seen.add(tuple(x))
            else:
                info["fit"] = model.fit(records)
                batch, info["selection"] = propose_batch(model, records, settings, pool_rng,
                                                         count=size, explore_rng=explore_rng)
            evaluate_batch(batch, iteration)
            save_boundary()
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "aborted",
                        error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        new_records = records[entry_count:]
        probe_new = records[inherited:]
        summary = {"status": manifest["status"], "kind": kind, "attempts": len(records),
                   **dict(Counter(r["status"] for r in records)),
                   **target_summary(records, settings.target_policy),
                   "new_attempts_this_process": len(new_records),
                   "inherited_initial_count": inherited,
                   "post_initial_probe_attempts": len(probe_new) if inherited else None,
                   "post_initial_probe_targets": sum(r["target"] for r in probe_new) if inherited else None,
                   "post_initial_probe_eq7": sum(r["target_eq7"] for r in probe_new) if inherited else None,
                   "coverage": coverage(records),
                   "origins": dict(Counter(r["origin"] for r in records)),
                   "validation_count": sum(r.get("validation", False) for r in records),
                   "wall_seconds_this_process": time.perf_counter() - started,
                   "note": "All targets verified by true observables; validation reused for early stopping, not a final unbiased test; occupancy is not complete coverage."}
        write_json(out / "manifest.json", manifest)
        write_json(out / "summary.json", summary)
        write_json(out / "training_history.json", history)
        notify("finished", status=manifest["status"])
    if model is not None and model.ready:
        torch.save(model.export(), out / "surrogates.pt")
    return {"records": records, "history": history, "summary": summary, "manifest": manifest}
