"""Public entry points. Ordinary scans always use real physics programs."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from .backend import PhysicsBackend, read_json, save_json
from .domain import score, validate_observables
from .historical import (target_decision, ScanSettings, run_scan as run_historical,
                         rejection_sample, ProposalExhausted)
from .historical_learning import Surrogates
from .settings import ModernSettings
from .learning import EnsembleSurrogates
from .acquisition import propose_batch
from .runner import run_scan, mark_validation, initial_split, checksum


def parser():
    p = argparse.ArgumentParser(prog="mssm-mls-modern", description="MLS methods for MSSM parameter scans")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("doctor", "point", "scan", "probe"):
        c = sub.add_parser(name)
        c.add_argument("--backend", type=Path, default=Path("configs/backend.local.json"))
        if name != "doctor":
            c.add_argument("--output", type=Path, required=True)
            c.add_argument("--target-policy", choices=("paper-higgs", "eq7-only"))
        if name == "point":
            c.add_argument("--M1", type=float, required=True)
            c.add_argument("--mu", type=float, required=True)
            c.add_argument("--tan-beta", type=float, required=True)
        if name in ("scan", "probe"):
            c.add_argument("--method", choices=("modern", "random", "historical") if name == "scan" else ("modern", "random"), default="modern")
            c.add_argument("--settings", type=Path)
            c.add_argument("--seed", type=int, default=0)
            c.add_argument("--cache", type=Path)
            c.add_argument("--reuse-run", type=Path, action="append", default=[])
        if name == "scan":
            c.add_argument("--budget", type=int)
            c.add_argument("--initial", type=int)
            c.add_argument("--resume", type=Path,
                           help="Complete-boundary checkpoint, same settings/code/physics, new output directory")
        if name == "probe":
            c.add_argument("--records", type=Path, default=Path("validation/initial1000_records.jsonl"))
            c.add_argument("--new-budget", type=int, default=315)
    c = sub.add_parser("replay", help="Saved-truth training and recommendations; no new true evaluations")
    c.add_argument("--records", type=Path, default=Path("validation/initial1000_records.jsonl"))
    c.add_argument("--settings", type=Path)
    c.add_argument("--method", choices=("modern", "historical"), default="modern")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs="+", type=Path)
    c.add_argument("--output", type=Path, required=True)
    return p


def load_records(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def load_settings(args):
    historical = args.method == "historical"
    path = args.settings or Path("configs/historical.json" if historical else "configs/modern.json")
    values = read_json(path)
    for key in ("budget", "initial"):
        override = getattr(args, key, None)
        if override is not None:
            values[key] = override
    if getattr(args, "target_policy", None) is not None:
        values["target_policy"] = args.target_policy
    if args.command == "probe":
        if type(args.new_budget) is not int or args.new_budget <= 0:
            raise ValueError("new-budget must be positive")
        values["budget"] = values["initial"] + args.new_budget
    return ScanSettings(**values) if historical else ModernSettings(**values)


def offline_replay(args):
    settings = load_settings(args)
    records = load_records(args.records)
    if len(records) != settings.initial:
        raise ValueError("Saved record count must equal configured initial size")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    report = {"kind": "saved_truth_replay_only", "method": args.method, "seed": args.seed,
              "settings": asdict(settings), "record_count": len(records),
              "input_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
              "new_physics_calls": 0, "full_scan_completed": False}
    if args.method == "historical":
        model = Surrogates(epochs=settings.epochs, learning_rate=settings.learning_rate,
                           batch_size=settings.batch_size, seed=args.seed)
        report["fit"] = model.fit(records)
        rng = np.random.default_rng(np.random.SeedSequence(args.seed).spawn(3)[1])
        try:
            batch, report["selection"] = rejection_sample(model, settings.guided, settings, rng, iteration=1)
            report["status"] = "recommendations_generated"
        except ProposalExhausted as exc:
            batch = []
            report.update(status="proposal_exhausted", selection=exc.diagnostics)
    else:
        records = mark_validation(records, settings, args.seed)
        if settings.surrogate_kind == 'gp':
            from .gp_learning import GPSurrogates
            model = GPSurrogates(settings, args.seed)
        else:
            model = EnsembleSurrogates(settings, args.seed)
        report["fit"] = model.fit(records)
        _, streams = initial_split(settings, args.seed)
        batch, report["selection"] = propose_batch(model, records, settings, streams[2], explore_rng=streams[3])
        report["status"] = "recommendations_generated_not_physically_evaluated"
    report["recommendations"] = [{**item, "x": np.asarray(item["x"]).tolist()} for item in batch]
    save_json(out / "report.json", report)
    import torch
    torch.save(model.export(), out / "surrogates.pt")
    return {k: v for k, v in report.items() if k not in ("fit", "recommendations")}


def compare_runs(paths, output):
    """Reject mixed truth definitions or physics identities; retain partial runs."""
    runs = [(path, read_json(path / "manifest.json"), read_json(path / "summary.json"),
             load_records(path / "records.jsonl")) for path in paths]
    policies = {m["settings"].get("target_policy", "eq7-only") for _, m, _, _ in runs}
    if len(policies) != 1:
        raise ValueError("Cannot compare different target policies")
    kinds = {m.get("kind", "fresh_scan") for _, m, _, _ in runs}
    identities = {checksum(read_json(p / "physics_identity.json")) for p, _, _, _ in runs}
    if len(policies) != 1 or len(kinds) != 1 or len(identities) != 1:
        raise ValueError("Comparison requires identical target policy, run kind and physics identity")
    if len({m["settings"]["budget"] for _, m, _, _ in runs}) != 1:
        raise ValueError("Comparison requires the same requested logical budget")
    initial_counts = {m["settings"]["initial"] for _, m, _, _ in runs}
    if len(initial_counts) != 1:
        raise ValueError("Comparison requires the same initial budget")
    initial_count = next(iter(initial_counts))
    if len({checksum([r["x"] for r in rows[:initial_count]]) for _, _, _, rows in runs}) != 1:
        raise ValueError("Use matching initial coordinates for a paired comparison")
    initial_truth = {checksum([{k: r.get(k) for k in ("x", "status", "observables", "hb_result")}
                               for r in rows[:initial_count]]) for _, _, _, rows in runs}
    if len(initial_truth) != 1:
        raise ValueError("Paired comparison requires identical initial true outcomes")
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    result = []
    for path, manifest, summary, rows in runs:
        label = f"{manifest['method']} seed {manifest['seed']} ({summary['status']})"
        x = np.arange(1, len(rows) + 1)
        y = np.cumsum([row["target"] for row in rows])
        axes[0].plot(x, y, label=label)
        axes[1].plot(x, y / x, label=label)
        result.append({"run_name": path.name, "summary": summary})
    for ax, ylabel in zip(axes, ("True allowed targets", "Cumulative target fraction")):
        ax.set(xlabel="Logical true-evaluation attempts", ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)
    save_json(output / "comparison.json", result)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "replay":
            print(json.dumps(offline_replay(args), indent=2))
            return
        if args.command == "compare":
            print(json.dumps(compare_runs(args.runs, args.output), indent=2))
            return
        backend = PhysicsBackend(args.backend, (args.output / "physics") if args.command != "doctor" else Path("runs/unused"))
        readiness = backend.doctor()
        if args.command == "doctor":
            print(json.dumps(readiness, indent=2))
            if not readiness["ready"]:
                raise SystemExit(2)
            return
        if not readiness["ready"]:
            raise RuntimeError("Physics tools are not ready; build them and run doctor")
        if args.command == "point":
            args.output.mkdir(parents=True, exist_ok=False)
            answer = backend.evaluate({"M1": args.M1, "mu": args.mu, "tan_beta": args.tan_beta}, 1)
            hit = False
            if answer["status"] == "ok":
                hit = bool(score(validate_observables(answer["observables"]).reshape(1, 4))[1][0])
            answer.update(target_decision(answer, hit, args.target_policy or "paper-higgs"))
            answer["eq7_target"] = answer["target_eq7"]
            save_json(args.output / "point.json", answer)
            print(json.dumps(answer, indent=2))
            if answer["status"] == "tool_error":
                raise SystemExit(2)
            return
        settings = load_settings(args)
        physical = {"config": readiness["config_sha256"], "adapter": readiness["adapter_sha256"],
                    "binaries": {key: readiness["checks"][key]["sha256"] for key in ("fh_driver", "omega_driver")}}
        evaluator = backend
        if args.reuse_run and args.cache is None:
            raise ValueError("reuse-run requires cache")
        if args.cache:
            from .execution import CachedBackend
            evaluator = CachedBackend(backend, args.cache, workers=1)
            for previous in args.reuse_run:
                evaluator.import_run(previous)
        initial_records = None
        probe_provenance = None
        if args.command == "probe":
            # Saved fixture is specifically tied to this historical profile.
            # A separate physical upgrade must not silently reuse these labels.
            example = read_json(Path("configs/backend.example.json"))
            for key in ("versions", "top_mass", "options", "micromegas"):
                if backend.config.get(key) != example[key]:
                    raise ValueError("Probe requires the recorded historical physics profile; use a fresh scan for changed physics")
            source = read_json(Path("validation/data_provenance.json"))
            digest = hashlib.sha256(args.records.read_bytes()).hexdigest()
            if digest != source["public_file_sha256"]:
                raise ValueError("Probe requires the published verified saved-initial fixture; use scan for new datasets")
            probe_provenance = {"fixture_sha256": digest,
                "inherited_initial_count": source["records"],
                "historical_binary_sha256": source["historical_binary_sha256"],
                "current_binary_sha256": physical["binaries"],
                "exact_original_binary_match": source["historical_binary_sha256"] == physical["binaries"],
                "scope": "Conditional follow-up to archived true initial evaluations; not a new full scan"}
            initial_records = load_records(args.records)
        try:
            if args.method == "historical":
                model = Surrogates(epochs=settings.epochs, learning_rate=settings.learning_rate,
                                   batch_size=settings.batch_size, seed=args.seed)
                result = run_historical(settings, evaluator, args.output, seed=args.seed, method="mls",
                                        models=model, checkpoint_path=args.output / "checkpoint.json",
                                        resume_from=args.resume, physics_fingerprint=checksum(physical))
            else:
                result = run_scan(settings, evaluator, args.output, seed=args.seed, method=args.method,
                                  physics_identity=physical, resume=getattr(args, "resume", None),
                                  initial_records=initial_records)
        finally:
            if (args.output / "manifest.json").exists():
                save_json(args.output / "physics_identity.json", physical)
                save_json(args.output / "physics_manifest.json", readiness | {"configuration": backend.config})
                if probe_provenance is not None:
                    save_json(args.output / "probe_provenance.json", probe_provenance)
                if args.cache:
                    save_json(args.output / "execution.json", evaluator.report())
        print(json.dumps(result["summary"], indent=2))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
