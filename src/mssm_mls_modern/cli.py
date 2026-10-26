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
from .operational_recovery import GuardedReplayBackend, OperationalSafetyStop, POLICY
from .reference.baseline import ScanSettings as ReferenceSettings, run_scan as run_reference


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
            c.add_argument("--method", choices=("modern", "random", "strong-reference", "historical") if name == "scan" else ("modern", "random"), default="modern",
                           help="strong-reference: standardized NN benchmark; historical: raw inputs plus log omega, a separate reconstruction")
            c.add_argument("--settings", type=Path)
            c.add_argument("--seed", type=int, default=0)
            c.add_argument("--cache", type=Path)
            c.add_argument("--reuse-run", type=Path, action="append", default=[])
        if name == "scan":
            c.add_argument("--budget", type=int)
            c.add_argument("--initial", type=int)
            c.add_argument("--resume", type=Path,
                           help="Rejected by the public CLI: partial-journal recovery requires an independently verified replay proof")
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
    reference = args.method == "strong-reference"
    filename = "strong-reference.json" if reference else "historical.json" if historical else "modern.json"
    path = args.settings or Path("configs") / filename
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
    # The public scan always uses the shared typed operational guard. Its
    # unknown-error limit is five; the coarse all-tool-error guard cannot stop
    # recognized numerical failures before the requested logical budget.
    if args.command == "scan":
        values["max_consecutive_tool_errors"] = values["budget"] + 1
    cls = ReferenceSettings if reference else ScanSettings if historical else ModernSettings
    return cls(**values)


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
        method_file = path / "method_identity.json"
        public_method = read_json(method_file)["public_method"] if method_file.exists() else manifest["method"]
        label = f"{public_method} seed {manifest['seed']} ({summary['status']})"
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
        if getattr(args, "resume", None) is not None:
            raise ValueError("Public --resume is disabled: a boundary checkpoint alone may omit completed partial-journal attempts. Preserve the original run; recovery requires a verified exact-prefix replay proof.")
        if args.command == "probe":
            raise ValueError("Public probe continuation is disabled: inherited truth requires a verified replay proof and attempt accounting. Use replay for offline analysis or scan for a fresh experiment.")
        if args.command != "doctor" and args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise ValueError("Output already contains data. Preserve this run; use a new directory for a new scan. Partial-journal recovery requires a verified replay proof.")
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
        # Guard failures are never relabelled as unphysical and never retried.
        guarded = GuardedReplayBackend(evaluator, args.output, physical)
        try:
            if args.method == "strong-reference":
                result = run_reference(settings, guarded, args.output, seed=args.seed, method="mls",
                                       checkpoint_path=args.output / "checkpoint.json",
                                       physics_fingerprint=checksum(physical), observer=guarded.observe)
            elif args.method == "historical":
                model = Surrogates(epochs=settings.epochs, learning_rate=settings.learning_rate,
                                   batch_size=settings.batch_size, seed=args.seed)
                result = run_historical(settings, guarded, args.output, seed=args.seed, method="mls",
                                        models=model, checkpoint_path=args.output / "checkpoint.json",
                                        resume_from=None, physics_fingerprint=checksum(physical), observer=guarded.observe)
            else:
                result = run_scan(settings, guarded, args.output, seed=args.seed, method=args.method,
                                  physics_identity=physical, resume=getattr(args, "resume", None),
                                  initial_records=initial_records, observer=guarded.observe)
        finally:
            if (args.output / "manifest.json").exists():
                method_labels = {
                    "modern": "GP active learning with resonance and HiggsBounds response features",
                    "random": "Uniform random scan",
                    "strong-reference": "Standardized NN MLS benchmark reference; not the historical raw-input version",
                    "historical": "Raw-input/log-omega reconstruction; not the strong benchmark reference",
                }
                save_json(args.output / "method_identity.json", {
                    "public_method": args.method, "description": method_labels[args.method],
                    "strong_benchmark_reference": args.method == "strong-reference",
                    "operational_policy": POLICY,
                    "runtime_source_sha256": {
                        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                        for name in ("cli.py", "runner.py", "operational_recovery.py",
                                     "reference/__init__.py", "reference/baseline.py",
                                     "reference/learning.py", "reference/domain.py")},
                    "unknown_error_limit": guarded.unknown_limit,
                    "coarse_error_limit": settings.max_consecutive_tool_errors,
                    "resume_scope": "Fresh scan; no partial-journal recovery accepted by public CLI"})
                save_json(args.output / "physics_identity.json", physical)
                save_json(args.output / "physics_manifest.json", readiness | {"configuration": backend.config})
                if probe_provenance is not None:
                    save_json(args.output / "probe_provenance.json", probe_provenance)
                if args.cache:
                    save_json(args.output / "execution.json", evaluator.report())
        print(json.dumps(result["summary"], indent=2))
    except OperationalSafetyStop as exc:
        print(f"Operational safety stop: {exc}. Completed attempts and diagnostic files were preserved.", file=sys.stderr)
        raise SystemExit(3) from None
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
