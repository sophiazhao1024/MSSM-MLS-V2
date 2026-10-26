"""Portable fresh benchmarks and narrow complete-boundary strong cap recovery.

No calculation starts on import. No workspace paths, implicit saved labels or
shared cache are used. Public --resume remains disabled. Recovery here is only
for a proved strong-reference ProposalExhausted at an exact complete boundary;
partial-journal, modern, historical and arbitrary interruption recovery refuse.
"""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[name]="1"
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"src"))
from mssm_mls_modern import cli
from mssm_mls_modern.backend import PhysicsBackend,save_json
from mssm_mls_modern.execution import CachedBackend
from mssm_mls_modern.operational_recovery import GuardedReplayBackend,OperationalSafetyStop,file_hash,canonical
from mssm_mls_modern.reference.baseline import ScanSettings,run_scan as run_reference
from mssm_mls_modern.reference.learning import Surrogates
from mssm_mls_modern.runner import checksum

CAPS=(200_000_000,1_000_000_000,5_000_000_000)


def load(path):
    def invalid(value):raise ValueError("Nonfinite JSON: "+value)
    return json.loads(Path(path).read_text(),parse_constant=invalid)


def payload_bytes(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()


def inventory():
    proof=load(PROJECT/"docs/core_preservation.json")
    wanted={**proof["source_core_sha256"],**proof["source_configuration_sha256"]}
    actual={str(p.relative_to(PROJECT)):file_hash(p) for tree,pattern in (("src","*.py"),("configs","*"))
            for p in sorted((PROJECT/tree).rglob(pattern)) if p.is_file() and (tree=="src" or p.suffix==".json" and p.name in {Path(n).name for n in proof["source_configuration_sha256"]})}
    # Local built backend files may be added by setup; every src module and all
    # five frozen scientific presets are required. Extra source is rejected.
    if actual!=wanted:raise ValueError("Frozen V5 source/preset inventory changed")
    return actual


def new_output(path):
    path=Path(path).resolve()
    if path.exists():raise ValueError("Use a new output path; preserve previous evidence")
    return path


def run_fresh(args):
    out=new_output(args.output);core=inventory()
    settings=Path(args.settings or PROJECT/"configs"/("strong-reference.json" if args.method=="strong-reference" else "modern.json")).resolve()
    request={"schema":1,"kind":"portable_fresh_benchmark","method":args.method,"seed":args.seed,
        "budget":args.budget,"initial":args.initial,"settings_file":str(settings),"settings_sha256":file_hash(settings),
        "source_and_presets_sha256":core,"tool_sha256":file_hash(__file__),"output":str(out),
        "cache":None if args.cache is None else str(args.cache.resolve()),
        "implicit_initial_fixture":False,"utc":datetime.now(timezone.utc).isoformat()}
    prestart=out.with_name(out.name+".prestart.json");prestart.parent.mkdir(parents=True,exist_ok=True)
    with prestart.open("x") as stream:json.dump(request,stream,indent=2);stream.write("\n")
    argv=["scan","--method",args.method,"--backend",str(args.backend.resolve()),"--settings",str(settings),
        "--seed",str(args.seed),"--budget",str(args.budget),"--initial",str(args.initial),"--output",str(out)]
    if args.cache is not None:argv += ["--cache",str(args.cache.resolve())]
    code=0
    try:
        cli.main(argv)
    except SystemExit as exc:
        code=int(exc.code or 0)
    finally:
        if (out/"manifest.json").exists():
            status=load(out/"manifest.json")["status"]
            save_json(out/"portable_run.json",request|{"prestart_sha256":file_hash(prestart),"status":status,"exit_code":code})
            save_json(out/"launcher_status.json",{"status":status,"exit_code":code,"utc":datetime.now(timezone.utc).isoformat()})
    return code


def cap_source(source,next_cap):
    """Revalidate source and derive ONLY the permitted cap field in memory."""
    source=Path(source).resolve();inventory()
    identity=load(source/"method_identity.json")
    if identity.get("public_method")!="strong-reference":raise ValueError("Only strong-reference cap recovery is supported")
    manifest,summary=load(source/"manifest.json"),load(source/"summary.json")
    if manifest.get("status")!=summary.get("status") or manifest.get("status")!="proposal_exhausted":
        raise ValueError("Only a closed ProposalExhausted run is eligible")
    envelope=load(source/"checkpoint.json");payload=envelope["payload"]
    if hashlib.sha256(payload_bytes(payload)).hexdigest()!=envelope.get("sha256"):
        raise ValueError("Source checkpoint checksum mismatch")
    old=payload["identity"]["settings"]["max_proposals"]
    if old not in CAPS[:-1] or next_cap!=CAPS[CAPS.index(old)+1]:raise ValueError("Use exactly the next cap: 200m -> 1b -> 5b")
    if not str(manifest.get("error","")).startswith("ProposalExhausted:") or f"from {old} proposals" not in manifest["error"]:
        raise ValueError("Failure did not exhaust the recorded cap")
    settings=payload["identity"]["settings"]
    if settings!=manifest["settings"] or settings.get("max_consecutive_tool_errors")!=settings["budget"]+1:
        raise ValueError("Source settings/typed guard mismatch")
    if payload.get("schema")!=1 or payload.get("kind")!="complete_iteration":raise ValueError("Not a complete boundary checkpoint")
    science={n:file_hash(PROJECT/"src/mssm_mls_modern/reference"/n) for n in ("baseline.py","learning.py","domain.py")}
    if payload["identity"].get("code_sha256")!=science or payload["identity"].get("method")!="mls":raise ValueError("Strong source identity mismatch")
    physical=load(source/"physics_identity.json")
    if payload["identity"]["physics_fingerprint"]!=checksum(physical):raise ValueError("Source physical identity mismatch")
    rows=[load_line for line in (source/"records.jsonl").read_text().splitlines() if line.strip() for load_line in [json.loads(line)]]
    if canonical(rows)!=canonical(payload["records"]):raise ValueError("Journal is not exactly the complete checkpoint; partial recovery is unsupported")
    n=len(rows);iteration=payload["completed_iteration"]
    if (type(iteration) is not int or iteration<0 or n!=min(settings["initial"]+iteration*(settings["guided"]+settings["random_extra"]),settings["budget"])
            or len(payload["history"])!=iteration or payload["fit_count"]!=iteration
            or [r.get("eval_id") for r in rows]!=list(range(1,n+1)) or summary.get("attempts")!=n):
        raise ValueError("Source is not a consistent complete boundary")
    failed_history=load(source/"training_history.json")
    if not failed_history or failed_history[-1].get("selection",{}).get("proposals")!=old:
        raise ValueError("Missing exact exhausted-cap diagnostics")
    derived=deepcopy(payload);derived["identity"]["settings"]["max_proposals"]=next_cap
    reversed_copy=deepcopy(derived);reversed_copy["identity"]["settings"]["max_proposals"]=old
    if canonical(reversed_copy)!=canonical(payload):raise ValueError("Checkpoint changed beyond proposal cap")
    bindings={name:file_hash(source/name) for name in ("manifest.json","summary.json","checkpoint.json","records.jsonl","training_history.json","physics_identity.json","method_identity.json")}
    return source,payload,derived,physical,bindings


def prepare_cap(source,output,next_cap,bundle):
    output=new_output(output);bundle=new_output(bundle)
    source,payload,derived,physical,bindings=cap_source(source,next_cap)
    paths=(source,output,bundle)
    if any(a==b or a.is_relative_to(b) or b.is_relative_to(a)
           for i,a in enumerate(paths) for b in paths[i+1:]):
        raise ValueError("Source, proof bundle and new run must be separate nonnested paths")
    bundle.mkdir(parents=True)
    save_json(bundle/"checkpoint.json",{"sha256":hashlib.sha256(payload_bytes(derived)).hexdigest(),"payload":derived})
    save_json(bundle/"settings.json",derived["identity"]["settings"])
    proof={"schema":1,"kind":"portable_strong_complete_boundary_cap_recovery","source_run":str(source),"output_run":str(output),
        "source_files_sha256":bindings,"source_and_presets_sha256":inventory(),"helper_sha256":file_hash(__file__),
        "seed":payload["identity"]["seed"],"before_cap":payload["identity"]["settings"]["max_proposals"],"after_cap":next_cap,
        "completed_attempts":len(payload["records"]),"physics_identity":physical,
        "checkpoint_file_sha256":file_hash(bundle/"checkpoint.json"),"settings_file_sha256":file_hash(bundle/"settings.json"),
        "changed_payload_fields":["identity.settings.max_proposals"],"all_other_payload_equal":True,
        "utc":datetime.now(timezone.utc).isoformat(),
        "scope":"No scan started. Complete boundary only; no partial-journal recovery, label changes, retry or budget reset. Future trajectory equivalence after a finite-cap exhaustion is not claimed; retain predecessor costs separately."}
    save_json(bundle/"proof.json",proof)
    return {"proof":str(bundle/"proof.json"),"proof_sha256":file_hash(bundle/"proof.json"),"output":str(output),"scan_started":False}


def resume_cap(proof_path,backend_path,cache=None,*,backend_factory=None,model_factory=None):
    proof_path=Path(proof_path).resolve();proof=load(proof_path);bundle=proof_path.parent
    if proof.get("kind")!="portable_strong_complete_boundary_cap_recovery" or proof.get("helper_sha256")!=file_hash(__file__):raise ValueError("Recovery proof/helper mismatch")
    out=new_output(proof["output_run"])
    source,payload,derived,physical,bindings=cap_source(proof["source_run"],proof["after_cap"])
    if bindings!=proof["source_files_sha256"] or inventory()!=proof["source_and_presets_sha256"]:
        raise ValueError("Recovery source or package changed")
    if proof["seed"]!=payload["identity"]["seed"] or proof["completed_attempts"]!=len(payload["records"]) or proof["physics_identity"]!=physical:
        raise ValueError("Recovery proof identity or budget mismatch")
    envelope=load(bundle/"checkpoint.json");settings_values=load(bundle/"settings.json")
    if (file_hash(bundle/"checkpoint.json")!=proof["checkpoint_file_sha256"] or file_hash(bundle/"settings.json")!=proof["settings_file_sha256"]
            or envelope!={"sha256":hashlib.sha256(payload_bytes(derived)).hexdigest(),"payload":derived}
            or settings_values!=derived["identity"]["settings"]):raise ValueError("Derived checkpoint/settings changed")
    backend=(backend_factory or PhysicsBackend)(backend_path,out/"physics");ready=backend.doctor()
    actual={"config":ready["config_sha256"],"adapter":ready["adapter_sha256"],"binaries":{n:ready["checks"][n]["sha256"] for n in ("fh_driver","omega_driver")}}
    if not ready["ready"] or actual!=physical:raise ValueError("Recovery requires the same real physics identity")
    evaluator=CachedBackend(backend,cache) if cache is not None else backend
    n=len(payload["records"])
    guarded=GuardedReplayBackend(evaluator,out,actual,source_run=source,checkpoint_count=n,prefix_count=n)
    settings=ScanSettings(**settings_values)
    model=(model_factory or Surrogates)(epochs=settings.epochs,learning_rate=settings.learning_rate,batch_size=settings.batch_size,seed=proof["seed"])
    code=0;error=None
    try:
        result=run_reference(settings,guarded,out,seed=proof["seed"],method="mls",models=model,
            checkpoint_path=out/"checkpoint.json",resume_from=bundle/"checkpoint.json",physics_fingerprint=checksum(actual),observer=guarded.observe)
        status=result["summary"]["status"]
    except OperationalSafetyStop as exc:
        code,error,status=3,str(exc),"operational_safety_stop"
    except KeyboardInterrupt:
        code,error,status=130,"KeyboardInterrupt","interrupted"
    except Exception as exc:
        code,error=2,f"{type(exc).__name__}: {exc}"
        status=load(out/"manifest.json")["status"] if (out/"manifest.json").exists() else "aborted"
    finally:
        if (out/"manifest.json").exists():
            guarded.flush();save_json(out/"physics_identity.json",actual)
            save_json(out/"physics_manifest.json",ready|{"configuration":backend.config})
            save_json(out/"method_identity.json",{"public_method":"strong-reference","description":"Strong NN reference with explicit complete-boundary cap recovery","operational_policy":"typed-darkomega-guard-v1"})
            save_json(out/"portable_run.json",{"kind":"portable_cap_continuation","source_run":str(source),"recovery_proof_sha256":file_hash(proof_path),"inherited_attempts":n,"logical_budget":settings.budget,"source_and_presets_sha256":inventory()})
            save_json(out/"launcher_status.json",{"status":status,"exit_code":code,"reason":error})
            if cache is not None:save_json(out/"execution.json",evaluator.report())
    return code


def main(argv=None):
    p=ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    c=sub.add_parser("run",help="Fresh real-physics scan; no implicit fixture or shared cache")
    c.add_argument("--method",choices=("modern","random","strong-reference"),required=True)
    c.add_argument("--backend",type=Path,required=True);c.add_argument("--output",type=Path,required=True)
    c.add_argument("--settings",type=Path);c.add_argument("--seed",type=int,default=0)
    c.add_argument("--budget",type=int,default=20000);c.add_argument("--initial",type=int,default=1000);c.add_argument("--cache",type=Path)
    c=sub.add_parser("prepare-cap",help="No physics: derive next-cap checkpoint after exact-boundary exhaustion")
    c.add_argument("--source-run",type=Path,required=True);c.add_argument("--output",type=Path,required=True)
    c.add_argument("--next-cap",type=int,choices=CAPS[1:],required=True);c.add_argument("--bundle",type=Path,required=True)
    c=sub.add_parser("resume-cap",help="Execute only a reverified complete-boundary cap proof")
    c.add_argument("--proof",type=Path,required=True);c.add_argument("--backend",type=Path,required=True);c.add_argument("--cache",type=Path)
    args=p.parse_args(argv)
    try:
        if args.command=="run":return run_fresh(args)
        if args.command=="prepare-cap":print(json.dumps(prepare_cap(args.source_run,args.output,args.next_cap,args.bundle),indent=2));return 0
        return resume_cap(args.proof,args.backend,args.cache)
    except (OSError,ValueError,RuntimeError,KeyError) as exc:
        print(f"Error: {exc}",file=sys.stderr);return 2


if __name__=="__main__":
    raise SystemExit(main())
