"""Portable control integration: invented backends, no MSSM program is invoked."""
from argparse import Namespace
from copy import deepcopy
from dataclasses import asdict,replace
from importlib.util import module_from_spec,spec_from_file_location
import json
from pathlib import Path

import numpy as np
import pytest

from mssm_mls_modern import cli
import mssm_mls_modern.reference.baseline as baseline
from mssm_mls_modern.runner import checksum

PROJECT=Path(__file__).resolve().parents[1]
spec=spec_from_file_location("portable_benchmark",PROJECT/"tools/benchmark.py")
bench=module_from_spec(spec);spec.loader.exec_module(bench)
IDENTITY={"config":"c","adapter":"a","binaries":{"fh_driver":"f","omega_driver":"o"}}


class Backend:
    calls=[]
    def __init__(self,*args):self.config={}
    def doctor(self):return {"ready":True,"config_sha256":"c","adapter_sha256":"a","checks":{n:{"sha256":v} for n,v in IDENTITY['binaries'].items()}}
    def evaluate(self,p,i):
        self.calls.append(i)
        return {"status":"ok","observables":{"mh":1.,"mH":126.,"chi2_hs":40.,"omega":.1186},"diagnostics":{"hb_result":1},"provenance":{},"reason":"synthetic software double"}


class Models:
    def __init__(self,**kw):self.fit_count=0
    def fit(self,rows):self.fit_count+=1;return {"fit_count":self.fit_count,"fixture":True}
    def predict(self,x,threshold):return np.tile([1.,126.,40.,.1186],(len(x),1)),np.ones(len(x))


def config(tmp_path,method,budget=6):
    values=json.loads((PROJECT/'configs'/('strong-reference.json' if method=='strong-reference' else 'modern.json')).read_text())
    values.update(initial=6,budget=budget,epochs=1)
    if method=='strong-reference':values.update(guided=3,random_extra=2,candidate_chunk=8)
    path=tmp_path/(method+'.json');path.write_text(json.dumps(values));return path


def args(tmp_path,method,budget=6):
    return Namespace(method=method,settings=config(tmp_path,method,budget),backend=tmp_path/'backend.json',output=tmp_path/'wrapped',seed=3,budget=budget,initial=6,cache=None)


@pytest.mark.parametrize('method',['modern','random','strong-reference'])
def test_fresh_control_preserves_public_cli_points_and_has_no_implicit_cache(tmp_path,monkeypatch,method):
    monkeypatch.setattr(cli,'PhysicsBackend',Backend);a=args(tmp_path,method)
    assert bench.run_fresh(a)==0
    direct=tmp_path/'direct'
    cli.main(['scan','--method',method,'--backend',str(a.backend),'--settings',str(a.settings),'--budget','6','--initial','6','--seed','3','--output',str(direct)])
    assert (a.output/'records.jsonl').read_bytes()==(direct/'records.jsonl').read_bytes()
    metadata=json.loads((a.output/'portable_run.json').read_text())
    assert metadata['implicit_initial_fixture'] is False and metadata['cache'] is None


@pytest.fixture
def exhausted(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'PhysicsBackend',Backend);monkeypatch.setattr(baseline,'Surrogates',Models)
    original=baseline.rejection_sample
    def cap(*values,**kwargs):
        settings=values[2]
        raise baseline.ProposalExhausted({'accepted':0,'requested':values[1],'proposals':settings.max_proposals})
    monkeypatch.setattr(baseline,'rejection_sample',cap)
    a=args(tmp_path,'strong-reference',budget=16)
    assert bench.run_fresh(a)==2
    monkeypatch.setattr(baseline,'rejection_sample',original)
    assert json.loads((a.output/'summary.json').read_text())['attempts']==6
    return a


def test_complete_boundary_cap_resume_matches_uninterrupted_new_cap_and_inherits_without_calls(exhausted,tmp_path):
    a=exhausted;out=tmp_path/'continued';bundle=tmp_path/'proof'
    original_checkpoint=(a.output/'checkpoint.json').read_bytes();original_journal=(a.output/'records.jsonl').read_bytes()
    report=bench.prepare_cap(a.output,out,1_000_000_000,bundle)
    assert report['scan_started'] is False
    before=len(Backend.calls)
    assert bench.resume_cap(bundle/'proof.json',a.backend,backend_factory=Backend,model_factory=Models)==0
    assert len(Backend.calls)-before==10
    settings=baseline.ScanSettings(**{**json.loads(a.settings.read_text()),'max_proposals':1_000_000_000,'max_consecutive_tool_errors':17})
    whole=baseline.run_scan(settings,Backend(),models=Models(),seed=3,method='mls',physics_fingerprint=checksum(IDENTITY))
    rows=[json.loads(line) for line in (out/'records.jsonl').read_text().splitlines()]
    assert rows==whole['records'] and len(rows)==16
    assert (a.output/'checkpoint.json').read_bytes()==original_checkpoint and (a.output/'records.jsonl').read_bytes()==original_journal
    guard=json.loads((out/'operational_recovery.json').read_text())
    assert guard['checkpoint_attempts']==6 and guard['new_evaluator_requests']==10 and guard['prefix_verified'] is True


@pytest.mark.parametrize('mutation',['journal_suffix','journal_value','checkpoint_value','wrong_method','wrong_status','skip_cap'])
def test_unsupported_or_corrupt_recovery_is_refused_before_backend(exhausted,tmp_path,mutation):
    source=exhausted.output;nextcap=1_000_000_000
    if mutation=='journal_suffix':
        with (source/'records.jsonl').open('a') as f:f.write((source/'records.jsonl').read_text().splitlines()[0]+'\n')
    elif mutation=='journal_value':
        rows=[json.loads(l) for l in (source/'records.jsonl').read_text().splitlines()];rows[0]['status']='tool_error'
        (source/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    elif mutation=='checkpoint_value':
        e=json.loads((source/'checkpoint.json').read_text());e['payload']['records'][0]['x'][0]+=1;(source/'checkpoint.json').write_text(json.dumps(e))
    elif mutation=='wrong_method':
        m=json.loads((source/'method_identity.json').read_text());m['public_method']='modern';(source/'method_identity.json').write_text(json.dumps(m))
    elif mutation=='wrong_status':
        m=json.loads((source/'manifest.json').read_text());m['status']='interrupted';(source/'manifest.json').write_text(json.dumps(m))
    else:nextcap=5_000_000_000
    before=len(Backend.calls)
    with pytest.raises(ValueError):bench.prepare_cap(source,tmp_path/'new',nextcap,tmp_path/'bundle')
    assert len(Backend.calls)==before


def test_changed_derived_settings_or_source_cannot_be_resumed(exhausted,tmp_path):
    bundle=tmp_path/'proof';bench.prepare_cap(exhausted.output,tmp_path/'new',1_000_000_000,bundle)
    values=json.loads((bundle/'settings.json').read_text());values['budget']+=1;(bundle/'settings.json').write_text(json.dumps(values))
    def forbidden(*a):raise AssertionError('No backend before proof validation')
    with pytest.raises(ValueError,match='Derived checkpoint'):
        bench.resume_cap(bundle/'proof.json',exhausted.backend,backend_factory=forbidden)


@pytest.mark.parametrize('nested',['source','output'])
def test_proof_bundle_cannot_write_inside_preserved_source_or_new_run(exhausted,tmp_path,nested):
    out=tmp_path/'new';bundle=(exhausted.output/'proof') if nested=='source' else out/'proof'
    with pytest.raises(ValueError,match='nonnested'):
        bench.prepare_cap(exhausted.output,out,1_000_000_000,bundle)
    assert not bundle.exists() and not out.exists()


@pytest.mark.parametrize('known',[False,True])
def test_cap_continuation_retains_typed_guard_and_failed_attempt_budget(exhausted,tmp_path,known):
    out=tmp_path/'continued';bundle=tmp_path/'proof'
    bench.prepare_cap(exhausted.output,out,1_000_000_000,bundle)
    class Failures(Backend):
        def evaluate(self,p,i):
            answer={'status':'tool_error','reason':'invented numerical signature' if known else 'invented unknown error','diagnostics':{},'provenance':{}}
            if known:
                answer['provenance']={'config_sha256':'c','adapter_sha256':'a','binary_sha256':IDENTITY['binaries'],'paper_version_match':True,
                    'higgs':{'status':'ok'},'commands':[{'stage':'higgs','returncode':0},{'stage':'omega','returncode':70}],
                    'micromegas':{'status':'technical_failure','reason':'darkOmega_failed','source_version':'4.3.2'}}
            return answer
    code=bench.resume_cap(bundle/'proof.json',exhausted.backend,backend_factory=Failures,model_factory=Models)
    rows=[json.loads(l) for l in (out/'records.jsonl').read_text().splitlines()]
    assert rows[:6]==[json.loads(l) for l in (exhausted.output/'records.jsonl').read_text().splitlines()]
    assert all(r['status']=='tool_error' and r['target'] is False for r in rows[6:])
    assert (code,len(rows))==((0,16) if known else (3,11))
    report=json.loads((out/'operational_recovery.json').read_text())
    assert report['checkpoint_attempts']==6
    assert report['recognized_failure_returns_this_process']==(10 if known else 0)
    assert report['unidentified_failure_streak']==(0 if known else 5)


def test_public_general_resume_stays_disabled_before_backend(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'PhysicsBackend',lambda *a: (_ for _ in ()).throw(AssertionError('Must not construct backend')))
    with pytest.raises(SystemExit) as exc:
        cli.main(['scan','--resume',str(tmp_path/'checkpoint.json'),'--output',str(tmp_path/'new')])
    assert exc.value.code==2


def test_core_and_presets_inventory_unchanged_and_public_data_recompute():
    assert len(bench.inventory())==24
    spec=spec_from_file_location('public_data_metrics',PROJECT/'tools/verify_benchmark_data.py')
    data=module_from_spec(spec);spec.loader.exec_module(data)
    result=data.verify()
    assert result['verified'] and result['old_protocol_qualified'] is False
    assert result['methods']['v5']['targets']==4642 and result['methods']['strong_reference']['targets']==4248
