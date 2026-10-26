"""Recompute the public seed0 comparison using only the Python standard library.

No physics, model fitting, private journal or computer-specific path is used.
Packing is a deterministic spacing-filtered count, not statistical independence.
"""
from collections import Counter,defaultdict
import csv
import hashlib
from itertools import product
import json
import math
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
BOUNDS=((2.,20.),(7000.,9000.),(2.,10.))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def packing(points,radius=.001):
    cells=defaultdict(list);kept=0;width=2*radius
    for point in sorted(points):
        key=tuple(math.floor(v/width) for v in point)
        if any(math.dist(point,p)<=radius for delta in product((-1,0,1),repeat=3)
               for p in cells.get(tuple(k+d for k,d in zip(key,delta)),())):continue
        cells[key].append(point);kept+=1
    return kept


def metrics(rows):
    status=Counter();origins={};points=[];seen=set();eq7_total=0;targets=0
    for i,r in enumerate(rows,1):
        if r['eval_id']!=i or r['seed']!=0:raise ValueError('Sequence/seed mismatch')
        x=[r[k] for k in ('M1','mu','tan_beta')]
        if any(type(v) not in (int,float) or not math.isfinite(v) or not lo<=v<=hi for v,(lo,hi) in zip(x,BOUNDS)):raise ValueError('Invalid coordinate')
        key=tuple(float(v).hex() for v in x)
        if key in seen:raise ValueError('Repeated exact coordinate')
        seen.add(key)
        if r['status'] not in ('ok','unphysical','tool_error'):raise ValueError('Unknown status')
        status[r['status']]+=1;eq7=False
        if r['status']=='ok':
            mh,mH,chi2,omega=[r[k] for k in ('mh','mH','chi2_hs','omega')]
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in (mh,mH,chi2,omega)) or mh<=0 or mH<=0 or chi2<0 or omega<=0:raise ValueError('Invalid observation')
            eq7=123.<mH<129. and chi2<112.7273 and .08209<omega<.15511
        elif any(r[k] is not None for k in ('mh','mH','chi2_hs','omega')):raise ValueError('Failed row has complete labels')
        hb=r['hb_result']
        if hb is not None and (type(hb) is not int or hb not in (-1,0,1)):raise ValueError('Invalid HB label')
        target=bool(eq7 and hb==1)
        if r['target_eq7'] is not bool(eq7) or r['target'] is not target or r['hb_allowed'] is not (hb==1):raise ValueError('Target flags disagree with truth')
        eq7_total+=eq7;targets+=target
        count=origins.setdefault(r['origin'],Counter());count['attempts']+=1;count[r['status']]+=1;count['targets']+=target;count['targets_eq7']+=eq7
        if target:points.append(tuple((v-lo)/(hi-lo) for v,(lo,hi) in zip(x,BOUNDS)))
    return {'attempts':len(rows),'unique_coordinates':len(seen),'targets':targets,'targets_eq7':eq7_total,
        'target_fraction':targets/len(rows),'eq7_fraction':eq7_total/len(rows),
        'status_counts':dict(status),'by_origin':{o:dict(c) for o,c in origins.items()},
        'grid_occupied':{str(b):len({tuple(min(b-1,math.floor(v*b)) for v in p) for p in points}) for b in (20,40,100)},
        'packing_radius':.001,'packing_count':packing(points),
        'packing_scope':'Lexicographic greedy distance>1e-3 in the full unit parameter box; not statistical independence or maximum packing'}


def verify():
    manifest=json.loads((PROJECT/'data/benchmark/provenance.json').read_text())
    summary=json.loads((PROJECT/'data/benchmark/summary.json').read_text());result={}
    for method in ('v5','strong_reference','random'):
        info=manifest['datasets'][method]
        path=PROJECT/'data/benchmark'/info['jsonl_file'];csvpath=PROJECT/'data/benchmark'/info['csv_file']
        if sha(path)!=info['jsonl_sha256'] or sha(csvpath)!=info['csv_sha256']:raise ValueError('Public data checksum mismatch')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if len(rows)!=20000:raise ValueError('Full budget required')
        actual=metrics(rows)
        if actual!=summary['methods'][method]:raise ValueError('Summary differs from independent data recomputation')
        with csvpath.open(newline='') as stream:csvrows=list(csv.DictReader(stream))
        if len(csvrows)!=len(rows):raise ValueError('CSV row count mismatch')
        for c,r in zip(csvrows,rows):
            expected={k:'' if v is None else str(v) for k,v in r.items()}
            if c!=expected:raise ValueError('CSV and JSONL values differ')
        result[method]=actual
    if summary['old_protocol_qualified'] is not False:raise ValueError('Old protocol failure must remain explicit')
    gain=(result['v5']['targets']/result['strong_reference']['targets']-1)
    if not math.isclose(summary['relative_target_gain_vs_strong'],gain,rel_tol=1e-14):raise ValueError('Gain mismatch')
    return {'verified':True,'methods':{m:{k:v[k] for k in ('attempts','targets','targets_eq7','grid_occupied','packing_count')} for m,v in result.items()},
        'single_development_seed_only':True,'old_protocol_qualified':False,'new_physics_calls':0}


if __name__=='__main__':
    print(json.dumps(verify(),indent=2))
