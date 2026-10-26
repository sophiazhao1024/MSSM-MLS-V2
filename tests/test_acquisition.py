"""Ranking/fallback tests use artificial predictions, never physics outcomes."""
from dataclasses import replace
import numpy as np
from mssm_mls_modern.settings import ModernSettings
from mssm_mls_modern.acquisition import acquisition, propose_batch, exact_key, batch_quotas, interleave_execution, candidate_pool, true_violation, normal_interval_log_probability, diverse_indices
from mssm_mls_modern.domain import score, BOUNDS


def prediction(n, *, invalid=False, hb=.8):
    mean=np.tile([10.,126.,100.,.1186],(n,1))
    transformed=mean.copy(); transformed[:,3]=np.log(transformed[:,3])
    if invalid: mean[:]=np.nan
    return {"mean":mean,"mean_transformed":transformed,
            "physical_probability":np.zeros(n),"hb_probability":np.full(n,hb),
            "uncertainty":np.ones(n)}


class Model:
    ready=True
    def __init__(self, invalid=False):self.invalid=invalid
    def predict(self,x):return prediction(len(x),invalid=self.invalid)


def test_soft_weights_do_not_relax_true_cuts_and_no_zero_probability_veto():
    p=prediction(1)
    p["mean"][0,2]=120.
    p["mean_transformed"][0,2]=120.
    exploit,uncertain=acquisition(p,ModernSettings())
    assert np.isfinite(exploit).all() and np.isfinite(uncertain).all()
    assert not score(p["mean"])[1][0]


def test_hb_probability_used_only_by_hb_target_policy():
    low,high=prediction(1,hb=.1),prediction(1,hb=.9)
    s=ModernSettings()
    assert acquisition(low,s)[0][0] < acquisition(high,s)[0][0]
    eq=replace(s,target_policy="eq7-only")
    assert acquisition(low,eq)[0][0] == acquisition(high,eq)[0][0]


def test_batch_counts_unique_and_exploration_independent_of_model():
    s=replace(ModernSettings(),pool_size=256)
    batch,report=propose_batch(Model(),[],s,np.random.default_rng(1),explore_rng=np.random.default_rng(2))
    assert report["selected_counts"] == {"exploit":63,"uncertain":21,"random":21,"uniform_fallback":0}
    assert len({exact_key(r["x"]) for r in batch}) == 105
    assert batch[0]["origin"] == "random"
    assert report["execution_order"] == "proportional_interleave_random_first"
    failed,fr=propose_batch(Model(invalid=True),[],s,np.random.default_rng(1),explore_rng=np.random.default_rng(2))
    good_random=[r["x"] for r in batch if r["origin"]=="random"]
    failed_random=[r["x"] for r in failed if r["origin"]=="random"]
    np.testing.assert_array_equal(good_random,failed_random)
    assert fr["fallback_count"] == 84
    assert fr["fallback_reason"] == "insufficient_finite_unique_ranked_candidates"
    # Additional fallback draws must not shift the reserved exploration stream.
    a,b=np.random.default_rng(42),np.random.default_rng(42)
    propose_batch(Model(),[],s,np.random.default_rng(9),explore_rng=a)
    propose_batch(Model(invalid=True),[],s,np.random.default_rng(9),explore_rng=b)
    np.testing.assert_array_equal(a.random(4),b.random(4))


def test_exact_duplicates_are_removed_from_candidate_pool(monkeypatch):
    import mssm_mls_modern.acquisition as module
    s=replace(ModernSettings(),pool_size=128)
    seen=[7.3,7600.,7.725]
    original=module.sample
    def fake_sample(n,rng):
        points=original(n,rng)
        if n==s.pool_size: points[:10]=seen
        return points
    monkeypatch.setattr(module,"sample",fake_sample)
    batch,report=propose_batch(Model(),[{"x":seen}],s,np.random.default_rng(1),explore_rng=np.random.default_rng(2))
    assert report["unique_pool_size"]==118
    assert exact_key(seen) not in {exact_key(r["x"]) for r in batch}


def test_truncated_batch_retains_exploration_and_notready_fallback():
    s=ModernSettings()
    assert batch_quotas(s,105)==(63,21,21)
    assert batch_quotas(s,1)==(0,0,1)
    batch,report=propose_batch(None,[],s,np.random.default_rng(1),explore_rng=np.random.default_rng(2),count=5)
    assert len(batch)==5 and report["fallback_count"]==4
    assert {r["origin"] for r in batch}=={"uniform_fallback","random"}


def test_diversity_radius_is_a_soft_scale_not_a_fill_constraint():
    s=replace(ModernSettings(),pool_size=128,diversity_radius=10.)
    batch,report=propose_batch(Model(),[],s,np.random.default_rng(1),explore_rng=np.random.default_rng(2))
    assert len(batch)==105 and report["diversity_relaxations"]==0
    assert report['selection_tradeoff']['selected_below_soft_radius']==84
    assert report['selection_tradeoff']['mean_log_repulsion_cost']>0
    assert report['diversity_mode']=='soft_log_nearest_anchor_repulsion'


def test_interleaving_preserves_candidates_and_within_category_rank():
    from itertools import groupby
    batch=[{"origin":origin,"rank":j,"x":[j,0,0]}
           for origin,count in [("exploit",63),("uncertain",21),("random",21)]
           for j in range(count)]
    original_order=[(item["origin"],item["rank"]) for item in batch]
    result=interleave_execution(batch)
    assert sorted(map(id,result))==sorted(map(id,batch))
    assert [(item["origin"],item["rank"]) for item in batch]==original_order
    assert result[0]["origin"]=="random"
    for origin,count in [("exploit",63),("uncertain",21),("random",21)]:
        assert [item["rank"] for item in result if item["origin"]==origin]==list(range(count))
    runs=[(origin,len(list(group))) for origin,group in groupby(item["origin"] for item in result)]
    assert not any(origin=="uncertain" and length>=5 for origin,length in runs)


def test_interleaving_fallback_is_deterministic_without_rng_consumption():
    batch=[{"origin":"uniform_fallback","rank":j} for j in range(84)]
    batch += [{"origin":"random","rank":j} for j in range(21)]
    before=np.random.get_state()
    first=interleave_execution(batch)
    second=interleave_execution(batch)
    after=np.random.get_state()
    assert first==second and first[0]["origin"]=="random"
    assert [item["rank"] for item in first if item["origin"]=="uniform_fallback"]==list(range(84))
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    assert interleave_execution([])==[]


def true_row(x, eval_id, *, omega=.12, chi2=100., hb=1):
    return {"x":x,"eval_id":eval_id,"status":"ok","hb_result":hb,
            "observables":{"mh":10.,"mH":126.,"chi2_hs":chi2,"omega":omega}}


def test_tiny_physical_probability_is_not_promoted_to_exploration_weight():
    a,b=prediction(1),prediction(1)
    a["physical_probability"][:]=1e-8
    b["physical_probability"][:]=.05
    low,lu=acquisition(a,ModernSettings())
    high,hu=acquisition(b,ModernSettings())
    assert high[0]-low[0] > 15.
    assert hu[0]-lu[0] > 15.
    assert ModernSettings().probability_floor == 1e-12


def test_true_score_unchanged_while_discovery_ranking_drops_mass_preference():
    p=prediction(2);p['physical_probability'][:]=.9
    p['mean'][:,0]=[40.,80.];p['mean_transformed'][:,0]=[40.,80.]
    original_weights,hits=score(p['mean'])
    assert hits.all() and original_weights[0] > original_weights[1]
    acquisition_weights,_=acquisition(p,ModernSettings())
    # Only the positive-mass soft-domain protection remains: no exp(-.15 mh).
    assert acquisition_weights[1] > acquisition_weights[0]


def test_solver_failure_probability_is_separate_and_penalizes_both_streams():
    a,b=prediction(1),prediction(1)
    for p in (a,b): p['physical_probability'][:]=.9
    a['solver_probability']=np.array([.01]);b['solver_probability']=np.array([.9])
    la,ua=acquisition(a,ModernSettings());lb,ub=acquisition(b,ModernSettings())
    assert la[0]<lb[0] and ua[0]<ub[0]


def test_local_pool_uses_only_current_true_rows_and_reflects_in_bounds():
    s=replace(ModernSettings(),pool_size=256,local_elites=2)
    # Synthetic boundary anchors, deliberately unrelated to real targets.
    rows=[true_row(BOUNDS[:,0].tolist(),1),true_row(BOUNDS[:,1].tolist(),2),
          true_row([11.,8000.,6.],3,omega=900.),
          {"x":[12.,8100.,7.],"eval_id":4,"status":"tool_error","observables":None},
          {"x":[10.,7900.,5.],"eval_id":5,"status":"unphysical","observables":None}]
    x,labels,info=candidate_pool(rows,s,np.random.default_rng(3))
    assert x.shape==(256,3) and len(labels)==256
    assert info['global_uniform']==128 and info['local_true_elite']==128
    assert info['elite_eval_ids']==[1,2]
    assert np.all(x>=BOUNDS[:,0]) and np.all(x<=BOUNDS[:,1])
    # Reflection does not collapse overshoots to an exact box boundary.
    assert not np.any(x[128:]==BOUNDS[:,0]) and not np.any(x[128:]==BOUNDS[:,1])
    assert len({exact_key(v) for v in x})==256


def test_local_pool_no_truth_means_global_and_elite_score_respects_real_hb():
    s=replace(ModernSettings(),pool_size=128)
    x,labels,info=candidate_pool([],s,np.random.default_rng(0))
    assert info['local_true_elite']==0 and info['global_uniform']==128
    allowed=true_row([11.,8000.,6.],1);excluded=dict(allowed,hb_result=0)
    unknown=dict(allowed,hb_result=-1)
    assert true_violation(allowed,s)==0.
    assert true_violation(excluded,s)>true_violation(allowed,s)
    assert true_violation(unknown,s)==true_violation(allowed,s)
    assert true_violation(excluded,replace(s,target_policy='eq7-only'))==0.


def test_local_pool_and_fallback_do_not_consume_independent_exploration_rng():
    s=replace(ModernSettings(),pool_size=256)
    rows=[true_row([11.,8000.,6.],1)]
    a,b=np.random.default_rng(42),np.random.default_rng(42)
    batch,report=propose_batch(Model(),rows,s,np.random.default_rng(9),explore_rng=a)
    fallback,fr=propose_batch(Model(invalid=True),rows,s,np.random.default_rng(9),explore_rng=b)
    np.testing.assert_array_equal(a.random(4),b.random(4))
    np.testing.assert_array_equal([v['x'] for v in batch if v['origin']=='random'],
                                  [v['x'] for v in fallback if v['origin']=='random'])
    assert sum(report['selected_candidate_sources'].values())==84


def test_gaussian_window_probability_is_stable_in_centre_and_extreme_tails():
    value=normal_interval_log_probability(np.array([0.]),np.array([1.]),-1.,1.)
    np.testing.assert_allclose(np.exp(value),[.6826894921370859],rtol=1e-13)
    # Direct floating CDF subtraction would produce zero in the positive tail.
    left=normal_interval_log_probability(np.array([0.]),np.array([1.]),-21.,-20.)
    right=normal_interval_log_probability(np.array([0.]),np.array([1.]),20.,21.)
    assert np.isfinite(left).all() and np.isfinite(right).all()
    np.testing.assert_allclose(left,right,rtol=1e-13)
    assert left[0] < -200.
    deterministic=normal_interval_log_probability(np.array([0.,1.,2.]),np.zeros(3),0.,2.)
    assert np.isneginf(deterministic[[0,2]]).all() and deterministic[1]==0.
    assert normal_interval_log_probability(np.array([0.]),np.zeros(1),0.,2.,include_lower=True)[0]==0.


def test_gp_window_ranking_uses_transformed_uncertainty_and_keeps_true_cuts():
    p=prediction(2)
    p['physical_probability'][:]=.9
    p['mean'][:,1]=130.;p['mean_transformed'][:,1]=130.
    p['std_transformed']=np.array([[.1,.1,.1,.05],[.1,2.,.1,.05]])
    values,_=acquisition(p,ModernSettings())
    # Both means fail the real mH upper cut; the second model allows credible
    # mass within the unchanged window and may merit an expensive evaluation.
    assert not score(p['mean'])[1].any()
    assert values[1]>values[0]
    assert np.isfinite(values).all()


def test_soft_repulsion_does_not_replace_high_utility_by_arbitrary_far_point():
    unit=np.array([[.5,.5,.5],[.5001,.5,.5],[.9,.9,.9]])
    pool=BOUNDS[:,0]+unit*np.ptp(BOUNDS,axis=1)
    scores=np.array([0.,-.1,-100.])
    indices,details=diverse_indices(pool,scores,2,[],.05,np.random.default_rng(1))
    assert indices==[0,1]
    assert details[1]['min_distance']<.05
    assert details[1]['penalized_acquisition']>scores[2]
    assert details[0]['min_distance'] is None
    assert details[1]['raw_utility_gap']==0.


def test_soft_repulsion_avoids_near_duplicates_when_utility_is_comparable():
    unit=np.array([[.5,.5,.5],[.500001,.5,.5],[.8,.8,.8]])
    pool=BOUNDS[:,0]+unit*np.ptp(BOUNDS,axis=1)
    scores=np.array([0.,-.01,-.1])
    indices,details=diverse_indices(pool,scores,2,[],.05,np.random.default_rng(1))
    assert indices==[0,2]
    assert details[1]['raw_utility_gap']>0
    assert details[1]['penalized_acquisition']<=details[1]['raw_acquisition']


def test_proposals_audit_gp_mode_and_per_point_log_tradeoffs():
    class GPModel(Model):
        def predict(self,x):
            p=prediction(len(x));p['physical_probability'][:]=.9
            p['std_transformed']=np.tile([.1,.2,1.,.1],(len(x),1))
            return p
    batch,report=propose_batch(GPModel(),[],replace(ModernSettings(),pool_size=256),
                               np.random.default_rng(3),explore_rng=np.random.default_rng(4))
    assert report['acquisition_mode']=='gp_marginal_feasibility'
    assert report['calibrated_joint_probability'] is False
    for r in batch:
        if r['origin'] in ('exploit','uncertain'):
            assert all(k in r for k in ('raw_acquisition','penalized_acquisition','min_distance'))
            assert r['raw_acquisition']>=r['penalized_acquisition']
