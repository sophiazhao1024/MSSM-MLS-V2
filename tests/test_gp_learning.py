"""Synthetic software contracts, never MSSM performance evidence."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from mssm_mls_modern.settings import ModernSettings
from mssm_mls_modern.gp_learning import ScalarGP, GPSurrogates, select_anchors, hb_ratio


def settings():
    return replace(ModernSettings(),initial=20,budget=30,epochs=3,patience=2,
                   hidden_layers=1,hidden_width=8,min_physical=2,gp_steps=10,gp_max_anchors=16)


def rows():
    result=[]
    for i in range(12):
        result.append({'x':[3+i,7100+60*i,2.5+.3*i],'status':'ok',
                       'validation':i>=10,'hb_result':i%2,
                       'observables':{'mh':10+i,'mH':124+.1*i,'chi2_hs':80+2*i,'omega':.1+.02*i}})
    result += [{'x':[17,8700,8],'status':'unphysical'},
               {'x':[18,8800,9],'status':'tool_error',
                'observables':{'mh':1e9,'mH':1e9,'chi2_hs':1e9,'omega':1e9}}]
    return result


def test_small_smooth_curve_is_interpolated_and_variance_is_finite():
    rng=np.random.default_rng(4)
    x=rng.uniform(-1,1,(24,3))
    y=np.sin(x[:,0])+x[:,1]**2-.3*x[:,2]
    gp=ScalarGP(steps=25)
    stats=gp.fit(x,y)
    predicted,spread=gp.predict(x)
    assert np.mean((predicted-y)**2)<1e-6
    assert np.isfinite(spread).all() and np.all(spread>=0)
    assert len(stats['lengthscales'])==3
    a,b=gp.predict(np.empty((0,3)))
    assert a.shape==b.shape==(0,)


def test_training_statuses_and_holdout_have_distinct_semantics():
    model=GPSurrogates(settings(),seed=2,gp_steps=10,max_anchors=16)
    report=model.fit(rows())
    assert report['physical_train']==10 and report['physical_validation']==2
    assert report['classifiers']['physical']['training_count']==11
    assert report['classifiers']['physical']['positive_count']==10
    assert report['classifiers']['success']['training_count']==11
    assert report['classifiers']['success']['negative_count']==1
    assert report['classifiers']['hb']['training_count']==10
    assert all(r['anchors']==10 for r in report['regressors'])
    expected=np.array([[r['observables'][k] for k in ('mh','mH','chi2_hs','omega')] for r in rows()[:10]])
    expected[:,3]=np.log(expected[:,3])
    expected[:,0]**=2
    np.testing.assert_allclose(model.mean,expected.mean(axis=0))
    result=model.predict([[7,7600,5],[15,8400,7]])
    assert result['mean'].shape==(2,4)
    assert result['std_standardized'].shape==(2,4)
    assert result['std_transformed'].shape==(2,4)
    assert [r.input_dim for r in model.regressors]==[3,3,3,4]
    assert result['solver_probability'].shape==(2,)
    assert np.isfinite(result['mean']).all()
    np.testing.assert_allclose(result['mean'][:,3],np.exp(result['mean_transformed'][:,3]))


def test_restricted_export_roundtrip(tmp_path):
    s=settings()
    model=GPSurrogates(s,seed=7,gp_steps=10,max_anchors=16)
    model.fit(rows())
    f=tmp_path/'gp.pt'
    torch.save(model.export(),f)
    restored=GPSurrogates(s,seed=7,gp_steps=10,max_anchors=16).load_export(torch.load(f,weights_only=True))
    x=[[7,7600,5],[15,8400,7]]
    for k,value in model.predict(x).items():
        np.testing.assert_array_equal(value,restored.predict(x)[k])
    with pytest.raises(ValueError):
        GPSurrogates(s,seed=8,gp_steps=10,max_anchors=16).load_export(model.export())


def test_anchor_selection_is_unique_bounded_and_uses_only_supplied_coordinates():
    physical=rows()[:10]
    indices=select_anchors(physical,6)
    assert len(indices)==len(set(indices))==6
    assert set(indices)<=set(range(len(physical)))
    np.testing.assert_array_equal(indices,select_anchors(physical,6))


def test_small_data_is_not_ready_and_no_physics_labels_are_fabricated():
    model=GPSurrogates(replace(settings(),gp_steps=5))
    r=model.fit(rows()[-2:])
    assert not r['ready'] and r['physical_train']==0
    with pytest.raises(RuntimeError): model.predict([[7,7600,5]])


def test_anchor_cap_retains_rare_low_violation_points_even_in_small_dataset():
    records=[]
    for i in range(30):
        records.append({'eval_id':i+1,'x':[3+.5*i,7100+50*i,2.5+.2*i],
                        'status':'ok','hb_result':1,
                        'observables':{'mh':15,'mH':125,'chi2_hs':250,'omega':500.}})
    for r in records[-4:]:
        r['observables'].update(chi2_hs=100,omega=.12)
    indices=select_anchors(records,8)
    assert set(range(26,30))<=set(indices)
    assert len(set(indices))==8


def test_four_dimensional_gp_gradient_matches_finite_difference():
    rng=np.random.default_rng(10)
    x=rng.uniform(-1,1,(20,4));y=np.sin(x[:,0])+x[:,3]**2
    model=ScalarGP(steps=10,input_dim=4);model.fit(x,y)
    z=rng.uniform(-.5,.5,(3,4));g=model.mean_gradient(z)
    for j in range(4):
        plus=z.copy();minus=z.copy();plus[:,j]+=1e-5;minus[:,j]-=1e-5
        p,_=model.predict(plus);m,_=model.predict(minus)
        np.testing.assert_allclose(g[:,j],(p-m)/2e-5,rtol=2e-6,atol=2e-6)


def test_candidate_detuning_uses_predicted_mass_and_propagates_its_error(monkeypatch):
    model=GPSurrogates(settings(),seed=1)
    model.fit(rows())
    mass2_standard=(100.-model.mean[0])/model.scale[0]
    monkeypatch.setattr(model.regressors[0],'predict',lambda x:(np.full(len(x),mass2_standard),np.full(len(x),.02)))
    seen=[]; original=model.regressors[3].predict
    def capture(x):
        seen.append(np.array(x));return original(x)
    monkeypatch.setattr(model.regressors[3],'predict',capture)
    result=model.predict([[7,7600,5]])
    assert result['mean'][0,0]==pytest.approx(10.)
    assert seen[0][0,3]==pytest.approx((14.-10.-model.detuning_mean)/model.detuning_scale)
    _,conditional=model.regressors[3].predict(seen[0])
    assert result['std_transformed'][0,3]>=conditional[0]*model.scale[3]
    assert result['mh_prediction_valid'][0]


def test_negative_predicted_mass_square_is_not_clipped_to_a_valid_higgs(monkeypatch):
    model=GPSurrogates(settings(),seed=1);model.fit(rows())
    negative=(-4.-model.mean[0])/model.scale[0]
    monkeypatch.setattr(model.regressors[0],'predict',lambda x:(np.full(len(x),negative),np.full(len(x),.01)))
    result=model.predict([[7,7600,5]])
    assert not result['mh_prediction_valid'][0]
    assert result['mean'][0,0]==pytest.approx(-2.)
    assert result['physical_probability'][0]<.5
    assert result['mh_square_positive_probability'][0]<.5
    assert np.isfinite(result['std_transformed']).all()


def ratio_rows():
    result=rows()
    for i,r in enumerate(result):
        if r['status']=='ok':
            r['diagnostics']={'hb_obsratio':float(np.exp(.12*(i-5.5)))}
            r['hb_result']=int(r['diagnostics']['hb_obsratio']<=1)
    return result


def test_logratio_model_replaces_binary_factor_and_survives_safe_restore(tmp_path):
    s=settings();model=GPSurrogates(s,seed=7);report=model.fit(ratio_rows())
    assert model.hb_regressor is not None and model.hb_regressor.input_dim==4
    assert report['hb_ratio']['enabled']
    assert not report['classifiers']['hb']['enabled']
    assert 'hb' not in model.classifiers
    assert report['kernel_input_dimensions']==[3,3,3,4,4]
    x=[[7,7600,5],[15,8400,7]];p=model.predict(x)
    assert np.isfinite(p['log_hb_ratio_mean']).all()
    assert np.all((p['hb_probability']>=0)&(p['hb_probability']<=1))
    f=tmp_path/'gp_hb.pt';torch.save(model.export(),f)
    restored=GPSurrogates(s,seed=7).load_export(torch.load(f,weights_only=True))
    for k,v in p.items():np.testing.assert_array_equal(v,restored.predict(x)[k])


def test_missing_zero_and_invalid_ratios_do_not_become_regression_or_exclusion_labels():
    r=ratio_rows()
    for row in r:
        row['diagnostics']={'hb_obsratio':0.}
    # Known binary HB labels remain available independently of missing ratios.
    m=GPSurrogates(settings(),seed=7);report=m.fit(r)
    assert not report['hb_ratio']['enabled']
    assert report['hb_ratio']['training_count']==0
    assert report['classifiers']['hb']['enabled']
    assert 'hb' in m.classifiers and m.hb_regressor is None
    for value in [None,0.,-1.,float('nan'),float('inf'),True]:
        assert hb_ratio({'status':'ok','diagnostics':{'hb_obsratio':value}}) is None
    assert hb_ratio({'status':'tool_error','diagnostics':{'hb_obsratio':.5}}) is None
    assert hb_ratio({'status':'ok','diagnostics':{'hb_obsratio':.5}})==.5


def test_hb_candidate_features_use_predicted_mass_without_oracle_lookup(monkeypatch):
    r=ratio_rows();m=GPSurrogates(settings(),seed=4);m.fit(r)
    # Change every supplied training record after fit. A prediction must use
    # only its x and fitted state, never inspect mutable truth records again.
    x=[[7,7600,5]];before=m.predict(x)
    for row in r:
        row['observables']={'mh':999.,'mH':999.,'chi2_hs':999.,'omega':999.}
        row['hb_result']=0;row['diagnostics']={'hb_obsratio':999.}
    for k,v in before.items():np.testing.assert_array_equal(v,m.predict(x)[k])
    # Force a first-stage predicted mass different from all real labels and
    # inspect the actual HB GP input to prove the extra coordinate follows it.
    z=(100.-m.mean[0])/m.scale[0]
    monkeypatch.setattr(m.regressors[0],'predict',lambda a:(np.full(len(a),z),np.full(len(a),.02)))
    seen=[];original=m.hb_regressor.predict
    def capture(a):seen.append(np.array(a));return original(a)
    monkeypatch.setattr(m.hb_regressor,'predict',capture)
    p=m.predict(x)
    assert seen[0][0,3]==pytest.approx((10.-m.hb_mass_mean)/m.hb_mass_scale)
    _,conditional=m.hb_regressor.predict(seen[0])
    assert p['log_hb_ratio_std'][0]>=conditional[0]*m.hb_log_scale
