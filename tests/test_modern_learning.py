"""Software fixtures only; these are not MSSM simulator results."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from torch import nn
from mssm_mls_modern.settings import ModernSettings
from mssm_mls_modern.learning import EnsembleSurrogates, train_with_holdout, hb_label


def settings(**kwargs):
    values=dict(initial=20, budget=30, ensemble_size=2, epochs=3, patience=2,
                hidden_layers=1, hidden_width=8, min_physical=2)
    values.update(kwargs)
    return replace(ModernSettings(), **values)


def data():
    records = []
    for i in range(10):
        records.append({"x": [3+i, 7100+70*i, 2.5+.3*i], "status": "ok",
                        "validation": i >= 8, "diagnostics": {"hb_result": i % 2},
                        "observables": {"mh": 10+i, "mH": 124+.2*i,
                                        "chi2_hs": 80+3*i, "omega": .1+.02*i}})
    records += [{"x": [17, 8700, 8], "status": "unphysical", "validation": False},
                {"x": [18, 8800, 9], "status": "unphysical", "validation": True},
                {"x": [19, 8900, 9.5], "status": "tool_error", "validation": False,
                 "observables": {"mh": 1e9, "mH": 1e9, "chi2_hs": 1e9, "omega": 1e9}}]
    return records


def test_input_scaling_and_train_only_label_statistics():
    assert np.array_equal(EnsembleSurrogates.inputs([[2,7000,2],[20,9000,10]]).numpy(),
                          [[-1,-1,-1],[1,1,1]])
    rows = data()
    rows[8]["observables"] = {"mh": 5000., "mH": 6000., "chi2_hs": 7000., "omega": 8000.}
    model = EnsembleSurrogates(settings(), seed=0)
    report = model.fit(rows)
    expected = model.transformed_labels(rows[:8])
    np.testing.assert_allclose(model.mean, expected.mean(axis=0))
    np.testing.assert_allclose(model.scale, expected.std(axis=0))
    assert report["physical_train"] == 8 and report["classifier_train"] == 9
    assert report["physical_validation"] == 2 and report["tool_errors_excluded"] == 1
    assert all(s["training_count_with_replacement"] == 8
               for m in report["members"] for s in m["regressors"])


def test_four_networks_are_independent_and_prediction_has_no_hard_gate():
    model = EnsembleSurrogates(settings(), seed=3)
    model.fit(data())
    all_models = [m for regs, _, _ in model.members for m in regs]
    addresses = [next(m.parameters()).data_ptr() for m in all_models]
    assert len(set(addresses)) == 8
    with torch.no_grad():
        for _, classifier, _ in model.members:
            for p in classifier.parameters(): p.zero_()
            classifier[-1].bias.fill_(-100.)
    prediction = model.predict([[7.3, 7600., 7.725]])
    assert prediction["physical_probability"][0] < .001
    assert np.isfinite(prediction["mean"]).all()
    np.testing.assert_allclose(prediction["mean"][:,3], np.exp(prediction["mean_transformed"][:,3]))


def test_hb_missing_or_single_class_is_neutral_not_negative():
    rows = data()
    for row in rows:
        row.pop("diagnostics", None)
    model = EnsembleSurrogates(settings(), seed=0)
    report = model.fit(rows)
    assert not report["hb_classifier_enabled"]
    assert np.array_equal(model.predict([[7.3,7600,7.725]])["hb_probability"], [1.])
    assert hb_label({"status":"ok","hb_result":-1}) is None
    assert hb_label({"status":"tool_error","hb_result":1}) is None
    assert hb_label({"status":"ok","hb_result":True}) is None
    for row in rows:
        row["hb_result"] = 1
    report = model.fit(rows)
    assert not report["hb_classifier_enabled"]


def test_small_training_dataset_is_explicitly_not_ready():
    model = EnsembleSurrogates(settings(min_physical=12), seed=0)
    report = model.fit(data())
    assert report["reason"] == "insufficient_physical_training_rows"
    assert model.mean is None and not model.ready
    with pytest.raises(RuntimeError): model.predict([[7.3,7600,7.725]])


def test_safe_export_roundtrip_preserves_predictions(tmp_path):
    s = settings()
    original = EnsembleSurrogates(s, seed=7)
    original.fit(data())
    path = tmp_path/"weights.pt"
    torch.save(original.export(), path)
    restored = EnsembleSurrogates(s, seed=7).load_export(torch.load(path, weights_only=True))
    x = [[7.3,7600,7.725],[15,8500,3]]
    for k,v in original.predict(x).items():
        np.testing.assert_array_equal(v, restored.predict(x)[k])
    with pytest.raises(ValueError):
        EnsembleSurrogates(s, seed=8).load_export(original.export())


def test_stopping_restores_best_validation_epoch():
    model = nn.Linear(3,1)
    with torch.no_grad(): model.weight.zero_(); model.bias.fill_(.5)
    x=torch.zeros((4,3)); y=torch.zeros(4)
    xv=torch.zeros((2,3)); yv=torch.full((2,),10.)
    s=replace(settings(),epochs=20,patience=2,learning_rate=.1)
    report=train_with_holdout(model,x,y,xv,yv,s,seed=0)
    assert report["best_epoch"] == 1
    assert report["epochs_executed"] == 3
    assert float(model.bias.detach()[0]) == pytest.approx(.4,abs=1e-6)


@pytest.mark.parametrize("change",[{"ensemble_size":1},{"random_extra":0},{"probability_floor":0},
    {"lnomega_soft_width":0},{"validation_fraction":.7},{"target_policy":"posterior"}])
def test_invalid_settings_rejected(change):
    with pytest.raises(ValueError): replace(ModernSettings(),**change)
