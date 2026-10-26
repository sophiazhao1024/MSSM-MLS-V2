"""Synthetic probability contracts only: no MSSM data or efficiency claim."""
from copy import deepcopy
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

from mssm_mls_modern.acquisition import (MH2_FACTORIZATION_MARKER, acquisition,
                                       log_bernoulli_entropy)
from mssm_mls_modern.gp_learning import (GPSurrogates,
                                       positive_mh_squared_log_probability)
from mssm_mls_modern.settings import ModernSettings


def prediction(n, *, gp=True):
    mean = np.tile([10., 126., 100., .12], (n, 1))
    transformed = mean.copy()
    transformed[:, 3] = np.log(transformed[:, 3])
    result = {'mean': mean, 'mean_transformed': transformed,
              'physical_probability': np.full(n, .8), 'hb_probability': np.ones(n),
              'solver_probability': np.ones(n), 'uncertainty': np.ones(n)}
    if gp:
        result['std_transformed'] = np.zeros((n, 4))
        result['std_transformed'][:, 0] = 1.
    return result


def factored(logq, raw=.8):
    logq = np.atleast_1d(np.asarray(logq, dtype=float))
    p = prediction(len(logq))
    p[MH2_FACTORIZATION_MARKER] = True
    p['physical_classifier_probability'] = np.broadcast_to(raw, logq.shape).astype(float).copy()
    p['log_mh_square_positive_probability'] = logq.copy()
    with np.errstate(under='ignore'):
        p['physical_probability'] = p['physical_classifier_probability'] * np.exp(logq)
    return p


def old_acquisition():
    path = Path(__file__).resolve().parent / '_snapshots/v4_acquisition.py'
    spec = spec_from_file_location('mssm_mls_modern._v4_entropy_contract_reference', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.acquisition


def mp_log_entropy(lp):
    mp = pytest.importorskip('mpmath')
    with mp.workdps(150):
        value = mp.mpf(float(lp))
        p = mp.exp(value)
        q = -mp.expm1(value)
        # Use the accurately represented small side near either endpoint.
        lq = mp.log1p(-p) if value < mp.log(mp.mpf('.5')) else mp.log(q)
        return float(mp.log(-p * value - q * lq))


def test_entropy_matches_high_precision_in_centre_and_extreme_log_tails():
    values = np.array([-1e6, -1000., -100., -10., -2., np.log(.5),
                       -.1, -1e-8, -1e-12, -1e-300, np.nextafter(0., -1.)])
    result = log_bernoulli_entropy(values)
    expected = np.array([mp_log_entropy(v) for v in values])
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, expected, rtol=2e-14, atol=2e-10)
    # exp(logp) underflows for these inputs, but log entropy must remain useful.
    assert result[0] > values[0] and result[1] > values[1]
    assert result[0] < result[1]


def test_entropy_has_exact_zero_probability_endpoints_and_expected_shapes():
    result = log_bernoulli_entropy(np.array([[-np.inf, 0.], [-0., -1.]]))
    assert result.shape == (2, 2)
    assert np.isneginf(result[0]).all() and np.isneginf(result[1, 0])
    assert np.isfinite(result[1, 1])
    assert np.shape(log_bernoulli_entropy(-1.)) == ()
    assert log_bernoulli_entropy(np.array([])).shape == (0,)
    np.testing.assert_allclose(log_bernoulli_entropy(np.array([-1, 0], dtype=int))[0],
                               np.log(np.exp(-1.) - (-np.expm1(-1.)) * np.log(-np.expm1(-1.))), rtol=1e-14)


def test_entropy_symmetry_and_half_probability_is_the_maximum():
    p = np.array([1e-12, .01, .1, .25, .5])
    left = log_bernoulli_entropy(np.log(p))
    right = log_bernoulli_entropy(np.log1p(-p))
    np.testing.assert_allclose(left, right, rtol=1e-13, atol=1e-13)
    assert np.all(np.diff(left) > 0)
    assert left[-1] == pytest.approx(np.log(np.log(2.)), abs=2e-15)


@pytest.mark.parametrize('value', [np.nan, np.inf, 1e-300, [0., .001],
    [1j], ['-.1'], [True], np.array([-.2], dtype=object), None])
def test_entropy_rejects_nonreal_or_out_of_domain_values(value):
    with pytest.raises((ValueError, TypeError)):
        log_bernoulli_entropy(value)


def test_gp_entropy_is_of_the_full_raw_feasibility_and_not_disagreement():
    p = prediction(3)
    p['physical_probability'] = np.array([.01, .5, .99])
    p['std_transformed'][:, 0] = 0.
    p['uncertainty'] = np.array([1e100, 0., 1e-100])
    exploit, uncertain = acquisition(p, ModernSettings())
    np.testing.assert_allclose(exploit, np.log([.01, .5, .99]), rtol=0., atol=1e-15)
    np.testing.assert_allclose(uncertain, log_bernoulli_entropy(exploit), rtol=0., atol=0.)
    assert uncertain[1] > uncertain[0] and uncertain[1] > uncertain[2]
    q = deepcopy(p); q['uncertainty'][:] = 1.
    for a, b in zip(acquisition(p, ModernSettings()), acquisition(q, ModernSettings())):
        np.testing.assert_array_equal(a, b)


def test_far_window_mean_has_less_entropy_even_with_enormous_disagreement():
    p = prediction(2)
    p['physical_probability'][:] = .5
    p['mean'][:, 1] = p['mean_transformed'][:, 1] = [126., 150.]
    p['std_transformed'][:, 1] = 1.
    p['uncertainty'][:] = [1., 1e100]
    exploit, uncertain = acquisition(p, ModernSettings())
    assert np.isfinite(exploit).all() and np.isfinite(uncertain).all()
    assert exploit[1] < exploit[0] - 100.
    assert uncertain[1] < uncertain[0] - 100.
    np.testing.assert_array_equal(uncertain, log_bernoulli_entropy(exploit))


def test_marked_mass_event_is_counted_once_and_legacy_combined_is_diagnostic():
    p = factored(np.log(.2), raw=.5)
    p['mean'][0, 0] = p['mean_transformed'][0, 0] = -2.
    p['std_transformed'][0, 0] = 3.
    first = acquisition(p, ModernSettings())
    assert first[0][0] == pytest.approx(np.log(.1), abs=1e-15)
    assert first[1][0] == pytest.approx(np.log(-.1 * np.log(.1) - .9 * np.log(.9)), abs=1e-14)
    for combined in (0., .01, 1.):
        other = deepcopy(p)
        other['physical_probability'][:] = combined
        other['std_transformed'][:, 0] = 900.
        other['mean'][:, 0] = other['mean_transformed'][:, 0] = 20.
        for a, b in zip(first, acquisition(other, ModernSettings())):
            np.testing.assert_array_equal(a, b)


def test_tiny_mass_probability_is_not_floored_and_exact_zero_stays_impossible():
    p = factored([-1000., -1e6, -np.inf, 0.], raw=.8)
    exploit, uncertain = acquisition(p, ModernSettings())
    np.testing.assert_allclose(exploit[[0, 1, 3]], np.log(.8) + np.array([-1000., -1e6, 0.]),
                               rtol=0., atol=1e-10)
    assert np.isfinite(exploit[[0, 1]]).all() and np.isfinite(uncertain[[0, 1]]).all()
    assert np.isneginf(exploit[2]) and np.isneginf(uncertain[2])
    assert exploit[1] < exploit[0] - 900000.
    np.testing.assert_array_equal(uncertain, log_bernoulli_entropy(exploit))


def test_raw_classifier_keeps_original_floor_separately_from_mass_factor():
    p = factored([-1000., -1000., -np.inf], raw=np.array([0., 1e-15, 0.]))
    settings = ModernSettings()
    exploit, uncertain = acquisition(p, settings)
    expected = np.log(settings.probability_floor) - 1000.
    assert exploit[0] == exploit[1] == pytest.approx(expected, abs=1e-12)
    assert np.isneginf(exploit[2]) and np.isneginf(uncertain[2])


@pytest.mark.parametrize('value', [None, 0, 1, 'true', np.bool_(True), np.array(True), [], {}])
def test_factorization_marker_is_strictly_a_python_boolean(value):
    p = factored(np.log(.2)); p[MH2_FACTORIZATION_MARKER] = value
    with pytest.raises(ValueError):
        acquisition(p, ModernSettings())


@pytest.mark.parametrize('missing', ['std_transformed', 'physical_classifier_probability',
                                   'log_mh_square_positive_probability'])
def test_true_marker_requires_its_complete_explicit_contract(missing):
    p = factored(np.log(.2)); del p[missing]
    with pytest.raises(ValueError):
        acquisition(p, ModernSettings())


@pytest.mark.parametrize('field,value', [
    ('physical_classifier_probability', np.array([-.1])),
    ('physical_classifier_probability', np.array([1.1])),
    ('physical_classifier_probability', np.array([np.nan])),
    ('physical_classifier_probability', np.array([.1, .2])),
    ('log_mh_square_positive_probability', np.array([np.nan])),
    ('log_mh_square_positive_probability', np.array([np.inf])),
    ('log_mh_square_positive_probability', np.array([.001])),
    ('log_mh_square_positive_probability', np.array([-.1, -.2])),
    ('physical_classifier_probability', np.array([.4 + 1j])),
    ('physical_classifier_probability', np.array([True])),
    ('physical_classifier_probability', np.array(['.4'])),
    ('log_mh_square_positive_probability', np.array([-.4 + 1j])),
    ('log_mh_square_positive_probability', np.array([False])),
    ('log_mh_square_positive_probability', np.array(['-.4'])),
])
def test_marked_factor_values_and_shapes_are_checked(field, value):
    p = factored(np.log(.2)); p[field] = value
    with pytest.raises(ValueError):
        acquisition(p, ModernSettings())


@pytest.mark.parametrize('policy', ['paper-higgs', 'eq7-only'])
@pytest.mark.parametrize('marker', ['absent', False])
def test_no_std_nn_both_scores_match_v4_value_for_value(policy, marker):
    p = prediction(7, gp=False)
    p['physical_probability'][:] = [0., 1e-15, .1, .8, 1., .5, .9]
    p['hb_probability'][:] = [.1, .5, 1., 0., .7, .8, .9]
    p['solver_probability'][:] = [1., .5, 0., .1, .9, 1., .5]
    p['uncertainty'][:] = [1., 0., .01, 10., 1e-20, 1., 3.]
    p['mean'][:, 0] = p['mean_transformed'][:, 0] = [10., -2., 0., 5., 100., 3., 8.]
    p['mean'][:, 1] = p['mean_transformed'][:, 1] = [126., 122., 130., 125., 128., 126., 127.]
    p['mean'][:, 2] = p['mean_transformed'][:, 2] = [100., 120., -1., 0., 90., 110., 95.]
    p['mean'][5, 0] = np.nan  # Legacy invalid-row behavior is also unchanged.
    if marker is not False:
        assert marker == 'absent'
    else:
        p[MH2_FACTORIZATION_MARKER] = False
        # Unavailable new-factor metadata does not override the legacy path.
        p['physical_classifier_probability'] = np.array([np.nan])
        p['log_mh_square_positive_probability'] = np.array([np.inf])
    settings = replace(ModernSettings(), target_policy=policy)
    old = old_acquisition()(p, settings)
    new = acquisition(p, settings)
    for a, b in zip(old, new):
        np.testing.assert_array_equal(a, b)


def test_unmarked_gp_exploitation_keeps_legacy_path_but_explores_feasibility_entropy():
    p = prediction(3)
    p['std_transformed'][:] = [[.2, 1., 4., .05], [1., 2., 10., .3], [2., .1, 1., .01]]
    p['mean'][:, 1] = p['mean_transformed'][:, 1] = [126., 131., 125.]
    settings = ModernSettings()
    old, _ = old_acquisition()(p, settings)
    for marker in ('absent', False):
        q = deepcopy(p)
        if marker is False:
            q[MH2_FACTORIZATION_MARKER] = False
        exploit, uncertain = acquisition(q, settings)
        np.testing.assert_array_equal(exploit, old)
        np.testing.assert_array_equal(uncertain, log_bernoulli_entropy(exploit))


def test_positive_squared_mass_log_probability_matches_high_precision_gaussian():
    mp = pytest.importorskip('mpmath')
    mean = np.array([[-50., -40., -20.], [0., 1., 20.]])
    sigma = np.ones_like(mean)
    with mp.workdps(150):
        expected = np.array([float(mp.log(mp.erfc(-mp.mpf(float(z)) / mp.sqrt(2)) / 2))
                             for z in mean.ravel()]).reshape(mean.shape)
    actual = positive_mh_squared_log_probability(mean, sigma)
    assert actual.shape == mean.shape and np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-300)
    assert actual[0, 0] < -1000. and actual[0, 1] < -800.
    assert actual[1, 2] < 0.  # The small positive-side complement is retained.


def test_zero_variance_squared_mass_event_uses_strict_positive_mean():
    mean = np.array([-1., -0., 0., np.nextafter(0., 1.), 1.])
    result = positive_mh_squared_log_probability(mean, np.zeros_like(mean))
    assert np.isneginf(result[:3]).all()
    np.testing.assert_array_equal(result[3:], [0., 0.])
    assert np.shape(positive_mh_squared_log_probability(1., 0.)) == ()
    assert positive_mh_squared_log_probability(np.array([]), np.array([])).shape == (0,)


@pytest.mark.parametrize('mean,sigma', [([0.], [-1.]), ([np.nan], [1.]), ([1.], [np.nan]),
    ([np.inf], [1.]), ([1.], [np.inf]), ([1., 2.], [1.]), ([True], [1.]),
    (['1'], [1.]), ([1j], [1.]), ([1.], [False]), (np.array([1.], dtype=object), [1.])])
def test_squared_mass_probability_rejects_malformed_gaussians(mean, sigma):
    with pytest.raises((ValueError, TypeError)):
        positive_mh_squared_log_probability(mean, sigma)


class StubGP:
    def __init__(self, mean, sigma):
        self.value, self.sigma = mean, sigma
    def predict(self, x):
        return np.full(len(x), self.value), np.full(len(x), self.sigma)
    def mean_gradient(self, x):
        return np.zeros_like(x)


def test_gp_prediction_exposes_raw_factors_and_retains_tiny_event_through_acquisition():
    model = GPSurrogates(ModernSettings(), seed=0)
    model.regressors = [StubGP(-40., 1.), StubGP(0., 0.), StubGP(0., 0.), StubGP(0., 0.)]
    model.mean = np.array([0., 126., 100., np.log(.12)])
    model.scale = model.prediction_scale = np.ones(4)
    model.detuning_mean, model.detuning_scale = 0., 1.
    model.constants = {'physical': .4, 'hb': 1., 'success': 1.}
    result = model.predict([[10., 8000., 6.]])
    assert result[MH2_FACTORIZATION_MARKER] is True
    assert result['physical_classifier_probability'][0] == .4
    assert result['physical_probability'][0] == 0.  # Legacy erf diagnostic underflows.
    logq = positive_mh_squared_log_probability(np.array([-40.]), np.ones(1))[0]
    assert np.isfinite(logq) and logq < -800.
    assert result['log_mh_square_positive_probability'][0] == logq
    exploit, uncertain = acquisition(result, ModernSettings())
    assert exploit[0] == pytest.approx(np.log(.4) + logq, abs=1e-12)
    assert np.isfinite(uncertain[0])
