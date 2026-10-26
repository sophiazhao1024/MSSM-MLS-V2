"""Synthetic mechanism tests only; no scan-efficiency or global-coverage claim."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest

from mssm_mls_modern.acquisition import (candidate_pool, quality_farthest_indices,
                                        true_target, true_violation)
from mssm_mls_modern.domain import BOUNDS
from mssm_mls_modern.gp_learning import select_anchors
from mssm_mls_modern.settings import ModernSettings


def row(unit, eval_id, *, target=True, hb=1):
    return {"eval_id": eval_id, "x": (BOUNDS[:, 0] + np.asarray(unit) * np.ptp(BOUNDS, axis=1)).tolist(),
            "status": "ok", "hb_result": hb,
            "observables": {"mh": 10., "mH": 125., "chi2_hs": 90. if target else 180., "omega": .12},
            "diagnostics": {"hb_result": hb, "hb_obsratio": .5 if hb == 1 else 2.}}


def separated_targets():
    early = [row([.2 + i * .00001, .2, .2], i + 1) for i in range(40)]
    late = [row([.8, .8, .8], 101), row([.8, .2, .8], 102), row([.2, .8, .8], 103)]
    return early + late


def test_late_separated_target_groups_enter_same_quality_local_centres():
    settings = replace(ModernSettings(), local_elites=4, pool_size=128)
    records = separated_targets()
    pool, labels, report = candidate_pool(records, settings, np.random.default_rng(4))
    # Stable arrival ranking previously selected only IDs 1..4. All three late
    # groups now participate while the known early group retains a centre.
    assert {101, 102, 103} <= set(report["elite_eval_ids"])
    assert any(i <= 40 for i in report["elite_eval_ids"])
    assert report["elite_true_targets"] == [True] * 4
    assert report["elite_violations"] == [0.] * 4
    assert report["global_uniform"] == report["local_true_elite"] == 64
    assert pool.shape == (128, 3) and len(labels) == 128
    assert np.all(pool >= BOUNDS[:, 0]) and np.all(pool <= BOUNDS[:, 1])


def test_local_selection_is_arrival_order_invariant_without_extra_rng_draws():
    records = separated_targets()
    settings = replace(ModernSettings(), local_elites=4, pool_size=128)
    a, b = np.random.default_rng(9), np.random.default_rng(9)
    first, labels_a, report_a = candidate_pool(records, settings, a)
    second, labels_b, report_b = candidate_pool(records[::-1], settings, b)
    np.testing.assert_array_equal(first, second)
    assert labels_a == labels_b and report_a == report_b
    np.testing.assert_array_equal(a.random(5), b.random(5))


def test_late_groups_enter_priority_anchor_half_and_global_half_still_covers_poor_rows():
    records = separated_targets()
    records += [row([.05, .05, .95], 201, target=False), row([.95, .05, .05], 202, target=False)]
    indices = select_anchors(records, 8)
    priority_ids = {records[i]["eval_id"] for i in indices[:4]}
    assert {101, 102, 103} <= priority_ids
    assert all(true_target(records[i]) for i in indices[:4])
    # The other half remains eligible to represent globally separated physical
    # non-target observations, rather than restricting every anchor to targets.
    assert {201, 202} <= {records[i]["eval_id"] for i in indices[4:]}
    assert len(indices) == len(set(indices)) == 8
    reversed_rows = records[::-1]
    reversed_indices = select_anchors(reversed_rows, 8)
    assert priority_ids == {reversed_rows[i]["eval_id"] for i in reversed_indices[:4]}


def test_quality_order_is_exact_and_never_traded_for_a_far_bad_point():
    coordinates = np.array([[.2, .2, .2], [.20001, .2, .2], [.99, .99, .99]])
    # The worse point is arbitrarily far, but exact quality takes precedence.
    assert quality_farthest_indices(coordinates, [0., 0., 1e-14], 2).tolist() == [0, 1]
    np.testing.assert_array_equal(quality_farthest_indices(coordinates, [0., 0., 0.], 2), [0, 2])
    assert quality_farthest_indices(coordinates, [0., 0., 0.], 0).size == 0
    with pytest.raises(ValueError):
        quality_farthest_indices(coordinates, [0., np.nan, 0.], 2)


def test_zero_violation_endpoints_and_unknown_hb_are_not_true_target_priority():
    settings = replace(ModernSettings(), local_elites=2, pool_size=128)
    records = [row([.01, .01, .01], 1), row([.99, .99, .99], 2, hb=-1),
               row([.4, .4, .4], 3), row([.41, .41, .41], 4)]
    records[0]["observables"]["mH"] = 123.  # strict window endpoint
    records[0]["target"] = records[1]["target"] = True  # fabricated flags are not trusted
    assert true_violation(records[0], settings) == true_violation(records[1], settings) == 0.
    assert not true_target(records[0]) and not true_target(records[1])
    _, _, report = candidate_pool(records, settings, np.random.default_rng(1))
    assert set(report["elite_eval_ids"]) == {3, 4}
    many = records + [row([.5 + i * .001, .5, .5], 10 + i, target=False) for i in range(5)]
    anchors = select_anchors(many, 4)
    assert set(anchors[:2]) == {2, 3}


def test_candidate_prediction_and_uncomputed_truth_cannot_create_elite_or_anchor():
    records = separated_targets()
    settings = replace(ModernSettings(), local_elites=4, pool_size=128)
    original, _, info = candidate_pool(records, settings, np.random.default_rng(1))
    altered = deepcopy(records)
    for record in altered:
        record["predicted"] = {"mh": 999., "mH": 999., "omega": 999., "chi2_hs": 999.}
        record["target"] = False  # selection independently rechecks existing true observables
    for status, i in (("tool_error", 1001), ("unphysical", 1002), ("pending", 1003)):
        future = row([.001, .999, .001], i)
        future.update(status=status, target=True)
        altered.append(future)
    observed, _, details = candidate_pool(altered, settings, np.random.default_rng(1))
    np.testing.assert_array_equal(original, observed)
    assert info["elite_eval_ids"] == details["elite_eval_ids"]
    assert not ({1001, 1002, 1003} & set(details["elite_eval_ids"]))
    # GP anchor selection receives only physical training rows via fit; adding
    # irrelevant prediction payloads must not alter its selected coordinates.
    np.testing.assert_array_equal(select_anchors(records, 8), select_anchors(altered[:len(records)], 8))


def test_actual_lowercase_hb_ratio_changes_only_known_exclusion_penalty():
    settings = ModernSettings()
    excluded = row([.5, .5, .5], 1, hb=0)
    excluded["diagnostics"]["hb_obsratio"] = float(np.exp(3.))
    assert true_violation(excluded, settings) == pytest.approx(3.)
    old_case = deepcopy(excluded)
    old_case["diagnostics"]["HB_obsratio"] = old_case["diagnostics"].pop("hb_obsratio")
    assert true_violation(old_case, settings) == true_violation(excluded, settings)
    excluded["hb_result"] = excluded["diagnostics"]["hb_result"] = 1
    assert true_violation(excluded, settings) == 0.


def test_global_proposals_and_rng_consumption_equal_v3_for_same_history():
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path
    original_path = Path(__file__).resolve().parent / "_snapshots/v3_acquisition.py"
    spec = spec_from_file_location("mssm_mls_modern._v3_coverage_test", original_path)
    original = module_from_spec(spec)
    spec.loader.exec_module(original)
    settings = replace(ModernSettings(), local_elites=4, pool_size=128)
    a, b = np.random.default_rng(41), np.random.default_rng(41)
    old_pool, old_labels, old_report = original.candidate_pool(separated_targets(), settings, a)
    new_pool, new_labels, new_report = candidate_pool(separated_targets(), settings, b)
    assert old_report["global_uniform"] == new_report["global_uniform"] == 64
    np.testing.assert_array_equal(old_pool[:64], new_pool[:64])
    assert old_labels == new_labels
    np.testing.assert_array_equal(a.random(8), b.random(8))
