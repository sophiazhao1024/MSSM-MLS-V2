"""Finite-pool soft ranking with explicit exploration and exact deduplication.

Selection is a discovery heuristic, not a replacement likelihood, posterior
sampler or relaxation of the true physical acceptance windows.
"""
import numpy as np
import torch
from itertools import groupby
from fractions import Fraction
from .domain import BOUNDS, CHI2_MAX, HIGGS_INTERVAL, OMEGA_INTERVAL, parameter_array, sample, validate_observables, score
from .learning import hb_label


MH2_FACTORIZATION_MARKER = "mh2_feasibility_factorization_available"


def log_bernoulli_entropy(log_probability):
    """Stable log[-p log(p)-(1-p)log(1-p)] from log(p), in nats.

    This is an entropy of an uncalibrated feasibility proxy, not mutual
    information or a calibrated posterior uncertainty. At p=0 or p=1 the
    entropy is zero, so its logarithm is -infinity.

    Symmetry lets us work with r=min(p,1-p)<=1/2. The second entropy term is
    (1-r)*r*[-log1p(-r)/r]. Its ratio tends to1 when exp(log(r)) underflows;
    keeping log(r) separately prevents tiny nonzero probabilities from losing
    their entropy ranking. The only limit branch is an exact floating zero.
    """
    raw = np.asarray(log_probability)
    if raw.dtype.kind not in "iuf":
        raise ValueError("Log probabilities must be real numeric values")
    value = np.asarray(raw, dtype=float)
    if np.any(np.isnan(value) | np.isposinf(value) | (value > 0)):
        raise ValueError("Log probabilities must lie in [-infinity,0]")
    result = np.full(value.shape, -np.inf)
    interior = np.isfinite(value) & (value < 0)
    if interior.any():
        lp = value[interior]
        # expm1 retains 1-p near p=1. If p is tiny, log(1-p) can round to0;
        # min(lp,logq) still keeps the accurately represented smaller event.
        logq = np.log(-np.expm1(lp))
        logr = np.minimum(lp, logq)
        with np.errstate(under="ignore"):
            r = np.exp(logr)
        ratio = np.ones_like(r)
        nonzero = r > 0
        ratio[nonzero] = -np.log1p(-r[nonzero]) / r[nonzero]
        log_first = logr + np.log(-logr)
        log_second = np.log1p(-r) + logr + np.log(ratio)
        result[interior] = np.logaddexp(log_first, log_second)
    return result


def normal_interval_log_probability(mean, standard_deviation, lower, upper, *, include_lower=False):
    """Stable marginal Gaussian window mass, not a calibrated joint probability.

    Switch to reflected survival tails on the positive side so subtracting
    two CDF values close to one does not erase a small interval probability.
    Zero predictive standard deviation is handled as a deterministic value.
    """
    mean = np.asarray(mean, dtype=float)
    standard_deviation = np.asarray(standard_deviation, dtype=float)
    if mean.shape != standard_deviation.shape or np.any(standard_deviation < 0):
        raise ValueError("Invalid Gaussian interval shapes or negative spread")
    result = np.full(mean.shape, -np.inf)
    deterministic = standard_deviation == 0
    above = mean >= lower if include_lower else mean > lower
    result[deterministic & above & (mean < upper)] = 0.
    positive = standard_deviation > 0
    if not positive.any():
        return result
    za = (lower - mean[positive]) / standard_deviation[positive]
    zb = (upper - mean[positive]) / standard_deviation[positive]
    logcdf = lambda z: torch.special.log_ndtr(torch.as_tensor(z.copy(), dtype=torch.float64)).numpy()
    if np.isneginf(lower):
        result[positive] = logcdf(zb)
    elif np.isposinf(upper):
        result[positive] = logcdf(-za)
    else:
        # Compare bounds' midpoint with zero without adding enormous tails.
        right_tail = mean[positive] < (lower + upper) / 2.
        larger = np.where(right_tail, -za, zb)
        smaller = np.where(right_tail, -zb, za)
        large_log, small_log = logcdf(larger), logcdf(smaller)
        delta = np.minimum(small_log - large_log, 0.)
        with np.errstate(divide="ignore", invalid="ignore"):
            logmass = large_log + np.log(-np.expm1(delta))
        result[positive] = logmass
    return result


def acquisition(prediction, settings):
    """GP log feasibility and feasibility-boundary entropy exploration.

    GP marginal distributions are a working model. Multiplying their masses
    ignores output correlations and does not claim calibrated joint coverage.
    The no-std NN compatibility path retains its original logistic cuts and
    log-disagreement score. A strict GP factorization marker selects the raw
    classifier and exact log P(mh squared>0) factors; the legacy combined
    physical_probability then remains a diagnostic rather than a scored term.
    """
    y = np.asarray(prediction["mean"], dtype=float)
    transformed = np.asarray(prediction["mean_transformed"], dtype=float)
    uncertainty = np.asarray(prediction["uncertainty"], dtype=float)
    physical = np.asarray(prediction["physical_probability"], dtype=float)
    hb = np.asarray(prediction["hb_probability"], dtype=float)
    n = len(y)
    if (y.shape != (n, 4) or transformed.shape != (n, 4)
            or any(v.shape != (n,) for v in (uncertainty, physical, hb))):
        raise ValueError("Inconsistent modern prediction shapes")
    solver = np.asarray(prediction.get("solver_probability", np.ones(n)), dtype=float)
    if solver.shape != (n,):
        raise ValueError("Inconsistent solver probability shape")
    finite = (np.isfinite(y).all(axis=1) & np.isfinite(transformed).all(axis=1)
              & np.isfinite(uncertainty) & (uncertainty >= 0)
              & np.isfinite(physical) & (physical >= 0) & (physical <= 1)
              & np.isfinite(hb) & (hb >= 0) & (hb <= 1)
              & np.isfinite(solver) & (solver >= 0) & (solver <= 1))
    spread = prediction.get("std_transformed")
    factorized = prediction.get(MH2_FACTORIZATION_MARKER, False)
    if type(factorized) is not bool:
        raise ValueError("The mh2 factorization marker must be a scalar boolean")
    if factorized and spread is None:
        raise ValueError("The mh2 factorization contract requires GP transformed spreads")
    if spread is not None:
        spread = np.asarray(spread, dtype=float)
        if spread.shape != (n, 4):
            raise ValueError("std_transformed must have N x 4 physical/log-label units")
        finite &= np.isfinite(spread).all(axis=1) & (spread >= 0).all(axis=1)
    if factorized:
        if ("physical_classifier_probability" not in prediction
                or "log_mh_square_positive_probability" not in prediction):
            raise ValueError("The marked GP prediction lacks its explicit physical factors")
        raw_classifier = np.asarray(prediction["physical_classifier_probability"])
        raw_log_mh2 = np.asarray(prediction["log_mh_square_positive_probability"])
        if raw_classifier.dtype.kind not in "iuf" or raw_log_mh2.dtype.kind not in "iuf":
            raise ValueError("Explicit GP physical factors must be real numeric arrays")
        classifier = np.asarray(raw_classifier, dtype=float)
        log_mh2_positive = np.asarray(raw_log_mh2, dtype=float)
        if classifier.shape != (n,) or log_mh2_positive.shape != (n,):
            raise ValueError("Inconsistent explicit GP physical-factor shapes")
        if (np.any(~np.isfinite(classifier) | (classifier < 0) | (classifier > 1))
                or np.any(np.isnan(log_mh2_positive) | np.isposinf(log_mh2_positive) | (log_mh2_positive > 0))):
            raise ValueError("Invalid explicit GP physical-factor values")
    exploit, uncertain = np.full(n, -np.inf), np.full(n, -np.inf)
    if finite.any():
        mh, mH, chi2 = y[finite, :3].T
        logomega = transformed[finite, 3]
        p = np.clip(physical[finite], settings.probability_floor, 1.)
        log_physical = (np.log(np.clip(classifier[finite], settings.probability_floor, 1.))
                        + log_mh2_positive[finite] if factorized else np.log(p))
        ph = (np.clip(hb[finite], settings.probability_floor, 1.)
              if settings.target_policy == "paper-higgs" else np.ones(len(p)))
        ps = np.clip(solver[finite], settings.probability_floor, 1.)
        if spread is not None:
            sd = spread[finite]
            log_soft_cuts = (
                (np.zeros(len(mh)) if factorized else normal_interval_log_probability(mh, sd[:, 0], 0., np.inf))
                + normal_interval_log_probability(mH, sd[:, 1], *HIGGS_INTERVAL)
                + normal_interval_log_probability(chi2, sd[:, 2], 0., CHI2_MAX, include_lower=True)
                + normal_interval_log_probability(logomega, sd[:, 3],
                                                  *np.log(OMEGA_INTERVAL)))
        else:
            margins = np.column_stack([
                (mH - HIGGS_INTERVAL[0]) / settings.mH_soft_width,
                (HIGGS_INTERVAL[1] - mH) / settings.mH_soft_width,
                (CHI2_MAX - chi2) / settings.chi2_soft_width,
                (logomega - np.log(OMEGA_INTERVAL[0])) / settings.lnomega_soft_width,
                (np.log(OMEGA_INTERVAL[1]) - logomega) / settings.lnomega_soft_width,
                mh / settings.mh_soft_width, chi2 / settings.chi2_soft_width])
            log_soft_cuts = -np.logaddexp(0., -margins).sum(axis=1)
        # This experiment seeks joint-constraint discoveries, not draws from
        # the original Eq.7 density. True Eq.7 acceptance remains unchanged.
        exploit[finite] = log_physical + np.log(ph) + np.log(ps) + log_soft_cuts
        if spread is not None:
            # The same raw feasibility product used for exploitation, before
            # within-batch distance repulsion, defines a Bernoulli proxy.
            uncertain[finite] = log_bernoulli_entropy(exploit[finite])
        else:
            with np.errstate(divide="ignore"):
                uncertain[finite] = np.log(uncertainty[finite]) + np.log(p) + np.log(ph) + np.log(ps)
    return exploit, uncertain


def true_target(record, target_policy="paper-higgs"):
    """Recheck existing truth with the unchanged strict windows and HB label.

    A zero distance from a window is not sufficient: exact cut endpoints and
    missing HB results are not known accepted points. No prediction is read.
    """
    if record.get("status") != "ok":
        return False
    try:
        values = validate_observables(record.get("observables"))
    except (TypeError, ValueError):
        return False
    return bool(score(values.reshape(1, 4))[1][0]) and (target_policy == "eq7-only" or hb_label(record) == 1)


def quality_farthest_indices(x, quality, count):
    """Preserve exact quality order; spread selections only within equal tiers.

    x is normalized by the known parameter box. There is no quality binning,
    metric-grid use, new distance threshold, or RNG draw. The first point and
    exact distance ties use coordinate lexicographic order, not arrival order.
    Once a tier is exhausted, the next tier maximizes distance to all selected
    points. This is a finite representative selection, not a coverage proof.
    """
    x = np.asarray(x, dtype=float)
    quality = np.asarray(quality, dtype=float)
    if quality.ndim == 1:
        quality = quality[:, None]
    if (x.ndim != 2 or quality.ndim != 2 or len(x) != len(quality)
            or not np.isfinite(x).all() or not np.isfinite(quality).all()
            or type(count) is not int or count < 0):
        raise ValueError("Invalid coordinates, quality tiers or selection count")
    order = sorted(range(len(x)), key=lambda i: (*quality[i], *x[i]))
    chosen = []
    minimum = np.full(len(x), np.inf)
    for _, indices in groupby(order, key=lambda i: tuple(quality[i])):
        candidates = np.asarray(list(indices), dtype=int)
        active = np.ones(len(candidates), dtype=bool)
        for _ in range(min(count - len(chosen), len(candidates))):
            distances = np.where(active, minimum[candidates], -np.inf)
            position = int(np.argmax(distances))
            index = int(candidates[position])
            chosen.append(index)
            active[position] = False
            minimum = np.minimum(minimum, np.sum((x - x[index])**2, axis=1))
        if len(chosen) == count:
            break
    return np.asarray(chosen, dtype=int)


def true_violation(record, settings):
    """Rank only already evaluated physical rows, without external coordinates.

    Dimensionless distances outside the original windows are summed. This is
    a candidate-generation heuristic, never a target decision. A known HB
    exclusion adds a penalty; missing HB information does not become a label.
    """
    if record.get("status") != "ok":
        return np.inf
    try:
        mh, mH, chi2, omega = validate_observables(record.get("observables"))
    except (TypeError, ValueError):
        return np.inf
    logomega = np.log(omega)
    value = (max(HIGGS_INTERVAL[0] - mH, mH - HIGGS_INTERVAL[1], 0.) / settings.mH_soft_width
             + max(chi2 - CHI2_MAX, 0.) / settings.chi2_soft_width
             + max(np.log(OMEGA_INTERVAL[0]) - logomega,
                   logomega - np.log(OMEGA_INTERVAL[1]), 0.) / settings.lnomega_soft_width)
    if settings.target_policy == "paper-higgs" and record.get("hb_result") == 0:
        ratio = record.get("hb_obsratio")
        # Accept only an explicitly available measured ratio. Do not infer
        # one from surrogate output or a missing/unknown HB result.
        if ratio is None and isinstance(record.get("diagnostics"), dict):
            ratio = record["diagnostics"].get("hb_obsratio", record["diagnostics"].get("HB_obsratio"))
        if isinstance(ratio, (int, float)) and np.isfinite(ratio) and ratio > 0:
            value += max(1., np.log(ratio))
        else:
            value += 1.
    return float(value)


def candidate_pool(records, settings, rng):
    """Mix global proposals with reflected Gaussian proposals around true elites.

    All centres are taken from this run's current evaluated records. No stored
    target coordinates, fitted historical weights, or external data enter.
    Scales are standard deviations in the unit parameter box. Reflection
    keeps proposals in bounds without an artificial boundary point mass.
    """
    ranked = [(true_violation(row, settings), i, row) for i, row in enumerate(records)]
    ranked = [entry for entry in ranked if np.isfinite(entry[0])]
    ranked.sort(key=lambda entry: (entry[0], entry[1]))
    unique = []
    seen = set()
    for violation, _, row in ranked:
        key = exact_key(row["x"])
        if key not in seen:
            unique.append((violation, row)); seen.add(key)
    elites = []
    if unique:
        coordinates = parameter_array([row["x"] for _, row in unique])
        unit = (coordinates - BOUNDS[:, 0]) / np.ptp(BOUNDS, axis=1)
        quality = [(violation, not true_target(row, settings.target_policy)) for violation, row in unique]
        indices = quality_farthest_indices(unit, quality, settings.local_elites)
        elites = [unique[i] for i in indices]
    local_count = int(settings.pool_size * settings.local_fraction) if elites else 0
    global_count = settings.pool_size - local_count
    global_pool = sample(global_count, rng)
    labels = ["global_uniform"] * global_count
    if local_count:
        centres = parameter_array([row["x"] for _, row in elites])
        unit = (centres - BOUNDS[:, 0]) / np.ptp(BOUNDS, axis=1)
        # Cycling assignments ensures small local pools still cover several
        # elites and scales; a permutation avoids order-based tie advantages.
        assignments = np.arange(local_count)
        rng.shuffle(assignments)
        elite_indices = assignments % len(elites)
        scale_indices = (assignments // len(elites)) % len(settings.local_scales)
        sigma = np.asarray(settings.local_scales)[scale_indices]
        u = unit[elite_indices] + rng.normal(size=(local_count, 3)) * sigma[:, None]
        reflected = np.mod(u, 2.)
        reflected = np.where(reflected <= 1., reflected, 2. - reflected)
        local_pool = BOUNDS[:, 0] + reflected * np.ptp(BOUNDS, axis=1)
        pool = np.concatenate([global_pool, local_pool])
        labels += ["local_true_elite"] * local_count
    else:
        pool = global_pool
    diagnostics = {"global_uniform": global_count, "local_true_elite": local_count,
                   "elite_eval_ids": [row.get("eval_id") for _, row in elites],
                   "elite_violations": [v for v, _ in elites],
                   "local_scales_unit_box": list(settings.local_scales),
                   "elite_source": "current_run_true_ok_records_only",
                   "elite_strategy": "exact_constraint_quality_then_true_target_then_farthest_ties",
                   "elite_true_targets": [true_target(row, settings.target_policy) for _, row in elites]}
    return pool, labels, diagnostics


def exact_key(x):
    """Exact float identities; neither rounded coordinates nor nearby points."""
    return tuple(float(v).hex() for v in x)


def _unique_uniform(count, rng, excluded):
    points, keys = [], set(excluded)
    # Finite guard matters for malformed RNG/test samplers; ordinary continuous
    # sampling almost never repeats a float64 point in this 3D box.
    attempts = 0
    while len(points) < count and attempts < max(1000, 100 * count):
        candidates = sample(max(1, count - len(points)), rng)
        attempts += len(candidates)
        for x in candidates:
            key = exact_key(x)
            if key not in keys:
                points.append(x)
                keys.add(key)
    if len(points) != count:
        raise RuntimeError("Uniform sampler could not generate enough unique points")
    return points


def diverse_indices(pool, scores, count, selected, radius, rng):
    """Greedy log-utility plus soft nearest-anchor repulsion.

    A radius is a soft scale, not a hard exclusion threshold. Each chosen point
    is scored by log_utility + log(1-exp(-(nearest_distance/radius)**2)).
    Very poor faraway candidates therefore cannot win merely by filling a hard
    distance quota. Exact duplicates still have negative-infinite repulsion.
    Distances refer to this batch's chosen and reserved random points, measured
    in the unit parameter box. Full-history nearest distances are a separate
    evaluation metric and are not represented by this field.
    """
    if count == 0:
        return [], []
    unit = (pool - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
    anchor = np.asarray(selected, dtype=float).reshape(-1, 3)
    anchor = (anchor - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
    scores = np.asarray(scores, dtype=float)
    tie = rng.random(len(pool))
    active = np.isfinite(scores)
    distance = (np.linalg.norm(unit[:, None] - anchor[None], axis=2).min(axis=1)
                if len(anchor) else np.full(len(pool), np.inf))
    chosen, details = [], []
    for _ in range(count):
        if radius > 0:
            with np.errstate(divide="ignore", over="ignore"):
                penalty = np.log(-np.expm1(-np.square(distance / radius)))
        else:
            penalty = np.zeros(len(pool))
        penalized = np.where(active, scores + penalty, -np.inf)
        best = np.max(penalized)
        if not np.isfinite(best):
            break
        candidates = np.flatnonzero(penalized == best)
        i = int(candidates[np.argmin(tie[candidates])])
        details.append({"raw_acquisition": float(scores[i]),
                        "penalized_acquisition": float(penalized[i]),
                        "min_distance": float(distance[i]) if np.isfinite(distance[i]) else None,
                        "raw_utility_gap": float(np.max(scores[active]) - scores[i])})
        chosen.append(i)
        active[i] = False
        distance = np.minimum(distance, np.linalg.norm(unit - unit[i], axis=1))
    return chosen, details


def batch_quotas(settings, count):
    """Preserve unfiltered exploration even in a truncated final batch."""
    if type(count) is not int or count < 1 or count > settings.batch_points:
        raise ValueError("count must be between 1 and the configured batch size")
    random_count = min(count, max(1, int(np.ceil(count * settings.random_extra / settings.batch_points))))
    remaining = count - random_count
    model_total = settings.exploit + settings.uncertain
    exploit_count = int(np.floor(remaining * settings.exploit / model_total))
    return exploit_count, remaining - exploit_count, random_count


def interleave_execution(batch):
    """Schedule fixed candidates proportionally, beginning with exploration.

This changes execution order only: no candidate, score, constraint or random
stream is changed. Preserve within-category rank order. A category-concatenated
batch could reach the consecutive-error guard before any reserved exploration
was evaluated; interleaving gives each category an early execution opportunity.
The same five-error guard still applies to the actual execution sequence.
"""
    groups = {}
    for item in batch:
        groups.setdefault(item["origin"], []).append(item)
    priority = {name: index for index, name in enumerate(
        ("random", "exploit", "uncertain", "uniform_fallback"))}
    unknown = sorted(set(groups) - set(priority))
    priority.update({name: len(priority) + i for i, name in enumerate(unknown)})
    scheduled = []
    for origin, items in groups.items():
        for j, item in enumerate(items):
            scheduled.append((Fraction(2*j + 1, 2*len(items)), priority[origin], j, item))
    ordered = [entry[-1] for entry in sorted(scheduled, key=lambda entry: entry[:3])]
    first_random = next((i for i, item in enumerate(ordered) if item["origin"] == "random"), None)
    if first_random is not None and first_random != 0:
        ordered.insert(0, ordered.pop(first_random))
    return ordered


def propose_batch(model, records, settings, rng, *, explore_rng, count=None):
    """Return a bounded batch and audit diagnostics.

`rng` is for the finite candidate pool/ranking; `explore_rng` is independent
and never consumed by prediction/ranking. Uniform exploration is generated
first, bypassing all surrogates, then model points avoid those exact inputs.
Fallback is explicitly labelled and never presented as a learned proposal.
"""
    count = settings.batch_points if count is None else count
    ne, nu, nr = batch_quotas(settings, count)
    prior = {exact_key(r["x"]) for r in records}
    diagnostics = {"requested": count, "quotas": {"exploit": ne, "uncertain": nu, "random": nr},
                   "pool_size": 0, "unique_pool_size": 0, "diversity_relaxations": 0,
                   "fallback_reason": None, "selection_is_posterior": False,
                   "execution_order": "proportional_interleave_random_first",
                   "diversity_mode": "soft_log_nearest_anchor_repulsion",
                   "min_distance_scope": "unit_box_distance_to_previously_selected_or_reserved_points_in_this_batch",
                   "acquisition_mode": "not_ready"}
    exploration = _unique_uniform(nr, explore_rng, prior)
    if model is None or not model.ready:
        points = _unique_uniform(ne + nu, rng, prior | {exact_key(x) for x in exploration})
        diagnostics.update(fallback_reason="insufficient_training_data", fallback_count=ne + nu,
                           selected_counts={"exploit": 0, "uncertain": 0,
                                            "random": nr, "uniform_fallback": ne + nu})
        batch = [{"x": x, "origin": "uniform_fallback", "predicted": None} for x in points]
        batch.extend({"x": x, "origin": "random", "predicted": None} for x in exploration)
        return interleave_execution(batch), diagnostics
    excluded = prior | {exact_key(x) for x in exploration}
    raw_pool, raw_sources, pool_info = candidate_pool(records, settings, rng)
    diagnostics["candidate_generation"] = pool_info
    unique_pool = []
    pool_sources = []
    seen = set(excluded)
    for x, source in zip(raw_pool, raw_sources):
        key = exact_key(x)
        if key not in seen:
            unique_pool.append(x); pool_sources.append(source); seen.add(key)
    pool = np.asarray(unique_pool, dtype=float).reshape(-1, 3)
    diagnostics.update(pool_size=len(raw_pool), unique_pool_size=len(pool))
    batch = []
    if len(pool):
        parameter_array(pool)
        prediction = model.predict(pool)
        diagnostics["acquisition_mode"] = ("gp_marginal_feasibility" if "std_transformed" in prediction
                                           else "logistic_compatibility")
        diagnostics["calibrated_joint_probability"] = False
        diagnostics["uncertain_acquisition_mode"] = ("uncalibrated_feasibility_boundary_log_entropy"
            if "std_transformed" in prediction else "legacy_log_disagreement_compatibility")
        exploit_scores, uncertainty_scores = acquisition(prediction, settings)
        diagnostics.update(finite_exploit_scores=int(np.isfinite(exploit_scores).sum()),
                           finite_uncertainty_scores=int(np.isfinite(uncertainty_scores).sum()))
        selected = []
        for origin, scores, needed in (("exploit", exploit_scores, ne),
                                        ("uncertain", uncertainty_scores, nu)):
            scores = scores.copy()
            scores[selected] = -np.inf
            anchor = exploration + [pool[i] for i in selected]
            indices, details = diverse_indices(pool, scores, needed, anchor,
                                                settings.diversity_radius, rng)
            selected.extend(indices)
            for i, audit in zip(indices, details):
                batch.append({"x": pool[i], "origin": origin,
                              **audit,
                              "candidate_source": pool_sources[i],
                              "predicted": prediction["mean"][i].tolist(),
                              "uncertainty": float(prediction["uncertainty"][i]),
                              "physical_probability": float(prediction["physical_probability"][i]),
                              "hb_probability": float(prediction["hb_probability"][i]),
                              "solver_probability": float(prediction.get("solver_probability", np.ones(len(pool)))[i])})
    missing = ne + nu - len(batch)
    if missing:
        excluded = prior | {exact_key(x) for x in exploration} | {exact_key(r["x"]) for r in batch}
        # A model-dependent fallback consumes the pool RNG, not the reserved
        # exploration RNG, so it cannot shift future exploration draws.
        fallback = _unique_uniform(missing, rng, excluded)
        batch.extend({"x": x, "origin": "uniform_fallback", "predicted": None} for x in fallback)
        diagnostics["fallback_reason"] = "insufficient_finite_unique_ranked_candidates"
    batch.extend({"x": x, "origin": "random", "predicted": None} for x in exploration)
    diagnostics["fallback_count"] = missing
    diagnostics["selected_counts"] = {key: sum(r["origin"] == key for r in batch)
                                       for key in ("exploit", "uncertain", "random", "uniform_fallback")}
    diagnostics["selected_candidate_sources"] = {
        source: sum(row.get("candidate_source") == source for row in batch)
        for source in ("global_uniform", "local_true_elite")}
    audits = [row for row in batch if "raw_acquisition" in row]
    if audits:
        penalties = np.array([row["raw_acquisition"] - row["penalized_acquisition"] for row in audits])
        gaps = np.array([row["raw_utility_gap"] for row in audits])
        distances = [row["min_distance"] for row in audits if row["min_distance"] is not None]
        diagnostics["selection_tradeoff"] = {
            "mean_log_repulsion_cost": float(penalties.mean()),
            "max_log_repulsion_cost": float(penalties.max()),
            "mean_raw_utility_gap": float(gaps.mean()),
            "max_raw_utility_gap": float(gaps.max()),
            "min_selected_distance": min(distances) if distances else None,
            "median_selected_distance": float(np.median(distances)) if distances else None,
            "selected_below_soft_radius": sum(d < settings.diversity_radius for d in distances)}
    if len(batch) != count or len({exact_key(r["x"]) for r in batch}) != count:
        raise RuntimeError("Internal modern batch size or uniqueness failure")
    return interleave_execution(batch), diagnostics
