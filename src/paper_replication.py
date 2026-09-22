"""Rigobon--Sack Tables 1--3 with their zero-mean IV specification.

The caller supplies the H/L observations and the full-period panel. This module
does not select events, controls, dates, signs, or a preferred normalization.
It uses average cross-products, no intercept, and equal-sized H/L samples.
Optional pair IDs control complete-case deletion only: the bootstrap resamples
independently within H and L, not pairs. Bootstrap diagnostics supplement the
paper's IV point estimates and absolute t-statistics; the war paper does not
specify a bootstrap procedure or the precise covariance estimator for its SEs.

Default units reproduce the paper's percentage-point Treasury anchor: a -0.25
scenario means a 25 bp fall. Basis-point inputs require reference_unit='bp'
and scenario=-25 explicitly. Input data must already have aligned intervals.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


METHODS = ("omega1", "omega2", "pooled")


def _binary(values: pd.Series) -> pd.Series:
    mapping = {True: True, False: False, 1: True, 0: False,
               "H": True, "L": False, "high": True, "low": False}
    if values.isna().any() or any(value not in mapping for value in values.unique()):
        raise ValueError("Regime must be a complete binary H/L, high/low, or 1/0 column")
    return values.map(mapping).astype(bool)


def _sample(panel: pd.DataFrame, reference: str, outcome: str, regime: str):
    if reference == outcome or regime in (reference, outcome):
        raise ValueError("Reference, outcome and regime must be distinct columns")
    if not panel.index.is_unique:
        raise ValueError("Panel index must identify unique market observations")
    high = _binary(panel[regime])
    values = panel[[reference, outcome]].apply(pd.to_numeric, errors="raise")
    finite = pd.Series(np.isfinite(values.to_numpy()).all(axis=1), index=panel.index)
    pairs_dropped = 0
    if "pair_id" in panel:
        if panel.pair_id.isna().any():
            raise ValueError("Missing pair_id in the selected estimation panel")
        audit = pd.DataFrame({"pair_id": panel.pair_id, "high": high, "finite": finite})
        groups = audit.groupby("pair_id", sort=False).agg(n=("high", "size"),
                    nh=("high", "sum"), complete=("finite", "all"))
        if not ((groups.n == 2) & (groups.nh == 1)).all():
            raise ValueError("Every pair_id must contain exactly one H and one L")
        keep = groups.index[groups.complete]
        pairs_dropped = int((~groups.complete).sum())
        finite = panel.pair_id.isin(keep)
    selected = values.loc[finite]
    labels = high.loc[finite]
    h = selected.loc[labels].to_numpy(float)
    l = selected.loc[~labels].to_numpy(float)
    if len(h) != len(l):
        raise ValueError("Original signed instruments require equal H/L counts; supply a balanced selection")
    if len(h) < 3:
        raise ValueError("At least three complete observations in each regime are required")
    return h, l, selected.index, {
        "n_high": len(h), "n_low": len(l), "n_estimation": len(selected),
        "n_rows_dropped": int((~finite).sum()), "n_pairs_dropped": pairs_dropped,
    }


def _ratio(numerator: float, denominator: float, scale: float) -> float:
    tol = 100 * np.finfo(float).eps * max(abs(scale), np.finfo(float).tiny)
    return float(numerator / denominator) if abs(denominator) > tol else np.nan


def _iv(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    if z.ndim == 1:
        z = z[:, None]
    rms = np.sqrt(np.mean(z * z, axis=0))
    active = rms > 0
    z = z[:, active] / rms[active]
    if not z.shape[1]:
        return {key: np.nan for key in ("coefficient", "se_classical", "se_hc1",
                "abs_t_classical", "abs_t_hc1", "first_stage_r2", "denominator")}
    inverse = np.linalg.pinv(z.T @ z, rcond=1e-12)
    fitted_x = z @ (inverse @ (z.T @ x))
    denominator = float(x @ fitted_x)
    coef = _ratio(float(fitted_x @ y), denominator, float(x @ x))
    residual = y - coef * x
    n = len(x)
    if np.isfinite(coef) and denominator > 0:
        classical = np.sqrt((residual @ residual) / (n - 1) / denominator)
        hc1 = np.sqrt(n / (n - 1) * np.sum((fitted_x * residual) ** 2) / denominator ** 2)
    else:
        classical = hc1 = np.nan
    def statistic(se):
        if not np.isfinite(coef) or not np.isfinite(se):
            return np.nan
        return abs(coef / se) if se > 0 else (np.inf if coef != 0 else np.nan)
    return {"coefficient": coef, "se_classical": float(classical), "se_hc1": float(hc1),
            "abs_t_classical": statistic(classical), "abs_t_hc1": statistic(hc1),
            "first_stage_r2": _ratio(denominator, float(x @ x), float(x @ x)),
            "denominator": denominator}


def _point(high: np.ndarray, low: np.ndarray) -> dict:
    mh, ml = high.T @ high / len(high), low.T @ low / len(low)
    delta = mh - ml
    a, b, c = float(delta[0, 0]), float(delta[0, 1]), float(delta[1, 1])
    stacked = np.vstack((high, low))
    x, y = stacked.T
    sign = np.r_[np.ones(len(high)), -np.ones(len(low))]
    instruments = (sign * x, sign * y, sign[:, None] * stacked)
    results = {"delta_xx": a, "delta_xy": b, "delta_yy": c,
               "second_moment_x_low": float(ml[0, 0]), "second_moment_x_high": float(mh[0, 0]),
               "second_moment_y_low": float(ml[1, 1]), "second_moment_y_high": float(mh[1, 1]),
               "var_ratio": _ratio(float(mh[0, 0]), float(ml[0, 0]), float(mh[0, 0])),
               "rank_one_determinant": a * c - b * b}
    for method, z in zip(METHODS, instruments):
        results.update({f"{method}_{key}": value for key, value in _iv(x, y, z).items()})
    # Explicit ratios expose the algebra and preserve its undefined cases.
    results["omega1_coefficient"] = _ratio(b, a, float(mh[0, 0] + ml[0, 0]))
    results["omega2_coefficient"] = _ratio(c, b, float(np.sqrt((mh[0, 0] + ml[0, 0]) * (mh[1, 1] + ml[1, 1]))))
    results["estimator_difference"] = results["omega1_coefficient"] - results["omega2_coefficient"]
    scale = np.sqrt(np.maximum(np.diag((mh + ml) / 2), 0))
    if (scale > 0).all():
        normalized = delta / np.outer(scale, scale)
        eigenvalues, eigenvectors = np.linalg.eigh(normalized)
        fit = max(eigenvalues[-1], 0) * np.outer(eigenvectors[:, -1], eigenvectors[:, -1])
        norm = np.linalg.norm(normalized, ord="fro")
        results["positive_rank_one_residual"] = float(np.linalg.norm(normalized-fit, ord="fro") / norm) if norm > 0 else np.nan
        results["leading_eigenvalue"] = float(eigenvalues[-1])
        results["second_eigenvalue"] = float(eigenvalues[0])
    else:
        results.update(positive_rank_one_residual=np.nan, leading_eigenvalue=np.nan, second_eigenvalue=np.nan)
    return results


def estimate_response(panel: pd.DataFrame, reference: str, outcome: str,
                      regime: str = "high", bootstrap: int = 1999, seed: int = 2603) -> dict:
    """Original uncentered IV coefficients plus separately labeled diagnostics.

    Classical and HC1 SEs are explicit alternative covariance conventions; the
    supplied paper reports t-statistics without documenting which convention it
    used. Neither is called an exact reconstruction of the published t-values.
    Bootstrap intervals are not weak-IV-robust confidence sets or rank tests.
    Raw numerical results are retained when the variance ordering is unsupported.
    """
    if not isinstance(bootstrap, (int, np.integer)) or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer")
    high, low, _, metadata = _sample(panel, reference, outcome, regime)
    point = _point(high, low)
    result = {"reference": reference, "outcome": outcome, **metadata, **point,
              "centered": False, "intercept": False, "moment_ddof": 0,
              "bootstrap_repetitions": int(bootstrap), "seed": int(seed),
              "bootstrap_design": "independent_within_regime"}
    fields = [f"{method}_coefficient" for method in METHODS] + [
        "delta_xx", "delta_xy", "delta_yy", "estimator_difference",
        "positive_rank_one_residual", "rank_one_determinant"]
    draws = {key: np.empty(bootstrap) for key in fields}
    rng = np.random.default_rng(seed)
    for b in range(bootstrap):
        h = high[rng.integers(0, len(high), size=len(high))]
        l = low[rng.integers(0, len(low), size=len(low))]
        estimate = _point(h, l)
        for key in fields:
            draws[key][b] = estimate[key]
    for key, values in draws.items():
        finite = values[np.isfinite(values)]
        lower, upper = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
        result[f"{key}_bootstrap_finite"] = len(finite)
        result[f"{key}_bootstrap_se"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan
        result[f"{key}_raw_ci_low"] = float(lower)
        result[f"{key}_raw_ci_high"] = float(upper)
    lower = result["delta_xx_raw_ci_low"]
    weak = point["delta_xx"] <= 0 or not np.isfinite(point["omega1_coefficient"])
    weak = weak or (bootstrap > 0 and (not np.isfinite(lower) or lower <= 0))
    result["weak_reference_variance_shift"] = bool(weak)
    result["delta_xx_bootstrap_pct_nonpositive"] = float(100 * np.mean(draws["delta_xx"] <= 0)) if bootstrap else np.nan
    low_b, high_b = result["delta_xy_raw_ci_low"], result["delta_xy_raw_ci_high"]
    weak_outcome = not np.isfinite(point["omega2_coefficient"]) or (bootstrap > 0 and low_b <= 0 <= high_b)
    result["weak_outcome_denominator"] = bool(weak_outcome)
    for method in METHODS:
        prefix = f"{method}_coefficient"
        enough = result[f"{prefix}_bootstrap_finite"] >= max(20, .9 * bootstrap)
        suppress = bool(bootstrap < 20 or weak or not enough or (method == "omega2" and weak_outcome))
        result[f"{method}_interval_suppressed"] = suppress
        for suffix in ("low", "high"):
            result[f"{method}_ci_{suffix}"] = np.nan if suppress else result[f"{prefix}_raw_ci_{suffix}"]
    result["diagnostic_interpretation"] = "Variance-shift and rank diagnostics do not establish the identifying assumptions."
    return result


def event_table(events: pd.DataFrame | None, high_dates: Sequence) -> pd.DataFrame:
    """Table 1 chronology; never infer a risk direction or rewrite an event.

    Supported input aliases are date/market_date, event/description and
    war_risk/risk_direction. Source and timing columns are retained for audit.
    Multiple events on one H date remain separate rows. Missing descriptions or
    directions remain missing, with a flag, rather than invented text or signs.
    """
    high_dates = pd.Index(high_dates).unique()
    if events is None:
        result = pd.DataFrame({"date": high_dates, "event": pd.NA, "war_risk": pd.NA})
    else:
        result = events.copy()
        aliases = {"market_date": "date", "description": "event", "risk_direction": "war_risk"}
        result = result.rename(columns={k: v for k, v in aliases.items() if k in result and v not in result})
        if "date" not in result:
            raise ValueError("Event metadata requires a date or market_date column")
        if isinstance(high_dates, pd.DatetimeIndex):
            result["date"] = pd.to_datetime(result.date, errors="raise").dt.normalize()
        result = result.loc[result.date.isin(high_dates)].copy()
        for column in ("event", "war_risk"):
            if column not in result:
                result[column] = pd.NA
        absent = high_dates.difference(pd.Index(result.date))
        if len(absent):
            result = pd.concat([result, pd.DataFrame({"date": absent, "event": pd.NA, "war_risk": pd.NA})], ignore_index=True)
    result["event_metadata_missing"] = result.event.isna() | result.event.astype("string").str.strip().eq("")
    columns = ["date", "event", "war_risk"] + [c for c in result if c not in ("date", "event", "war_risk")]
    return result[columns].sort_values("date", kind="stable").reset_index(drop=True)


def replicate_tables(estimation_panel: pd.DataFrame, outcomes: Sequence[str], *,
                     full_panel: pd.DataFrame, events: pd.DataFrame | None = None,
                     reference: str = "two_year", regime: str = "high",
                     scenario: float = -.25, reference_unit: str = "percentage_point",
                     units: Mapping[str, str] | None = None,
                     labels: Mapping[str, str] | None = None,
                     unavailable: Mapping[str, str] | None = None,
                     bootstrap: int = 1999, seed: int = 2603) -> dict:
    """Return table1, table2, table3, diagnostics and metadata without file I/O.

    ``estimation_panel`` must be the caller's equal-sized H/L selection.
    ``full_panel`` must contain every aligned financial interval in the stated
    study window, including unselected L dates, and its own binary H indicator.
    Missing values are excluded explicitly; no prices or returns are imputed.
    Table 3 uses the full panel's available H count and actual sum of squared
    changes. All shares remain raw, unbounded diagnostics under assumptions.
    ``unavailable`` maps an explicitly unavailable outcome to its documented
    reason; that row remains in both financial tables with missing estimates.
    """
    outcomes = list(outcomes)
    unavailable = dict(unavailable or {})
    labels = dict(labels or {})
    if set(unavailable) - set(outcomes):
        raise ValueError("Every unavailable variable must appear in outcomes")
    if len(set(outcomes)) != len(outcomes) or reference in outcomes:
        raise ValueError("outcomes must be unique and must not include the reference")
    if not np.isfinite(scenario) or scenario == 0:
        raise ValueError("scenario must be a finite nonzero reference-market change")
    expected = {"percentage_point": -.25, "pp": -.25, "bp": -25., "basis_point": -25.}
    if reference_unit not in expected:
        raise ValueError("Declare the Treasury reference unit as percentage_point or bp")
    if not np.isclose(scenario, expected[reference_unit]):
        raise ValueError("Original Table 2 normalization requires a 25 bp decline in the declared unit")
    if not full_panel.index.is_unique:
        raise ValueError("Full panel index must identify unique market observations")
    selected_regime = _binary(estimation_panel[regime])
    full_regime = _binary(full_panel[regime])
    if not estimation_panel.index.isin(full_panel.index).all():
        raise ValueError("Selected estimation dates must belong to the full panel")
    if not np.array_equal(selected_regime.to_numpy(), full_regime.reindex(estimation_panel.index).to_numpy()):
        raise ValueError("H/L labels disagree between the selected and full panels")
    for column in [reference] + outcomes:
        if column in unavailable:
            continue
        left = pd.to_numeric(estimation_panel[column], errors="raise").to_numpy(float)
        right = pd.to_numeric(full_panel.loc[estimation_panel.index, column], errors="raise").to_numpy(float)
        if not np.allclose(left, right, equal_nan=True):
            raise ValueError(f"Selected and full-panel values disagree for {column}")
    units = dict(units or {})
    rows, variance, diagnostic = [], [], []
    # Reference row follows Table 3: moments displayed, attribution cells blank.
    reference_sample = estimation_panel.copy()
    alias = "__reference_copy_for_complete_case_selection__"
    if alias in reference_sample:
        raise ValueError(f"Reserved column name: {alias}")
    reference_sample[alias] = reference_sample[reference]
    ref_h, ref_l, _, _ = _sample(reference_sample, reference, alias, regime)
    variance.append({"variable": reference, "label": labels.get(reference, reference),
        "available": True, "unit_squared": f"({reference_unit})^2",
        "variance_low": float(np.mean(ref_l[:, 0] ** 2)), "variance_high": float(np.mean(ref_h[:, 0] ** 2)),
        "predicted_variance_change": np.nan, "high_variance_share_pct": np.nan,
        "all_variance_share_pct": np.nan, "n_high": len(ref_h), "n_low": len(ref_l)})
    for j, outcome in enumerate(outcomes):
        if outcome in unavailable:
            rows.append({"variable": outcome, "label": labels.get(outcome, outcome),
                "unit": units.get(outcome, "input unit"), "available": False,
                "availability_reason": unavailable[outcome], "n_high": 0, "n_low": 0,
                **{f"{method}_{suffix}": np.nan for method in METHODS for suffix in
                   ("effect", "effect_se_classical", "effect_se_hc1", "abs_t_classical", "abs_t_hc1",
                    "effect_bootstrap_se", "effect_ci_low", "effect_ci_high")}})
            variance.append({"variable": outcome, "label": labels.get(outcome, outcome),
                "unit_squared": f"({units.get(outcome, 'input unit')})^2", "available": False,
                "availability_reason": unavailable[outcome], "variance_low": np.nan,
                "variance_high": np.nan, "predicted_variance_change": np.nan,
                "high_variance_share_pct": np.nan, "all_variance_share_pct": np.nan})
            continue
        # Use common random numbers so identical anchor samples receive the
        # same variance-shift diagnostic across outcome rows.
        result = estimate_response(estimation_panel, reference, outcome, regime, bootstrap, seed)
        diagnostic.append(result)
        row = {"variable": outcome, "label": labels.get(outcome, outcome), "available": True,
               "unit": units.get(outcome, "input unit"),
               "n_high": result["n_high"], "n_low": result["n_low"], "scenario": scenario,
               "weak_reference_variance_shift": result["weak_reference_variance_shift"]}
        for method in METHODS:
            row[f"{method}_effect"] = scenario * result[f"{method}_coefficient"]
            for se in ("classical", "hc1"):
                row[f"{method}_effect_se_{se}"] = abs(scenario) * result[f"{method}_se_{se}"]
                row[f"{method}_abs_t_{se}"] = result[f"{method}_abs_t_{se}"]
            row[f"{method}_effect_bootstrap_se"] = abs(scenario) * result[f"{method}_coefficient_bootstrap_se"]
            bounds = sorted([scenario * result[f"{method}_ci_low"], scenario * result[f"{method}_ci_high"]])
            row[f"{method}_effect_ci_low"], row[f"{method}_effect_ci_high"] = bounds
        rows.append(row)
        values = full_panel[[reference, outcome]].apply(pd.to_numeric, errors="raise")
        complete = np.isfinite(values.to_numpy()).all(axis=1)
        y_all = values.loc[complete, outcome]
        n_h_full = int(full_regime.loc[complete].sum())
        all_second = float(np.mean(y_all ** 2))
        q = result["pooled_coefficient"] ** 2 * result["delta_xx"]
        high_share = 100 * _ratio(q, result["second_moment_y_high"], result["second_moment_y_high"])
        all_share = 100 * _ratio(n_h_full * q, float(np.sum(y_all ** 2)), float(np.sum(y_all ** 2)))
        variance.append({"variable": outcome, "label": labels.get(outcome, outcome), "available": True,
            "unit_squared": f"({units.get(outcome, 'input unit')})^2",
            "variance_low": result["second_moment_y_low"], "variance_high": result["second_moment_y_high"],
            "predicted_variance_change": q, "high_variance_share_pct": high_share,
            "all_variance_share_pct": all_share, "n_high": result["n_high"], "n_low": result["n_low"],
            "full_period_n": len(y_all), "full_period_n_high": n_h_full,
            "full_period_second_moment": all_second,
            "reference_second_moment_increase": result["delta_xx"],
            "weak_reference_variance_shift": result["weak_reference_variance_shift"],
            "share_outside_0_100": bool(q < 0 or high_share > 100 or all_share > 100),
            "conditional_diagnostic_only": True})
    return {"table1": event_table(events, estimation_panel.index[selected_regime]),
            "table2": pd.DataFrame(rows), "table3": pd.DataFrame(variance),
            "diagnostics": pd.DataFrame(diagnostic),
            "metadata": {"reference": reference, "reference_unit": reference_unit,
                "scenario": scenario, "centered": False, "intercept": False,
                "instruments": ["s*x", "s*y", "[s*x,s*y]"], "moment_divisor": "regime n",
                "bootstrap_design": "independent_within_regime",
                "bootstrap_repetitions": int(bootstrap),
                "common_draws_for_identical_outcome_samples": True,
                "se_conventions": ["classical no-intercept IV, residual df=n-1", "HC1 IV, n/(n-1) correction"],
                "variance_interpretation": "Conditional lower-bound calculation requires stable nuisance moments, one changing war-shock variance, stable loadings and serially independent daily changes.",
                "bootstrap_interpretation": "Supplementary diagnostics; not a source-paper bootstrap recipe or weak-IV-robust confidence set."}}
