"""Rigobon--Sack covariance-difference estimators for financial changes.

Regimes and matching must be supplied by the caller. No event selection, sign
orientation, or economic shock normalization occurs here. Moment denominators
are n (not n-1). By default observations are demeaned within each regime; set
center=False for the original paper's zero-mean/second-moment convention.

Intervals are percentile bootstrap diagnostics. The rank-one residual interval
is not a formal rank test. The matched-pair bootstrap preserves H/L matches but
does not model dependence between different pairs. Weak-normalization results
retain raw estimates and raw intervals, while reported inference is suppressed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _regime_indicator(values: pd.Series) -> pd.Series:
    """Read a two-regime indicator without treating nonempty 'low' as True."""
    if values.isna().any():
        raise ValueError("Regime labels contain missing values")
    mapping = {
        True: True, False: False, 1: True, 0: False,
        "H": True, "L": False, "high": True, "low": False,
        "High": True, "Low": False, "true": True, "false": False,
    }
    invalid = [v for v in values.unique() if v not in mapping]
    if invalid:
        raise ValueError(f"Unknown regime labels: {invalid}")
    return values.map(mapping).astype(bool)


def _prepare(frame, columns, regime):
    if len(set(columns)) != len(columns):
        raise ValueError("Financial variable columns must be distinct")
    missing = set(columns + [regime]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if regime in columns:
        raise ValueError("Regime column must differ from financial columns")
    indicator = _regime_indicator(frame[regime])
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="raise")
    finite = pd.Series(np.isfinite(numeric.to_numpy()).all(axis=1), index=frame.index)
    dropped_pairs = 0
    paired = "pair_id" in frame.columns
    if paired:
        if frame.pair_id.isna().any():
            raise ValueError("pair_id contains missing values")
        metadata = pd.DataFrame({"pair_id": frame.pair_id.to_numpy(),
                                 "high": indicator.to_numpy(),
                                 "finite": finite.to_numpy()})
        counts = metadata.groupby("pair_id", sort=False).agg(
            size=("high", "size"), highs=("high", "sum"), finite=("finite", "all"))
        if not ((counts["size"] == 2) & (counts.highs == 1)).all():
            raise ValueError("Each pair_id must have exactly one H row and one L row")
        keep_ids = counts.index[counts.finite]
        dropped_pairs = int((~counts.finite).sum())
        finite = frame.pair_id.isin(keep_ids)
    selected = numeric.loc[finite].copy()
    selected["_high"] = indicator.loc[finite].to_numpy()
    if paired:
        selected["_pair_id"] = frame.loc[finite, "pair_id"].to_numpy()
        # Preserve the caller's chronological pair ordering, even when group
        # rows are interleaved differently. Both arrays use the same ID order.
        order = pd.unique(selected._pair_id)
        high = selected.loc[selected._high].set_index("_pair_id").loc[order, columns].to_numpy(float)
        low = selected.loc[~selected._high].set_index("_pair_id").loc[order, columns].to_numpy(float)
    else:
        high = selected.loc[selected._high, columns].to_numpy(float)
        low = selected.loc[~selected._high, columns].to_numpy(float)
    if min(len(high), len(low)) < 3:
        raise ValueError("At least three complete observations per regime are required")
    return high, low, {
        "n_high": len(high), "n_low": len(low),
        "n_pairs": len(high) if paired else 0,
        "n_rows_dropped": int(len(frame) - len(selected)),
        "n_pairs_dropped": dropped_pairs,
        "bootstrap_design": "matched_pairs" if paired else "independent_within_regime",
    }


def _moments(values, center):
    residuals = values - values.mean(axis=0) if center else values
    return residuals, residuals.T @ residuals / len(residuals)


def _divide(numerator, denominator, scale):
    tolerance = np.finfo(float).eps * 100 * max(float(abs(scale)), np.finfo(float).tiny)
    return float(numerator / denominator) if abs(denominator) > tolerance else float("nan")


def _rank_diagnostics(high_cov, low_cov):
    """Unit-invariant comparison with the nearest positive rank-one matrix."""
    pooled_diagonal = np.diag((high_cov + low_cov) / 2)
    scale = np.sqrt(np.maximum(pooled_diagonal, 0))
    if np.any(scale == 0):
        return {
            "rank_one_residual": float("nan"),
            "eigenvalues": np.full(len(scale), np.nan),
            "standardized_contrast": np.full_like(high_cov, np.nan),
            "standardization_scale": scale,
        }
    contrast = (high_cov - low_cov) / np.outer(scale, scale)
    eigenvalues, eigenvectors = np.linalg.eigh(contrast)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    fit = max(float(eigenvalues[0]), 0.0) * np.outer(eigenvectors[:, 0], eigenvectors[:, 0])
    norm = np.linalg.norm(contrast, ord="fro")
    residual = np.linalg.norm(contrast - fit, ord="fro") / norm if norm > 1e-14 else np.nan
    return {
        "rank_one_residual": float(residual),
        "eigenvalues": eigenvalues,
        "standardized_contrast": contrast,
        "standardization_scale": scale,
    }


def _point_estimates(high, low, center):
    hc, oh = _moments(high, center)
    lc, ol = _moments(low, center)
    change = oh - ol
    a, b, c = float(change[0, 0]), float(change[0, 1]), float(change[1, 1])
    d_reference = _divide(b, a, oh[0, 0] + ol[0, 0])
    d_outcome = _divide(c, b, np.sqrt((oh[0, 0] + ol[0, 0]) * (oh[1, 1] + ol[1, 1])))

    observations = np.vstack((hc, lc))
    # The n/n_regime weights give each conditional covariance equal weight,
    # including when no matching is supplied and group sizes differ.
    weights = np.r_[np.full(len(hc), len(observations) / len(hc)),
                    np.full(len(lc), -len(observations) / len(lc))]
    instruments = observations * weights[:, None]
    rms = np.sqrt((instruments ** 2).mean(axis=0))
    instruments = instruments[:, rms > 0]
    if instruments.shape[1]:
        instruments = instruments / rms[rms > 0]
        zz = instruments.T @ instruments / len(observations)
        zx = instruments.T @ observations[:, 0] / len(observations)
        zy = instruments.T @ observations[:, 1] / len(observations)
        inverse = np.linalg.pinv(zz, rcond=1e-12)
        denominator = float(zx @ inverse @ zx)
        numerator = float(zx @ inverse @ zy)
        d_pooled = _divide(numerator, denominator, np.mean(observations[:, 0] ** 2))
        instrument_rank = int(np.linalg.matrix_rank(zz, tol=1e-12))
        first_stage_r2 = _divide(denominator, np.mean(observations[:, 0] ** 2),
                                 np.mean(observations[:, 0] ** 2))
    else:
        denominator, d_pooled, first_stage_r2, instrument_rank = np.nan, np.nan, np.nan, 0
    ranks = _rank_diagnostics(oh, ol)
    variance_ratio = float(oh[0, 0] / ol[0, 0]) if ol[0, 0] > 0 else (float("inf") if oh[0, 0] > 0 else np.nan)
    return {
        "d_reference": d_reference, "d_outcome": d_outcome, "d_pooled": d_pooled,
        "delta_var_x": a, "delta_cov_xy": b, "delta_var_y": c,
        "var_x_high": float(oh[0, 0]), "var_x_low": float(ol[0, 0]),
        "var_y_high": float(oh[1, 1]), "var_y_low": float(ol[1, 1]),
        "cov_xy_high": float(oh[0, 1]), "cov_xy_low": float(ol[0, 1]),
        "var_ratio": variance_ratio, "pooled_denominator": denominator,
        "instrument_rank": instrument_rank, "first_stage_r2": first_stage_r2,
        "rank_one_residual": ranks["rank_one_residual"],
        "rank_one_leading_eigenvalue": float(ranks["eigenvalues"][0]),
        "rank_one_second_eigenvalue": float(ranks["eigenvalues"][1]),
    }


def _percentiles(values):
    finite = np.asarray(values)[np.isfinite(values)]
    if len(finite) == 0:
        return np.nan, np.nan
    return tuple(float(v) for v in np.quantile(finite, [0.025, 0.975]))


def estimate_pair(frame, x, y, regime="high", bootstrap=999, seed=2603, center=True):
    """Estimate y's response relative to financial anchor x.

    ``regime`` names a column containing bools, 0/1, or H/L (high/low).
    With ``pair_id``, each pair must contain one observation in each regime;
    missing x/y removes the whole pair. Otherwise available rows are used with
    balanced conditional-moment instrument weights and stratified resampling.

    Flat output includes d estimates, moments, bootstrap denominator intervals,
    sign frequencies, raw ratio intervals, and conservative reported intervals.
    ``weak_identification`` is True when the reference variance increase is
    nonpositive or its bootstrap interval includes zero. This diagnostic checks
    the required positive variance shift; it is not a formal weak-IV test.
    No p-values or causal interpretation are mechanically assigned.
    """
    if not isinstance(bootstrap, (int, np.integer)) or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer")
    high, low, metadata = _prepare(frame, [x, y], regime)
    point = _point_estimates(high, low, center)
    result = {"x": x, "y": y, "centered": bool(center), "moment_ddof": 0,
              "bootstrap_repetitions": int(bootstrap), "seed": int(seed), **metadata, **point}
    fields = ("d_reference", "d_outcome", "d_pooled", "delta_var_x", "delta_cov_xy",
              "delta_var_y", "pooled_denominator", "rank_one_residual")
    draws = {field: np.empty(bootstrap, dtype=float) for field in fields}
    rng = np.random.default_rng(seed)
    paired = metadata["bootstrap_design"] == "matched_pairs"
    for iteration in range(bootstrap):
        hi = rng.integers(0, len(high), size=len(high))
        li = hi if paired else rng.integers(0, len(low), size=len(low))
        sampled = _point_estimates(high[hi], low[li], center)
        for field in fields:
            draws[field][iteration] = sampled[field]
    for field in fields:
        lower, upper = _percentiles(draws[field])
        result[f"{field}_raw_ci_low"] = lower
        result[f"{field}_raw_ci_high"] = upper
        result[f"{field}_bootstrap_finite"] = int(np.isfinite(draws[field]).sum())
        if not field.startswith("d_"):
            result[f"{field}_ci_low"] = lower
            result[f"{field}_ci_high"] = upper
    result["delta_var_x_pct_nonpositive"] = float(100 * np.mean(draws["delta_var_x"] <= 0)) if bootstrap else np.nan
    result["delta_var_x_probability_nonpositive"] = result["delta_var_x_pct_nonpositive"] / 100
    result["delta_cov_xy_pct_nonpositive"] = float(100 * np.mean(draws["delta_cov_xy"] <= 0)) if bootstrap else np.nan
    point_bad = not np.isfinite(point["d_reference"]) or point["delta_var_x"] <= 0
    reference_weak = point_bad or (bootstrap > 0 and result["delta_var_x_ci_low"] <= 0)
    cov_low, cov_high = result["delta_cov_xy_ci_low"], result["delta_cov_xy_ci_high"]
    outcome_weak = not np.isfinite(point["d_outcome"]) or (bootstrap > 0 and cov_low <= 0 <= cov_high)
    result["weak_identification"] = bool(reference_weak)
    result["weak_reference_denominator"] = bool(reference_weak)
    result["weak_outcome_denominator"] = bool(outcome_weak)
    result["inference_available"] = bool(bootstrap >= 20 and not reference_weak)
    for field in ("d_reference", "d_outcome", "d_pooled"):
        enough = result[f"{field}_bootstrap_finite"] >= max(20, .9 * bootstrap)
        weak = reference_weak or (field == "d_outcome" and outcome_weak)
        suppressed = bootstrap < 20 or weak or not enough
        result[f"{field}_inference_suppressed"] = bool(suppressed)
        result[f"{field}_ci_low"] = np.nan if suppressed else result[f"{field}_raw_ci_low"]
        result[f"{field}_ci_high"] = np.nan if suppressed else result[f"{field}_raw_ci_high"]
    result["inference_status"] = (
        "bootstrap_not_run" if bootstrap == 0 else
        "too_few_bootstrap_repetitions" if bootstrap < 20 else
        "weak_or_nonpositive_reference_variance_shift" if reference_weak else
        "percentile_bootstrap_conditional_on_regime_design"
    )
    return result


def covariance_rank_one(frame, columns, regime="high", center=True):
    """Describe a standardized multivariate covariance contrast; no rank test.

    Every variable uses the same complete rows (and complete pairs when given).
    Scaling is sqrt((diag(Omega_H) + diag(Omega_L))/2). Negative eigenvalues are
    retained. The approximation is the nearest positive rank-one matrix, as
    required by the one-factor positive-variance-shift population restriction.
    """
    columns = list(columns)
    if len(columns) < 2:
        raise ValueError("At least two financial variables are required")
    high, low, metadata = _prepare(frame, columns, regime)
    _, hc = _moments(high, center)
    _, lc = _moments(low, center)
    ranks = _rank_diagnostics(hc, lc)
    return {
        "columns": columns, "centered": bool(center), "moment_ddof": 0,
        **metadata,
        "eigenvalues": ranks["eigenvalues"].tolist(),
        "rank_one_residual": ranks["rank_one_residual"],
        "standardization_scale": ranks["standardization_scale"].tolist(),
        "standardized_covariance_contrast": ranks["standardized_contrast"].tolist(),
        "covariance_contrast": (hc - lc).tolist(),
        "diagnostic_only": True,
    }
