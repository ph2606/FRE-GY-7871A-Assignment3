"""Identification checks with known shocks, including failure diagnostics."""
import numpy as np
import pandas as pd
import pytest

from src.heteroskedasticity import covariance_rank_one, estimate_pair


def known_system(repetitions=80, paired=True, shifts=False):
    # Balanced orthogonal shocks: nuisance creates substantial OLS confounding.
    signs = np.tile(np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]), (repetitions, 1))
    chunks = []
    for is_high, war_scale in ((True, 3), (False, 1)):
        war, nuisance = war_scale * signs[:, 0], signs[:, 1]
        x = war + nuisance
        y = -2 * war + 3 * nuisance
        z = .5 * war - nuisance
        if shifts and is_high:
            x, y, z = x + 8, y - 40, z + 3
        part = pd.DataFrame({"x": x, "y": y, "z": z, "high": is_high})
        if paired:
            part["pair_id"] = np.arange(len(signs))
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True)


def test_recovers_known_response_despite_ols_confounding():
    frame = known_system()
    result = estimate_pair(frame, "x", "y", bootstrap=0)
    assert result["delta_var_x"] == pytest.approx(8)
    assert result["delta_cov_xy"] == pytest.approx(-16)
    assert result["delta_var_y"] == pytest.approx(32)
    for method in ("d_reference", "d_outcome", "d_pooled"):
        assert result[method] == pytest.approx(-2)
    assert np.polyfit(frame.x, frame.y, 1)[0] == pytest.approx(-7 / 6)
    assert result["rank_one_residual"] < 1e-12


def test_centering_handles_different_regime_means():
    frame = known_system(shifts=True)
    centered = estimate_pair(frame, "x", "y", bootstrap=0)
    original_second_moments = estimate_pair(frame, "x", "y", bootstrap=0, center=False)
    assert centered["d_pooled"] == pytest.approx(-2)
    assert original_second_moments["d_reference"] != pytest.approx(-2)
    assert original_second_moments["delta_var_x"] == pytest.approx(72)


def test_unequal_groups_use_balanced_conditional_moment_weights():
    frame = known_system(paired=False)
    high = frame.loc[frame.high]
    uneven = pd.concat([frame, high, high], ignore_index=True)
    result = estimate_pair(uneven, "x", "y", bootstrap=0)
    assert result["n_high"] == 3 * result["n_low"]
    assert result["d_reference"] == pytest.approx(-2)
    assert result["d_pooled"] == pytest.approx(-2)


def test_bootstrap_preserves_pairs_and_retains_denominator_diagnostics():
    result = estimate_pair(known_system(120), "x", "y", bootstrap=199)
    assert result["bootstrap_design"] == "matched_pairs"
    assert result["delta_var_x_ci_low"] > 0
    assert result["delta_var_x_pct_nonpositive"] == 0
    assert not result["d_reference_inference_suppressed"]
    assert result["d_reference_ci_low"] < -2 < result["d_reference_ci_high"]
    assert 0 <= result["rank_one_residual_ci_low"] <= result["rank_one_residual_ci_high"]


def test_negative_variance_shift_keeps_raw_estimate_but_suppresses_inference():
    frame = known_system()
    frame["high"] = ~frame.high
    result = estimate_pair(frame, "x", "y", bootstrap=99)
    assert result["delta_var_x"] < 0
    assert result["weak_identification"]
    assert result["delta_var_x_pct_nonpositive"] == 100
    assert result["d_reference"] == pytest.approx(-2)
    assert np.isfinite(result["d_reference_raw_ci_low"])
    assert np.isnan(result["d_reference_ci_low"])
    assert result["rank_one_residual"] == pytest.approx(1)


def test_zero_shift_is_unidentified():
    frame = known_system()
    high = frame.loc[frame.high, ["x", "y", "z"]].to_numpy()
    frame.loc[~frame.high, ["x", "y", "z"]] = high
    result = estimate_pair(frame, "x", "y", bootstrap=49)
    assert result["delta_var_x"] == 0
    assert np.isnan(result["d_reference"])
    assert np.isnan(result["d_pooled"])
    assert result["weak_identification"]


def test_pairwise_missing_values_remove_both_observations():
    frame = known_system()
    frame.loc[0, "y"] = np.nan
    result = estimate_pair(frame, "x", "y", bootstrap=0)
    assert result["n_rows_dropped"] == 2
    assert result["n_pairs_dropped"] == 1
    assert result["n_high"] == result["n_low"] == 319
    with pytest.raises(ValueError, match="exactly one H"):
        estimate_pair(frame.iloc[1:], "x", "y", bootstrap=0)


def test_unit_changes_preserve_standardized_rank_diagnostics():
    frame = known_system()
    expected = covariance_rank_one(frame, ["x", "y", "z"])
    frame["x"] *= 100
    frame["y"] *= .01
    actual = covariance_rank_one(frame, ["x", "y", "z"])
    np.testing.assert_allclose(actual["eigenvalues"], expected["eigenvalues"], atol=1e-12)
    result = estimate_pair(frame, "x", "y", bootstrap=0)
    assert result["d_pooled"] == pytest.approx(-.0002)


def test_second_heteroskedastic_factor_breaks_rank_one_restriction():
    frame = known_system()
    rows = frame.loc[frame.high]
    # Change nuisance variance as well as war variance: contrary to assumptions.
    signs = np.tile([-1, 1, -1, 1], len(rows) // 4)
    frame.loc[frame.high, "y"] += 10 * signs
    ranks = covariance_rank_one(frame, ["x", "y", "z"])
    assert ranks["rank_one_residual"] > .05
    result = estimate_pair(frame, "x", "y", bootstrap=0)
    assert abs(result["d_reference"] - result["d_outcome"]) > .1


def test_string_regimes_are_not_cast_by_truthiness():
    frame = known_system()
    frame["high"] = np.where(frame.high, "H", "L")
    assert estimate_pair(frame, "x", "y", bootstrap=0)["d_pooled"] == pytest.approx(-2)
    frame.loc[0, "high"] = "unspecified"
    with pytest.raises(ValueError, match="Unknown regime"):
        estimate_pair(frame, "x", "y", bootstrap=0)
