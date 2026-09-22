"""Original-moment, table-denominator and unit checks with known shocks."""
import numpy as np
import pandas as pd
import pytest

from src.paper_replication import estimate_response, event_table, replicate_tables


def system(repetitions=20):
    signs = np.tile([[-1, -1], [-1, 1], [1, -1], [1, 1]], (repetitions, 1))
    rows = []
    for high, scale in [(True, 3), (False, 1)]:
        war, nuisance = scale * signs[:, 0], signs[:, 1]
        rows.append(pd.DataFrame({"x": war + nuisance, "y": -2 * war + 3 * nuisance,
                                 "high": high, "pair_id": np.arange(len(signs))}))
    result = pd.concat(rows, ignore_index=True).astype({'x':float,'y':float})
    result.index = pd.bdate_range("2026-01-01", periods=len(result))
    return result


def tables(panel, full=None, **kwargs):
    return replicate_tables(panel, ["y"], full_panel=panel if full is None else full,
                            reference="x", bootstrap=0, units={"y": "percent"}, **kwargs)


def test_original_three_estimators_recover_response_despite_ols_confounding():
    panel = system()
    result = estimate_response(panel, "x", "y", bootstrap=0)
    assert result["delta_xx"] == pytest.approx(8)
    assert result["delta_xy"] == pytest.approx(-16)
    assert result["delta_yy"] == pytest.approx(32)
    for method in ("omega1", "omega2", "pooled"):
        assert result[f"{method}_coefficient"] == pytest.approx(-2)
    assert np.dot(panel.x, panel.y) / np.dot(panel.x, panel.x) != pytest.approx(-2)
    assert result["positive_rank_one_residual"] < 1e-12


def test_original_cross_products_do_not_subtract_regime_means():
    panel = system()
    panel.loc[panel.high, "x"] += 8
    panel.loc[panel.high, "y"] -= 40
    result = estimate_response(panel, "x", "y", bootstrap=0)
    assert result["delta_xx"] == pytest.approx(72)
    assert result["delta_xy"] == pytest.approx(-336)
    assert result["omega1_coefficient"] == pytest.approx(-336 / 72)
    assert not result["centered"]
    assert not result["intercept"]


def test_no_intercept_iv_matches_direct_projection_and_explicit_se_conventions():
    panel = system()
    panel.loc[panel.high, "y"] += .7  # Avoid relying only on a zero-mean exact model.
    result = estimate_response(panel, "x", "y", bootstrap=0)
    x, y = panel[["x", "y"]].to_numpy().T
    sign = np.where(panel.high, 1, -1)
    z = np.column_stack((sign * x, sign * y))
    projected = z @ np.linalg.solve(z.T @ z, z.T @ x)
    coefficient = projected @ y / (projected @ x)
    residual = y - coefficient * x
    expected_classical = np.sqrt(residual @ residual / (len(x)-1) / (projected @ x))
    expected_hc1 = np.sqrt(len(x)/(len(x)-1) * np.sum((projected*residual)**2) / (projected @ x)**2)
    assert result["pooled_coefficient"] == pytest.approx(coefficient)
    assert result["pooled_se_classical"] == pytest.approx(expected_classical)
    assert result["pooled_se_hc1"] == pytest.approx(expected_hc1)


def test_full_period_variance_uses_unselected_days_and_actual_high_count():
    panel = system()
    extra = pd.DataFrame({"x": [1., 2.], "y": [50., -70.], "high": [False, False]},
                         index=pd.bdate_range(panel.index[-1] + pd.Timedelta(days=1), periods=2))
    full = pd.concat([panel, extra])
    result = tables(panel, full)["table3"].set_index("variable").loc["y"]
    q = 32
    assert result.predicted_variance_change == pytest.approx(q)
    assert result.variance_low == pytest.approx(13)
    assert result.variance_high == pytest.approx(45)
    assert result.high_variance_share_pct == pytest.approx(100*q/45)
    assert result.all_variance_share_pct == pytest.approx(100 * panel.high.sum() * q / np.sum(full.y**2))
    assert result.full_period_n == len(full)
    assert result.all_variance_share_pct != pytest.approx(100 * panel.high.sum() * q / np.sum(panel.y**2))


def test_unit_conversion_preserves_25bp_effect_and_variance_share():
    panel = system()
    original = tables(panel)
    bp = panel.copy()
    bp["x"] *= 100
    converted = tables(bp, reference_unit="bp", scenario=-25)
    for method in ("omega1", "omega2", "pooled"):
        assert original["table2"].iloc[0][f"{method}_effect"] == pytest.approx(converted["table2"].iloc[0][f"{method}_effect"])
    assert original["table3"].iloc[1].all_variance_share_pct == pytest.approx(converted["table3"].iloc[1].all_variance_share_pct)
    with pytest.raises(ValueError, match="25 bp"):
        tables(bp, reference_unit="bp")


def test_missing_member_removes_complete_pair_but_does_not_rebalance_without_ids():
    panel = system()
    panel.iloc[0, panel.columns.get_loc("y")] = np.nan
    result = estimate_response(panel, "x", "y", bootstrap=0)
    assert result["n_high"] == result["n_low"] == 79
    assert result["n_rows_dropped"] == 2
    with pytest.raises(ValueError, match="equal H/L"):
        estimate_response(panel.drop(columns="pair_id"), "x", "y", bootstrap=0)


def test_bootstrap_is_stratified_reproducible_and_not_matched_pair_resampling():
    panel = system()
    paired_metadata = estimate_response(panel, "x", "y", bootstrap=49, seed=42)
    no_pairs = estimate_response(panel.drop(columns="pair_id"), "x", "y", bootstrap=49, seed=42)
    assert paired_metadata["bootstrap_design"] == "independent_within_regime"
    assert paired_metadata["pooled_coefficient_bootstrap_se"] == pytest.approx(no_pairs["pooled_coefficient_bootstrap_se"])
    assert paired_metadata["delta_xx_raw_ci_low"] == no_pairs["delta_xx_raw_ci_low"]


def test_unsupported_variance_order_keeps_raw_results_and_suppresses_intervals():
    panel = system()
    panel["high"] = ~panel.high
    result = estimate_response(panel, "x", "y", bootstrap=49)
    assert result["delta_xx"] < 0
    assert result["weak_reference_variance_shift"]
    assert result["pooled_coefficient"] == pytest.approx(-2)
    assert np.isfinite(result["pooled_coefficient_raw_ci_low"])
    assert np.isnan(result["pooled_ci_low"])
    assert result["positive_rank_one_residual"] == pytest.approx(1)
    variance = tables(panel)["table3"].set_index("variable").loc["y"]
    assert variance.predicted_variance_change < 0  # Do not clip an invalid bound.
    assert variance.share_outside_0_100


def test_table_one_keeps_multiple_events_without_inventing_risk_directions():
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    events = pd.DataFrame({"market_date": [dates[0], dates[0]],
                           "description": ["Shipping stopped", "Oil storage damaged"],
                           "source_url": ["source-a", "source-b"]})
    result = event_table(events, dates)
    assert len(result) == 3
    assert result.war_risk.isna().all()
    assert result.event_metadata_missing.sum() == 1
    assert result.source_url.dropna().tolist() == ["source-a", "source-b"]


def test_unavailable_original_variable_stays_in_table_as_missing_with_reason():
    panel = system()
    result = replicate_tables(panel, ["y", "liquidity"], full_panel=panel, reference="x", bootstrap=0,
        labels={"liquidity": "Ten-year on-the-run liquidity premium"},
        units={"liquidity": "percentage_point"}, unavailable={"liquidity": "Public series not available"})
    row = result["table2"].set_index("variable").loc["liquidity"]
    assert row.label == "Ten-year on-the-run liquidity premium"
    assert not row.available
    assert pd.isna(row.pooled_effect)
    assert row.availability_reason == "Public series not available"
    assert result["table3"].set_index("variable").loc["liquidity", "variance_high"] != 0


def test_full_panel_disagreement_is_rejected_instead_of_using_inconsistent_units():
    panel = system()
    full = panel.copy()
    full["y"] *= 100
    with pytest.raises(ValueError, match="values disagree"):
        tables(panel, full)


def test_identical_anchor_samples_share_variance_shift_bootstrap_draws():
    panel=system()
    panel['other']=panel.y*2+panel.x*.3
    result=replicate_tables(panel,['y','other'],full_panel=panel,reference='x',bootstrap=99)
    diagnostics=result['diagnostics']
    for name in ['delta_xx_raw_ci_low','delta_xx_raw_ci_high','delta_xx_bootstrap_pct_nonpositive']:
        assert diagnostics[name].nunique()==1
