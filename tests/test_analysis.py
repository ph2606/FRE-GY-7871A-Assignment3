"""Cross-module checks for information intervals, matching, and units."""
import numpy as np
import pandas as pd
import pytest

from src import analysis
from src.heteroskedasticity import estimate_pair


def test_publication_clock_ignores_inserted_non_equity_calendar_rows(monkeypatch):
    """A weekend row in another market must not erase Monday's news change."""
    rng = np.random.default_rng(17)
    dates = pd.bdate_range("2026-01-02", periods=85)
    previous = pd.Series(dates, index=dates).shift(1)
    frame = pd.DataFrame({
        "sp500": rng.normal(size=len(dates)),
        "asset": rng.normal(size=len(dates)),
        "war_articles_publication": rng.integers(1, 20, len(dates)),
        "negative_pct_publication": rng.uniform(0, 10, len(dates)),
        "sp500_prior_date": previous,
        "asset_prior_date": previous,
        "interval_days": (pd.Series(dates, index=dates) - previous).dt.days.fillna(1),
    }, index=dates)
    monkeypatch.setattr(analysis, "OUTCOMES", ["asset"])
    baseline, baseline_samples = analysis.regressions(frame, pd.DataFrame(), version="publication_clock")
    calendar = pd.date_range(dates.min(), dates.max(), freq="D")
    augmented = frame.reindex(calendar)
    actual, actual_samples = analysis.regressions(augmented, pd.DataFrame(), version="publication_clock")
    assert actual_samples.date.tolist() == baseline_samples.date.tolist()
    np.testing.assert_allclose(actual.coefficient, baseline.coefficient, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.p_value, baseline.p_value, rtol=1e-12, atol=1e-12)


def test_event_matching_is_independent_of_return_magnitudes(monkeypatch):
    dates = pd.bdate_range("2026-01-01", "2026-04-15")
    previous = pd.Series(dates, index=dates).shift(1)
    frame = pd.DataFrame({"sp500": 1., "two_year": 1., "war_attention": 1.,
                          "sp500_prior_date": previous, "two_year_prior_date": previous}, index=dates)
    monkeypatch.setattr(analysis, "EVENTS", [
        {"market_date": "2026-01-14", "timing_uncertain": False},
        {"market_date": "2026-02-17", "timing_uncertain": False},
        {"market_date": "2026-03-02", "timing_uncertain": False},
    ])
    expected, expected_audit = analysis.make_pairs(frame)
    rng = np.random.default_rng(51)
    changed = frame.copy()
    changed["sp500"] = rng.normal(size=len(frame)) * 100
    changed["two_year"] = rng.normal(size=len(frame)) * 1000
    actual, actual_audit = analysis.make_pairs(changed)
    pd.testing.assert_frame_equal(actual[["date", "high", "pair_id"]], expected[["date", "high", "pair_id"]])
    pd.testing.assert_frame_equal(actual_audit, expected_audit)
    assert actual.loc[~actual.high, "date"].is_unique
    assert (actual_audit.gap_calendar_days <= 35).all()
    assert ((actual_audit.high_date < "2026-02-28") == (actual_audit.low_date < "2026-02-28")).all()


def test_regression_excludes_a_current_day_whose_lag_has_a_different_interval(monkeypatch):
    rng = np.random.default_rng(71)
    dates = pd.bdate_range("2026-01-02", periods=85)
    previous = pd.Series(dates, index=dates).shift(1)
    frame = pd.DataFrame({
        "sp500": rng.normal(size=len(dates)),
        "asset": rng.normal(size=len(dates)),
        "d_log_attention": rng.normal(size=len(dates)),
        "d_negative_pct": rng.normal(size=len(dates)),
        "sp500_prior_date": previous,
        "asset_prior_date": previous.copy(),
        "interval_days": (pd.Series(dates, index=dates) - previous).dt.days.fillna(1),
    }, index=dates)
    # Day 40 has a longer outcome return interval. Day 41 is itself aligned,
    # but cannot use day 40's return as a consistently aligned one-session lag.
    frame.loc[dates[40], "asset_prior_date"] -= pd.Timedelta(days=1)
    monkeypatch.setattr(analysis, "OUTCOMES", ["asset"])
    _, samples = analysis.regressions(frame, pd.DataFrame())
    used = set(samples.date)
    assert str(dates[40].date()) not in used
    assert str(dates[41].date()) not in used
    assert str(dates[39].date()) in used
    assert str(dates[42].date()) in used


def _financial_fixture():
    signs = np.tile(np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]), (2, 1))
    rows = []
    dates = pd.bdate_range("2026-01-05", periods=26)
    for high, scale in [(True, 3), (False, 1)]:
        for pid, (war, nuisance) in enumerate(signs):
            i = len(rows)
            prior = dates[i] - pd.offsets.BDay()
            rows.append({"date": dates[i], "two_year": scale * war + nuisance,
                         "sp500": -2 * scale * war + 3 * nuisance,
                         "high": high, "pair_id": pid,
                         "two_year_prior_date": prior, "sp500_prior_date": prior})
    pairs = pd.DataFrame(rows)
    panel = pairs.drop(columns=["high", "pair_id"]).set_index("date")
    # Additional nonmatched days must enter the full-period denominator.
    extra_dates = dates[len(pairs):]
    extra = pd.DataFrame({"two_year": 2., "sp500": np.arange(len(extra_dates)) + 20.,
                          "two_year_prior_date": extra_dates - pd.offsets.BDay(),
                          "sp500_prior_date": extra_dates - pd.offsets.BDay()}, index=extra_dates)
    return pairs, pd.concat([panel, extra])


def test_scenario_uses_basis_points_and_variance_denominator_uses_all_dates(monkeypatch):
    pairs, panel = _financial_fixture()
    monkeypatch.setattr(analysis, "OUTCOMES", ["two_year", "sp500"])
    result = analysis.estimate_set(panel, pairs, bootstrap=0).iloc[0]
    assert result.d_pooled == pytest.approx(-2)
    assert result.d_pooled_scenario == pytest.approx(50)  # -25 bp * -2 outcome units/bp
    assert result.incremental_variance == pytest.approx(32)
    expected = 100 * 8 * 32 / (len(panel) * panel.sp500.var(ddof=0))
    assert result.all_variance_share == pytest.approx(expected)
    assert result.all_sample_n == len(panel)


def test_model_sample_drops_whole_pair_when_market_intervals_differ():
    pairs, _ = _financial_fixture()
    pairs.loc[0, "sp500_prior_date"] -= pd.Timedelta(days=1)
    selected = analysis.model_sample(pairs, "sp500", "two_year")
    result = estimate_pair(selected, "two_year", "sp500", bootstrap=0)
    assert result["n_pairs"] == 7
    assert result["n_pairs_dropped"] == 1
    assert result["n_rows_dropped"] == 2
