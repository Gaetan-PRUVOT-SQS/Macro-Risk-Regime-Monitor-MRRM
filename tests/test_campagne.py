"""Campagne ISTQB / ISO 29119 sur le calcul. Les oracles viennent de la règle, pas du code vectorisé."""

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

import mrrm_core as mrrm


@pytest.fixture
def market_df():
    rng = np.random.default_rng(0)
    n = 600
    index = pd.bdate_range("2010-01-01", periods=n)
    spx = 1000 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    vix = np.clip(15 + rng.normal(0, 4, n).cumsum() * 0.05 + 5, 9, 80)
    return pd.DataFrame(
        {
            "SPX": spx,
            "VIX": vix,
            "UST10Y": 45 + rng.normal(0, 1, n).cumsum() * 0.1,
            "UST3M": 40 + rng.normal(0, 1, n).cumsum() * 0.1,
            "HYG": 90 * np.exp(np.cumsum(rng.normal(0, 0.004, n))),
            "IEF": 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n))),
        },
        index=index,
    )


def _frame(columns, periods=12):
    index = pd.bdate_range("2020-01-01", periods=periods)
    data = {name: values for name, values in columns.items()}
    return pd.DataFrame(data, index=index)


def test_zscore_matches_previous_window():
    """Le jour n'entre pas dans sa moyenne ni dans son écart-type."""
    values = np.arange(1, 15, dtype=float)
    values[4:9] = [1, 2, 3, 4, 5]
    values[9:14] = [1, 2, 3, 4, 100]
    frame = _frame({"VIX": values}, periods=14)
    _smoothed, _raw, z_scores, _loadings, _explained, _mean = mrrm.compute_stress_index(
        frame, 4, 3.0, 1, 3,
    )
    mild = (5.0 - 2.5) / np.sqrt(5.0 / 3.0)
    assert np.isclose(z_scores["VIX"].iloc[z_scores.index.get_loc(frame.index[8])], mild)
    assert np.isclose(mild, 1.9364916731037085)
    assert z_scores["VIX"].loc[frame.index[13]] == 3.0

    for stamp in z_scores.index:
        pos = frame.index.get_loc(stamp)
        window = values[pos - 4:pos]
        sigma = window.std(ddof=1)
        expected = float(np.clip((values[pos] - window.mean()) / sigma, -3.0, 3.0))
        assert np.isclose(z_scores["VIX"].loc[stamp], expected)


def test_zscore_ignores_shift_and_scale(market_df):
    indicators = mrrm.build_stress_indicators(market_df, 21)
    shifted = indicators.copy()
    shifted["Credit"] = indicators["Credit"] * 4.0 + 9.0
    _s1, raw_base, z_base, _l1, _e1, _m1 = mrrm.compute_stress_index(
        indicators, 30, 5.0, 3, 30,
    )
    _s2, raw_shift, z_shift, _l2, _e2, _m2 = mrrm.compute_stress_index(
        shifted, 30, 5.0, 3, 30,
    )
    assert np.allclose(z_base["Credit"], z_shift["Credit"], atol=1e-8)
    assert np.allclose(raw_base, raw_shift, atol=1e-6)


def test_column_order_keeps_the_index(market_df):
    indicators = mrrm.build_stress_indicators(market_df, 21)
    reordered = indicators[["RVol", "Curve", "VIX", "Credit"]]
    _s1, raw_base, _z1, load_base, _e1, _m1 = mrrm.compute_stress_index(
        indicators, 30, 3.0, 3, 30,
    )
    _s2, raw_order, _z2, load_order, _e2, _m2 = mrrm.compute_stress_index(
        reordered, 30, 3.0, 3, 30,
    )
    assert np.allclose(raw_base, raw_order, atol=1e-6)
    assert np.allclose(load_base.sort_index(), load_order.sort_index(), atol=1e-6)
    assert load_order["VIX"] >= 0


def test_interior_scores_match_window_pca(market_df):
    indicators = mrrm.build_stress_indicators(market_df, 21)
    pca_window = 40
    _smoothed, raw, z_scores, loadings, _explained, _mean = mrrm.compute_stress_index(
        indicators, 30, 3.0, 5, pca_window,
    )
    positions = [pca_window, len(raw) // 2, len(raw) - 1]
    sign_at = z_scores.columns.get_loc("VIX")
    for pos in positions:
        window = z_scores.iloc[pos - pca_window:pos]
        today = z_scores.iloc[pos].to_numpy()
        model = PCA(n_components=window.shape[1]).fit(window.to_numpy())
        component = model.components_[0].copy()
        if component[sign_at] < 0:
            component *= -1
        expected = float((today - model.mean_) @ component)
        assert np.isclose(raw.iloc[pos], expected, atol=1e-6)
    assert loadings["VIX"] >= 0


@pytest.mark.parametrize(
    ("long_yield", "short_yield", "expected"),
    [
        pytest.param([15.0, 15.0], [15.0, 15.0], 0.0, id="both-at-15"),
        pytest.param([15.1, 15.1], [4.0, 4.0], -(1.51 - 0.4), id="long-above-15"),
        pytest.param([4.0, 4.0], [15.1, 15.1], -(0.4 - 1.51), id="short-above-15"),
        pytest.param([40.0, 40.0], [20.0, 20.0], -2.0, id="both-times-ten"),
        pytest.param([4.0, 1.0], [1.0, 4.0], -3.0, id="both-percent"),
    ],
)
def test_yield_pair_scale_decision_table(long_yield, short_yield, expected):
    frame = _frame(
        {
            "SPX": [100.0, 101.0],
            "VIX": [15.0, 16.0],
            "UST10Y": long_yield,
            "UST3M": short_yield,
        },
        periods=2,
    )
    curve = mrrm.build_stress_indicators(frame, 2)["Curve"]
    assert np.isclose(curve.iloc[0], expected)


def test_window_bounds_are_independent(market_df):
    indicators = mrrm.build_stress_indicators(market_df, 21)
    with pytest.raises(ValueError):
        mrrm.compute_stress_index(indicators, 1, 3.0, 5, 40)
    with pytest.raises(ValueError):
        mrrm.compute_stress_index(indicators, 40, 3.0, 5, 1)
    smoothed, raw, _z, _loadings, explained, _mean = mrrm.compute_stress_index(
        indicators, 2, 3.0, 1, 2,
    )
    assert len(raw) > 10
    assert np.allclose(smoothed, raw)
    assert np.isclose(explained.sum(), 1.0)


def _quantile_with_labels(monkeypatch, labels, min_duration):
    pattern = list(labels)

    def fake(stress, n_states):
        assert len(stress) == len(pattern)
        return pd.Series(pattern, index=stress.index)

    monkeypatch.setattr(mrrm, "_quantile_labels", fake)
    stress = pd.Series(np.linspace(-1.0, 1.0, len(pattern)))
    regime, probs, tag = mrrm.detect_regimes(stress, 2, "Quantile", min_duration)
    assert tag == "Quantile"
    assert probs is None
    return regime


def test_closed_flicker_just_under_minimum_is_absorbed(monkeypatch):
    regime = _quantile_with_labels(
        monkeypatch,
        ["RISK_ON"] * 4 + ["STRESS"] * 2 + ["RISK_ON"] * 4,
        3,
    )
    assert (regime == "RISK_ON").all()


def test_closed_segment_at_minimum_stays(monkeypatch):
    regime = _quantile_with_labels(
        monkeypatch,
        ["RISK_ON"] * 4 + ["STRESS"] * 3 + ["RISK_ON"] * 4,
        3,
    )
    assert regime.iloc[4:7].tolist() == ["STRESS", "STRESS", "STRESS"]
    assert (regime.iloc[:4] == "RISK_ON").all()
    assert (regime.iloc[7:] == "RISK_ON").all()


def test_short_segment_at_the_start_stays(monkeypatch):
    regime = _quantile_with_labels(
        monkeypatch,
        ["STRESS", "STRESS"] + ["RISK_ON"] * 6,
        4,
    )
    assert regime.iloc[:2].tolist() == ["STRESS", "STRESS"]
    assert (regime.iloc[2:] == "RISK_ON").all()


def test_min_duration_one_keeps_a_single_day(monkeypatch):
    regime = _quantile_with_labels(
        monkeypatch,
        ["RISK_ON", "STRESS", "RISK_ON", "RISK_ON"],
        1,
    )
    assert regime.tolist() == ["RISK_ON", "STRESS", "RISK_ON", "RISK_ON"]


def _linear_quantile(values, probability):
    """Quantile linéaire de type 7, posé à part du code de production."""
    ordered = np.sort(np.asarray(values, dtype=float))
    if ordered.size == 1:
        return float(ordered[0])
    position = probability * (ordered.size - 1)
    left = int(position)
    right = min(left + 1, ordered.size - 1)
    weight = position - left
    return float(ordered[left] * (1.0 - weight) + ordered[right] * weight)


def _prefix_quantile_labels(values, n_states):
    labels = []
    for end in range(1, len(values) + 1):
        sample = values[:end]
        current = sample[-1]
        if n_states == 2:
            threshold = _linear_quantile(sample, 0.5)
            labels.append("RISK_ON" if current < threshold else "STRESS")
        else:
            low = _linear_quantile(sample, 0.33)
            high = _linear_quantile(sample, 0.67)
            if current < low:
                labels.append("RISK_ON")
            elif current > high:
                labels.append("STRESS")
            else:
                labels.append("NEUTRAL")
    return labels


def test_quantile_two_states_boundary():
    """Égal au seuil du préfixe : Stress. Le premier point est son propre seuil."""
    stress = pd.Series([5.0, 1.0, 1.0])
    regime, probs, tag = mrrm.detect_regimes(stress, 2, "Quantile", 1)
    assert tag == "Quantile"
    assert probs is None
    assert regime.tolist() == ["STRESS", "RISK_ON", "STRESS"]
    assert regime.tolist() == _prefix_quantile_labels(stress.tolist(), 2)


def test_quantile_three_states_boundaries():
    stress = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    regime, _probs, tag = mrrm.detect_regimes(stress, 3, "Quantile", 1)
    assert tag == "Quantile"
    assert regime.iloc[0] == "NEUTRAL"
    assert regime.iloc[-1] == "STRESS"
    assert regime.tolist() == _prefix_quantile_labels(stress.tolist(), 3)


def test_quantile_prefix_ignores_later_shock():
    history = [1.0, 1.2, 0.8, 1.1, 0.7, 1.0, 1.3]
    shock = history + [40.0, 45.0, -20.0]
    before, _, _ = mrrm.detect_regimes(pd.Series(history), 3, "Quantile", 1)
    after, _, _ = mrrm.detect_regimes(pd.Series(shock), 3, "Quantile", 1)
    assert before.tolist() == after.iloc[: len(history)].tolist()
    assert before.tolist() == _prefix_quantile_labels(history, 3)


def test_missing_spx_session_is_not_a_zero_return():
    index = pd.bdate_range("2020-01-01", periods=16)
    prices = np.array(
        [
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0,
            108.0, 109.0, 110.0, 111.0, 140.0, 112.0, 113.0, 114.0,
        ]
    )
    holed = pd.Series(prices, index=index)
    holed.iloc[12] = np.nan
    frame = mrrm._prepare_market_frame(
        {"SPX": holed, "VIX": pd.Series(np.full(16, 18.0), index=index)}
    )
    assert index[12] not in frame.index
    got = mrrm.build_stress_indicators(frame, 8)["RVol"].iloc[-1]
    observed_prices = np.delete(prices, 12)
    returns = np.diff(np.log(observed_prices))
    expected = float(returns[-8:].std(ddof=1) * np.sqrt(252) * 100)
    filled = pd.Series(prices, index=index).ffill()
    filled_vol = float(np.diff(np.log(filled.to_numpy()))[-8:].std(ddof=1) * np.sqrt(252) * 100)
    assert np.isclose(got, expected)
    assert not np.isclose(filled_vol, expected)


def test_events_just_outside_the_sample_stay_within_tolerance():
    index = pd.to_datetime(["2020-01-03", "2020-01-06"])
    stress = pd.Series([1.5, -0.2], index=index)
    regime = pd.Series(["STRESS", "RISK_ON"], index=index)
    events = {
        "2020-01-02": "day before",
        "2020-01-01": "two days before",
        "2020-01-04": "saturday",
        "2020-01-07": "day after",
        "2020-01-08": "two days after",
        "2020-01-03": "exact",
    }
    rows = mrrm.reference_event_rows(stress, regime, events, tolerance_days=1)
    by_date = {row["Date"]: row for row in rows}
    assert set(by_date) == {"2020-01-02", "2020-01-04", "2020-01-07", "2020-01-03"}
    assert by_date["2020-01-02"]["Matched"] == "2020-01-03"
    assert by_date["2020-01-02"]["Above average"] is True
    assert by_date["2020-01-04"]["Matched"] == "2020-01-03"
    assert by_date["2020-01-07"]["Matched"] == "2020-01-06"
    assert by_date["2020-01-07"]["Regime"] == "RISK_ON"
    assert by_date["2020-01-03"]["Stress"] == 1.5


def test_zero_stress_is_not_above_average():
    index = pd.to_datetime(["2020-01-03"])
    rows = mrrm.reference_event_rows(
        pd.Series([0.0], index=index),
        pd.Series(["NEUTRAL"], index=index),
        {"2020-01-03": "flat"},
        tolerance_days=0,
    )
    assert rows[0]["Above average"] is False
    assert rows[0]["Regime"] == "NEUTRAL"


def test_bad_start_does_not_download(monkeypatch):
    def download(*_args, **_kwargs):
        raise AssertionError("download called")

    monkeypatch.setattr(mrrm.yf, "download", download)
    with pytest.raises(ValueError):
        mrrm.fetch_market_data("2020-01-01; rm -rf /")


def test_unpadded_month_reaches_download_as_canonical(monkeypatch):
    seen = {}

    def download(ticker, start, progress=False, auto_adjust=True):
        seen["start"] = start
        index = pd.bdate_range("2020-01-01", periods=4)
        return pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0]}, index=index)

    monkeypatch.setattr(mrrm.yf, "download", download)
    _frame, errors = mrrm.fetch_market_data("2020-1-01")
    assert errors == []
    assert seen["start"] == "2020-01-01"


def test_start_date_calendar_edges():
    assert mrrm._normalize_start("2024-02-29") == "2024-02-29"
    assert mrrm._normalize_start("2020-1-01") == "2020-01-01"
    for bad in ("2023-02-29", "../2020-01-01", None, "   "):
        with pytest.raises(ValueError):
            mrrm._normalize_start(bad)


def test_prepare_keeps_last_duplicate_and_sorts():
    day = pd.Timestamp("2020-01-02")
    earlier = pd.Timestamp("2020-01-01")
    index = pd.DatetimeIndex([day, earlier, day])
    frame = mrrm._prepare_market_frame(
        {
            "SPX": pd.Series([1.0, 10.0, 3.0], index=index),
            "VIX": pd.Series([15.0, 16.0, 17.0], index=index),
        }
    )
    assert list(frame.index) == [earlier, day]
    assert frame.loc[day, "SPX"] == 3.0
    assert frame.loc[earlier, "VIX"] == 16.0


def test_one_leg_does_not_build_a_factor():
    frame = _frame(
        {
            "SPX": np.linspace(100, 110, 8),
            "VIX": np.linspace(12, 20, 8),
            "HYG": np.linspace(80, 70, 8),
            "UST10Y": np.full(8, 4.0),
        },
        periods=8,
    )
    indicators = mrrm.build_stress_indicators(frame, 3)
    assert "Credit" not in indicators.columns
    assert "Curve" not in indicators.columns
    assert set(indicators.columns) == {"VIX", "RVol"}


def test_flat_zscore_window_raises():
    values = [0.0, 1.0]
    for _ in range(8):
        values.append((values[-1] + values[-2]) / 2.0)
    frame = _frame({"VIX": values}, periods=len(values))
    with pytest.raises(ValueError, match="no variance"):
        mrrm.compute_stress_index(frame, 2, 3.0, 1, 2)


def test_download_omits_kwargs_the_signature_lacks(monkeypatch):
    seen = {}

    def download(ticker, start, progress=False, auto_adjust=True):
        seen["start"] = start
        seen["auto_adjust"] = auto_adjust
        index = pd.bdate_range("2021-01-01", periods=5)
        return pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)

    monkeypatch.setattr(mrrm.yf, "download", download)
    frame, errors = mrrm.fetch_market_data("2021-01-01")
    assert errors == []
    assert seen["start"] == "2021-01-01"
    assert seen["auto_adjust"] is True
    assert set(frame.columns) == set(mrrm.TICKERS)


def test_sign_falls_through_to_credit():
    rng = np.random.default_rng(3)
    hyg = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 40)))
    frame = _frame(
        {
            "SPX": np.full(40, 100.0),
            "VIX": np.full(40, 18.0),
            "HYG": hyg,
            "IEF": np.full(40, 100.0),
        },
        periods=40,
    )
    _smoothed, raw, z_scores, loadings, _explained, _mean = mrrm.compute_stress_index(
        mrrm.build_stress_indicators(frame, 5),
        8,
        3.0,
        2,
        8,
    )
    assert "VIX" not in z_scores.columns
    assert "RVol" not in z_scores.columns
    assert loadings["Credit"] >= 0
    assert np.isclose(abs(loadings["Credit"]), 1.0)
    assert len(raw) > 5


def test_rvol_window_of_two():
    prices = np.array([100.0, 110.0, 90.0, 95.0])
    frame = _frame({"SPX": prices, "VIX": [15.0, 16.0, 17.0, 18.0]}, periods=4)
    got = mrrm.build_stress_indicators(frame, 2)["RVol"].iloc[-1]
    window = np.diff(np.log(prices))[-2:]
    expected = float(window.std(ddof=1) * np.sqrt(252) * 100)
    assert np.isclose(got, expected)


def test_n_states_one_raises():
    with pytest.raises(ValueError):
        mrrm.detect_regimes(pd.Series([1.0, 2.0, 3.0]), 1, "Quantile", 1)
