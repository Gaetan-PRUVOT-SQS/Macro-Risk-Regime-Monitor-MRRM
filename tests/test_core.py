import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

import mrrm_core as mrrm
from mrrm_plots import (
    plot_heatmap,
    plot_loadings,
    plot_regime_probabilities,
    plot_spx_overlay,
    plot_stress_index,
    plot_zscore_overlay,
)


@pytest.fixture
def market_df():
    """Synthetic prices, six series, fixed seed."""
    rng = np.random.default_rng(0)
    n = 600
    idx = pd.bdate_range("2010-01-01", periods=n)
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
        index=idx,
    )


@pytest.fixture
def stress_series():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2010-01-01", periods=600)
    vals = np.concatenate(
        [rng.normal(-1.0, 0.5, 300), rng.normal(2.0, 0.7, 300)]
    )
    return pd.Series(vals, index=idx)


@pytest.fixture
def _idx5():
    return pd.bdate_range("2020-01-01", periods=5)


def test_rescale_yield_divides_when_x10():
    out = mrrm._rescale_yield(pd.Series([45.0, 46.0, 44.0]))
    assert np.isclose(out.iloc[0], 4.5)


def test_rescale_yield_passthrough_when_percent():
    series = pd.Series([4.5, 4.6, 4.4])
    pd.testing.assert_series_equal(mrrm._rescale_yield(series), series)


def test_rescale_yield_boundary_fifteen():
    kept = mrrm._rescale_yield(pd.Series([15.0, 15.0]))
    divided = mrrm._rescale_yield(pd.Series([15.1, 15.1]))
    assert kept.iloc[0] == 15.0
    assert np.isclose(divided.iloc[0], 1.51)


def test_extract_close_multiindex_close(_idx5):
    df = pd.DataFrame(
        {("Close", "^VIX"): [10, 11, 12, 13, 14], ("Open", "^VIX"): [1] * 5},
        index=_idx5,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    assert mrrm._extract_close(df, "^VIX").tolist() == [10, 11, 12, 13, 14]


def test_extract_close_ticker_then_field(_idx5):
    df = pd.DataFrame({("^VIX", "Close"): [10, 11, 12, 13, 14]}, index=_idx5)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    assert mrrm._extract_close(df, "^VIX").tolist() == [10, 11, 12, 13, 14]


def test_extract_close_single_level_close(_idx5):
    df = pd.DataFrame({"Close": [10, 11, 12, 13, 14], "Open": [1] * 5}, index=_idx5)
    assert mrrm._extract_close(df, "^VIX").tolist() == [10, 11, 12, 13, 14]


def test_extract_close_adj_close_fallback(_idx5):
    df = pd.DataFrame({"Adj Close": [10, 11, 12, 13, 14]}, index=_idx5)
    assert mrrm._extract_close(df, "^VIX").tolist() == [10, 11, 12, 13, 14]


def test_extract_close_multiindex_adj_close(_idx5):
    df = pd.DataFrame({("Adj Close", "HYG"): [10, 11, 12, 13, 14]}, index=_idx5)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    assert mrrm._extract_close(df, "HYG").tolist() == [10, 11, 12, 13, 14]


def test_extract_close_empty_raises():
    with pytest.raises(ValueError):
        mrrm._extract_close(pd.DataFrame(), "X")


def test_extract_close_unknown_ticker_raises(_idx5):
    df = pd.DataFrame({("Close", "SPY"): [1, 2, 3, 4, 5]}, index=_idx5)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    with pytest.raises(ValueError):
        mrrm._extract_close(df, "^VIX")


def test_build_stress_indicators_full(market_df):
    indicators = mrrm.build_stress_indicators(market_df, 21)
    assert set(indicators.columns) == {"VIX", "Credit", "Curve", "RVol"}
    assert len(indicators) == len(market_df)


def test_build_stress_indicators_subset(market_df):
    frame = market_df.drop(columns=["HYG", "IEF", "UST10Y", "UST3M"])
    indicators = mrrm.build_stress_indicators(frame, 21)
    assert set(indicators.columns) == {"VIX", "RVol"}


def test_build_stress_indicators_missing_required():
    frame = pd.DataFrame({"VIX": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        mrrm.build_stress_indicators(frame, 2)


def test_vol_window_rejects_one(market_df):
    with pytest.raises(ValueError):
        mrrm.build_stress_indicators(market_df, 1)


def test_credit_rises_when_hyg_falls():
    idx = pd.bdate_range("2020-01-01", periods=5)
    frame = pd.DataFrame(
        {
            "SPX": [100, 100, 100, 100, 100],
            "VIX": [15, 15, 15, 15, 15],
            "HYG": [100, 100, 80, 80, 80],
            "IEF": [100, 100, 100, 100, 100],
        },
        index=idx,
    )
    credit = mrrm.build_stress_indicators(frame, 2)["Credit"]
    assert credit.iloc[2] > credit.iloc[1]
    assert np.isclose(credit.iloc[2], -np.log(0.8))


def test_curve_shares_scale_when_bill_median_is_low():
    idx = pd.bdate_range("2020-01-01", periods=3)
    frame = pd.DataFrame(
        {
            "SPX": [100, 100, 100],
            "VIX": [15, 15, 15],
            "UST10Y": [40.0, 40.0, 10.0],
            "UST3M": [10.0, 10.0, 40.0],
        },
        index=idx,
    )
    curve = mrrm.build_stress_indicators(frame, 2)["Curve"]
    assert np.isclose(curve.iloc[0], -3.0)
    assert np.isclose(curve.iloc[2], 3.0)


def test_curve_percent_quote_not_rescaled():
    idx = pd.bdate_range("2020-01-01", periods=3)
    frame = pd.DataFrame(
        {
            "SPX": [100, 100, 100],
            "VIX": [15, 15, 15],
            "UST10Y": [4.0, 4.0, 1.0],
            "UST3M": [1.0, 1.0, 4.0],
        },
        index=idx,
    )
    curve = mrrm.build_stress_indicators(frame, 2)["Curve"]
    assert np.isclose(curve.iloc[0], -3.0)
    assert np.isclose(curve.iloc[2], 3.0)


def test_rvol_matches_annualized_log_std():
    rng = np.random.default_rng(2)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 40)))
    idx = pd.bdate_range("2020-01-01", periods=40)
    frame = pd.DataFrame({"SPX": prices, "VIX": np.full(40, 18.0)}, index=idx)
    got = mrrm.build_stress_indicators(frame, 10)["RVol"].iloc[-1]
    window = np.diff(np.log(prices))[-10:]
    expected = window.std(ddof=1) * np.sqrt(252) * 100
    assert np.isclose(got, expected)


def test_compute_stress_index_outputs(market_df):
    indicators = mrrm.build_stress_indicators(market_df, 21)
    smoothed, raw, z_scores, loadings, explained, _mean = mrrm.compute_stress_index(
        indicators, 120, 3.0, 10, 120,
    )
    assert not smoothed.isna().any()
    assert len(smoothed) == len(raw) == len(z_scores)
    assert np.isclose(explained.sum(), 1.0, atol=1e-6)
    assert (z_scores.abs() <= 3.0 + 1e-9).all().all()
    assert loadings["VIX"] >= 0


def test_last_score_matches_window_pca(market_df):
    indicators = mrrm.build_stress_indicators(market_df, 21)
    pca_window = 50
    _smoothed, raw, z_scores, loadings, _explained, window_mean = mrrm.compute_stress_index(
        indicators, 40, 3.0, 5, pca_window,
    )
    window = z_scores.iloc[-(pca_window + 1):-1]
    today = z_scores.iloc[-1]
    model = PCA(n_components=window.shape[1]).fit(window.to_numpy())
    component = model.components_[0].copy()
    if component[window.columns.get_loc("VIX")] < 0:
        component *= -1
    expected = float((today.to_numpy() - model.mean_) @ component)
    assert np.isclose(raw.iloc[-1], expected, atol=1e-6)
    assert np.allclose(loadings.to_numpy(), component, atol=1e-6)
    assert np.allclose(window_mean.to_numpy(), model.mean_, atol=1e-6)


def test_contributions_sum_to_raw(market_df):
    indicators = mrrm.build_stress_indicators(market_df, 21)
    _smoothed, raw, z_scores, loadings, _explained, window_mean = mrrm.compute_stress_index(
        indicators, 40, 3.0, 5, 50,
    )
    parts = mrrm.latest_contributions(z_scores, loadings, window_mean)
    assert np.isclose(parts.sum(), raw.iloc[-1], atol=1e-6)


def test_past_stress_ignores_future(market_df):
    mid = len(market_df) // 2
    prefix = mrrm.build_stress_indicators(market_df.iloc[:mid], 21)
    full = mrrm.build_stress_indicators(market_df, 21)
    _s1, raw_prefix, _, _, _, _ = mrrm.compute_stress_index(prefix, 60, 3.0, 5, 60)
    _s2, raw_full, _, _, _, _ = mrrm.compute_stress_index(full, 60, 3.0, 5, 60)
    common = raw_prefix.index.intersection(raw_full.index)
    assert len(common) > 30
    assert np.allclose(raw_prefix.loc[common], raw_full.loc[common], atol=1e-8)
    assert np.allclose(_s1.loc[common], _s2.loc[common], atol=1e-8)


def test_last_day_does_not_rewrite_history(market_df):
    indicators = mrrm.build_stress_indicators(market_df, 21)
    shocked = indicators.copy()
    shocked.loc[shocked.index[-1], "VIX"] = shocked["VIX"].iloc[-1] + 50
    _s1, raw_base, _, _, _, _ = mrrm.compute_stress_index(indicators, 40, 3.0, 5, 40)
    _s2, raw_shock, _, _, _, _ = mrrm.compute_stress_index(shocked, 40, 3.0, 5, 40)
    shared = raw_base.index.intersection(raw_shock.index)[:-1]
    assert len(shared) > 30
    assert np.allclose(raw_base.loc[shared], raw_shock.loc[shared], atol=1e-8)


def test_constant_vix_is_dropped(market_df):
    frame = market_df.copy()
    frame["VIX"] = 18.0
    indicators = mrrm.build_stress_indicators(frame, 21)
    _s, raw, z_scores, loadings, _e, _m = mrrm.compute_stress_index(
        indicators, 40, 3.0, 5, 40,
    )
    assert len(raw) > 0
    assert "VIX" not in z_scores.columns
    assert "VIX" not in loadings.index
    assert loadings["RVol"] >= 0


def test_compute_stress_index_too_few_obs_raises(market_df):
    indicators = mrrm.build_stress_indicators(market_df.head(150), 21)
    with pytest.raises(ValueError):
        mrrm.compute_stress_index(indicators, 140, 3.0, 10, 140)


def test_build_label_map_two_states():
    assert mrrm._build_label_map([0, 1], 2) == {0: "RISK_ON", 1: "STRESS"}


def test_build_label_map_three_states():
    assert mrrm._build_label_map([2, 0, 1], 3) == {
        2: "RISK_ON",
        0: "NEUTRAL",
        1: "STRESS",
    }


def test_smooth_regimes_removes_short_flicker():
    regime = pd.Series(
        ["RISK_ON", "RISK_ON", "RISK_ON", "STRESS", "RISK_ON", "RISK_ON", "RISK_ON", "RISK_ON"],
    )
    out = mrrm._smooth_regimes(regime, min_duration=3)
    assert "STRESS" not in out.to_numpy()


def test_smooth_regimes_keeps_open_segment():
    regime = pd.Series(["RISK_ON"] * 6 + ["STRESS"])
    out = mrrm._smooth_regimes(regime, min_duration=5)
    assert out.iloc[-1] == "STRESS"
    assert (out.iloc[:6] == "RISK_ON").all()


def test_order_probs_canonical_order():
    frame = pd.DataFrame({"STRESS": [0.2], "RISK_ON": [0.5], "NEUTRAL": [0.3]})
    assert list(mrrm._order_probs(frame).columns) == ["RISK_ON", "NEUTRAL", "STRESS"]


def test_order_probs_subset():
    frame = pd.DataFrame({"STRESS": [0.6], "RISK_ON": [0.4]})
    assert list(mrrm._order_probs(frame).columns) == ["RISK_ON", "STRESS"]


def test_detect_regimes_gmm(stress_series):
    regime, probs, tag = mrrm.detect_regimes(stress_series, 3, "GMM", 5)
    assert tag == "GMM"
    assert len(regime) == len(stress_series)
    assert list(probs.columns) == ["RISK_ON", "NEUTRAL", "STRESS"]
    assert set(regime.unique()).issubset({"RISK_ON", "NEUTRAL", "STRESS"})


def test_detect_regimes_quantile_two(stress_series):
    regime, probs, tag = mrrm.detect_regimes(stress_series, 2, "Quantile", 5)
    assert tag == "Quantile"
    assert probs is None
    assert set(regime.unique()).issubset({"RISK_ON", "STRESS"})


def test_detect_regimes_quantile_three(stress_series):
    regime, probs, tag = mrrm.detect_regimes(stress_series, 3, "Quantile", 5)
    assert tag == "Quantile"
    assert probs is None
    assert set(regime.unique()).issubset({"RISK_ON", "NEUTRAL", "STRESS"})


def test_detect_regimes_quantile_with_nans(stress_series):
    series = stress_series.copy()
    series.iloc[::10] = np.nan
    regime, probs, tag = mrrm.detect_regimes(series, 2, "Quantile", 1)
    assert tag == "Quantile"
    assert probs is None
    assert len(regime) == int(series.notna().sum())
    assert regime.index.equals(series.dropna().index)


def test_detect_regimes_hmm(stress_series):
    if not mrrm._HMM_AVAILABLE:
        pytest.skip("hmmlearn not installed")
    regime, probs, tag = mrrm.detect_regimes(stress_series, 2, "HMM", 5)
    assert tag == "HMM"
    assert len(regime) == len(stress_series)
    assert list(probs.columns) == [
        column for column in ("RISK_ON", "NEUTRAL", "STRESS") if column in probs.columns
    ]
    assert {"RISK_ON", "STRESS"}.issubset(set(regime.unique()))


def test_hmm_labels_follow_model_means(stress_series, monkeypatch):
    class FakeHMM:
        def __init__(self, *args, **kwargs):
            self.means_ = np.array([[0.0], [5.0], [-2.0]])

        def fit(self, samples):
            return self

        def score(self, samples):
            return 1.0

        def predict(self, samples):
            count = len(samples)
            return np.array([0, 1] * (count // 2) + [0] * (count % 2))

        def predict_proba(self, samples):
            states = self.predict(samples)
            probs = np.zeros((len(samples), 3))
            probs[np.arange(len(samples)), states] = 1.0
            return probs

    monkeypatch.setattr(mrrm, "_HMM_AVAILABLE", True)
    monkeypatch.setattr(mrrm, "GaussianHMM", FakeHMM)
    regime, _probs, tag = mrrm.detect_regimes(stress_series, 3, "HMM", 1)
    assert tag == "HMM"
    assert (regime.to_numpy()[0::2] == "NEUTRAL").all()
    assert (regime.to_numpy()[1::2] == "STRESS").all()


def test_gmm_failure_falls_back_to_quantile(stress_series, monkeypatch):
    def boom(self, X, y=None):
        raise ValueError("singular")

    monkeypatch.setattr(mrrm.GaussianMixture, "fit", boom)
    regime, probs, tag = mrrm.detect_regimes(stress_series, 2, "GMM", 1)
    assert tag == "Quantile (model fit failed)"
    assert probs is None
    assert set(regime.unique()).issubset({"RISK_ON", "STRESS"})


def test_hmm_failure_uses_gmm(stress_series, monkeypatch):
    class BadHMM:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, samples):
            raise RuntimeError("nope")

    monkeypatch.setattr(mrrm, "_HMM_AVAILABLE", True)
    monkeypatch.setattr(mrrm, "GaussianHMM", BadHMM)
    _regime, probs, tag = mrrm.detect_regimes(stress_series, 2, "HMM", 1)
    assert tag.startswith("GMM (HMM fit failed")
    assert probs is not None


def test_unknown_method_raises(stress_series):
    with pytest.raises(ValueError):
        mrrm.detect_regimes(stress_series, 2, "kmeans", 1)


def test_n_states_out_of_range_raises(stress_series):
    with pytest.raises(ValueError):
        mrrm.detect_regimes(stress_series, 4, "Quantile", 1)


def test_normalize_start_strips_and_checks_calendar():
    assert mrrm._normalize_start(" 2007-01-01 ") == "2007-01-01"
    today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    assert mrrm._normalize_start(today) == today
    tomorrow = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for bad in ("", "2007/01/01", "yesterday", "2020-13-01", tomorrow):
        with pytest.raises(ValueError):
            mrrm._normalize_start(bad)


def _prices(index, start=10.0):
    return pd.Series(np.linspace(start, start + len(index) - 1, len(index)), index=index)


def test_prepare_fills_five_day_gap_and_leaves_the_sixth():
    index = pd.bdate_range("2020-01-01", periods=30)
    filled = _prices(index, 80)
    filled.iloc[10:15] = np.nan
    holed = _prices(index, 80)
    holed.iloc[10:16] = np.nan
    base = {"SPX": _prices(index, 100), "VIX": _prices(index, 15)}
    five = mrrm._prepare_market_frame({**base, "HYG": filled})
    six = mrrm._prepare_market_frame({**base, "HYG": holed})
    assert five["HYG"].isna().sum() == 0
    assert six["HYG"].isna().sum() == 1


def test_prepare_drops_series_that_stopped_and_strips_tz():
    index = pd.bdate_range("2020-01-01", periods=30, tz="UTC")
    hyg = _prices(index, 80)
    hyg.iloc[-10:] = np.nan
    frame = mrrm._prepare_market_frame(
        {
            "SPX": _prices(index, 100),
            "VIX": _prices(index, 15),
            "HYG": hyg,
        }
    )
    assert frame.index.tz is None
    assert frame.index[0] == pd.Timestamp("2020-01-01")
    assert "HYG" not in frame.columns
    assert len(frame) == 30


def test_fetch_skips_failed_ticker_and_reads_both_layouts(monkeypatch):
    def download(ticker, **kwargs):
        assert kwargs["start"] == "2021-01-01"
        assert kwargs["auto_adjust"] is True
        assert kwargs["repair"] is True
        assert kwargs["threads"] is False
        if ticker == "HYG":
            raise RuntimeError("down")
        index = pd.bdate_range("2021-01-01", periods=40)
        values = np.linspace(10, 20, 40)
        if ticker == "IEF":
            columns = pd.MultiIndex.from_tuples([(ticker, "Close")])
        else:
            columns = pd.MultiIndex.from_tuples([("Close", ticker)])
        return pd.DataFrame(values, index=index, columns=columns)

    monkeypatch.setattr(mrrm.yf, "download", download)
    frame, errors = mrrm.fetch_market_data("2021-01-01")
    assert {"SPX", "VIX", "IEF", "UST10Y", "UST3M"} <= set(frame.columns)
    assert "HYG" not in frame.columns
    assert frame.index.tz is None
    assert any("HYG" in message for message in errors)


def test_reference_events_tolerance_and_sign():
    index = pd.to_datetime(["2020-01-01", "2020-01-20"])
    stress = pd.Series([-1.0, 2.0], index=index)
    regime = pd.Series(["RISK_ON", "STRESS"], index=index)
    events = {
        "2020-01-06": "five days",
        "2020-01-07": "six days",
        "2019-01-01": "before",
        "2020-01-20": "exact",
    }
    rows = mrrm.reference_event_rows(stress, regime, events, tolerance_days=5)
    by_date = {row["Date"]: row for row in rows}
    assert set(by_date) == {"2020-01-06", "2020-01-20"}
    assert by_date["2020-01-06"]["Above average"] is False
    assert by_date["2020-01-06"]["Matched"] == "2020-01-01"
    assert by_date["2020-01-20"]["Above average"] is True
    assert by_date["2020-01-20"]["Regime"] == "STRESS"
    assert "Pass" not in by_date["2020-01-20"]


def test_reference_events_timezone_index():
    index = pd.bdate_range("2020-03-02", periods=5, tz="UTC")
    stress = pd.Series([0.4, 0.2, -0.1, 0.3, 0.5], index=index)
    regime = pd.Series(["STRESS"] * 5, index=index)
    rows = mrrm.reference_event_rows(
        stress, regime, {"2020-03-02": "crash"}, tolerance_days=0,
    )
    assert len(rows) == 1
    assert rows[0]["Above average"] is True
    assert rows[0]["Matched"] == "2020-03-02"


def test_plot_stress_marks_regimes():
    index = pd.bdate_range("2020-01-01", periods=4)
    raw = pd.Series([0.1, 0.2, 1.5, 1.4], index=index)
    regime = pd.Series(["RISK_ON", "RISK_ON", "STRESS", "STRESS"], index=index)
    figure = plot_stress_index(raw, raw, regime)
    assert "STRESS" in [trace.name for trace in figure.data]


def test_bad_windows_and_empty_factors_raise(market_df):
    indicators = mrrm.build_stress_indicators(market_df, 21)
    for args in ((1, 3.0, 5, 40), (40, 0, 5, 40), (40, 3.0, 0, 40)):
        with pytest.raises(ValueError):
            mrrm.compute_stress_index(indicators, *args)
    defaulted = mrrm.compute_stress_index(indicators, 40, 3.0, 5)
    assert len(defaulted[1]) > 30
    empty = pd.DataFrame({"VIX": np.nan, "RVol": np.nan}, index=indicators.index[:30])
    with pytest.raises(ValueError, match="usable"):
        mrrm.compute_stress_index(empty, 5, 3.0, 2, 5)


def test_zero_prices_do_not_crash():
    index = pd.bdate_range("2020-01-01", periods=8)
    frame = pd.DataFrame(
        {
            "SPX": [0, 100, 101, 102, 103, 104, 105, 106],
            "VIX": [15, 16, 17, 18, 19, 20, 21, 22],
            "HYG": [100, 100, 100, 100, 100, 100, 100, 100],
            "IEF": [0, 100, 100, 100, 100, 100, 100, 100],
        },
        index=index,
    )
    indicators = mrrm.build_stress_indicators(frame, 3)
    assert pd.isna(indicators["Credit"].iloc[0])
    assert np.isclose(indicators["Credit"].iloc[1], 0.0)


def test_min_duration_and_empty_stress(stress_series):
    with pytest.raises(ValueError):
        mrrm.detect_regimes(stress_series, 2, "Quantile", 0)
    with pytest.raises(ValueError, match="No stress"):
        mrrm.detect_regimes(pd.Series([np.nan, np.nan]), 2, "Quantile", 1)


def test_hmm_missing_uses_gmm(stress_series, monkeypatch):
    monkeypatch.setattr(mrrm, "_HMM_AVAILABLE", False)
    _regime, probs, tag = mrrm.detect_regimes(stress_series, 2, "HMM", 1)
    assert tag == "GMM (hmmlearn missing)"
    assert probs is not None


def test_prepare_rejects_missing_or_empty_required():
    index = pd.bdate_range("2020-01-01", periods=3)
    with pytest.raises(RuntimeError, match="Missing required"):
        mrrm._prepare_market_frame({"VIX": _prices(index)})
    with pytest.raises(RuntimeError, match="after alignment"):
        mrrm._prepare_market_frame(
            {
                "SPX": pd.Series([np.nan, np.nan, np.nan], index=index),
                "VIX": pd.Series([np.nan, np.nan, np.nan], index=index),
            }
        )


def test_fetch_empty_close_is_a_missing_series(monkeypatch):
    def download(ticker, **kwargs):
        index = pd.bdate_range("2021-01-01", periods=3)
        if ticker == "^GSPC":
            return pd.DataFrame({"Close": [np.nan, np.nan, np.nan]}, index=index)
        return pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=index)

    monkeypatch.setattr(mrrm.yf, "download", download)
    with pytest.raises(RuntimeError, match="SPX"):
        mrrm.fetch_market_data("2021-01-01")


def test_fetch_all_failed(monkeypatch):
    def download(ticker, **kwargs):
        raise RuntimeError("down")

    monkeypatch.setattr(mrrm.yf, "download", download)
    with pytest.raises(RuntimeError, match="All downloads"):
        mrrm.fetch_market_data("2021-01-01")


def test_extract_skips_empty_close_then_uses_adj(_idx5):
    frame = pd.DataFrame(
        {("Close", "^VIX"): [np.nan] * 5, ("Adj Close", "^VIX"): [1, 2, 3, 4, 5]},
        index=_idx5,
    )
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    assert mrrm._extract_close(frame, "^VIX").tolist() == [1, 2, 3, 4, 5]


def test_extract_flat_column_name(_idx5):
    frame = pd.DataFrame({("Close_^VIX",): [1, 2, 3, 4, 5]}, index=_idx5)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    assert mrrm._extract_close(frame, "^VIX").tolist() == [1, 2, 3, 4, 5]
    blank = pd.DataFrame({"Close": [np.nan] * 5}, index=_idx5)
    with pytest.raises(ValueError):
        mrrm._extract_close(blank, "^VIX")


def test_reference_events_empty_and_bad_tolerance():
    with pytest.raises(ValueError):
        mrrm.reference_event_rows(pd.Series(dtype=float), pd.Series(dtype=float), {}, -1)
    assert mrrm.reference_event_rows(pd.Series(dtype=float), pd.Series(dtype=float), {"2020-01-01": "x"}) == []


def test_plot_heatmap_and_missing_spx():
    index = pd.bdate_range("2020-01-01", periods=5)
    z_scores = pd.DataFrame({"VIX": [0.1, -0.2, 0.3, 1.2, -0.4]}, index=index)
    figure = plot_heatmap(z_scores, days=90)
    assert figure.layout.title.text.startswith("90-session")
    assert plot_spx_overlay(pd.DataFrame(index=index), z_scores["VIX"], z_scores["VIX"]) is None
    overlay = plot_zscore_overlay(z_scores)
    assert overlay.layout.title.text.startswith("Z-scored")
    loadings = plot_loadings(pd.Series({"VIX": 0.4, "RVol": -0.2}), np.array([0.6, 0.4]))
    assert "PC1" in loadings.layout.title.text
    assert plot_regime_probabilities(None) is None
    probs = pd.DataFrame({"RISK_ON": [0.7, 0.2], "STRESS": [0.3, 0.8]}, index=index[:2])
    assert plot_regime_probabilities(probs) is not None
    market = pd.DataFrame({"SPX": [100, 101, 102, 103, 104]}, index=index)
    regime = pd.Series(["RISK_ON", "STRESS", "STRESS", "NEUTRAL", "RISK_ON"], index=index)
    painted = plot_spx_overlay(market, z_scores["VIX"], regime)
    assert painted is not None
    assert "STRESS" in [trace.name for trace in painted.data]
    other = pd.bdate_range("2021-01-01", periods=1)
    assert plot_spx_overlay(market, pd.Series([0.1], index=other), pd.Series(["RISK_ON"], index=other)) is None
