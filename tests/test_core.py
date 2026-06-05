import numpy as np
import pandas as pd
import pytest

import mrrm


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def market_df():
    """Synthetic price frame with all six series and enough history."""
    rng = np.random.default_rng(0)
    n = 600
    idx = pd.bdate_range("2010-01-01", periods=n)
    spx = 1000 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    vix = np.clip(15 + rng.normal(0, 4, n).cumsum() * 0.05 + 5, 9, 80)
    return pd.DataFrame(
        {
            "SPX": spx,
            "VIX": vix,
            "UST10Y": 45 + rng.normal(0, 1, n).cumsum() * 0.1,  # x10-quoted style
            "UST3M": 40 + rng.normal(0, 1, n).cumsum() * 0.1,
            "HYG": 90 * np.exp(np.cumsum(rng.normal(0, 0.004, n))),
            "IEF": 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n))),
        },
        index=idx,
    )


@pytest.fixture
def stress_series():
    """Bimodal stress series: low regime then high regime."""
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2010-01-01", periods=600)
    vals = np.concatenate(
        [rng.normal(-1.0, 0.5, 300), rng.normal(2.0, 0.7, 300)]
    )
    return pd.Series(vals, index=idx)


# --------------------------------------------------------------------------
# _rescale_yield
# --------------------------------------------------------------------------

def test_rescale_yield_divides_when_x10():
    s = pd.Series([45.0, 46.0, 44.0])
    out = mrrm._rescale_yield(s)
    assert np.isclose(out.iloc[0], 4.5)


def test_rescale_yield_passthrough_when_percent():
    s = pd.Series([4.5, 4.6, 4.4])
    out = mrrm._rescale_yield(s)
    pd.testing.assert_series_equal(out, s)


# --------------------------------------------------------------------------
# _extract_close
# --------------------------------------------------------------------------

@pytest.fixture
def _idx5():
    return pd.bdate_range("2020-01-01", periods=5)


def test_extract_close_multiindex_close(_idx5):
    df = pd.DataFrame(
        {("Close", "^VIX"): [10, 11, 12, 13, 14], ("Open", "^VIX"): [1] * 5},
        index=_idx5,
    )
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


# --------------------------------------------------------------------------
# build_stress_indicators
# --------------------------------------------------------------------------

def test_build_stress_indicators_full(market_df):
    ind = mrrm.build_stress_indicators(market_df, 21, 21)
    assert set(ind.columns) == {"VIX", "Credit", "dCurve", "RVol"}
    assert len(ind) == len(market_df)


def test_build_stress_indicators_subset(market_df):
    df = market_df.drop(columns=["HYG", "IEF", "UST10Y", "UST3M"])
    ind = mrrm.build_stress_indicators(df, 21, 21)
    assert set(ind.columns) == {"VIX", "RVol"}


# --------------------------------------------------------------------------
# compute_stress_index
# --------------------------------------------------------------------------

def test_compute_stress_index_outputs(market_df):
    ind = mrrm.build_stress_indicators(market_df, 21, 21)
    smoothed, raw, z, loadings, ev = mrrm.compute_stress_index(ind, 120, 3.0, 10)
    assert not smoothed.isna().any()
    assert len(smoothed) == len(raw) == len(z)
    assert np.isclose(ev.sum(), 1.0, atol=1e-6)
    assert (z.abs() <= 3.0 + 1e-9).all().all()  # clip respected


def test_compute_stress_index_sign_convention(market_df):
    ind = mrrm.build_stress_indicators(market_df, 21, 21)
    _, raw, z, _, _ = mrrm.compute_stress_index(ind, 120, 3.0, 10)
    # positive index must align with VIX (stress)
    assert raw.corr(z["VIX"]) >= 0


def test_compute_stress_index_too_few_obs_raises(market_df):
    ind = mrrm.build_stress_indicators(market_df.head(150), 21, 21)
    with pytest.raises(ValueError):
        mrrm.compute_stress_index(ind, 140, 3.0, 10)


# --------------------------------------------------------------------------
# label / smoothing helpers
# --------------------------------------------------------------------------

def test_build_label_map_two_states():
    assert mrrm._build_label_map([0, 1], 2) == {0: "RISK_ON", 1: "STRESS"}


def test_build_label_map_three_states():
    assert mrrm._build_label_map([2, 0, 1], 3) == {
        2: "RISK_ON",
        0: "NEUTRAL",
        1: "STRESS",
    }


def test_smooth_regimes_removes_short_flicker():
    idx = pd.RangeIndex(8)
    reg = pd.Series(
        ["RISK_ON", "RISK_ON", "RISK_ON", "STRESS", "RISK_ON", "RISK_ON", "RISK_ON", "RISK_ON"],
        index=idx,
    )
    out = mrrm._smooth_regimes(reg, min_duration=3)
    assert "STRESS" not in out.values  # 1-day flicker absorbed


def test_state_means_empty_state_is_inf(stress_series):
    states = np.zeros(len(stress_series), dtype=int)  # state 1 never occurs
    means = mrrm._state_means(stress_series, states, 2)
    assert means[1] == np.inf
    assert np.isfinite(means[0])


def test_order_probs_canonical_order():
    df = pd.DataFrame(
        {"STRESS": [0.2], "RISK_ON": [0.5], "NEUTRAL": [0.3]}
    )
    assert list(mrrm._order_probs(df).columns) == ["RISK_ON", "NEUTRAL", "STRESS"]


def test_order_probs_subset():
    df = pd.DataFrame({"STRESS": [0.6], "RISK_ON": [0.4]})
    assert list(mrrm._order_probs(df).columns) == ["RISK_ON", "STRESS"]


# --------------------------------------------------------------------------
# detect_regimes
# --------------------------------------------------------------------------

def test_detect_regimes_gmm(stress_series):
    reg, probs, tag = mrrm.detect_regimes(stress_series, 3, "GMM", 5)
    assert tag == "GMM"
    assert len(reg) == len(stress_series)
    assert list(probs.columns) == ["RISK_ON", "NEUTRAL", "STRESS"]
    assert set(reg.unique()).issubset({"RISK_ON", "NEUTRAL", "STRESS"})


def test_detect_regimes_quantile_two(stress_series):
    reg, probs, tag = mrrm.detect_regimes(stress_series, 2, "Quantile", 5)
    assert tag == "Quantile"
    assert probs is None
    assert set(reg.unique()).issubset({"RISK_ON", "STRESS"})


def test_detect_regimes_quantile_three(stress_series):
    reg, probs, tag = mrrm.detect_regimes(stress_series, 3, "Quantile", 5)
    assert tag == "Quantile"
    assert probs is None
    assert set(reg.unique()).issubset({"RISK_ON", "NEUTRAL", "STRESS"})


def test_detect_regimes_hmm(stress_series):
    if not mrrm._HMM_AVAILABLE:
        pytest.skip("hmmlearn not installed")
    reg, probs, tag = mrrm.detect_regimes(stress_series, 2, "HMM", 5)
    assert tag == "HMM"
    assert len(reg) == len(stress_series)
    assert list(probs.columns) == [c for c in ("RISK_ON", "NEUTRAL", "STRESS") if c in probs.columns]
    # bimodal input: both regimes should be detected
    assert {"RISK_ON", "STRESS"}.issubset(set(reg.unique()))
