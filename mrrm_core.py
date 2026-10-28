"""Stress index and regime classification for the MRRM dashboard."""

import inspect
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

try:
    from hmmlearn.hmm import GaussianHMM

    _HMM_AVAILABLE = True
except ImportError:
    GaussianHMM = None
    _HMM_AVAILABLE = False

TICKERS = {
    "SPX": "^GSPC",
    "VIX": "^VIX",
    "UST10Y": "^TNX",
    "UST3M": "^IRX",
    "HYG": "HYG",
    "IEF": "IEF",
}

_REQUIRED = ("SPX", "VIX")
_FILL_LIMIT = 5
_SIGN_PRIORITY = ("VIX", "RVol", "Credit", "Curve")
_CANON_ORDER = ("RISK_ON", "NEUTRAL", "STRESS")
_METHODS = ("HMM", "GMM", "Quantile")

REFERENCE_EVENTS = {
    "2008-10-10": "GFC, Lehman aftermath",
    "2011-08-08": "US debt ceiling crisis",
    "2015-08-24": "CNY devaluation shock",
    "2018-02-06": "Volmageddon",
    "2020-03-16": "COVID-19 crash",
    "2022-06-13": "Fed rate hiking cycle",
    "2024-08-05": "JPY carry unwind",
}


def _extract_close(frame: pd.DataFrame, ticker: str) -> pd.Series:
    """Close column, whatever order yfinance gave the levels."""
    if frame is None or frame.empty:
        raise ValueError("Empty dataframe")

    if isinstance(frame.columns, pd.MultiIndex):
        pairs = (
            (field, ticker) for field in ("Close", "Adj Close")
        )
        swapped = (
            (ticker, field) for field in ("Close", "Adj Close")
        )
        for key in (*pairs, *swapped):
            if key in frame.columns:
                series = frame[key].dropna()
                if not series.empty:
                    return series

        flat = frame.copy()
        flat.columns = [
            "_".join(str(part) for part in col).strip("_") for col in frame.columns
        ]
        for candidate in (
            f"Close_{ticker}",
            f"Adj Close_{ticker}",
            f"{ticker}_Close",
            f"{ticker}_Adj Close",
        ):
            if candidate in flat.columns:
                series = flat[candidate].dropna()
                if not series.empty:
                    return series
        raise ValueError(f"Cannot resolve Close for {ticker} in MultiIndex")

    for field in ("Close", "Adj Close"):
        if field in frame.columns:
            series = frame[field].dropna()
            if not series.empty:
                return series
    raise ValueError("No Close column found")


def _rescale_yield(series: pd.Series) -> pd.Series:
    """Yahoo sometimes quotes a CBOE yield index as yield times 10."""
    if np.nanmedian(series) > 15:
        return series / 10.0
    return series


def _yields_in_percent(long_yield: pd.Series, short_yield: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Same scale for both legs. One median above 15 means the pair is times 10."""
    if np.nanmedian(long_yield) > 15 or np.nanmedian(short_yield) > 15:
        return long_yield / 10.0, short_yield / 10.0
    return long_yield, short_yield


def _normalize_start(start_date: str) -> str:
    if not isinstance(start_date, str) or not start_date.strip():
        raise ValueError("Start date is empty")
    text = start_date.strip()
    try:
        stamp = pd.to_datetime(text, format="%Y-%m-%d")
    except (ValueError, TypeError) as exc:
        raise ValueError("Start date must be YYYY-MM-DD") from exc
    if stamp > pd.Timestamp.today().normalize():
        raise ValueError("Start date is in the future")
    return stamp.strftime("%Y-%m-%d")


def _download(ticker: str, start_date: str):
    params = inspect.signature(yf.download).parameters
    accepts_any = any(
        item.kind == inspect.Parameter.VAR_KEYWORD for item in params.values()
    )
    kwargs = {"start": start_date, "progress": False, "auto_adjust": True}
    for name, value in (
        ("repair", True),
        ("threads", False),
        ("ignore_tz", True),
    ):
        if accepts_any or name in params:
            kwargs[name] = value
    return yf.download(ticker, **kwargs)


def _prepare_market_frame(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    """Align series, fill short holes, drop a series that no longer updates."""
    missing = [name for name in _REQUIRED if name not in series_map]
    if missing:
        raise RuntimeError(f"Missing required series: {missing}")

    combined = pd.concat(series_map, axis=1).sort_index()
    if isinstance(combined.index, pd.DatetimeIndex) and combined.index.tz is not None:
        combined.index = combined.index.tz_localize(None)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.ffill(limit=_FILL_LIMIT)

    anchor = combined.dropna(subset=list(_REQUIRED))
    if anchor.empty:
        raise RuntimeError("Missing required series after alignment")
    last = anchor.index.max()

    for column in list(combined.columns):
        if column in _REQUIRED:
            continue
        last_valid = combined[column].last_valid_index()
        if last_valid is None or last_valid < last:
            combined = combined.drop(columns=column)

    return combined.loc[:last].dropna(subset=list(_REQUIRED))


def fetch_market_data(start_date: str) -> tuple[pd.DataFrame, list[str]]:
    """Download adjusted closes and align them."""
    start_date = _normalize_start(start_date)
    series_map: dict[str, pd.Series] = {}
    errors: list[str] = []

    for name, ticker in TICKERS.items():
        try:
            raw = _download(ticker, start_date)
            series = _extract_close(raw, ticker)
            series = pd.to_numeric(series, errors="coerce").dropna()
            if series.empty:
                raise ValueError("No valid data points")
            series_map[name] = series
        except Exception as exc:
            errors.append(f"{name} ({ticker}): {exc}")

    if not series_map:
        raise RuntimeError("All downloads failed:\n" + "\n".join(errors))
    return _prepare_market_frame(series_map), errors


def build_stress_indicators(df: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    """Factors signed so that a higher value means more stress."""
    if vol_window < 2:
        raise ValueError("vol_window must be >= 2")
    missing = [name for name in _REQUIRED if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required series: {missing}")

    indicators: dict[str, pd.Series] = {"VIX": df["VIX"]}

    if "HYG" in df.columns and "IEF" in df.columns:
        treasury = df["IEF"].replace(0, np.nan)
        ratio = (df["HYG"] / treasury).where(lambda values: values > 0)
        indicators["Credit"] = -np.log(ratio)

    if "UST10Y" in df.columns and "UST3M" in df.columns:
        long_yield, short_yield = _yields_in_percent(df["UST10Y"], df["UST3M"])
        slope = long_yield - short_yield
        indicators["Curve"] = -slope

    price = df["SPX"].where(df["SPX"] > 0)
    log_return = np.log(price).diff()
    indicators["RVol"] = (
        log_return.rolling(vol_window, min_periods=vol_window).std(ddof=1)
        * np.sqrt(252)
        * 100
    )
    return pd.DataFrame(indicators, index=df.index)


def _stress_sign_index(columns: pd.Index) -> int | None:
    for name in _SIGN_PRIORITY:
        if name in columns:
            return int(columns.get_loc(name))
    return None


def _rolling_pc1(
    values: np.ndarray,
    window: int,
    sign_idx: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PC1 at t from the previous `window` rows only, then project row t."""
    count, width = values.shape
    cumulative = np.zeros((count + 1, width))
    cumulative[1:] = np.cumsum(values, axis=0)
    outer = values[:, :, None] * values[:, None, :]
    cumulative_outer = np.zeros((count + 1, width, width))
    cumulative_outer[1:] = np.cumsum(outer, axis=0)

    scores = np.full(count, np.nan)
    last_comp = None
    last_mean = None
    last_ratio = None

    for index in range(window, count):
        sum_x = cumulative[index] - cumulative[index - window]
        sum_xx = cumulative_outer[index] - cumulative_outer[index - window]
        mean = sum_x / window
        cov = (sum_xx - window * np.outer(mean, mean)) / (window - 1)
        cov = (cov + cov.T) / 2
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = np.clip(evals[order], 0, None)
        total = float(evals.sum())
        if total <= 1e-12:
            continue
        component = evecs[:, order[0]].copy()
        if sign_idx is not None and component[sign_idx] < 0:
            component *= -1
        scores[index] = float((values[index] - mean) @ component)
        last_comp = component
        last_mean = mean
        last_ratio = evals / total

    if last_comp is None or last_mean is None or last_ratio is None:
        raise ValueError("PCA window has no variance")
    return scores, last_comp, last_mean, last_ratio


def compute_stress_index(
    indicators: pd.DataFrame,
    lookback: int,
    clip_z: float,
    ema_span: int,
    pca_window: int | None = None,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.Series, np.ndarray, pd.Series]:
    """Rolling z-score and rolling PC1. Past dates ignore later rows.

    Returns smoothed, raw, z-scores, latest loadings, latest variance ratios,
    and the mean of the latest PCA window.
    """
    if pca_window is None:
        pca_window = lookback
    if lookback < 2 or pca_window < 2:
        raise ValueError("windows must be >= 2")
    if clip_z <= 0:
        raise ValueError("clip_z must be positive")
    if ema_span < 1:
        raise ValueError("ema_span must be >= 1")

    z_scores = pd.DataFrame(index=indicators.index)
    for column in indicators.columns:
        rolled = indicators[column].rolling(lookback, min_periods=lookback)
        center = rolled.mean().shift(1)
        sigma = rolled.std(ddof=1).shift(1)
        sigma = sigma.mask(sigma <= 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            scored = (indicators[column] - center) / sigma
        z_scores[column] = scored.clip(-clip_z, clip_z)

    z_scores = z_scores.dropna(axis=1, how="all")
    if z_scores.shape[1] == 0:
        raise ValueError("No usable stress factor")
    valid = z_scores.dropna()
    if len(valid) <= pca_window:
        raise ValueError(
            f"Only {len(valid)} valid observations, need more than {pca_window}"
        )

    scores, component, mean, ratio = _rolling_pc1(
        valid.to_numpy(dtype=float),
        pca_window,
        _stress_sign_index(valid.columns),
    )
    raw = pd.Series(scores, index=valid.index).dropna()
    if raw.empty:
        raise ValueError("PCA window has no variance")
    smoothed = raw.ewm(span=ema_span, min_periods=1).mean()
    loadings = pd.Series(component, index=valid.columns)
    window_mean = pd.Series(mean, index=valid.columns)
    return smoothed, raw, valid.loc[raw.index], loadings, ratio, window_mean


def latest_contributions(
    z_scores: pd.DataFrame,
    loadings: pd.Series,
    window_mean: pd.Series,
) -> pd.Series:
    """Contribution of the latest z-score. The parts sum to the raw index."""
    current = z_scores.iloc[-1]
    return ((current - window_mean) * loadings).sort_values(ascending=False)


def _build_label_map(ordered_states: list, n_states: int) -> dict:
    """Map state indices already sorted from low stress to high stress."""
    if n_states == 2:
        return {ordered_states[0]: "RISK_ON", ordered_states[1]: "STRESS"}
    mapping = {ordered_states[0]: "RISK_ON", ordered_states[-1]: "STRESS"}
    for state in ordered_states[1:-1]:
        mapping[state] = "NEUTRAL"
    return mapping


def _order_probs(probs_df: pd.DataFrame) -> pd.DataFrame:
    return probs_df[[column for column in _CANON_ORDER if column in probs_df.columns]]


def _smooth_regimes(regime: pd.Series, min_duration: int) -> pd.Series:
    """Drop a finished flicker. The regime still open at the end stays."""
    values = regime.to_numpy().copy()
    count = len(values)
    start = 0
    while start < count:
        end = start
        while end < count and values[end] == values[start]:
            end += 1
        if (end - start) < min_duration and start > 0 and end < count:
            values[start:end] = values[start - 1]
        start = end
    return pd.Series(values, index=regime.index)


def _labeled_from_means(means, states, probs, index, n_states, min_duration, tag):
    order = list(np.argsort(np.asarray(means, dtype=float).reshape(-1)))
    labels = _build_label_map(order, n_states)
    regime = pd.Series([labels[int(state)] for state in states], index=index)
    columns = [labels[state] for state in range(n_states)]
    probs_df = _order_probs(pd.DataFrame(probs, index=index, columns=columns))
    return _smooth_regimes(regime, min_duration), probs_df, tag


def _fit_hmm(samples: np.ndarray, n_states: int):
    best_model = None
    best_score = -np.inf
    error = None
    for seed in range(5):
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=300,
                random_state=seed * 42 + 7,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                model.fit(samples)
            score = float(model.score(samples))
            if score > best_score:
                best_score = score
                best_model = model
        except Exception as exc:
            error = exc
    return best_model, error


def _quantile_labels(stress: pd.Series, n_states: int) -> pd.Series:
    if n_states == 2:
        threshold = stress.quantile(0.50)
        labels = np.where(stress < threshold, "RISK_ON", "STRESS")
    else:
        low, high = stress.quantile(0.33), stress.quantile(0.67)
        labels = np.select(
            [stress < low, stress > high],
            ["RISK_ON", "STRESS"],
            default="NEUTRAL",
        )
    return pd.Series(labels, index=stress.index)


def detect_regimes(
    stress: pd.Series,
    n_states: int,
    method: str,
    min_duration: int,
) -> tuple[pd.Series, pd.DataFrame | None, str]:
    """HMM, then GMM, then quantiles. NaNs are left out of the sample."""
    if n_states not in (2, 3):
        raise ValueError("n_states must be 2 or 3")
    if method not in _METHODS:
        raise ValueError(f"Unknown method: {method}")
    if min_duration < 1:
        raise ValueError("min_duration must be >= 1")

    clean = stress.dropna()
    if clean.empty:
        raise ValueError("No stress observations")
    samples = clean.to_numpy(dtype=float).reshape(-1, 1)
    index = clean.index
    hmm_error = None

    if method == "HMM" and _HMM_AVAILABLE:
        model, hmm_error = _fit_hmm(samples, n_states)
        if model is not None:
            return _labeled_from_means(
                model.means_,
                model.predict(samples),
                model.predict_proba(samples),
                index,
                n_states,
                min_duration,
                "HMM",
            )

    if method in ("HMM", "GMM"):
        try:
            mixture = GaussianMixture(
                n_components=n_states,
                covariance_type="diag",
                n_init=10,
                random_state=42,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                mixture.fit(samples)
            if method == "GMM":
                tag = "GMM"
            elif not _HMM_AVAILABLE:
                tag = "GMM (hmmlearn missing)"
            else:
                reason = f": {type(hmm_error).__name__}" if hmm_error else ""
                tag = f"GMM (HMM fit failed{reason})"
            return _labeled_from_means(
                mixture.means_,
                mixture.predict(samples),
                mixture.predict_proba(samples),
                index,
                n_states,
                min_duration,
                tag,
            )
        except Exception:
            tag = "Quantile (model fit failed)"
            regime = _quantile_labels(clean, n_states)
            return _smooth_regimes(regime, min_duration), None, tag

    regime = _quantile_labels(clean, n_states)
    return _smooth_regimes(regime, min_duration), None, "Quantile"


def _naive_index(series: pd.Series) -> pd.Series:
    if isinstance(series.index, pd.DatetimeIndex) and series.index.tz is not None:
        copy = series.copy()
        copy.index = copy.index.tz_localize(None)
        return copy
    return series


def reference_event_rows(
    stress: pd.Series,
    regime: pd.Series,
    events: dict[str, str],
    tolerance_days: int = 5,
) -> list[dict]:
    """Nearest session for each event. Above average means the index is positive."""
    if tolerance_days < 0:
        raise ValueError("tolerance_days must be >= 0")
    stress = _naive_index(stress)
    regime = _naive_index(regime).reindex(stress.index)
    if len(stress) == 0:
        return []

    rows = []
    first, last = stress.index[0], stress.index[-1]
    horizon = pd.Timedelta(days=tolerance_days)
    for date_str, event_name in events.items():
        target = pd.Timestamp(date_str)
        if target < first - horizon or target > last + horizon:
            continue
        position = int(stress.index.get_indexer([target], method="nearest")[0])
        if position < 0:
            continue
        matched = stress.index[position]
        if abs((matched - target).days) > tolerance_days:
            continue
        value = float(stress.iloc[position])
        rows.append(
            {
                "Date": date_str,
                "Matched": matched.strftime("%Y-%m-%d"),
                "Event": event_name,
                "Stress": value,
                "Regime": regime.iloc[position],
                "Above average": value > 0,
            }
        )
    return rows
