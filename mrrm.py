"""
Macro Risk Regime Monitor.

    streamlit run mrrm.py
"""

import traceback

import numpy as np
import pandas as pd
import streamlit as st

import mrrm_core as core
import mrrm_plots as plots

st.set_page_config(page_title="MRRM", layout="wide")
st.title("Macro Risk Regime Monitor")
st.caption(
    "Rolling z-score, rolling PCA and quantile regimes use only data known that day. "
    "HMM and GMM history is fit on the loaded sample and can be revised."
)

if not core._HMM_AVAILABLE:
    st.info("HMM needs hmmlearn. The page falls back to GMM.")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def run_monitor(
    start_date: str,
    lookback: int,
    vol_window: int,
    clip_z: float,
    ema_span: int,
    pca_window: int,
    n_states: int,
    method: str,
    min_duration: int,
):
    market, errors = core.fetch_market_data(start_date)
    indicators = core.build_stress_indicators(market, vol_window)
    smoothed, raw, z_scores, loadings, explained, window_mean = core.compute_stress_index(
        indicators, lookback, clip_z, ema_span, pca_window,
    )
    regime, probs, tag = core.detect_regimes(smoothed, n_states, method, min_duration)
    return (
        market, errors, smoothed, raw, z_scores, loadings, explained,
        window_mean, regime, probs, tag,
    )


with st.sidebar:
    st.header("Parameters")
    start_date = st.text_input("Start date", value="2007-01-01")
    lookback = st.slider(
        "Z-score lookback (trading days)",
        63, 756, 252, step=21,
        help="Mean and volatility are the previous window, not including today.",
    )
    pca_window = st.slider(
        "PCA window (trading days)",
        126, 756, 504, step=21,
        help="Loadings use this many past observations. Older index values stay put.",
    )
    vol_window = st.slider("Volatility window (trading days)", 10, 63, 21)
    clip_z = st.slider("Z-score clip", 1.0, 5.0, 3.0, step=0.5)
    ema_span = st.slider("EMA span (trading days)", 1, 42, 10)

    st.subheader("Regime detection")
    n_states = st.radio("States", [2, 3], index=1, horizontal=True)
    available_methods = ["HMM", "GMM", "Quantile"]
    default_method = 0 if core._HMM_AVAILABLE else 1
    method = st.radio("Method", available_methods, index=default_method, horizontal=True)
    min_duration = st.slider(
        "Min regime duration (trading days)",
        1, 21, 5,
        help="Short finished regimes are absorbed. The regime still open is kept.",
    )

    if st.button("Clear cache"):
        st.cache_data.clear()

try:
    with st.spinner("Loading market data..."):
        (
            market_data, fetch_errors, stress, stress_raw, z_scores, loadings,
            explained_var, window_mean, regime, regime_probs, method_tag,
        ) = run_monitor(
            start_date, lookback, vol_window, clip_z, ema_span, pca_window,
            n_states, method, min_duration,
        )
    if fetch_errors:
        st.warning("Some tickers failed:\n" + "\n".join(fetch_errors))

    st.success(
        f"{stress.index[0].date()} → {stress.index[-1].date()} · "
        f"{len(stress):,} obs · {len(z_scores.columns)} factors · "
        f"PC1 = {explained_var[0] * 100:.1f}% · {method_tag}"
    )

    current_stress = stress.iloc[-1]
    current_regime = regime.iloc[-1]
    current_z = z_scores.iloc[-1]
    contributions = core.latest_contributions(z_scores, loadings, window_mean)

    col_regime, col_stress = st.columns(2)
    col_regime.metric("Current regime", current_regime)
    col_stress.metric("Stress index", f"{current_stress:.2f}")

    z_cols = st.columns(len(z_scores.columns))
    for i, name in enumerate(z_scores.columns):
        z_cols[i].metric(f"Z({name})", f"{current_z[name]:.2f}")

    st.plotly_chart(
        plots.plot_stress_index(stress_raw, stress, regime), width="stretch",
    )
    st.plotly_chart(plots.plot_zscore_overlay(z_scores), width="stretch")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            plots.plot_loadings(loadings, explained_var), width="stretch",
        )
    with right:
        prob_fig = plots.plot_regime_probabilities(regime_probs)
        if prob_fig:
            st.plotly_chart(prob_fig, width="stretch")
        else:
            st.info("Probabilities are not available in quantile mode.")

    st.plotly_chart(plots.plot_heatmap(z_scores, days=90), width="stretch")

    spx_fig = plots.plot_spx_overlay(market_data, stress, regime)
    if spx_fig:
        st.plotly_chart(spx_fig, width="stretch")

    st.subheader("Stress decomposition (latest observation)")
    decomp = pd.DataFrame({
        "Z-score": current_z,
        "PC1 loading": loadings,
        "Contribution": contributions,
    }).sort_values("Contribution", ascending=False)
    st.dataframe(decomp.style.format("{:.3f}"), width="stretch")

    with st.expander("Explained variance by component"):
        var_table = pd.DataFrame({
            "Component": [f"PC{i + 1}" for i in range(len(explained_var))],
            "Variance (%)": explained_var * 100,
            "Cumulative (%)": np.cumsum(explained_var) * 100,
        })
        st.dataframe(
            var_table.style.format(
                {"Variance (%)": "{:.1f}", "Cumulative (%)": "{:.1f}"}
            ),
            width="stretch",
        )

    st.subheader("Regime distribution")
    counts = regime.value_counts()
    pct = (counts / len(regime) * 100).round(1)
    st.dataframe(pd.DataFrame({"Count": counts, "%": pct}), width="stretch")

    with st.expander("Known stress dates"):
        st.caption(
            "Positive means above the recent norm. "
            "Regime labels can change when the sample grows. The index does not."
        )
        rows = core.reference_event_rows(stress, regime, core.REFERENCE_EVENTS)
        if rows:
            table = pd.DataFrame(rows)
            table["Stress"] = table["Stress"].map(lambda value: f"{value:.2f}")
            table["Above average"] = table["Above average"].map(
                lambda flag: "yes" if flag else "no"
            )
            st.dataframe(table, width="stretch")
        else:
            st.caption("No reference date in the selected range.")

    export = pd.DataFrame({
        "stress_index": stress,
        "stress_raw": stress_raw,
        "regime": regime,
    })
    for column in z_scores.columns:
        export[f"z_{column}"] = z_scores[column]

    st.download_button(
        "Download CSV",
        export.to_csv().encode("utf-8"),
        "mrrm_export.csv",
        "text/csv",
    )

    with st.expander("Recent data"):
        st.dataframe(export.tail(10), width="stretch")

except Exception as exc:
    st.error(str(exc))
    with st.expander("Detail"):
        st.code(traceback.format_exc())
