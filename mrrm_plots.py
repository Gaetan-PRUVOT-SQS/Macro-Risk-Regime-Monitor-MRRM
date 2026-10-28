"""Plotly figures for the stress dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REGIME_PALETTE = {
    "RISK_ON": "#00C853",
    "NEUTRAL": "#FFD600",
    "STRESS": "#FF1744",
}


def _base_layout(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=30, r=30, t=50, b=30),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    return fig


def plot_stress_index(raw: pd.Series, smooth: pd.Series, regime: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=raw.index, y=raw.to_numpy(), mode="lines",
        name="Raw", line=dict(width=0.5, color="rgba(150,150,150,0.2)"),
    ))
    fig.add_trace(go.Scatter(
        x=smooth.index, y=smooth.to_numpy(), mode="lines",
        name="Smoothed", line=dict(width=1.5, color="rgba(255,255,255,0.45)"),
    ))
    aligned = regime.reindex(smooth.index)
    for label, color in REGIME_PALETTE.items():
        mask = (aligned == label).to_numpy()
        if mask.any():
            fig.add_trace(go.Scatter(
                x=smooth.index[mask],
                y=smooth.to_numpy()[mask],
                mode="markers",
                name=label,
                marker=dict(color=color, size=3),
            ))
    return _base_layout(fig, "PCA macro stress index", 420)


def plot_zscore_overlay(z_scores: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column in z_scores.columns:
        fig.add_trace(go.Scatter(
            x=z_scores.index, y=z_scores[column], name=column, line=dict(width=1),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
    return _base_layout(fig, "Z-scored stress indicators", 350)


def plot_heatmap(z_scores: pd.DataFrame, days: int = 90) -> go.Figure:
    recent = z_scores.tail(days).T
    fig = go.Figure(go.Heatmap(
        z=recent.to_numpy(),
        x=recent.columns.strftime("%Y-%m-%d"),
        y=list(recent.index),
        colorscale="RdYlGn_r",
        zmid=0,
    ))
    return _base_layout(fig, f"{days}-session stress heatmap", 300)


def plot_loadings(loadings: pd.Series, explained: np.ndarray) -> go.Figure:
    ordered = loadings.sort_values()
    colors = ["#FF1744" if value > 0 else "#00C853" for value in ordered.to_numpy()]
    fig = go.Figure(go.Bar(
        x=ordered.to_numpy(), y=list(ordered.index), orientation="h", marker_color=colors,
    ))
    share = explained[0] * 100
    return _base_layout(
        fig,
        f"PCA loadings, current window, PC1 = {share:.1f}%",
        280,
    )


def plot_regime_probabilities(probs_df: pd.DataFrame | None) -> go.Figure | None:
    if probs_df is None:
        return None
    fig = go.Figure()
    for label in ("RISK_ON", "NEUTRAL", "STRESS"):
        if label in probs_df.columns:
            fig.add_trace(go.Scatter(
                x=probs_df.index,
                y=probs_df[label],
                name=label,
                stackgroup="one",
                line=dict(width=0),
                fillcolor=REGIME_PALETTE.get(label, "gray"),
            ))
    fig.update_yaxes(range=[0, 1])
    return _base_layout(fig, "Regime probabilities", 300)


def plot_spx_overlay(
    market_data: pd.DataFrame,
    stress: pd.Series,
    regime: pd.Series,
) -> go.Figure | None:
    if "SPX" not in market_data.columns:
        return None
    common = stress.index.intersection(market_data.index)
    if len(common) == 0:
        return None
    spx = market_data.loc[common, "SPX"]
    aligned = regime.reindex(common)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spx.index, y=spx.to_numpy(), mode="lines",
        name="S&P 500", line=dict(width=1, color="rgba(200,200,200,0.25)"),
    ))
    for label, color in REGIME_PALETTE.items():
        mask = (aligned == label).to_numpy()
        if mask.any():
            fig.add_trace(go.Scatter(
                x=spx.index[mask],
                y=spx.to_numpy()[mask],
                mode="markers",
                name=label,
                marker=dict(color=color, size=2),
            ))
    fig.update_yaxes(type="log")
    return _base_layout(fig, "S&P 500, regime overlay", 380)
