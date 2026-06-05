# Macro Risk Regime Monitor (MRRM)

Real-time macro-financial stress dashboard that combines PCA-based composite stress indexing with statistical regime detection.

<!-- ![Dashboard](docs/screenshots/dashboard.png) -->

## Overview

MRRM aggregates four market-based stress factors into a single composite index using Principal Component Analysis, then classifies market conditions into discrete regimes (Risk-On, Neutral, Stress) via Hidden Markov Models or Gaussian Mixture Models.

The approach draws from institutional methodologies — notably the [OFR Financial Stress Index](https://www.financialresearch.gov/financial-stress-index/) and Kritzman, Page & Turkington's regime detection framework (*Financial Analysts Journal*, 2012).

## Methodology

### Stress Factors

| Factor | Source | Orientation |
|--------|--------|-------------|
| **VIX** | CBOE Volatility Index | Level — direct implied vol measure |
| **RVol** | S&P 500 returns | 21-day realized volatility, annualized |
| **Credit** | HYG / IEF ratio | Inverted momentum — HY underperformance = stress |
| **dCurve** | 10Y − 3M Treasury slope | Rate of flattening — rapid inversion = stress |

### Pipeline

1. **Rolling z-scores** — Each factor is standardized against a 252-day rolling window (configurable), clipped at ±3σ to limit outlier influence.

2. **PCA aggregation** — The first principal component (PC1) captures the common variance across factors. A sign convention enforces positive correlation with VIX (positive = stress). Typical PC1 explained variance: 50–60%.

3. **EMA smoothing** — An exponential moving average (default 10-day span) reduces daily noise before regime classification.

4. **Regime detection** — The smoothed stress index is classified using:
   - **HMM** (preferred): models temporal persistence via state transition probabilities
   - **GMM** (fallback): independent daily classification, no memory
   - **Quantile** (simple fallback): fixed percentile thresholds

5. **Regime smoothing** — A minimum duration filter (default 5 days) eliminates spurious flickers.

### Validation

The model is back-tested against known stress events:

| Date | Event | Expected |
|------|-------|----------|
| 2008-10-10 | GFC — Lehman aftermath | STRESS |
| 2011-08-08 | US debt ceiling crisis | STRESS |
| 2015-08-24 | CNY devaluation shock | STRESS |
| 2018-02-06 | Volmageddon | STRESS |
| 2020-03-16 | COVID-19 crash | STRESS |
| 2022-06-13 | Fed rate hiking cycle | STRESS |
| 2024-08-05 | JPY carry unwind | STRESS |

> **Note on look-ahead.** The PCA loadings, z-score statistics, and the
> HMM/GMM regime models are fitted **in-sample over the full history** loaded.
> Historical regime labels therefore use information that was not available in
> real time, and the back-test above is illustrative, not an out-of-sample
> result. For live use the latest observation is informative; the historical
> reconstruction is not a tradeable signal. A walk-forward (expanding-window)
> refit would be required to remove this bias.

## Quick Start

```bash
git clone https://github.com/<your-username>/mrrm.git
cd mrrm
pip install -r requirements.txt
streamlit run mrrm.py
```

The dashboard opens at `http://localhost:8501`. Data is fetched from Yahoo Finance on first load and cached for 6 hours.

### Parameters

All parameters are adjustable via the sidebar:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Start date | 2007-01-01 | — | Historical data start |
| Z-score lookback | 252 days | 63–756 | Rolling standardization window |
| Momentum window | 21 days | 5–63 | Credit & curve momentum period |
| Volatility window | 21 days | 10–63 | Realized vol estimation window |
| Z-score clip | 3.0 | 1.0–5.0 | Outlier clipping threshold |
| EMA span | 10 days | 1–42 | Stress index smoothing |
| States | 3 | 2–3 | Number of regime states |
| Method | HMM | HMM/GMM/Quantile | Classification algorithm |
| Min duration | 5 days | 1–21 | Regime flicker filter |

## Dashboard Components

- **PCA Stress Index** — Time series with raw (gray) and EMA-smoothed values, color-coded by regime
- **Z-scored Indicators** — Overlay of all four standardized factors
- **PCA Loadings** — PC1 factor weights with explained variance
- **Regime Probabilities** — Stacked area chart of state probabilities (HMM/GMM)
- **90-Day Heatmap** — Recent stress levels by factor
- **S&P 500 Overlay** — Price history colored by detected regime (log scale)
- **Stress Decomposition** — Factor-level attribution table (z-score × loading)
- **Event Back-test** — Validation against historical stress episodes

## Requirements

- Python 3.10+
- Dependencies: see `requirements.txt`
- `hmmlearn` is optional but recommended for temporal regime persistence

## License

[MIT](LICENSE)

---

## 🇫🇷 Version française

### Aperçu

MRRM est un tableau de bord de suivi du stress macro-financier en temps réel. Il agrège quatre facteurs de stress (VIX, volatilité réalisée, spread de crédit, delta de courbe des taux) en un indice composite via Analyse en Composantes Principales, puis classifie les conditions de marché en régimes discrets (Risk-On, Neutre, Stress) par modèle de Markov caché ou mélange gaussien.

### Démarrage rapide

```bash
git clone https://github.com/<your-username>/mrrm.git
cd mrrm
pip install -r requirements.txt
streamlit run mrrm.py
```

### Méthodologie

1. **Z-scores glissants** sur fenêtre de 252 jours, clippés à ±3σ
2. **ACP** — La première composante principale capture la variance commune. Convention de signe : positif = stress
3. **Lissage EMA** (10 jours par défaut) avant classification
4. **Détection de régimes** — HMM (préféré), GMM (fallback), ou quantiles
5. **Filtrage de durée minimale** — élimine les régimes trop courts

### Facteurs de stress

| Facteur | Source | Logique |
|---------|--------|---------|
| **VIX** | Indice de volatilité CBOE | Niveau — mesure directe de vol implicite |
| **RVol** | Rendements S&P 500 | Volatilité réalisée 21j, annualisée |
| **Credit** | Ratio HYG/IEF | Momentum inversé — sous-performance HY = stress |
| **dCurve** | Pente 10Y − 3M | Vitesse d'aplatissement — inversion rapide = stress |

### Licence

[MIT](LICENSE)
