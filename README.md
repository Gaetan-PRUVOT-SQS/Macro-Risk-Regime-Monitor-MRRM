# Macro Risk Regime Monitor (MRRM)

[Français](#français) · [English](#english)

Dashboard de stress macro-financier (PCA, régimes dynamiques).

Macro-financial stress dashboard with dynamic PCA regimes.

---

## Français

### Présentation

MRRM agrège plusieurs facteurs de stress de marché en un **indice composite** via une Analyse en Composantes Principales (PCA), puis classe les conditions de marché en régimes discrets (Risk-On, Neutre, Stress) à l’aide de modèles statistiques (HMM / GMM).

L’approche s’inspire de méthodologies institutionnelles — notamment l’[OFR Financial Stress Index](https://www.financialresearch.gov/financial-stress-index/) et le cadre de détection de régimes de Kritzman, Page & Turkington (*Financial Analysts Journal*, 2012).

### Facteurs de stress (aperçu)

| Facteur | Source | Orientation |
|---------|--------|-------------|
| **VIX** | Indice de volatilité CBOE | Niveau — vol implicite |
| **RVol** | Rendements S&P 500 | Vol réalisée 21 jours, annualisée |

### Objectifs

- Suivre le stress macro-financier en temps quasi réel
- Visualiser les régimes de marché
- Combiner PCA et détection de régimes de façon lisible

### Stack

- Python
- Analyse de données / visualisation (dashboard)

### Lancer

Voir les scripts et dépendances du dépôt (`requirements`, notebooks ou app dashboard selon la structure du projet).

```bash
pip install -r requirements.txt
# puis lancer l'application / le notebook principal
```

---

## English

### Overview

MRRM aggregates several market stress factors into a **composite index** using Principal Component Analysis (PCA), then classifies market conditions into discrete regimes (Risk-On, Neutral, Stress) with statistical models (HMM / GMM).

The approach draws from institutional methodologies — notably the [OFR Financial Stress Index](https://www.financialresearch.gov/financial-stress-index/) and Kritzman, Page & Turkington’s regime framework (*Financial Analysts Journal*, 2012).

### Stress factors (overview)

| Factor | Source | Orientation |
|--------|--------|-------------|
| **VIX** | CBOE Volatility Index | Level — implied vol |
| **RVol** | S&P 500 returns | 21-day realized vol, annualized |

### Goals

- Track macro-financial stress in near real time
- Visualize market regimes
- Combine PCA and regime detection in a readable dashboard

### Stack

- Python
- Data analysis / visualization (dashboard)

### Run

See repository scripts and dependencies (`requirements`, notebooks, or main app).

```bash
pip install -r requirements.txt
# then start the main app / notebook
```
