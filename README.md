# Macro Risk Regime Monitor (MRRM)

Page locale qui suit le stress du marché. Elle télécharge quelques séries Yahoo, en fait un indice du jour, puis range la période en Risk-On, Neutre ou Stress.

Local page that tracks market stress. It downloads a few Yahoo series, turns them into a daily index, then labels the period Risk-On, Neutral or Stress.

## Français

La page se lance en local. La date de départ se règle dans la barre latérale, 2007 par défaut. Cette barre règle aussi la longueur des fenêtres, le nombre de régimes et la méthode. Les cours restent en cache six heures. Le bouton Clear cache relance le téléchargement.

En haut, le régime en cours et le niveau de l'indice. Dessous : la courbe, les facteurs en z-score, les poids du jour, une heatmap, le S&P 500 coloré par régime, et un tableau de dates de crise connues (2008, 2011, 2015, 2018, 2020, 2022, 2024). Un bouton exporte le CSV. Une série optionnelle qui ne joint plus la fin est laissée de côté, avec un avertissement. S&P 500 et VIX sont obligatoires.

Les cours viennent de Yahoo Finance, prix ajustés.

| Nom | Ticker | Lecture |
|-----|--------|---------|
| VIX | `^VIX` | volatilité implicite, en niveau |
| Credit | HYG / IEF | moins le log du ratio. Le high yield qui se dégrade fait monter le facteur |
| Curve | `^TNX` et `^IRX` | l'opposé de la pente 10 ans moins 3 mois. Une courbe plate ou inversée fait monter le facteur |
| RVol | `^GSPC` | volatilité réalisée du S&P 500, log-rendements des séances imprimées, fenêtre glissante, annualisée |

L'indice monte quand le stress monte. Chaque facteur est d'abord comparé aux séances d'avant : la moyenne et l'écart-type s'arrêtent à la veille, pour qu'un choc du jour ne fabrique pas sa propre norme. Ces z-scores sont résumés par le premier axe d'une ACP calculée seulement sur les séances passées (504 par défaut, proche des 500 séances utilisées par l'OFR). Le poids du VIX est gardé positif. Zéro, c'est la norme récente. Au-dessus, le stress dépasse cette norme.

Le régime préféré est un HMM à deux ou trois états (Kritzman, Page et Turkington, 2012). Si l'ajustement échoue, la page passe à un GMM, puis à des quantiles. Un régime déjà terminé et trop court est absorbé par le précédent. Celui qui est encore en cours reste affiché, même s'il vient de commencer. Le HMM et le GMM sont estimés sur tout l'échantillon chargé : ces étiquettes peuvent changer quand l'historique s'allonge. Les quantiles coupent avec les séances connues ce jour-là, jour inclus. Une séance du S&P 500 sans cours sort du calcul de la volatilité. Les valeurs déjà calculées de l'indice ne bougent pas.

L'OFR publie son indice à partir de 33 séries. Ici il y en a quatre. L'idée reprise est la même : un niveau qui monte avec le stress, des poids estimés sur le passé, et aucune séance future dans le calcul du jour.

### Lancer

```bash
uv venv .venv
uv pip install -r requirements.txt -r requirements-dev.txt --python .venv/bin/python
.venv/bin/streamlit run mrrm.py
```

La page écoute en local. Les tests :

```bash
.venv/bin/pytest
```

La campagne, les techniques et le dernier passage sont dans [tests/README.md](tests/README.md).

### Fichiers

| Fichier | Rôle |
|---------|------|
| `mrrm.py` | la page Streamlit |
| `mrrm_core.py` | téléchargement, indice, régimes |
| `mrrm_plots.py` | graphiques |
| `tests/` | suite et campagne |

## English

The page runs locally. The start date is set in the sidebar, 2007 by default. The sidebar also sets the windows, the number of regimes and the method. Prices stay in cache for six hours. Clear cache downloads them again.

The top of the page shows the current regime and the index level. Below that: the curve, factor z-scores, today's loadings, a heatmap, the S&P 500 colored by regime, and a table of known stress dates (2008, 2011, 2015, 2018, 2020, 2022, 2024). A button exports the CSV. An optional series that no longer reaches the end is dropped, with a warning. S&P 500 and VIX are required.

Prices come from Yahoo Finance, adjusted.

| Name | Ticker | Reading |
|------|--------|---------|
| VIX | `^VIX` | implied volatility, as a level |
| Credit | HYG / IEF | minus the log of the ratio. Weaker high yield raises the factor |
| Curve | `^TNX` and `^IRX` | opposite of the 10-year minus 3-month slope. A flat or inverted curve raises the factor |
| RVol | `^GSPC` | S&P 500 realized volatility from log returns of printed sessions, rolling window, annualized |

The index rises when stress rises. Each factor is compared with earlier sessions: mean and standard deviation stop at yesterday, so today's shock does not build its own norm. Those z-scores are summarized by the first principal component, fit on past sessions only (default 504, near the 500 sessions used by the OFR). The VIX loading stays positive. Zero is the recent norm. Above zero, stress is higher than that norm.

The preferred regime model is a two or three state HMM. If the fit fails, the page uses a GMM, then quantiles. A short regime that has already ended is absorbed. The one still open stays on screen, even if it just started. HMM and GMM labels are fit on the whole loaded sample and can change when history grows. Quantile cuts use only the sessions known that day, including that day. A S&P 500 session with no print stays out of realized volatility. Past index values do not.

The OFR index uses 33 series. This page uses four. The shared idea is a level that rises with stress, loadings fit on the past, and no future session in today's calculation.

### Run

```bash
uv venv .venv
uv pip install -r requirements.txt -r requirements-dev.txt --python .venv/bin/python
.venv/bin/streamlit run mrrm.py
```

The page listens locally. Tests:

```bash
.venv/bin/pytest
```

The campaign, the techniques and the last run are in [tests/README.md](tests/README.md).

### Files

| File | Role |
|------|------|
| `mrrm.py` | Streamlit page |
| `mrrm_core.py` | download, index, regimes |
| `mrrm_plots.py` | figures |
| `tests/` | suite and campaign |
