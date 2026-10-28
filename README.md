# Macro Risk Regime Monitor (MRRM)

Dashboard de stress de marché. Quatre facteurs, un indice par ACP glissante, puis un régime.

Market stress dashboard. Four factors, a rolling-PCA index, then a regime.

## Français

L'indice du jour n'utilise que les données connues ce jour-là. Allonger l'historique ne réécrit pas les dates déjà calculées.

| Facteur | Série | Lecture |
|---------|--------|---------|
| VIX | `^VIX` | niveau |
| Credit | HYG / IEF | −log du ratio. Plus le crédit high yield est faible, plus le facteur monte |
| Curve | `^TNX` − `^IRX` | l'opposé de la pente 10 ans − 3 mois |
| RVol | `^GSPC` | vol réalisée, log-rendements, fenêtre glissante, annualisée |

Le z-score compare le jour à la fenêtre précédente (le jour n'entre pas dans sa norme). Les poids d'ACP sont ceux des séances passées (504 par défaut, proche des 500 séances de l'OFR). Le poids du VIX est forcé positif.

Le régime préféré est un HMM (Kritzman, Page et Turkington, 2012). S'il ne s'ajuste pas, la page passe au GMM, puis à des quantiles. L'historique des régimes est estimé sur l'échantillon chargé : il peut changer. L'indice, non.

```bash
uv venv .venv
uv pip install -r requirements.txt -r requirements-dev.txt --python .venv/bin/python
.venv/bin/streamlit run mrrm.py
.venv/bin/pytest
```

Campagne de test : [tests/README.md](tests/README.md).

## English

The index on a date uses only data known that day. Extending history does not rewrite earlier dates.

| Factor | Series | Reading |
|--------|--------|---------|
| VIX | `^VIX` | level |
| Credit | HYG / IEF | minus log of the ratio. Weaker high yield credit raises the factor |
| Curve | `^TNX` minus `^IRX` | opposite of the 10-year minus 3-month slope |
| RVol | `^GSPC` | realized vol from log returns, rolling window, annualized |

The z-score compares today with the previous window (today is not in its own norm). PCA loadings use past sessions only (default 504, near the OFR 500). The VIX loading is kept positive.

The preferred regime model is an HMM. If it does not fit, the page uses a GMM, then quantiles. Regime history is fit on the loaded sample and can change. The index does not.

```bash
uv venv .venv
uv pip install -r requirements.txt -r requirements-dev.txt --python .venv/bin/python
.venv/bin/streamlit run mrrm.py
.venv/bin/pytest
```

Test campaign: [tests/README.md](tests/README.md).
