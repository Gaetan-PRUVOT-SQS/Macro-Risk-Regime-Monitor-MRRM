# Campagne de test

Campagne de test composant sur l'indice de stress et les régimes. Le cadre est l'ISTQB CTFL 4.0.1 et l'ISO/IEC/IEEE 29119, en version courte : les risques choisissent les techniques, les cas portent le modèle dans leur nom, le plan et le détail vivent avec les tests. Ce README sert à relancer la campagne et à lire le dernier passage.

Les tests restent dans `tests/`, pour le dépôt à venir.

## Lancer

Depuis la racine du projet, avec le virtualenv déjà installé :

```bash
.venv/bin/pytest
.venv/bin/python -m coverage run --branch -m pytest -q
.venv/bin/python -m coverage report --include='mrrm_core.py,mrrm_plots.py'
```

`pytest.ini` met la racine sur le `PYTHONPATH`. La suite ne téléphone pas à Yahoo. Un faux `download` remplace le réseau.

## Base de test

- Monin, The OFR Financial Stress Index, OFR Working Paper 17-04, 2017. Une date ne dépend pas de ce qui arrive après. Un indicateur monte quand le stress monte.
- Présentation OFR au FRAC, 20 juillet 2017. Les poids viennent des séances passées, puis on projette le jour.
- Kritzman, Page et Turkington, Financial Analysts Journal, 2012. Un régime de Markov tient mieux qu'un seuil fixe, à cause de la persistance.
- Le contrat du code : le z-score du jour exclut le jour, l'ACP aussi, le poids du VIX (puis RVol, Credit, Curve) reste positif, la tolérance d'une date de crise est un nombre de jours calendaires inclusif, la chaîne envoyée au téléchargement est une date `YYYY-MM-DD`.

## Risques

| Risque | Gravité | Réponse |
|--------|---------|---------|
| L'indice passé bouge quand le futur arrive, ou le jour entre dans sa propre norme | haute | propriété sur un préfixe, oracle chiffré du z-score, dos à dos avec sklearn sur plusieurs dates |
| Mauvais signe, mauvaise échelle de taux, facteur qui s'éteint trop tôt | haute | table de décision et valeur calculée à la main |
| Régime du jour effacé, ou une date de crise perdue au bord de l'échantillon | haute | transitions et limites, y compris juste hors de la plage |
| Date ou série hostile qui part telle quelle vers le téléchargement | moyenne | la date invalide n'appelle pas `download`, la date acceptée est canonisée |

Le niveau système (ouvrir la page Streamlit) n'est pas dans cette campagne.

## Techniques et cas

| Modèle | Technique | Fichier et cas |
|--------|-----------|----------------|
| M1 indice causal | métamorphique | `test_core.py` : `test_past_stress_ignores_future`, `test_last_day_does_not_rewrite_history` |
| M2 dernier score ACP | oracle sklearn | `test_last_score_matches_window_pca` |
| M3 facteurs | partitions, valeur connue | `test_credit_rises_when_hyg_falls`, `test_curve_shares_scale_when_bill_median_is_low`, `test_curve_percent_quote_not_rescaled`, `test_rvol_matches_annualized_log_std` |
| M4 régimes | table de décision | `test_detect_regimes_*`, repli HMM, GMM, quantile |
| M5 données | limites | médiane 15, trou de 5 et de 6, série morte, colonnes Yahoo |
| M6 dates | limites | écart de 5 jours gardé, écart de 6 retiré |
| M7 norme du jour | oracle à la main, métamorphique | `test_campagne.py` : `test_zscore_matches_previous_window`, `test_zscore_ignores_shift_and_scale`, `test_column_order_keeps_the_index` |
| M8 ACP intérieure | dos à dos sklearn | `test_interior_scores_match_window_pca` |
| M9 échelle des deux taux | table, MC/DC du `ou` | `test_yield_pair_scale_decision_table` (5 règles) |
| M10 fenêtres | MC/DC du `ou`, limite | `test_window_bounds_are_independent` |
| M11 durée d'un régime | transitions, MC/DC du `et` | flicker juste sous le minimum, pile au minimum, début court conservé, durée 1 |
| M12 quantiles | limites | `test_quantile_two_states_boundary`, `test_quantile_three_states_boundaries` |
| M13 bords de l'échantillon | limites | `test_events_just_outside_the_sample_stay_within_tolerance`, `test_zero_stress_is_not_above_average` |
| M14 entrées hostiles | supposition d'erreur | date, mois sans zéro, doublon, une seule jambe, z plat, signature yfinance, signe sans VIX, vol sur 2 séances, 1 état |

`coverage.py` ne mesure pas le MC/DC. Pour les trois décisions à plusieurs conditions, chaque condition a un cas qui change le résultat à elle seule.

## Dernier passage

2026-10-28, `.venv/bin/pytest` via coverage.

- 87 cas passés, 0 échoué, 0 sauté, 1,50 s.
- 48 avertissements, tous dans `hmmlearn` avec NumPy 2.5.
- Branches : `mrrm_core.py` 97 %, `mrrm_plots.py` 99 %, total 98 %.
- Compilation des trois modules et de `tests/test_campagne.py` : ok.
- Scan de la baseline sur le code et les deux fichiers de test : sortie 0.

Incident corrigé dans cette campagne. Une date de crise placée un jour avant la première séance, ou un jour après la dernière, disparaissait du tableau alors que la tolérance était de 1 jour. Le samedi situé entre les deux séances, lui, restait. Le cas a échoué avant le correctif (`1 failed` : il manquait `2020-01-02` et `2020-01-07`), puis la suite complète est passée. La fenêtre comparée à l'échantillon est élargie de la tolérance. L'écart en jours reste le juge.

Un autre rouge venait de l'oracle, pas du produit. `2020-1-01` est accepté et renvoyé en `2020-01-01`. C'est cette chaîne qui est donnée à yfinance. Le test vérifie ça.

## Ce qui reste

- L'historique des régimes est ajusté sur l'échantillon chargé. Il peut changer. L'indice, non.
- Quatre proxys de marché, pas les 33 séries de l'OFR.
- `mrrm.py` n'est pas dans la suite : l'importer lance Streamlit. La page n'a pas été ouverte.
- La tolérance compte des jours calendaires entiers.
- Lignes encore hors suite : import de hmmlearn manquant, série vide après conversion numérique, facteur sans colonne de signe, score ACP vide, séance introuvable pour une date. Une suite verte baisse la chance d'un défaut. Elle n'en prouve pas l'absence.
- Trois `except Exception` restent aux frontières téléchargement et modèle (`ruff` BLE001). Ils évitent qu'une série morte ou un ajustement raté ferment la page.

## English

Component test campaign for the stress index and the regimes. The frame is ISTQB CTFL 4.0.1 and ISO/IEC/IEEE 29119, kept short. This file is how to rerun the campaign and read the last run.

```bash
.venv/bin/pytest
```

The suite does not call Yahoo. A fake `download` stands in for the network.

Test basis: Monin, OFR Working Paper 17-04 (2017); OFR FRAC slides, 20 July 2017; Kritzman, Page and Turkington, FAJ 2012. A date uses only data known that day. The z-score and the PCA window both leave today out. The VIX loading stays positive. A crisis date is kept when a session falls inside the tolerance, including one sitting just before the first session or just after the last.

Last run, 2026-10-28: 87 passed, 0 failed, 0 skipped, 1.50s. Branch coverage 98% (`mrrm_core.py` 97%, `mrrm_plots.py` 99%). One product defect was fixed in this campaign: dates just outside the sample and still inside the tolerance were dropped. The failing test is `test_events_just_outside_the_sample_stay_within_tolerance`.

Regime history is fit on the loaded sample and can change. The index does not. The Streamlit page is not part of the suite.
