# Campagne de test

Campagne de test composant sur l'indice de stress et les régimes. Le cadre est l'ISTQB CTFL 4.0.1 et l'ISO/IEC/IEEE 29119, en version courte : les risques choisissent les techniques, les cas portent le modèle dans leur nom, le plan et le détail vivent avec les tests. Ce README sert à relancer la campagne et à lire le dernier passage.

Les tests restent dans `tests/`.

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
- Présentation OFR au FRAC, 20 juillet 2017. Les poids viennent des séances passées. Le jour est projeté ensuite.
- Kritzman, Page et Turkington, Financial Analysts Journal, 2012. Un régime de Markov tient mieux qu'un seuil fixe, à cause de la persistance.
- Holló, Kremer et Lo Duca, ECB Working Paper 1426, 2012. La distribution connue à la date sert aux quantiles. L'indice de la page reste l'ACP glissante de l'OFR.
- Le contrat du code : le z-score du jour exclut le jour, l'ACP aussi. Le poids du VIX (puis RVol, Credit, Curve) reste positif. La tolérance d'une date de crise est un nombre de jours calendaires inclusif. La chaîne envoyée au téléchargement est une date `YYYY-MM-DD`. Une séance SPX sans cours, ou avec un cours nul, sort de la volatilité. Les niveaux (VIX, crédit, pente) gardent le dernier cours connu, cinq séances au plus. Le seuil Quantile d'une date est celui du préfixe qui finit ce jour-là, aujourd'hui inclus. HMM et GMM restent ajustés sur tout l'échantillon chargé.

## Risques

| Risque | Gravité | Réponse |
|--------|---------|---------|
| L'indice passé bouge quand le futur arrive, ou le jour entre dans sa propre norme | haute | propriété sur un préfixe, oracle chiffré du z-score, dos à dos avec sklearn sur plusieurs dates |
| Mauvais signe, mauvaise échelle de taux, facteur qui s'éteint trop tôt | haute | table de décision et valeur calculée à la main |
| Une séance SPX manquante, ou un cours nul, compte comme un rendement nul | haute | oracle des log-rendements observés, limite de cinq séances |
| Un choc futur reclasse un Quantile déjà passé, ou l'égalité au seuil change de camp | haute | préfixe, échantillon plat, repli GMM sur les mêmes bornes |
| Régime du jour effacé, ou une date de crise perdue au bord de l'échantillon | haute | transitions et limites, y compris juste hors de la plage |
| Date ou série hostile qui part telle quelle vers le téléchargement | moyenne | la date invalide n'appelle pas `download`, la date acceptée est canonisée |

Le niveau système (ouvrir la page Streamlit) reste hors de cette campagne.

## Techniques et cas

| Modèle | Technique | Fichier et cas |
|--------|-----------|----------------|
| M1 indice causal | métamorphique | `test_core.py` : `test_past_stress_ignores_future`, `test_last_day_does_not_rewrite_history` |
| M2 dernier score ACP | oracle sklearn | `test_last_score_matches_window_pca` |
| M3 facteurs | partitions, valeur connue | `test_credit_rises_when_hyg_falls`, `test_curve_shares_scale_when_bill_median_is_low`, `test_curve_percent_quote_not_rescaled`, `test_rvol_matches_annualized_log_std` |
| M4 régimes | table de décision | `test_detect_regimes_*`, repli HMM et GMM |
| M5 données | limites | médiane 15, trou de niveau de 5 et de 6, série morte, colonnes Yahoo, `test_level_gap_stays_filled_beside_a_missing_spx_session` |
| M6 dates | limites | écart de 5 jours gardé, écart de 6 retiré |
| M7 norme du jour | oracle à la main, métamorphique | `test_campagne.py` : `test_zscore_matches_previous_window`, `test_zscore_ignores_shift_and_scale`, `test_column_order_keeps_the_index` |
| M8 ACP intérieure | dos à dos sklearn | `test_interior_scores_match_window_pca` |
| M9 échelle des deux taux | table, MC/DC du `ou` | `test_yield_pair_scale_decision_table` (5 règles) |
| M10 fenêtres | MC/DC du `ou`, limite | `test_window_bounds_are_independent` |
| M11 durée d'un régime | transitions, MC/DC du `et` | flicker juste sous le minimum, pile au minimum, début court conservé, durée 1, `test_duration_filter_rewrites_a_finished_quantile_day` |
| M12 quantiles | limites, métamorphique | `test_quantile_two_states_boundary`, `test_quantile_three_states_boundaries`, `test_quantile_prefix_ignores_later_shock`, `test_quantile_constant_sample_stays_on_the_cut`, `test_quantile_ignores_a_missing_stress_point`, `test_gmm_failure_uses_the_same_prefix_cuts` |
| M13 bords de l'échantillon | limites | `test_events_just_outside_the_sample_stay_within_tolerance`, `test_zero_stress_is_not_above_average` |
| M14 entrées hostiles | supposition d'erreur | date, mois sans zéro, doublon, une seule jambe, z plat, signature yfinance, signe sans VIX, vol sur 2 séances, 1 état |
| M15 séances de vol | oracle à la main, supposition d'erreur | `test_missing_spx_session_is_not_a_zero_return`, `test_five_missing_spx_sessions_stay_out_of_vol`, `test_non_positive_spx_print_is_left_out_of_vol` |

`coverage.py` ne mesure pas le MC/DC. Pour les trois décisions à plusieurs conditions, chaque condition a un cas qui change le résultat à elle seule.

## Dernier passage

2026-10-28, `.venv/bin/python -m coverage run --branch -m pytest -q`, puis `coverage report -m --include='mrrm_core.py,mrrm_plots.py'`.

- 96 cas passés, 0 échoué, 0 sauté, 1,78 s.
- 48 avertissements, tous dans `hmmlearn` avec NumPy 2.5.
- Branches : `mrrm_core.py` 98 % (342 instructions, 7 manquées, 144 branches, 5 partielles), `mrrm_plots.py` 99 %, total 98 %.
- Lignes encore ouvertes : `mrrm_core.py` 16-18, 171, 232, 325, 528, branchement 76→68. `mrrm_plots.py` 125→123.
- Compilation de `mrrm.py`, `mrrm_core.py`, `mrrm_plots.py`, `tests/test_campagne.py` et `tests/test_core.py` : ok.
- Scan de la baseline sur `mrrm_core.py`, `mrrm.py`, `mrrm_plots.py`, `tests/test_campagne.py`, `tests/README.md` et `README.md` : sortie 0.
- `ruff check tests/test_campagne.py` : ok. Les quatre `BLE001` déjà connus restent dans `mrrm_core.py` et `mrrm.py`.

Les deux cas de vol et de préfixe Quantile ont d'abord échoué sur l'ancien calcul. Le trou du 17 janvier 2020 restait dans le cadre, cours recopié. Le préfixe Quantile changeait dès qu'un choc était collé à la fin. Après le correctif, ces quatre cas et la suite passaient (89 cas, 1,52 s).

Sept cas voisins complètent le même contrat : cinq séances SPX vides, cours SPX nul, trou de niveau à côté d'un trou SPX, échantillon plat, trou dans la série de stress, filtre de durée sur un vrai jour Quantile, repli GMM sur les bornes du préfixe. Écrits après le correctif, ils sont verts au premier passage, avec le reste de la suite (96 cas).

Incident corrigé plus tôt dans cette campagne. Une date de crise placée un jour avant la première séance, ou un jour après la dernière, disparaissait du tableau alors que la tolérance était de 1 jour. Le samedi situé entre les deux séances restait. Le cas a échoué avant le correctif (`1 failed` : il manquait `2020-01-02` et `2020-01-07`), puis la suite complète est passée. La fenêtre comparée à l'échantillon est élargie de la tolérance. L'écart en jours reste le juge.

Un autre rouge venait de l'oracle. `2020-1-01` est accepté et renvoyé en `2020-01-01`. Cette chaîne part vers yfinance. Le test vérifie ça.

## Ce qui reste

- HMM et GMM sont ajustés sur l'échantillon chargé. L'historique de ces deux méthodes peut changer quand la période chargée s'allonge.
- Le seuil Quantile d'une date est celui du préfixe qui finit ce jour-là, aujourd'hui inclus.
- Le filtre de durée peut encore absorber un segment Quantile déjà fini quand la durée minimale dépasse 1. Le cas `test_duration_filter_rewrites_a_finished_quantile_day` fixe ce reste : sur `[5, 1, 1]`, la durée 1 donne Stress, Risk-On, Stress, et la durée 2 donne trois Stress.
- Quatre proxys de marché. L'OFR en publie 33. L'indice de la page reste l'ACP glissante.
- `mrrm.py` reste hors de la suite : l'importer lance Streamlit. La page n'a pas été ouverte.
- La tolérance compte des jours calendaires entiers.
- Lignes encore hors suite : import de hmmlearn manquant, série vide après conversion numérique, facteur sans colonne de signe, score ACP vide, séance introuvable pour une date. Une suite verte baisse la chance d'un défaut. Elle n'en prouve pas l'absence.
- Quatre `except Exception` restent aux frontières : téléchargement, ajustement HMM, repli GMM, page (`ruff` BLE001). Ils évitent qu'une série morte ou un ajustement raté ferment la page.

## English

Component test campaign for the stress index and the regimes. The frame is ISTQB CTFL 4.0.1 and ISO/IEC/IEEE 29119, kept short. The table above is the campaign. This section restates the contract and the last run.

```bash
.venv/bin/pytest
```

The suite does not call Yahoo. A fake `download` stands in for the network.

Test basis: Monin, OFR Working Paper 17-04 (2017); OFR FRAC slides, 20 July 2017; Kritzman, Page and Turkington, FAJ 2012; Holló, Kremer and Lo Duca, ECB Working Paper 1426 (2012), for the distribution known on that date. The page index stays the rolling OFR-style PCA.

A date uses only data known that day. The z-score and the PCA window both leave today out. The VIX loading stays positive. Realized volatility uses SPX sessions that printed a positive price. A missing session, and a zero print, stay out of that volatility. Level series (VIX, credit, curve) still keep the last known print, five sessions at most. Quantile cuts use the prefix that ends that day, that day included. HMM and GMM stay fit on the loaded sample. A crisis date is kept when a session falls inside the tolerance, including one sitting just before the first session or just after the last.

Models M12 and M15 in the table cover the quantile prefix and the printed sessions. M11 also records that the duration filter can still absorb a finished short quantile day.

Last run, 2026-10-28: 96 passed, 0 failed, 0 skipped, 1.78s. 48 warnings, all inside `hmmlearn` on NumPy 2.5. Branch coverage 98% (`mrrm_core.py` 98%, 342 statements, 7 missed, 144 branches, 5 partial; `mrrm_plots.py` 99%). Open lines: `mrrm_core.py` 16-18, 171, 232, 325, 528, branch 76→68; `mrrm_plots.py` 125→123.

Two earlier product defects stay in the suite. Dates just outside the sample and still inside the tolerance were dropped (`test_events_just_outside_the_sample_stay_within_tolerance`). A missing SPX session was copied forward and a later shock relabeled past quantile days. Those four cases failed on the old calculation, then the suite passed (89 cases, 1.52s). Seven neighbor cases of the same contract are green on their first run, inside the 96.

HMM and GMM history can change when the loaded sample grows. Quantile cuts do not. The duration filter can still rewrite a finished short quantile segment when the minimum duration is above 1. The Streamlit page is not part of the suite. Four `except Exception` boundaries remain (download, HMM fit, GMM fallback, page).
