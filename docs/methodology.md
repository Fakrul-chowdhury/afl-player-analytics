### Data

- **Player match statistics:** [AFL Tables](https://afltables.com/afl/afl_index.html) match pages for every AFL match from 2021 to 2026 (one page per match: 23 statistics per player plus age and career games).
- **Results cross-check:** the [Squiggle API](https://api.squiggle.com.au/), used to independently confirm every match's goals, behinds and final score.
- **Collection:** one request every 2.5 seconds with a descriptive User-Agent; every page is cached so it is fetched only once.
- **Data integrity:** nothing is imputed, simulated or padded.
  - Zero counts appear as blank cells on AFL Tables and are read as 0.
  - Named substitutes who never took the field (no time-on-ground figure) are kept in the raw table but excluded from modelling, because they have no performance to predict.
- **Anomaly handling (rule-based, `afl.clean`):**
  - A match whose score differs between AFL Tables and Squiggle keeps its AFL Tables values but is flagged `source_conflict` and left out of the evaluation metrics. One match is affected, by a single behind.
  - When a team in a match has more than one substitute-on marker, the substitution markers for that match are set to null (unknown). One match is affected.

### Derived quantities

- **AFL Fantasy points** are calculated from the real stats using the official scoring: kick 3, handball 2, mark 3, tackle 4, free for 1, free against −3, hitout 1, goal 6, behind 1.
- **Inferred role** (Ruck / Midfielder / Forward / Defender / Wing-Utility) is needed because no free source publishes positions.
  - It comes from k-means clustering (k = 5) on each player's *previous* 10-match profile: hitouts, goals, marks inside 50, rebound 50s, one-percenters, clearances, inside 50s, contested marks, tackles and uncontested possessions.
  - The clustering is fitted on the 2021–2024 training seasons only.
  - Players with fewer than 3 prior games are labelled *Unknown*.

### Features (all strictly pre-match)

Every rolling feature is computed on matches *before* the one being predicted (`shift(1)` within each player).

- **Player form:** rolling 3/5/10-match averages of 16 stats, a 10-match standard deviation, the last-match value and the season-to-date average.
- **Player profile:** age, career games before the match, games in the dataset, days since the last match, and whether the player was a substitute last match.
- **Context:** home/away, final, team, opponent and venue (categorical); the team's rolling disposals; and the opponent's rolling disposals and fantasy points conceded.

### Modelling

- **Targets:** next-match disposals and AFL Fantasy points.
- **Time-based split:**
  - Train on 2021–2024.
  - Tune on 2025 (Ridge alpha; early stopping for the boosted models).
  - Refit on 2021–2025 with the chosen settings.
  - Evaluate on the 2026 season. All tuning and model selection use 2025 only; see the README limitations for how often 2026 was scored.
- **Models:**
  - Ridge regression (one-hot categoricals, median imputation with missing-value indicators).
  - LightGBM, with a small grid over `num_leaves` × `min_child_samples`.
  - CatBoost.
  - An ensemble: the mean of the three models.
- **Model selection:** the model with the lowest RMSE on the 2025 validation season, chosen before the 2026 test season was scored.
- **Baselines:** the player's last-5-match average, and their season-to-date average. When a player has no history, the baselines fall back to the average debut output in the training data.
- **Metrics:** RMSE, MAE and R² on the 2026 test matches. Results are also reported by inferred role and for full-match established players.
- **Prediction intervals (80%):** empirical 10th/90th-percentile residuals of the selected model on 2025, computed within quintiles of the predicted value. Test coverage: 81.9% for disposals, 82.6% for fantasy points.

### Explaining individual predictions ("Why this prediction?")

- `afl.explain` refits the final LightGBM for each target with exactly the hyperparameters stored in `results/metrics.json` (same features, same 2021–2025 rows, same seed). Before saving anything it checks that the refit reproduces the LightGBM predictions already in `predictions.parquet`; the maximum difference is 0.0 for both targets, so **no model result changes**.
- Contributions come from LightGBM's built-in `pred_contrib=True` (TreeSHAP); no extra library is used. For every 2026 prediction, base value + contributions = the LightGBM prediction.
- To keep the deployed app small, each prediction stores its 12 largest contributions plus one row holding the exact sum of the other 56, so the waterfall still adds up.
- The headline model is the ensemble; only its LightGBM member is explained, and the page says so. Contributions describe what the model relied on, not what caused the result.

### Team strength (Elo) and home-ground advantage

- **Elo** is computed from every result (win = 1, draw = 0.5, loss = 0) in date order. Every team starts at 1,500 in round 1, 2021. Expected result = 1 / (1 + 10^(−(home − away + HGA)/400)); after each match the winner takes K × (result − expected) from the loser. Between seasons each rating keeps a fraction of its distance from 1,500.
- **K, the home bonus and the carry-over** were chosen by grid search (K 20–60, home bonus 0–120, carry-over 0.5–1.0) to minimise the Brier score on 2022–2024 (2021 is burn-in). Chosen: K = 40, home bonus = 60 Elo points, carry-over = 70%; the home-bonus optimum is inside the grid, not at its edge.
- **Out of sample (2025–2026, 434 matches):** Elo tips 70.7% of decided matches correctly vs 57.4% for always tipping the home team; Brier score 0.185 (0.25 = coin flip).
- **Home-ground advantage by team** = (average margin as the designated home team − average margin as the away team) ÷ 2, home-and-away rounds, with a 95% confidence interval from the standard error of the two means. "Home" is AFL Tables' designated home team, so shared grounds and relocated fixtures dilute the effect; most intervals overlap.
- The one match with a cross-source score conflict has the same winner in both sources, so Elo is unaffected; it is excluded from the margin-based analyses.

### What wins games

- One row per match (home minus away, so no match is double-counted), using the team totals printed on each AFL Tables match page.
- For every stat: Pearson correlation between the stat difference and the final margin, the least-squares slope (points of margin per unit of difference) and how often the team ahead on the stat won.
- Goals, behinds, rushed behinds and Brownlow votes are excluded because they *are* the score or are awarded after it. The results are associations, not causes (for example, rebound 50s correlate negatively because the team under pressure defends more).

### Player-level views

- **Similarity map:** per-game averages on 16 stats (fantasy points excluded because it is a weighted sum of the others) for players with at least N games in the chosen seasons, standardised, then projected onto the first two principal components with a NumPy SVD. "Most similar" uses Euclidean distance across all 16 standardised stats, not the 2-D picture; similarity = 1 / (1 + distance).
- **League spread:** every qualifying player's per-game average as a dot, one row per stat with its own scale. Vertical position is fixed random jitter with no meaning.
- **Age curves:** league mean and interquartile range of single-match output by whole year of age (ages with at least 150 player-matches), with the player's average at each age overlaid. Six seasons is a cross-section, not a career curve, and older ages contain only the survivors.
- **Form calendar:** a season × round grid coloured by the player's output. Qualifying and elimination finals share a column (same week). Rounds the player missed are drawn as empty dashed cells and nothing is filled in.
- **With / without:** for each season in which the player played at least once for their main club, the club's win rate and average margin in matches with and without them. Descriptive only: absences coincide with injuries, opposition and form, and samples without a regular player are often tiny.

### Not built: match momentum

Match momentum charts need the quarter-by-quarter scoring progression from the AFL Tables match pages. The processed tables store only final scores and player totals, and the raw HTML cache (`data/raw/`, git-ignored) was not available in the environment used for this update, so this view was not built rather than approximated. It can be added by extending `afl.parse` to read the scoring-progression table from the cached pages.

### Design

- The UI is inspired by [Analytics Dashboard by Lindsay (@lho)](https://www.figma.com/@lho), Figma Community, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): light grey canvas, white rounded cards with soft shadows, KPI tiles with sparklines, a filter bar, and a sidebar with a highlighted active item. Colours, type and spacing live in `.streamlit/config.toml` (separate light and dark themes) and `app/style.css`; layout helpers are in `app/ui.py`.
- **Chart colours were adapted, not copied.** The design's blue, orange-red and green were re-stepped to `#3d6be0`, `#e8603c` and `#17a673` and checked with a palette validator (OKLab lightness band, chroma floor, colour-vision-deficiency separation for every pair, normal-vision separation, and ≥ 3:1 contrast). All checks pass on both card surfaces (`#ffffff` light, `#1c2030` dark). The worst CVD pair (green vs coral, ΔE 8.7 deutan) is never the only cue: legends, labels and position always carry the meaning too.
- Diverging scales run coral → neutral grey → blue; sequential scales use one hue, light to dark, with separate steps for dark mode.
