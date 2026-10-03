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
