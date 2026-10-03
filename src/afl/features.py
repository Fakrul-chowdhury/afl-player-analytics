"""Feature engineering. Every feature for a match uses only matches played before it.

Output: data/processed/features.parquet (one row per player-match, with targets).
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from afl.config import PROCESSED, RESULTS

TRAIN_END = 2024  # role clustering is fitted on 2021-2024 only (the training period)

# Official AFL Fantasy scoring weights, applied to the real recorded stats.
FANTASY_WEIGHTS = {"kicks": 3, "handballs": 2, "marks": 3, "tackles": 4, "frees_for": 1,
                   "frees_against": -3, "hitouts": 1, "goals": 6, "behinds": 1}
FORM_STATS = ["disposals", "kicks", "handballs", "marks", "tackles", "clearances",
              "contested_possessions", "uncontested_possessions", "inside_50s", "goals",
              "hitouts", "rebound_50s", "marks_inside_50", "one_percenters", "fantasy_points",
              "pct_time_played"]
WINDOWS = (3, 5, 10)
ROLE_STATS = ["hitouts", "goals", "marks_inside_50", "rebound_50s", "one_percenters", "clearances",
              "inside_50s", "contested_marks", "tackles", "uncontested_possessions"]
TARGETS = ["disposals", "fantasy_points"]


def add_fantasy(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        fantasy_points=pl.sum_horizontal(pl.col(k) * w for k, w in FANTASY_WEIGHTS.items()))


def _prior(col: str, w: int, fn: str = "mean") -> pl.Expr:
    base = pl.col(col).shift(1)
    roll = base.rolling_mean(w, min_samples=1) if fn == "mean" else base.rolling_std(w, min_samples=3)
    return roll.over("player_id")


def player_form(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort("date", "match_id")
    exprs = [_prior(s, w).alias(f"{s}_avg{w}") for s in FORM_STATS for w in WINDOWS]
    exprs += [_prior(t, 10, "std").alias(f"{t}_std10") for t in TARGETS]
    exprs += [pl.col(t).shift(1).over("player_id").alias(f"{t}_last") for t in TARGETS]
    exprs += [
        pl.col(t).shift(1).cum_sum().over("player_id", "season")
        .truediv(pl.int_range(pl.len()).over("player_id", "season")).alias(f"{t}_season_avg")
        for t in TARGETS
    ]
    exprs += [
        pl.int_range(pl.len()).over("player_id").alias("games_in_data"),
        (pl.col("career_games") - 1).alias("career_games_before"),
        (pl.col("age_days") / 365.25).alias("age_years"),
        (pl.col("date") - pl.col("date").shift(1)).over("player_id").dt.total_days().alias("days_since_last"),
        pl.col("sub_status").shift(1).over("player_id").is_in(["on", "off"]).fill_null(False).alias("sub_last_match"),
    ]
    return df.with_columns(exprs).with_columns(
        pl.col("disposals_season_avg", "fantasy_points_season_avg").fill_nan(None))


def team_context(df: pl.DataFrame) -> pl.DataFrame:
    """Rolling 5-match team output and opponent concessions (prior matches only)."""
    tm = df.group_by("match_id", "date", "team", "opponent").agg(
        team_disp=pl.col("disposals").sum(), team_fp=pl.col("fantasy_points").sum())
    tm = tm.join(tm.select("match_id", pl.col("team").alias("opponent"),
                           pl.col("team_disp").alias("opp_disp"), pl.col("team_fp").alias("opp_fp")),
                 on=["match_id", "opponent"]).sort("date", "match_id")
    roll = lambda c: pl.col(c).shift(1).rolling_mean(5, min_samples=1).over("team")  # noqa: E731
    tm = tm.with_columns(team_disp_for5=roll("team_disp"), team_disp_against5=roll("opp_disp"),
                         team_fp_against5=roll("opp_fp"))
    own = tm.select("match_id", "team", "team_disp_for5")
    opp = tm.select("match_id", pl.col("team").alias("opponent"),
                    pl.col("team_disp_against5").alias("opp_disp_conceded5"),
                    pl.col("team_fp_against5").alias("opp_fp_conceded5"))
    return df.join(own, on=["match_id", "team"], how="left").join(opp, on=["match_id", "opponent"], how="left")


def infer_roles(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """K-means on each player's prior 10-match stat profile. Fitted on training seasons only."""
    prof_cols = [f"_r_{s}" for s in ROLE_STATS]
    df = df.sort("date", "match_id", "team", "player_name").with_columns(
        [pl.col(s).shift(1).rolling_mean(10, min_samples=3).over("player_id").alias(f"_r_{s}") for s in ROLE_STATS])
    has = pl.all_horizontal(pl.col(c).is_not_null() for c in prof_cols)
    fit = df.filter(has & (pl.col("season") <= TRAIN_END)).select(prof_cols).to_numpy()
    scaler = StandardScaler().fit(fit)
    km = KMeans(n_clusters=5, n_init=20, random_state=42).fit(scaler.transform(fit))
    cent = scaler.inverse_transform(km.cluster_centers_)
    c = {s: cent[:, i] for i, s in enumerate(ROLE_STATS)}
    names, left = {}, set(range(5))
    for label, score in [("Ruck", c["hitouts"]),
                         ("Forward", c["goals"] + c["marks_inside_50"]),
                         ("Defender", c["rebound_50s"] + c["one_percenters"]),
                         ("Midfielder", c["clearances"])]:
        k = max(left, key=lambda i: score[i])
        names[k] = label
        left.remove(k)
    names[left.pop()] = "Wing/Utility"

    X = df.filter(has).select(prof_cols).to_numpy()
    labels = km.predict(scaler.transform(X))
    role = np.full(df.height, "Unknown", dtype=object)
    role[np.flatnonzero(df.select(has).to_series().to_numpy())] = [names[l] for l in labels]
    centroids = {names[k]: {s: round(float(c[s][k]), 2) for s in ROLE_STATS} for k in range(5)}
    return df.with_columns(role=pl.Series(role.tolist())).drop(prof_cols), centroids


def main() -> None:
    # Unused substitutes (listed, never took the field) have no performance to model.
    df = add_fantasy(pl.read_parquet(PROCESSED / "player_stats.parquet").filter(pl.col("took_field")))
    df = player_form(df)
    df = team_context(df)
    df, centroids = infer_roles(df)
    df = df.with_columns(is_final=~pl.col("round_label").str.contains(r"^\d+$"))
    df.write_parquet(PROCESSED / "features.parquet")
    # Slim copy for the dashboard (keeps the deployed repo small).
    stats = [c for c in pl.read_parquet_schema(PROCESSED / "player_stats.parquet") if c not in ("took_field",)]
    df.select(stats + ["fantasy_points", "role", "is_final"]).write_parquet(PROCESSED / "app_player_matches.parquet")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "role_centroids.json").write_text(json.dumps(centroids, indent=2))
    print(df.shape, df.group_by("role").len().sort("len").to_dicts())


if __name__ == "__main__":
    main()
