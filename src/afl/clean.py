"""Apply documented, rule-based handling of anomalies found by validation.

Rules (detected from the data, never hard-coded to a match):
1. Source conflict: a match whose goals/behinds/score differ between AFL Tables and
   Squiggle keeps its AFL Tables values but is flagged `source_conflict = true`
   (matches and player rows). Flagged rows are excluded from model evaluation.
2. Unverifiable substitution markers: AFL rules allow one activated substitute per
   team. If any team in a match has more than one substitute-on marker, the
   markers for that whole match are set to null (unknown). Stats are untouched.

No statistic is changed, imputed or synthesised. Writes back to data/processed/
and records what was applied in reports/cleaning.json.
"""
from __future__ import annotations

import json

import polars as pl

from afl.config import PROCESSED, REPORTS
from afl.squiggle import fetch_games

SCORE_COLS = [("home_goals", "hgoals"), ("home_behinds", "hbehinds"), ("home_score", "hscore"),
              ("away_goals", "agoals"), ("away_behinds", "abehinds"), ("away_score", "ascore")]


def main() -> None:
    matches = pl.read_parquet(PROCESSED / "matches.parquet").drop("source_conflict", strict=False)
    players = pl.read_parquet(PROCESSED / "player_stats.parquet").drop("source_conflict", strict=False)
    sq = fetch_games()

    joined = matches.join(sq, left_on=["season", "home_team", "away_team"], right_on=["season", "hteam", "ateam"])
    joined = joined.filter((pl.col("date").dt.date() - pl.col("date_right")).dt.total_days().abs() <= 1)
    differs = pl.any_horizontal(pl.col(a) != pl.col(b) for a, b in SCORE_COLS)
    conflicts = joined.filter(differs).select(
        "match_id", "season", "round_label", "home_team", "away_team",
        *[c for pair in SCORE_COLS for c in pair])
    conflict_ids = conflicts["match_id"].to_list()

    multi_sub = (players.group_by("match_id", "team").agg((pl.col("sub_status") == "on").sum().alias("on"))
                 .filter(pl.col("on") > 1)["match_id"].unique().to_list())
    nulled = players.filter(pl.col("match_id").is_in(multi_sub) & pl.col("sub_status").is_not_null()).height

    matches = matches.with_columns(source_conflict=pl.col("match_id").is_in(conflict_ids))
    players = players.with_columns(
        source_conflict=pl.col("match_id").is_in(conflict_ids),
        sub_status=pl.when(pl.col("match_id").is_in(multi_sub)).then(None).otherwise(pl.col("sub_status")),
    )
    matches.write_parquet(PROCESSED / "matches.parquet")
    players.write_parquet(PROCESSED / "player_stats.parquet")

    report = {
        "source_conflict_matches": conflicts.to_dicts(),
        "source_conflict_player_rows": players.filter(pl.col("source_conflict")).height,
        "sub_markers_nulled_matches": sorted(multi_sub),
        "sub_markers_nulled_rows": nulled,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "cleaning.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "source_conflict_matches"}),
          f"conflicts={conflict_ids}")


if __name__ == "__main__":
    main()
