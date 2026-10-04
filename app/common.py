"""Shared data access and helpers for all dashboard pages."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afl import charts  # noqa: E402,F401
from afl.charts import STAT_LABELS, pretty_feature  # noqa: E402,F401

PROC, RESULTS, REPORTS = ROOT / "data" / "processed", ROOT / "results", ROOT / "reports"
BASELINE = "Baseline: last-5 average"


@st.cache_resource
def db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for view, file in (("pm", "app_player_matches"), ("matches", "matches"), ("preds", "predictions")):
        con.execute(f"CREATE VIEW {view} AS SELECT * FROM read_parquet('{(PROC / f'{file}.parquet').as_posix()}')")
    con.execute(f"CREATE VIEW explain AS SELECT * FROM read_parquet('{(PROC / 'explanations.parquet').as_posix()}')")
    # One row per team per match, with result from that team's perspective.
    con.execute("""
        CREATE VIEW team_games AS
        SELECT match_id, season, date, round_label, venue, attendance, home_team AS team, away_team AS opponent,
               home_score AS pf, away_score AS pa, TRUE AS is_home,
               regexp_matches(round_label, '^[0-9]+$') AS home_and_away FROM matches
        UNION ALL
        SELECT match_id, season, date, round_label, venue, attendance, away_team, home_team,
               away_score, home_score, FALSE, regexp_matches(round_label, '^[0-9]+$') FROM matches""")
    return con


@st.cache_data
def sql(query: str, params: tuple = ()) -> pd.DataFrame:
    return db().execute(query, list(params)).df()


@st.cache_data
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def chart(c) -> None:
    # Transparent background so charts sit directly on the card surface in both themes.
    st.altair_chart(c.configure(background="transparent"), width="stretch")


SEASONS = sql("SELECT DISTINCT season FROM matches ORDER BY 1")["season"].tolist()
TEAMS = sql("SELECT DISTINCT team FROM pm ORDER BY 1")["team"].tolist()
FORM_STATS = ["disposals", "fantasy_points", "kicks", "handballs", "marks", "tackles", "goals", "clearances",
              "contested_possessions", "inside_50s", "rebound_50s", "hitouts", "pct_time_played"]
PROFILE_STATS = ["disposals", "kicks", "handballs", "marks", "tackles", "goals", "goal_assists", "clearances",
                 "contested_possessions", "uncontested_possessions", "inside_50s", "rebound_50s",
                 "marks_inside_50", "contested_marks", "one_percenters", "hitouts", "fantasy_points"]
STYLE_STATS = ["disposals", "kicks", "handballs", "marks", "contested_possessions", "uncontested_possessions",
               "clearances", "inside_50s", "tackles", "rebound_50s", "one_percenters", "contested_marks", "goals"]


# --------------------------------------------------------------------------- shared helpers
@st.cache_data
def ladder(season: int) -> pd.DataFrame:
    df = sql("""
        SELECT team AS Team, count(*) AS P, sum((pf > pa)::INT) AS W, sum((pf < pa)::INT) AS L,
               sum((pf = pa)::INT) AS D, sum(pf) AS PF, sum(pa) AS PA,
               round(100.0 * sum(pf) / sum(pa), 1) AS "%",
               sum(CASE WHEN pf > pa THEN 4 WHEN pf = pa THEN 2 ELSE 0 END)::INT AS Pts
        FROM team_games WHERE season = ? AND home_and_away GROUP BY 1 ORDER BY Pts DESC, "%" DESC""", (season,))
    df.index = range(1, len(df) + 1)
    return df


@st.cache_data
def ladder_progression(season: int) -> pd.DataFrame:
    g = sql("SELECT team, CAST(round_label AS INT) AS round, pf, pa FROM team_games "
            "WHERE season = ? AND home_and_away", (season,))
    g["pts"] = np.select([g.pf > g.pa, g.pf == g.pa], [4, 2], 0)
    rows = []
    for r in sorted(g["round"].unique()):
        cum = g[g["round"] <= r].groupby("team")[["pts", "pf", "pa"]].sum().reindex(TEAMS, fill_value=0)
        cum["pct"] = np.where(cum["pa"] > 0, 100 * cum["pf"] / cum["pa"].where(cum["pa"] > 0, 1), 0.0)
        cum = cum.sort_values(["pts", "pct"], ascending=False)
        cum["position"] = range(1, len(cum) + 1)
        rows.append(cum.reset_index(names="team").assign(round=r))
    return pd.concat(rows)


def player_options() -> pd.DataFrame:
    return sql("""
        SELECT player_id, any_value(player_name) AS player_name, arg_max(team, date) AS team, count(*) AS games,
               max(season) AS last_season
        FROM pm GROUP BY player_id ORDER BY games DESC""")


def label_for(players: pd.DataFrame) -> dict:
    dup = players["player_name"].duplicated(keep=False)
    return {r.player_id: f"{r.player_name} ({r.team}{', ' + r.player_id.split('/')[-1] if d else ''})"
            for r, d in zip(players.itertuples(), dup)}


@st.cache_data
def season_pool(seasons: tuple, min_games: int = 10) -> pd.DataFrame:
    """Per-game averages for every player in the given seasons (with latest team and inferred role)."""
    ph = ",".join("?" * len(seasons))
    df = sql(f"""
        SELECT player_id, any_value(player_name) AS player_name, arg_max(team, date) AS team,
               arg_max(role, date) AS role, count(*) AS games,
               {', '.join(f'avg({c}) AS {c}' for c in PROFILE_STATS)}
        FROM pm WHERE season IN ({ph}) GROUP BY player_id""", seasons)
    return df[df["games"] >= min_games].set_index("player_id")


def default_index(options: list, value, fallback: int = 0) -> int:
    return options.index(value) if value in options else fallback


def player_picker(container, key: str, default: str = "N/Nick_Daicos", label: str = "Player") -> str:
    players = player_options()
    labels = label_for(players)
    ids = players["player_id"].tolist()
    return container.selectbox(label, ids, format_func=labels.get, index=default_index(ids, default), key=key)


@st.cache_data
def table(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROC / f"{name}.parquet")


@st.cache_data
def elo() -> tuple[pd.DataFrame, dict]:
    from afl import insights
    return insights.elo_ratings(table("matches"))


def is_home_and_away(round_label: pd.Series) -> pd.Series:
    return round_label.str.fullmatch(r"\d+")


def stat_name(s: str) -> str:
    return STAT_LABELS.get(s, s.replace("_", " ").capitalize())
