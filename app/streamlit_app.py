"""AFL Player Performance Analytics — Streamlit dashboard."""
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

from afl import charts  # noqa: E402
from afl.charts import STAT_LABELS, pretty_feature  # noqa: E402

PROC, RESULTS, REPORTS = ROOT / "data" / "processed", ROOT / "results", ROOT / "reports"
st.set_page_config(page_title="AFL Player Analytics", page_icon="🏉", layout="wide")
BASELINE = "Baseline: last-5 average"
st.markdown("<style>[data-testid='stMetricValue'] {font-size: 1.55rem;}</style>", unsafe_allow_html=True)


@st.cache_resource
def db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for view, file in (("pm", "app_player_matches"), ("matches", "matches"), ("preds", "predictions")):
        con.execute(f"CREATE VIEW {view} AS SELECT * FROM read_parquet('{(PROC / f'{file}.parquet').as_posix()}')")
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
    st.altair_chart(c, width="stretch")


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


# --------------------------------------------------------------------------- pages
def page_overview() -> None:
    st.title("Season overview")
    season = st.selectbox("Season", SEASONS[::-1])
    m = sql("SELECT * FROM matches WHERE season = ?", (season,))
    prev = sql("SELECT avg(home_score + away_score) AS s FROM matches WHERE season = ?", (season - 1,))["s"].iat[0]
    gf = m[m["round_label"] == "Grand Final"]
    lad = ladder(season)
    m["margin"] = (m["home_score"] - m["away_score"]).abs()
    big = m.loc[m["margin"].idxmax()]
    winner, loser = ((big.home_team, big.away_team) if big.home_score > big.away_score
                     else (big.away_team, big.home_team))

    k = st.columns(6)
    if not gf.empty:
        g = gf.iloc[0]
        w = g.home_team if g.home_score > g.away_score else g.away_team
        k[0].metric("Premiers", w, f"GF {max(g.home_score, g.away_score)}–{min(g.home_score, g.away_score)}",
                    delta_color="off", delta_arrow="off")
    k[1].metric("Minor premiers", lad["Team"].iat[0], f"{lad['Pts'].iat[0]} pts, {lad['%'].iat[0]}%", delta_color="off", delta_arrow="off")
    k[2].metric("Matches", len(m), f"{(~m['round_label'].str.isdigit()).sum()} finals", delta_color="off", delta_arrow="off")
    avg_total = m.eval("home_score + away_score").mean()
    k[3].metric("Points per match", f"{avg_total:.1f}",
                f"{avg_total - prev:+.1f} vs {season - 1}" if pd.notna(prev) else None)
    k[4].metric("Biggest win", f"{int(big.margin)} pts", f"{winner} d. {loser}", delta_color="off", delta_arrow="off")
    k[5].metric("Average crowd", f"{m['attendance'].mean():,.0f}", f"{m['attendance'].sum() / 1e6:.2f}M total",
                delta_color="off", delta_arrow="off")

    st.subheader(f"{season} ladder race")
    c1, c2 = st.columns([3, 1])
    c1.caption("Ladder position after each home-and-away round, computed from results (4 pts per win, percentage "
               "as tie-break). Hover a line to follow a team; pick a team to pin it.")
    hl = c2.selectbox("Highlight team", ["(none)"] + TEAMS, key="ov_hl",
                      index=1 + TEAMS.index(lad["Team"].iat[0]))
    chart(charts.ladder_bump(ladder_progression(season), None if hl == "(none)" else hl))

    l, r = st.columns(2)
    with l:
        st.subheader("How close were the games?")
        st.caption(f"Winning margins, {season}. Median {m['margin'].median():.0f} points; "
                   f"{(m['margin'] <= 12).mean():.0%} of matches decided by two goals or less.")
        chart(charts.histogram(m, "margin", "Winning margin (points)", 6))
    with r:
        st.subheader("Where the crowds went")
        crowd = (m.groupby("venue").agg(avg_crowd=("attendance", "mean"), games=("match_id", "count"))
                 .reset_index().query("games >= 3").nlargest(12, "avg_crowd"))
        st.caption("Average attendance by venue (venues with 3+ matches), as recorded by AFL Tables.")
        chart(charts.crowd_bars(crowd))

    st.subheader("How the game has changed, 2021–2026")
    trend = sql("""
        SELECT t.season, avg(t.pf) AS points, avg(s.disposals) AS disposals, avg(s.tackles) AS tackles,
               avg(s.contested_possessions) AS contested
        FROM team_games t JOIN (
            SELECT match_id, team, sum(disposals) disposals, sum(tackles) tackles,
                   sum(contested_possessions) contested_possessions
            FROM pm GROUP BY 1, 2) s USING (match_id, team)
        GROUP BY 1 ORDER BY 1""")
    cols = st.columns(4)
    st.caption("League averages per team per game, all matches. Each panel has its own y-axis scale.")
    for col, (y, title) in zip(cols, [("points", "Points"), ("disposals", "Disposals"), ("tackles", "Tackles"),
                                      ("contested", "Contested possessions")]):
        with col:
            chart(charts.season_trend(trend, y, title))


def page_teams() -> None:
    st.title("Team insights")
    c1, c2 = st.columns([1, 2])
    season = c1.selectbox("Season", SEASONS[::-1])
    lad = ladder(season)
    team = c2.selectbox("Focus team", TEAMS, index=TEAMS.index(lad["Team"].iat[0]))

    profile = sql(f"""
        SELECT team, count(DISTINCT match_id) games,
               {', '.join(f'sum({c}) / count(DISTINCT match_id) AS {c}' for c in STYLE_STATS)}
        FROM pm WHERE season = ? GROUP BY team""", (season,))
    against = sql("""
        SELECT opponent AS team, sum(inside_50s) / count(DISTINCT match_id) inside_50s_against,
               sum(disposals) / count(DISTINCT match_id) disposals_against,
               sum(goals) / count(DISTINCT match_id) goals_against
        FROM pm WHERE season = ? GROUP BY opponent""", (season,))
    prof = profile.merge(against, on="team")

    left, right = st.columns([5, 7])
    with left:
        st.subheader(f"{season} home-and-away ladder")
        st.caption("Computed from match results: 4 points per win, 2 per draw; ties broken by percentage (PF/PA).")
        st.dataframe(lad, width="stretch", height=36 * len(lad) + 40)
    with right:
        st.subheader("Team stat comparison")
        metric = st.selectbox("Statistic (per game)", STYLE_STATS, format_func=lambda s: STAT_LABELS[s])
        chart(charts.team_bars(prof, metric, f"{STAT_LABELS[metric]} per game", team))

    st.subheader("Team style fingerprint")
    st.caption("Per-game team totals (all matches incl. finals); the number in each cell is the raw value. Colour = "
               "standard deviations from the league average: blue above, red below. Teams in ladder order.")
    cols = STYLE_STATS + ["inside_50s_against", "goals_against"]
    long = prof.melt(id_vars="team", value_vars=cols, var_name="stat")
    long["z"] = long.groupby("stat")["value"].transform(lambda s: (s - s.mean()) / s.std())
    names = {**STAT_LABELS, "inside_50s_against": "I50 against", "goals_against": "Goals against",
             "contested_possessions": "Contested", "uncontested_possessions": "Uncontested",
             "contested_marks": "Cont. marks", "one_percenters": "1%ers", "rebound_50s": "Rebound 50s",
             "inside_50s": "Inside 50s"}
    long["stat"] = long["stat"].map(names)
    chart(charts.style_heatmap(long, lad["Team"].tolist()))

    st.subheader(f"{team}: match-by-match margins, {season}")
    g = sql("""SELECT row_number() OVER (ORDER BY date) AS n, round_label, opponent, venue, pf, pa, pf - pa AS margin
               FROM team_games WHERE season = ? AND team = ? ORDER BY date""", (season, team))
    g["result"] = np.select([g.margin > 0, g.margin < 0], ["Win", "Loss"], "Draw")
    g["score"] = g["pf"].astype(str) + "–" + g["pa"].astype(str)
    wins, losses = (g.result == "Win").sum(), (g.result == "Loss").sum()
    st.caption(f"{wins} wins, {losses} losses; average margin {g.margin.mean():+.1f}. Finals included. Hover a bar.")
    chart(charts.margin_bars(g, team))

    st.subheader("Territory battle")
    st.caption("Inside 50s for vs against per game. Bottom-right is best.")
    chart(charts.team_scatter(prof, "inside_50s", "inside_50s_against", "Inside 50s per game",
                              "Inside 50s conceded per game", team))
    with st.expander("Table view"):
        st.dataframe(prof.round(1).sort_values("inside_50s", ascending=False), width="stretch", hide_index=True)


def page_player() -> None:
    st.title("Player form trends")
    players = player_options()
    labels = label_for(players)
    ids = players["player_id"].tolist()
    c1, c2, c3 = st.columns([3, 2, 2])
    pid = c1.selectbox("Player", ids, format_func=labels.get, index=default_index(ids, "N/Nick_Daicos"))
    stat = c2.selectbox("Statistic", FORM_STATS, format_func=lambda s: STAT_LABELS[s])
    seasons = c3.multiselect("Seasons", SEASONS, default=SEASONS)
    df = sql("""SELECT p.*, CASE WHEN t.pf > t.pa THEN 'Win' WHEN t.pf < t.pa THEN 'Loss' ELSE 'Draw' END AS result
                FROM pm p JOIN team_games t USING (match_id, team) WHERE p.player_id = ? ORDER BY p.date""", (pid,))
    df = df[df["season"].isin(seasons)]
    if df.empty:
        st.info("No matches for this selection.")
        return
    label = STAT_LABELS[stat]
    last = df[df["season"] == df["season"].max()]
    ls = int(last["season"].iat[0])
    k = st.columns(6)
    k[0].metric("Matches", len(df))
    k[1].metric(f"{label} / game ({ls})", f"{last[stat].mean():.1f}",
                f"{last[stat].mean() - df[stat].mean():+.1f} vs selection avg")
    k[2].metric(f"Fantasy / game ({ls})", f"{last['fantasy_points'].mean():.1f}")
    k[3].metric(f"Best {label.lower()}", f"{df[stat].max():.0f}")
    k[4].metric("Goals (selection)", int(df["goals"].sum()))
    k[5].metric("Latest inferred role", df["role"].iat[-1])

    chart(charts.form_trend(df, stat))

    l, r = st.columns([5, 4])
    with l:
        pool_season = st.selectbox("Compare against league in season", sorted(df["season"].unique())[::-1],
                                   key="pf_season")
        pool = season_pool((int(pool_season),), min_games=10)
        if pid in pool.index:
            role = pool.at[pid, "role"]
            peers = pool[pool["role"] == role] if role != "Unknown" else pool
            pct = peers[PROFILE_STATS].rank(pct=True) * 100
            prof = pd.DataFrame({"stat": [STAT_LABELS[s] for s in PROFILE_STATS],
                                 "value": [peers.at[pid, s] for s in PROFILE_STATS],
                                 "pct": [pct.at[pid, s] for s in PROFILE_STATS]})
            st.subheader("Profile vs role peers")
            st.caption(f"Percentile rank among {len(peers)} players inferred as **{role}** with 10+ games in "
                       f"{pool_season}. Blue = above the median (dashed line).")
            chart(charts.percentile_bars(prof, "Percentile among role peers"))
        else:
            st.info(f"Fewer than 10 games in {pool_season}: no peer comparison.")
    with r:
        st.subheader(f"{label}: splits")
        split = st.radio("Split by", ["Opponent", "Home / away", "Result", "Venue"], horizontal=True)
        col = {"Opponent": "opponent", "Home / away": "where", "Result": "result", "Venue": "venue"}[split]
        d = df.assign(where=np.where(df["is_home"], "Home", "Away"))
        sp = d.groupby(col).agg(value=(stat, "mean"), games=(stat, "size")).reset_index()
        if col == "venue":
            sp = sp[sp["games"] >= 3]
        st.caption("Dashed orange line = overall average for the selection. Hover for games played.")
        chart(charts.split_bars(sp, col, label, df[stat].mean()))
        st.subheader("Distribution")
        chart(charts.histogram(df, stat, f"{label} per match", 2 if df[stat].max() > 15 else 1))

    preds = sql("SELECT * FROM preds WHERE player_id = ? ORDER BY date", (pid,))
    if not preds.empty:
        m = load_json(str(RESULTS / "metrics.json"))
        st.subheader("2026 out-of-sample predictions")
        tgt = st.radio("Target", ["disposals", "fantasy_points"], horizontal=True, format_func=lambda s: STAT_LABELS[s])
        model = m["targets"][tgt]["selected_model"]
        p = preds[preds["target"] == tgt]
        inside = ((p["actual"] >= p["pi_lower"]) & (p["actual"] <= p["pi_upper"])).mean()
        st.caption(f"Each prediction uses only matches before that game ({model}, trained on 2021–2025). "
                   f"This player's actual result fell inside the 80% interval in {inside:.0%} of 2026 matches; "
                   f"MAE {np.abs(p['actual'] - p[model]).mean():.1f} vs {np.abs(p['actual'] - p[BASELINE]).mean():.1f} "
                   f"for the last-5 baseline.")
        chart(charts.prediction_band(p, STAT_LABELS[tgt], model))

    st.subheader("Season averages")
    avg = df.groupby("season")[["disposals", "kicks", "handballs", "marks", "tackles", "goals", "clearances",
                                "contested_possessions", "inside_50s", "fantasy_points", "pct_time_played"]].mean()
    avg.insert(0, "games", df.groupby("season").size())
    st.dataframe(avg.round(1).rename(columns=STAT_LABELS), width="stretch")
    with st.expander("Match log"):
        st.dataframe(df[["date", "season", "round_label", "team", "opponent", "result", "disposals", "kicks",
                         "handballs", "marks", "tackles", "goals", "behinds", "clearances", "fantasy_points",
                         "pct_time_played", "sub_status"]].sort_values("date", ascending=False),
                     width="stretch", hide_index=True)


def page_h2h() -> None:
    st.title("Head-to-head")
    tab_p, tab_t = st.tabs(["Player vs player", "Team vs team"])
    with tab_p:
        players = player_options()
        labels = label_for(players)
        ids = players["player_id"].tolist()
        c1, c2, c3 = st.columns([3, 3, 2])
        p1 = c1.selectbox("Player A", ids, index=default_index(ids, "N/Nick_Daicos"), format_func=labels.get)
        p2 = c2.selectbox("Player B", ids, index=default_index(ids, "Z/Zach_Merrett", 1), format_func=labels.get)
        seasons = c3.multiselect("Seasons", SEASONS, default=SEASONS[-2:], key="h2h_seasons")
        if p1 == p2 or not seasons:
            st.info("Pick two different players and at least one season.")
        else:
            ph = ",".join("?" * len(seasons))
            df = sql(f"SELECT * FROM pm WHERE player_id IN (?, ?) AND season IN ({ph})", (p1, p2, *seasons))
            n1, n2 = labels[p1], labels[p2]
            df["player"] = df["player_id"].map({p1: n1, p2: n2})
            pool = season_pool(tuple(seasons), min_games=10)
            pool = pd.concat([pool, season_pool(tuple(seasons), min_games=1).loc[
                [p for p in (p1, p2) if p not in pool.index]]])
            pct = pool[PROFILE_STATS].rank(pct=True) * 100
            avg = pd.DataFrame([{"player": name, "stat": STAT_LABELS[s], "value": pool.at[pid_, s],
                                 "pct": pct.at[pid_, s]}
                                for pid_, name in ((p1, n1), (p2, n2)) for s in PROFILE_STATS])
            games = df.groupby("player").size()
            st.caption(f"Games in selection: {n1} {games.get(n1, 0)}, {n2} {games.get(n2, 0)}")
            l, r = st.columns([1, 1])
            with l:
                st.subheader("Per-game averages vs the league")
                st.caption(f"Percentile among {len(pool):,} players with 10+ games in the selected seasons; "
                           "hover for the raw per-game average.")
                chart(charts.head_to_head(avg, n1, n2))
            with r:
                st.subheader("Form over time")
                stat = st.selectbox("Statistic", FORM_STATS, format_func=lambda s: STAT_LABELS[s], key="h2h_stat")
                df = df.sort_values("date")
                df["rolling"] = df.groupby("player")[stat].transform(lambda s: s.rolling(5, min_periods=1).mean())
                chart(charts.overlay_trend(df, stat, n1, n2))
                st.subheader("Consistency")
                st.caption("Spread of single-match output: box = middle 50% of games, white line = median.")
                chart(charts.distribution_compare(df, stat, [n1, n2]))
            meet = sql("""
                SELECT a.date, a.season, a.round_label, a.team AS team_a, a.disposals AS disp_a, a.goals AS goals_a,
                       a.fantasy_points AS fantasy_a, b.team AS team_b, b.disposals AS disp_b, b.goals AS goals_b,
                       b.fantasy_points AS fantasy_b
                FROM pm a JOIN pm b ON a.match_id = b.match_id AND a.team <> b.team
                WHERE a.player_id = ? AND b.player_id = ? ORDER BY a.date DESC""", (p1, p2))
            st.subheader(f"Direct meetings ({len(meet)}, all seasons)")
            if meet.empty:
                st.caption("These players have not played against each other in 2021–2026.")
            else:
                st.caption(f"Disposals in meetings: {n1} averaged {meet.disp_a.mean():.1f}, {n2} {meet.disp_b.mean():.1f}.")
                st.dataframe(meet, width="stretch", hide_index=True)
            with st.expander("Table view"):
                st.dataframe(avg.pivot(index="stat", columns="player", values=["value", "pct"]).round(2), width="stretch")

    with tab_t:
        c1, c2 = st.columns(2)
        t1 = c1.selectbox("Team A", TEAMS, index=default_index(TEAMS, "Collingwood"))
        t2 = c2.selectbox("Team B", TEAMS, index=default_index(TEAMS, "Brisbane Lions", 1))
        games = sql("""SELECT row_number() OVER (ORDER BY date) AS n, date, season, round_label, venue, opponent,
                              pf, pa, pf - pa AS margin
                       FROM team_games WHERE team = ? AND opponent = ? ORDER BY date""", (t1, t2))
        if games.empty or t1 == t2:
            st.info("No matches between these teams in 2021–2026.")
        else:
            k = st.columns(4)
            k[0].metric("Meetings", len(games))
            k[1].metric(f"{t1} wins", int((games.margin > 0).sum()))
            k[2].metric(f"{t2} wins", int((games.margin < 0).sum()))
            k[3].metric(f"Avg margin ({t1})", f"{games.margin.mean():+.1f}")
            games["result"] = np.select([games.margin > 0, games.margin < 0], ["Win", "Loss"], "Draw")
            games["score"] = games["pf"].astype(str) + "–" + games["pa"].astype(str)
            chart(charts.margin_bars(games, t1))
            st.dataframe(games.drop(columns=["n", "opponent"]).sort_values("date", ascending=False),
                         width="stretch", hide_index=True)


def page_leaders() -> None:
    st.title("League leaders")
    c1, c2, c3 = st.columns([1, 2, 1])
    season = c1.selectbox("Season", SEASONS[::-1], key="ld_season")
    stat = c2.selectbox("Statistic", PROFILE_STATS, format_func=lambda s: STAT_LABELS[s], key="ld_stat")
    min_games = c3.slider("Minimum games", 1, 20, 10)
    pool = season_pool((season,), min_games=min_games).reset_index()
    pool["label"] = pool["player_name"] + " (" + pool["team"] + ")"

    l, r = st.columns(2)
    with l:
        st.subheader(f"{STAT_LABELS[stat]} per game")
        st.caption(f"Top 15, {season}, all matches incl. finals, {min_games}+ games.")
        chart(charts.leader_bars(pool.nlargest(15, stat), stat, f"{STAT_LABELS[stat]} per game"))
    with r:
        totals = sql("""
            SELECT any_value(player_name) || ' (' || arg_max(team, date) || ')' AS label, count(*) AS games,
                   sum(goals) AS goals, sum(brownlow_votes) AS votes
            FROM pm WHERE season = ? AND regexp_matches(round_label, '^[0-9]+$') GROUP BY player_id""", (season,))
        st.subheader("Goals, home-and-away")
        st.caption("Season totals, home-and-away matches only (as for the Coleman Medal).")
        chart(charts.leader_bars(totals.nlargest(10, "goals"), "goals", "Goals", ".0f"))
        st.subheader("Brownlow votes")
        st.caption("Umpires' 3-2-1 votes as recorded by AFL Tables (eligibility rules not applied).")
        chart(charts.leader_bars(totals.nlargest(10, "votes"), "votes", "Votes", ".0f"))

    st.subheader("Player landscape by inferred role")
    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("X axis", PROFILE_STATS, index=PROFILE_STATS.index("contested_possessions"),
                     format_func=lambda s: STAT_LABELS[s])
    y = c2.selectbox("Y axis", PROFILE_STATS, index=PROFILE_STATS.index("uncontested_possessions"),
                     format_func=lambda s: STAT_LABELS[s])
    roles = ["Midfielder", "Forward", "Defender", "Ruck", "Wing/Utility"]
    role = c3.selectbox("Highlight role", roles)
    pts = pool[pool["role"] != "Unknown"]
    st.caption(f"{len(pts)} players with {min_games}+ games in {season}; role = latest k-means role from the "
               "previous 10 matches. Top six of the highlighted role on the Y axis are labelled.")
    chart(charts.role_scatter(pts, x, y, role))


def page_models() -> None:
    st.title("Prediction models")
    m = load_json(str(RESULTS / "metrics.json"))
    st.markdown(
        "Predicting each player's **next-match disposals** and **AFL Fantasy points** from information available "
        "before the bounce. **Train** 2021–2024 · **validate/tune** 2025 · **test** 2026 (held out; not used for "
        "tuning or model selection).")
    tgt = st.radio("Target", list(m["targets"]), horizontal=True, format_func=lambda s: STAT_LABELS[s])
    res = m["targets"][tgt]
    best = res["selected_model"]
    tab = pd.DataFrame(res["test_2026"]).T.reset_index(names="model")
    base_rmse = tab.loc[tab["model"] == BASELINE, "rmse"].iat[0]
    sel_row = tab[tab["model"] == best].iloc[0]
    iv = res["interval_80"]
    k = st.columns(4)
    k[0].metric("Selected model", best)
    k[1].metric("Test RMSE", f"{sel_row.rmse:.2f}", f"{sel_row.rmse / base_rmse - 1:+.1%} vs last-5 baseline",
                delta_color="inverse")
    k[2].metric("Test MAE", f"{sel_row.mae:.2f}")
    k[3].metric("80% interval coverage", f"{iv['test_2026_coverage']:.1%}", f"avg width {iv['mean_width']:.1f}",
                delta_color="off", delta_arrow="off")

    c1, c2 = st.columns([3, 2])
    with c1:
        metric = st.radio("Metric", ["rmse", "mae"], horizontal=True, format_func=str.upper)
        chart(charts.model_metric_bars(tab, metric))
    with c2:
        tab["RMSE vs last-5 baseline"] = (tab["rmse"] / base_rmse - 1).map("{:+.1%}".format)
        st.dataframe(tab.drop(columns="n").rename(columns={"rmse": "RMSE", "mae": "MAE", "r2": "R²"})
                     .set_index("model").round(3), width="stretch")
        st.caption(f"{int(tab['n'].iat[0]):,} player-matches in the 2026 test season. Selected model = lowest "
                   "RMSE on the 2025 validation season.")

    preds = sql("SELECT * FROM preds WHERE target = ? AND NOT source_conflict", (tgt,))
    label = STAT_LABELS[tgt].lower()
    model = st.selectbox("Model for diagnostics", [c for c in charts.MODEL_ORDER if c in preds.columns],
                         index=charts.MODEL_ORDER.index(best))
    preds["residual"] = preds["actual"] - preds[model]
    st.caption("Left: density of every 2026 prediction against what happened. Right: predictions grouped into ten "
               "equal-sized bins; each dot is the bin's mean prediction vs mean actual, with a 95% confidence bar. "
               "Dots on the dashed line mean the model is unbiased across the whole range.")
    l, r = st.columns(2)
    with l:
        chart(charts.predicted_vs_actual(preds, model, label))
    with r:
        preds["decile"] = pd.qcut(preds[model], 10, labels=range(1, 11))
        cal = preds.groupby("decile", observed=True).agg(
            pred_mean=(model, "mean"), actual_mean=("actual", "mean"), sd=("actual", "std"), n=("actual", "size"))
        cal["lo"] = cal.actual_mean - 1.96 * cal.sd / np.sqrt(cal.n)
        cal["hi"] = cal.actual_mean + 1.96 * cal.sd / np.sqrt(cal.n)
        chart(charts.calibration(cal.reset_index(), model, label))

    l, r = st.columns(2)
    with l:
        st.subheader("Error across the 2026 season")
        rounds = preds[preds["round_label"].str.isdigit()].assign(round=lambda d: d.round_label.astype(int))
        er = pd.concat([rounds.groupby("round").apply(
            lambda d, c=c: pd.Series({"mae": np.abs(d.actual - d[c]).mean(), "n": len(d)}), include_groups=False)
            .reset_index().assign(model=c) for c in (model, BASELINE)])
        st.caption("Mean absolute error per home-and-away round: the model's edge over the baseline is consistent "
                   "rather than driven by a few rounds.")
        chart(charts.error_by_round(er, model))
    with r:
        st.subheader("Residuals")
        st.caption(f"Mean residual {preds.residual.mean():+.2f} (bias); "
                   f"{(preds.residual.abs() <= (5 if tgt == 'disposals' else 20)).mean():.0%} of predictions within "
                   f"{5 if tgt == 'disposals' else 20} {label}.")
        chart(charts.residual_hist(preds, label, 1 if tgt == "disposals" else 5))

    l, r = st.columns(2)
    rows = []
    for role, d in res["test_2026_by_role"].items():
        for name in (model, BASELINE):
            rows.append({"role": role, "model": name, "mae": d[name]["mae"], "n": d[name]["n"]})
    with l:
        st.subheader("Error by inferred role")
        chart(charts.error_by_group(pd.DataFrame(rows), "role", "Mean absolute error"))
    with r:
        st.subheader("Biggest surprises of 2026")
        st.caption("Largest gaps between actual output and the pre-match prediction.")
        cols = ["player_name", "team", "opponent", "round_label", "actual", model]
        tabs = st.tabs(["Outperformed", "Underperformed"])
        tabs[0].dataframe(preds.nlargest(10, "residual")[cols].round(1), width="stretch", hide_index=True)
        tabs[1].dataframe(preds.nsmallest(10, "residual")[cols].round(1), width="stretch", hide_index=True)

    with st.expander("Validation (2025) scores used for tuning, and interval method"):
        st.dataframe(pd.DataFrame(res["validation_2025"]).T.round(3), width="stretch")
        st.json({"tuning": res["tuning"], "interval_80": iv})


def page_importance() -> None:
    st.title("Feature importance")
    imp = pd.read_parquet(RESULTS / "feature_importance.parquet")
    tgt = st.radio("Target", imp["target"].unique().tolist(), horizontal=True, format_func=lambda s: STAT_LABELS[s])
    d = imp[imp["target"] == tgt].copy()
    d["family"] = d["feature"].map(charts.family_of)
    fam = d.groupby("family").agg(gain_share=("lgbm_gain_share", "sum"), features=("feature", "size")).reset_index()
    l, r = st.columns([4, 5])
    with l:
        st.subheader("What kind of information matters")
        st.caption("LightGBM split gain summed by feature family. Recent form carries most of the signal; "
                   "context features add a smaller, real increment.")
        chart(charts.family_bars(fam))
    with r:
        st.subheader("Top individual features")
        st.caption("Permutation importance on the 2026 test set: how much RMSE worsens when a feature is shuffled "
                   "(mean ± std over 5 repeats). Model-agnostic and measured on unseen data.")
        d["feature"] = d["feature"].map(pretty_feature)
        top = d.nlargest(1, "permutation_rmse_increase").iloc[0]
        zoom = st.toggle(f"Zoom in: hide the dominant feature ({top.feature}, +{top.permutation_rmse_increase:.2f} RMSE)")
        chart(charts.importance_bars(d[d["feature"] != top.feature] if zoom else d))
    with st.expander("All features (permutation, LightGBM gain share, CatBoost importance)"):
        st.dataframe(d.drop(columns="target").sort_values("permutation_rmse_increase", ascending=False).round(4),
                     width="stretch", hide_index=True)


def page_about() -> None:
    st.title("Data & methodology")
    st.markdown((ROOT / "docs" / "methodology.md").read_text(encoding="utf-8"))
    v = load_json(str(REPORTS / "validation.json"))
    st.subheader("Validation summary")
    st.json({k: v[k] for k in ("rows", "squiggle_match", "score_rebuild_mismatches",
                               "duplicate_player_match_rows", "unused_substitutes")})
    st.markdown("Full report: `reports/VALIDATION.md` in the repository.")


pg = st.navigation([
    st.Page(page_overview, title="Season overview", icon="🏆", url_path="overview", default=True),
    st.Page(page_teams, title="Team insights", icon="📊", url_path="teams"),
    st.Page(page_player, title="Player form", icon="📈", url_path="player"),
    st.Page(page_h2h, title="Head-to-head", icon="⚖️", url_path="head-to-head"),
    st.Page(page_leaders, title="League leaders", icon="🥇", url_path="leaders"),
    st.Page(page_models, title="Prediction models", icon="🎯", url_path="models"),
    st.Page(page_importance, title="Feature importance", icon="🔍", url_path="importance"),
    st.Page(page_about, title="Data & methodology", icon="📚", url_path="about"),
])
st.sidebar.caption("Data: AFL Tables (player stats) · Squiggle API (results cross-check). Seasons 2021–2026.")
pg.run()
