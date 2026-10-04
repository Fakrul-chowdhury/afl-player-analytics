"""League pages: season overview and league leaders."""
from __future__ import annotations

import streamlit as st

import common as c
import ui
from common import PROFILE_STATS, SEASONS, STAT_LABELS, TEAMS, charts


def page_overview() -> None:
    ui.header("Season overview", "Premiers, the ladder race, margins and crowds for any season, 2021–2026.")
    with ui.filter_bar():
        f = st.columns([1, 3])
        season = f[0].selectbox("Season", SEASONS[::-1])
    m = c.sql("SELECT * FROM matches WHERE season = ?", (season,))
    by_season = c.sql("""SELECT season, avg(home_score + away_score) AS pts, avg(attendance) AS crowd
                         FROM matches GROUP BY 1 ORDER BY 1""")
    upto = by_season[by_season["season"] <= season]
    prev = by_season.loc[by_season["season"] == season - 1, "pts"]
    gf = m[m["round_label"] == "Grand Final"]
    lad = c.ladder(season)
    m["margin"] = (m["home_score"] - m["away_score"]).abs()
    big = m.loc[m["margin"].idxmax()]
    winner, loser = ((big.home_team, big.away_team) if big.home_score > big.away_score
                     else (big.away_team, big.home_team))

    k = st.columns(6)
    if not gf.empty:
        g = gf.iloc[0]
        w = g.home_team if g.home_score > g.away_score else g.away_team
        ui.kpi(k[0], "Premiers", w, f"GF {max(g.home_score, g.away_score)}–{min(g.home_score, g.away_score)}",
               neutral=True)
    else:
        ui.kpi(k[0], "Premiers", "—", "Grand Final not in data", neutral=True)
    ui.kpi(k[1], "Minor premiers", lad["Team"].iat[0], f"{lad['Pts'].iat[0]} pts, {lad['%'].iat[0]}%", neutral=True)
    ui.kpi(k[2], "Matches", len(m), f"{(~m['round_label'].str.isdigit()).sum()} finals", neutral=True)
    avg_total = m.eval("home_score + away_score").mean()
    ui.kpi(k[3], "Points per match", f"{avg_total:.1f}",
           f"{avg_total - prev.iat[0]:+.1f} vs {season - 1}" if len(prev) else None, spark=upto["pts"].round(1),
           help="Sparkline: average points per match in each season up to the selected one.")
    ui.kpi(k[4], "Biggest win", f"{int(big.margin)} pts", f"{winner} d. {loser}", neutral=True)
    ui.kpi(k[5], "Average crowd", f"{m['attendance'].mean():,.0f}", f"{m['attendance'].sum() / 1e6:.2f}M total",
           neutral=True, spark=upto["crowd"].round(0),
           help="Sparkline: average recorded attendance in each season up to the selected one.")

    with ui.card(f"{season} ladder race"):
        a, b = st.columns([3, 1])
        a.caption("Ladder position after each home-and-away round, computed from results (4 pts per win, "
                  "percentage as tie-break). Hover a line to follow a team; pick a team to pin it.")
        hl = b.selectbox("Highlight team", ["(none)"] + TEAMS, key="ov_hl",
                         index=1 + TEAMS.index(lad["Team"].iat[0]))
        c.chart(charts.ladder_bump(c.ladder_progression(season), None if hl == "(none)" else hl))

    left, right = st.columns(2)
    with left, ui.card("How close were the games?",
                       f"Winning margins, {season}. Median {m['margin'].median():.0f} points; "
                       f"{(m['margin'] <= 12).mean():.0%} of matches decided by two goals or less."):
        c.chart(charts.histogram(m, "margin", "Winning margin (points)", 6))
    with right, ui.card("Where the crowds went",
                        "Average attendance by venue (venues with 3+ matches), as recorded by AFL Tables."):
        crowd = (m.groupby("venue").agg(avg_crowd=("attendance", "mean"), games=("match_id", "count"))
                 .reset_index().query("games >= 3").nlargest(12, "avg_crowd"))
        c.chart(charts.crowd_bars(crowd))

    trend = c.sql("""
        SELECT t.season, avg(t.pf) AS points, avg(s.disposals) AS disposals, avg(s.tackles) AS tackles,
               avg(s.contested_possessions) AS contested
        FROM team_games t JOIN (
            SELECT match_id, team, sum(disposals) disposals, sum(tackles) tackles,
                   sum(contested_possessions) contested_possessions
            FROM pm GROUP BY 1, 2) s USING (match_id, team)
        GROUP BY 1 ORDER BY 1""")
    with ui.card("How the game has changed, 2021–2026",
                 "League averages per team per game, all matches. Each panel has its own y-axis scale."):
        cols = st.columns(4)
        for col, (y, title) in zip(cols, [("points", "Points"), ("disposals", "Disposals"), ("tackles", "Tackles"),
                                          ("contested", "Contested possessions")]):
            with col:
                c.chart(charts.season_trend(trend, y, title))


def page_leaders() -> None:
    ui.header("League leaders", "Per-game and season-total leaderboards, plus every player mapped by role.")
    with ui.filter_bar():
        f = st.columns([1, 2, 1])
        season = f[0].selectbox("Season", SEASONS[::-1], key="ld_season")
        stat = f[1].selectbox("Statistic", PROFILE_STATS, format_func=lambda s: STAT_LABELS[s], key="ld_stat")
        min_games = f[2].slider("Minimum games", 1, 20, 10)
    pool = c.season_pool((season,), min_games=min_games).reset_index()
    pool["label"] = pool["player_name"] + " (" + pool["team"] + ")"
    totals = c.sql("""
        SELECT any_value(player_name) || ' (' || arg_max(team, date) || ')' AS label, count(*) AS games,
               sum(goals) AS goals, sum(brownlow_votes) AS votes
        FROM pm WHERE season = ? AND regexp_matches(round_label, '^[0-9]+$') GROUP BY player_id""", (season,))

    left, right = st.columns(2)
    with left, ui.card(f"{STAT_LABELS[stat]} per game",
                       f"Top 15, {season}, all matches incl. finals, {min_games}+ games."):
        c.chart(charts.leader_bars(pool.nlargest(15, stat), stat, f"{STAT_LABELS[stat]} per game"))
    with right:
        with ui.card("Goals, home-and-away", "Season totals, home-and-away matches only (as for the Coleman Medal)."):
            c.chart(charts.leader_bars(totals.nlargest(10, "goals"), "goals", "Goals", ".0f"))
        with ui.card("Brownlow votes", "Umpires' 3-2-1 votes as recorded by AFL Tables (eligibility rules not applied)."):
            c.chart(charts.leader_bars(totals.nlargest(10, "votes"), "votes", "Votes", ".0f"))

    with ui.card("Player landscape by inferred role"):
        f = st.columns(3)
        x = f[0].selectbox("X axis", PROFILE_STATS, index=PROFILE_STATS.index("contested_possessions"),
                           format_func=lambda s: STAT_LABELS[s])
        y = f[1].selectbox("Y axis", PROFILE_STATS, index=PROFILE_STATS.index("uncontested_possessions"),
                           format_func=lambda s: STAT_LABELS[s])
        roles = ["Midfielder", "Forward", "Defender", "Ruck", "Wing/Utility"]
        role = f[2].selectbox("Highlight role", roles)
        pts = pool[pool["role"] != "Unknown"]
        st.caption(f"{len(pts)} players with {min_games}+ games in {season}; role = latest k-means role from the "
                   "previous 10 matches. Top six of the highlighted role on the Y axis are labelled.")
        c.chart(charts.role_scatter(pts, x, y, role))
        csv = pool.drop(columns=["label"]).round(2).to_csv(index=False).encode()
        st.download_button("Download per-game table (CSV)", csv, f"afl_{season}_per_game.csv", "text/csv",
                           icon=":material/download:")
