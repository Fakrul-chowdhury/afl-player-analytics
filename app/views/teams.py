"""Team pages: team insights, team strength (Elo + home advantage), what wins games."""
from __future__ import annotations

import numpy as np
import streamlit as st

import common as c
import ui
from afl import insights
from common import SEASONS, STAT_LABELS, STYLE_STATS, TEAMS, charts


def page_teams() -> None:
    ui.header("Team insights", "Ladder, style fingerprint, match margins and territory for every club.")
    with ui.filter_bar():
        f = st.columns([1, 2, 2])
        season = f[0].selectbox("Season", SEASONS[::-1])
        lad = c.ladder(season)
        team = f[1].selectbox("Focus team", TEAMS, index=TEAMS.index(lad["Team"].iat[0]))
        metric = f[2].selectbox("Statistic (per game)", STYLE_STATS, format_func=lambda s: STAT_LABELS[s])

    profile = c.sql(f"""
        SELECT team, count(DISTINCT match_id) games,
               {', '.join(f'sum({s}) / count(DISTINCT match_id) AS {s}' for s in STYLE_STATS)}
        FROM pm WHERE season = ? GROUP BY team""", (season,))
    against = c.sql("""
        SELECT opponent AS team, sum(inside_50s) / count(DISTINCT match_id) inside_50s_against,
               sum(disposals) / count(DISTINCT match_id) disposals_against,
               sum(goals) / count(DISTINCT match_id) goals_against
        FROM pm WHERE season = ? GROUP BY opponent""", (season,))
    prof = profile.merge(against, on="team")

    left, right = st.columns([5, 7])
    with left, ui.card(f"{season} home-and-away ladder",
                       "Computed from match results: 4 points per win, 2 per draw; ties broken by percentage."):
        st.dataframe(lad, width="stretch", height=36 * len(lad) + 40)
    with right, ui.card("Team stat comparison", f"{STAT_LABELS[metric]} per game, {season}; {team} highlighted."):
        c.chart(charts.team_bars(prof, metric, f"{STAT_LABELS[metric]} per game", team))

    with ui.card("Team style fingerprint",
                 "Per-game team totals (all matches incl. finals); the number in each cell is the raw value. Colour = "
                 "standard deviations from the league average: blue above, coral below. Teams in ladder order."):
        cols = STYLE_STATS + ["inside_50s_against", "goals_against"]
        long = prof.melt(id_vars="team", value_vars=cols, var_name="stat")
        long["z"] = long.groupby("stat")["value"].transform(lambda s: (s - s.mean()) / s.std())
        names = {**STAT_LABELS, "inside_50s_against": "I50 against", "goals_against": "Goals against",
                 "contested_possessions": "Contested", "uncontested_possessions": "Uncontested",
                 "contested_marks": "Cont. marks", "one_percenters": "1%ers", "rebound_50s": "Rebound 50s",
                 "inside_50s": "Inside 50s"}
        long["stat"] = long["stat"].map(names)
        c.chart(charts.style_heatmap(long, lad["Team"].tolist()))

    g = c.sql("""SELECT row_number() OVER (ORDER BY date) AS n, round_label, opponent, venue, pf, pa, pf - pa AS margin
                 FROM team_games WHERE season = ? AND team = ? ORDER BY date""", (season, team))
    g["result"] = np.select([g.margin > 0, g.margin < 0], ["Win", "Loss"], "Draw")
    g["score"] = g["pf"].astype(str) + "–" + g["pa"].astype(str)
    wins, losses = (g.result == "Win").sum(), (g.result == "Loss").sum()
    left, right = st.columns(2)
    with left, ui.card(f"{team}: match-by-match margins, {season}",
                       f"{wins} wins, {losses} losses; average margin {g.margin.mean():+.1f}. Finals included."):
        c.chart(charts.margin_bars(g, team))
    with right, ui.card("Territory battle", "Inside 50s for vs against per game. Bottom-right is best."):
        c.chart(charts.team_scatter(prof, "inside_50s", "inside_50s_against", "Inside 50s per game",
                                    "Inside 50s conceded per game", team))
    with st.expander("Table view"):
        st.dataframe(prof.round(1).sort_values("inside_50s", ascending=False), width="stretch", hide_index=True)


def page_strength() -> None:
    ui.header("Team strength", "An Elo rating computed from every result since 2021, and each club's home advantage.")
    hist, info = c.elo()
    current = hist.sort_values("date").groupby("team").last().sort_values("elo", ascending=False)
    with ui.filter_bar():
        f = st.columns([3, 1])
        teams = f[0].multiselect("Teams to highlight (up to 3)", TEAMS, default=current.index[:3].tolist(),
                                 max_selections=3)
        seasons = f[1].select_slider("Seasons", SEASONS, value=(SEASONS[0], SEASONS[-1]))
    k = st.columns(4)
    ui.kpi(k[0], "Highest rated now", current.index[0], f"Elo {current['elo'].iat[0]:.0f}", neutral=True)
    ui.kpi(k[1], "Lowest rated now", current.index[-1], f"Elo {current['elo'].iat[-1]:.0f}", neutral=True)
    ui.kpi(k[2], "Tips correct, 2025–26", f"{info['test_accuracy']:.1%}",
           f"{info['test_accuracy'] - info['test_home_team_win_rate']:+.1%} vs always tipping home team",
           help="Out of sample: parameters were tuned on 2022–2024 only. Draws excluded.")
    ui.kpi(k[3], "Brier score, 2025–26", f"{info['test_brier']:.3f}", "0.25 = coin flip; lower is better",
           neutral=True)

    h = hist[hist["season"].between(*seasons)]
    with ui.card("Elo rating over time",
                 f"Every team starts at 1,500 in round 1, 2021 (so 2021 is a settling-in season). After each match "
                 f"the winner takes K × (result − expected) points from the loser; the home team gets a "
                 f"{info['hga']}-point bonus when computing the expected result; between seasons each rating keeps "
                 f"{info['carry']:.0%} of its distance from 1,500. K = {info['k']}. These three settings were chosen "
                 f"by grid search to minimise the Brier score on 2022–2024. Grey lines = other teams; dashed = 1,500."):
        c.chart(charts.elo_lines(h, teams))

    hga = insights.home_advantage(c.table("matches"))
    left, right = st.columns([3, 2])
    with left, ui.card("Home-ground advantage by team",
                       "(Average margin as the home team − average margin as the away team) ÷ 2, home-and-away rounds "
                       "2021–2026. Bars show the estimate in points per game; whiskers are 95% confidence intervals. "
                       "\"Home\" is the designated home team, so shared grounds (e.g. Victorian clubs at the MCG) and "
                       "relocated games dilute the effect; most intervals overlap, so rankings are not precise."):
        c.chart(charts.hga_bars(hga))
    with right, ui.card("Current ratings", "Latest Elo after each team's most recent match in the data."):
        t = current[["elo", "date"]].assign(elo=lambda d: d.elo.round(0).astype(int),
                                            date=lambda d: d.date.dt.date)
        t.index.name = "Team"
        st.dataframe(t.rename(columns={"elo": "Elo", "date": "Last match"}), width="stretch",
                     height=36 * len(t) + 40)


def page_what_wins() -> None:
    ui.header("What wins games", "How the difference between two teams on each stat relates to the final margin.")
    with ui.filter_bar():
        f = st.columns([2, 2])
        seasons = f[0].multiselect("Seasons", SEASONS, default=SEASONS)
    if not seasons:
        st.info("Pick at least one season.")
        return
    m = c.table("matches")
    m = m[m["season"].isin(seasons)]
    corr, per_match = insights.stat_vs_margin(m, c.table("team_totals"))
    corr["label"] = corr["stat"].map(c.stat_name)
    corr["r_pos"] = corr["r"].clip(lower=0)
    order = corr.sort_values("r", ascending=False)["stat"].tolist()
    stat = f[1].selectbox("Stat for the scatter plot", order, format_func=c.stat_name,
                          index=c.default_index(order, "inside_50s"))

    left, right = st.columns([5, 6])
    with left, ui.card("Correlation with the final margin",
                       f"{len(per_match):,} matches, each counted once (home minus away). Goals, behinds, rushed "
                       "behinds and Brownlow votes are left out because they are the score or are awarded after it. "
                       "Correlation is not causation: goal assists and marks inside 50 are tied to scoring by "
                       "definition, and negative stats like rebound 50s mostly reflect being under pressure."):
        c.chart(charts.stat_correlations(corr))
    row = corr.set_index("stat").loc[stat]
    with right, ui.card(f"{c.stat_name(stat)} difference vs margin",
                        f"r = {row.r:+.2f}. Each extra unit of difference goes with {row.slope:+.2f} points of margin "
                        f"on average (orange least-squares line). The team ahead on this stat won "
                        f"{row.win_pct:.0%} of decided matches where the stat was not level."):
        d = per_match.assign(diff=per_match[stat],
                             match=lambda x: x.season.astype(str) + " R" + x.round_label + ": " + x.home_team
                             + " v " + x.away_team)
        c.chart(charts.diff_scatter(d, c.stat_name(stat)))
    with st.expander("Table view"):
        st.dataframe(corr.drop(columns=["r_pos", "stat"]).set_index("label").sort_values("r", ascending=False)
                     .rename(columns={"r": "r", "slope": "Margin per +1", "win_pct": "Win % when ahead",
                                      "n": "Matches"}).round(3), width="stretch")
