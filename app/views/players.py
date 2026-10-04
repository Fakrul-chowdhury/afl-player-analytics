"""Player pages: player form (with calendar, spread, age, with/without), similar players, head-to-head."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import common as c
import ui
from afl import insights
from common import BASELINE, FORM_STATS, PROFILE_STATS, SEASONS, STAT_LABELS, TEAMS, charts

# Qualifying and elimination finals are played in the same week, so they share a column.
FINALS = {"Wildcard Final": "WF", "Qualifying Final": "F1", "Elimination Final": "F1", "Semi Final": "SF",
          "Preliminary Final": "PF", "Grand Final": "GF"}
SIM_STATS = [s for s in PROFILE_STATS if s != "fantasy_points"]  # fantasy points is a sum of the others


def _round_col(label: str) -> str:
    return FINALS.get(label, label)


def _player_matches(pid: str) -> pd.DataFrame:
    return c.sql("""SELECT p.*, t.pf, t.pa,
                           CASE WHEN t.pf > t.pa THEN 'Win' WHEN t.pf < t.pa THEN 'Loss' ELSE 'Draw' END AS result
                    FROM pm p JOIN team_games t USING (match_id, team) WHERE p.player_id = ? ORDER BY p.date""",
                 (pid,))


def page_player() -> None:
    ui.header("Player form", "Match-by-match form, calendar, league context, age curve and team impact.")
    with ui.filter_bar():
        f = st.columns([3, 2, 2])
        pid = c.player_picker(f[0], "pf_player")
        stat = f[1].selectbox("Statistic", FORM_STATS, format_func=lambda s: STAT_LABELS[s])
        seasons = f[2].multiselect("Seasons", SEASONS, default=SEASONS)
    full = _player_matches(pid)
    df = full[full["season"].isin(seasons)]
    if df.empty:
        st.info("No matches for this selection.")
        return
    label = STAT_LABELS[stat]
    last = df[df["season"] == df["season"].max()]
    ls = int(last["season"].iat[0])
    per_season = df.groupby("season")[[stat, "fantasy_points"]].mean()
    k = st.columns(6)
    ui.kpi(k[0], "Matches", len(df), f"{df['season'].nunique()} seasons", neutral=True)
    ui.kpi(k[1], f"{label} / game ({ls})", f"{last[stat].mean():.1f}",
           f"{last[stat].mean() - df[stat].mean():+.1f} vs selection avg", spark=per_season[stat].round(1),
           help="Sparkline: season averages across the selection.")
    ui.kpi(k[2], f"Fantasy / game ({ls})", f"{last['fantasy_points'].mean():.1f}",
           spark=per_season["fantasy_points"].round(1), help="Sparkline: season averages across the selection.")
    ui.kpi(k[3], f"Best {label.lower()}", f"{df[stat].max():.0f}")
    ui.kpi(k[4], "Goals (selection)", int(df["goals"].sum()))
    ui.kpi(k[5], "Latest inferred role", df["role"].iat[-1])

    tabs = st.tabs(["Form", "Form calendar", "League spread", "Age curve", "With / without", "2026 predictions",
                    "Season log"])
    with tabs[0]:
        _form_tab(df, pid, stat, label)
    with tabs[1]:
        _calendar_tab(df, stat, label, seasons)
    with tabs[2]:
        _spread_tab(df, pid)
    with tabs[3]:
        _age_tab(full, stat, label)
    with tabs[4]:
        _with_without_tab(full, seasons)
    with tabs[5]:
        _predictions_tab(pid)
    with tabs[6]:
        avg = df.groupby("season")[["disposals", "kicks", "handballs", "marks", "tackles", "goals", "clearances",
                                    "contested_possessions", "inside_50s", "fantasy_points", "pct_time_played"]].mean()
        avg.insert(0, "games", df.groupby("season").size())
        with ui.card("Season averages"):
            st.dataframe(avg.round(1).rename(columns=STAT_LABELS), width="stretch")
        with ui.card("Match log"):
            st.dataframe(df[["date", "season", "round_label", "team", "opponent", "result", "disposals", "kicks",
                             "handballs", "marks", "tackles", "goals", "behinds", "clearances", "fantasy_points",
                             "pct_time_played", "sub_status"]].sort_values("date", ascending=False),
                         width="stretch", hide_index=True)


def _form_tab(df: pd.DataFrame, pid: str, stat: str, label: str) -> None:
    with ui.card():
        c.chart(charts.form_trend(df, stat))
    left, right = st.columns([5, 4])
    with left, ui.card("Profile vs role peers"):
        pool_season = st.selectbox("Compare against league in season", sorted(df["season"].unique())[::-1],
                                   key="pf_season")
        pool = c.season_pool((int(pool_season),), min_games=10)
        if pid in pool.index:
            role = pool.at[pid, "role"]
            peers = pool[pool["role"] == role] if role != "Unknown" else pool
            pct = peers[PROFILE_STATS].rank(pct=True) * 100
            prof = pd.DataFrame({"stat": [STAT_LABELS[s] for s in PROFILE_STATS],
                                 "value": [peers.at[pid, s] for s in PROFILE_STATS],
                                 "pct": [pct.at[pid, s] for s in PROFILE_STATS]})
            st.caption(f"Percentile rank among {len(peers)} players inferred as **{role}** with 10+ games in "
                       f"{pool_season}. Blue = above the median (dashed line).")
            c.chart(charts.percentile_bars(prof, "Percentile among role peers"))
        else:
            st.info(f"Fewer than 10 games in {pool_season}: no peer comparison.")
    with right:
        with ui.card(f"{label}: splits", "Dashed orange line = overall average for the selection. Hover for games."):
            split = st.radio("Split by", ["Opponent", "Home / away", "Result", "Venue"], horizontal=True)
            col = {"Opponent": "opponent", "Home / away": "where", "Result": "result", "Venue": "venue"}[split]
            d = df.assign(where=np.where(df["is_home"], "Home", "Away"))
            sp = d.groupby(col).agg(value=(stat, "mean"), games=(stat, "size")).reset_index()
            if col == "venue":
                sp = sp[sp["games"] >= 3]
            c.chart(charts.split_bars(sp, col, label, df[stat].mean()))
        with ui.card("Distribution"):
            c.chart(charts.histogram(df, stat, f"{label} per match", 2 if df[stat].max() > 15 else 1))


def _calendar_tab(df: pd.DataFrame, stat: str, label: str, seasons: list[int]) -> None:
    league = c.sql("SELECT DISTINCT season, round_label FROM matches")
    league = league[league["season"].isin(seasons)]
    digits = sorted({int(r) for r in league["round_label"] if r.isdigit()})
    finals = list(dict.fromkeys(v for k, v in FINALS.items() if k in set(league["round_label"])))
    order = [str(r) for r in digits] + finals
    played = df.assign(col=df["round_label"].map(_round_col), value=df[stat])
    league = league.assign(col=league["round_label"].map(_round_col)).drop_duplicates(["season", "col"])
    grid = league.drop(columns="round_label").merge(
        played[["season", "col", "round_label", "value", "opponent", "result"]], on=["season", "col"], how="left")
    grid["round_label"] = grid["round_label"].fillna(grid["col"])
    missing = int(grid["value"].isna().sum())
    grid["opponent"] = grid["opponent"].fillna("Did not play")
    grid["result"] = grid["result"].fillna("—")
    with ui.card(f"{label} by round and season",
                 f"Each cell is one round; stronger blue = higher {label.lower()}. Empty dashed cells are rounds the "
                 f"league played but this player did not ({missing} in the selection: injury, omission, bye, or not yet/no longer "
                 "on a list, or the team missed that finals week). Nothing is filled in for missing games. Finals: WF "
                 "wildcard, F1 qualifying/elimination, SF semi, PF preliminary, GF grand final."):
        base = grid.assign(value=grid["value"])
        empty = base[base["value"].isna()]
        chart = charts.form_calendar(base.dropna(subset=["value"]), label, order)
        bg = alt.Chart(empty).mark_rect(cornerRadius=3, fill=charts.T().surface, stroke=charts.T().neutral,
                                        strokeWidth=1, strokeDash=[3, 2]).encode(
            x=alt.X("col:N", sort=order), y=alt.Y("season:O", sort="descending"),
            tooltip=[alt.Tooltip("season:O"), alt.Tooltip("round_label:N", title="Round"),
                     alt.Tooltip("opponent:N", title="")])
        c.chart(bg + chart)


def _spread_tab(df: pd.DataFrame, pid: str) -> None:
    season = st.selectbox("Season", sorted(df["season"].unique())[::-1], key="spread_season")
    pool = c.season_pool((int(season),), min_games=10)
    if pid not in pool.index:
        st.info(f"Fewer than 10 games in {season}: the player is not in the comparison pool.")
        return
    rows = []
    rng = np.random.default_rng(7)  # fixed vertical jitter so dots do not overlap; carries no meaning
    jitter = pd.Series(rng.uniform(-1, 1, len(pool)), index=pool.index)
    jitter[pid] = 0.0
    for s in PROFILE_STATS:
        pct = pool[s].rank(pct=True) * 100
        rows.append(pd.DataFrame({"stat": STAT_LABELS[s], "value": pool[s], "pct": pct, "jitter": jitter,
                                  "label": pool["player_name"] + " (" + pool["team"] + ")",
                                  "is_player": pool.index == pid}))
    d = pd.concat(rows).reset_index(drop=True)
    d["pct_text"] = np.where(d["is_player"], d["pct"].round(0).astype(int).astype(str) + "th pct", "")
    name = pool.at[pid, "player_name"]
    with ui.card(f"Where {name} sits in the league, {season}",
                 f"Every dot is one of {len(pool)} players with 10+ games in {season}, at their per-game average; "
                 f"{name} is the large blue dot, labelled with their percentile. Each row has its own scale. "
                 "Vertical position is random jitter only, to stop dots overlapping."):
        c.chart(charts.league_spread(d, name))


def _age_tab(full: pd.DataFrame, stat: str, label: str) -> None:
    pm = c.sql(f"SELECT age_days, {stat} FROM pm WHERE age_days IS NOT NULL")
    league = insights.age_profile(pm, stat)
    me = full.dropna(subset=["age_days"]).assign(age=lambda x: (x.age_days // 365.25).astype(int))
    mine = me.groupby("age")[stat].agg(mean="mean", games="size").reset_index()
    name = full["player_name"].iat[-1]
    with ui.card(f"{label} by age: {name} vs the league",
                 "Dashed line = league average per match at each whole year of age (all player-matches 2021–2026, "
                 "ages with 150+ player-matches); grey band = middle 50% of single-match values. Blue = this "
                 "player's average at each age. This is a cross-section of six seasons, not a lifetime curve: "
                 "players who are still playing at 33 are the survivors, which lifts the older ages."):
        c.chart(charts.age_curve(league, mine, label, name))


def _with_without_tab(full: pd.DataFrame, seasons: list[int]) -> None:
    teams = full.groupby(["season", "team"]).size().reset_index(name="n")
    main = teams.sort_values("n").groupby("season").tail(1)
    main = main[main["season"].isin(seasons)]
    played = set(full["match_id"])
    tg = c.sql("SELECT team, season, match_id, pf, pa FROM team_games")
    parts = [insights.with_without(tg, played, r.team, [r.season]) for r in main.itertuples()]
    g = pd.concat(parts) if parts else pd.DataFrame()
    if g.empty or (g["group"] == "Without player").sum() == 0:
        st.info("The player's team played no matches without them in the selected seasons, so there is nothing "
                "to compare.")
        return
    agg = g.groupby("group").agg(games=("win", "size"), win=("win", "mean"), margin=("margin", "mean"))
    k = st.columns(4)
    w = agg.loc["With player"]
    wo = agg.loc["Without player"] if "Without player" in agg.index else None
    ui.kpi(k[0], "Team win rate with", f"{w.win:.0%}", f"{int(w.games)} games", neutral=True)
    ui.kpi(k[1], "Team win rate without", f"{wo.win:.0%}", f"{int(wo.games)} games", neutral=True)
    ui.kpi(k[2], "Avg margin with", f"{w.margin:+.1f}", neutral=True)
    ui.kpi(k[3], "Avg margin without", f"{wo.margin:+.1f}", neutral=True)
    by = g.groupby(["season", "group"]).agg(games=("win", "size"), win=("win", "mean"),
                                            margin=("margin", "mean")).reset_index()
    by["season"] = by["season"].astype(str)
    note = (" Small samples: with fewer than ~10 games without the player, the difference is mostly noise."
            if wo.games < 10 else "")
    cap = ("Team results in seasons where the player played at least once for that club (their main club that "
           "season), split by whether they took the field. Numbers above bars = team games. This is descriptive, "
           "not causal: absences coincide with injuries, opponents, team changes and form." + note)
    left, right = st.columns(2)
    with left, ui.card("Win rate with vs without", cap):
        c.chart(charts.with_without(by, "win", "Team win rate", ".0%"))
    with right, ui.card("Average margin with vs without", "Points per game from the club's perspective."):
        c.chart(charts.with_without(by, "margin", "Average margin (points)", "+.0f"))


def _predictions_tab(pid: str) -> None:
    preds = c.sql("SELECT * FROM preds WHERE player_id = ? ORDER BY date", (pid,))
    if preds.empty:
        st.info("No 2026 matches for this player, so no out-of-sample predictions.")
        return
    m = c.load_json(str(c.RESULTS / "metrics.json"))
    tgt = st.radio("Target", ["disposals", "fantasy_points"], horizontal=True, format_func=lambda s: STAT_LABELS[s],
                   key="pf_pred_tgt")
    model = m["targets"][tgt]["selected_model"]
    p = preds[preds["target"] == tgt]
    inside = ((p["actual"] >= p["pi_lower"]) & (p["actual"] <= p["pi_upper"])).mean()
    with ui.card("2026 out-of-sample predictions",
                 f"Each prediction uses only matches before that game ({model}, trained on 2021–2025). "
                 f"This player's actual result fell inside the 80% interval in {inside:.0%} of 2026 matches; "
                 f"MAE {np.abs(p['actual'] - p[model]).mean():.1f} vs {np.abs(p['actual'] - p[BASELINE]).mean():.1f} "
                 "for the last-5 baseline. See *Why this prediction?* for a feature-by-feature breakdown."):
        c.chart(charts.prediction_band(p, STAT_LABELS[tgt], model))


def page_similar() -> None:
    ui.header("Similar players", "A map of every player's statistical profile, and who plays most like whom.")
    with ui.filter_bar():
        f = st.columns([3, 2, 1])
        pid = c.player_picker(f[0], "sim_player")
        seasons = f[1].multiselect("Seasons pooled", SEASONS, default=SEASONS[-2:], key="sim_seasons")
        min_games = f[2].number_input("Min games", 5, 40, 10, key="sim_min")
    if not seasons:
        st.info("Pick at least one season.")
        return
    pool = c.season_pool(tuple(seasons), min_games=int(min_games))
    if pid not in pool.index:
        st.info(f"This player has fewer than {min_games} games in the selected seasons.")
        return
    coords, var, Z = insights.similarity(pool, SIM_STATS)
    near = insights.nearest(Z, pool.index, pid, n=10)
    d = pool.join(coords).reset_index()
    d["group"] = np.where(d.player_id == pid, "Selected",
                          np.where(d.player_id.isin(near.index), "Most similar", "Other players"))
    d["rank"] = d["group"].map({"Selected": 0, "Most similar": 1, "Other players": 2})
    d["label"] = d["player_name"] + " (" + d["team"] + ")"
    # Label the selected player and the three closest only; the rest are in tooltips and the table.
    d["name"] = np.where(d.player_id.isin([pid, *near.index[:3]]), d["player_name"], "")
    name = pool.at[pid, "player_name"]
    left, right = st.columns([3, 2])
    with left, ui.card("Player similarity map",
                       f"{len(pool)} players with {min_games}+ games in {', '.join(map(str, seasons))}. Each player's "
                       f"per-game averages on {len(SIM_STATS)} stats are standardised and projected onto two "
                       f"principal components (PCA). Axes explain {var[0]:.0%} and {var[1]:.0%} of the variation, so "
                       "nearby dots are similar but the full comparison (right) uses all stats."):
        c.chart(charts.similarity_map(d, f"Component 1 ({var[0]:.0%} of variation)",
                                      f"Component 2 ({var[1]:.0%} of variation)"))
    with right, ui.card(f"Players most like {name}",
                        "Ranked by distance across all standardised stats (0 = identical). Similarity = "
                        "1 / (1 + distance)."):
        t = pool.loc[near.index, ["player_name", "team", "role", "games"]].assign(
            distance=near.values, similarity=1 / (1 + near.values))
        st.dataframe(t.round(2).rename(columns={"player_name": "Player", "team": "Team", "role": "Role",
                                                "games": "Games", "distance": "Distance"}),
                     width="stretch", hide_index=True,
                     column_config={"similarity": st.column_config.ProgressColumn(
                         "Similarity", min_value=0.0, max_value=1.0, format="%.2f")})
        top = near.index[0]
        z = pd.DataFrame({"stat": [STAT_LABELS[s] for s in SIM_STATS],
                          name: Z[pool.index.get_loc(pid)], pool.at[top, "player_name"]: Z[pool.index.get_loc(top)]})
        st.caption(f"Closest match: **{pool.at[top, 'player_name']}**. Standardised values (0 = league average):")
        st.dataframe(z.set_index("stat").round(2), width="stretch")


def page_h2h() -> None:
    ui.header("Head-to-head", "Two players or two clubs, side by side.")
    tab_p, tab_t = st.tabs(["Player vs player", "Team vs team"])
    with tab_p:
        with ui.filter_bar():
            f = st.columns([3, 3, 2])
            p1 = c.player_picker(f[0], "h2h_a", "N/Nick_Daicos", "Player A")
            p2 = c.player_picker(f[1], "h2h_b", "Z/Zach_Merrett", "Player B")
            seasons = f[2].multiselect("Seasons", SEASONS, default=SEASONS[-2:], key="h2h_seasons")
        if p1 == p2 or not seasons:
            st.info("Pick two different players and at least one season.")
        else:
            _h2h_players(p1, p2, seasons)
    with tab_t:
        f = st.columns(2)
        t1 = f[0].selectbox("Team A", TEAMS, index=c.default_index(TEAMS, "Collingwood"))
        t2 = f[1].selectbox("Team B", TEAMS, index=c.default_index(TEAMS, "Brisbane Lions", 1))
        games = c.sql("""SELECT row_number() OVER (ORDER BY date) AS n, date, season, round_label, venue, opponent,
                                pf, pa, pf - pa AS margin
                         FROM team_games WHERE team = ? AND opponent = ? ORDER BY date""", (t1, t2))
        if games.empty or t1 == t2:
            st.info("No matches between these teams in 2021–2026.")
            return
        k = st.columns(4)
        ui.kpi(k[0], "Meetings", len(games))
        ui.kpi(k[1], f"{t1} wins", int((games.margin > 0).sum()))
        ui.kpi(k[2], f"{t2} wins", int((games.margin < 0).sum()))
        ui.kpi(k[3], f"Avg margin ({t1})", f"{games.margin.mean():+.1f}")
        games["result"] = np.select([games.margin > 0, games.margin < 0], ["Win", "Loss"], "Draw")
        games["score"] = games["pf"].astype(str) + "–" + games["pa"].astype(str)
        with ui.card(f"{t1} v {t2}: every meeting", "Margin from Team A's perspective. Hover a bar for details."):
            c.chart(charts.margin_bars(games, t1))
            st.dataframe(games.drop(columns=["n", "opponent"]).sort_values("date", ascending=False),
                         width="stretch", hide_index=True)


def _h2h_players(p1: str, p2: str, seasons: list[int]) -> None:
    labels = c.label_for(c.player_options())
    ph = ",".join("?" * len(seasons))
    df = c.sql(f"SELECT * FROM pm WHERE player_id IN (?, ?) AND season IN ({ph})", (p1, p2, *seasons))
    n1, n2 = labels[p1], labels[p2]
    df["player"] = df["player_id"].map({p1: n1, p2: n2})
    pool = c.season_pool(tuple(seasons), min_games=10)
    pool = pd.concat([pool, c.season_pool(tuple(seasons), min_games=1).loc[
        [p for p in (p1, p2) if p not in pool.index]]])
    pct = pool[PROFILE_STATS].rank(pct=True) * 100
    avg = pd.DataFrame([{"player": name, "stat": STAT_LABELS[s], "value": pool.at[pid_, s], "pct": pct.at[pid_, s]}
                        for pid_, name in ((p1, n1), (p2, n2)) for s in PROFILE_STATS])
    games = df.groupby("player").size()
    st.caption(f"Games in selection: {n1} {games.get(n1, 0)}, {n2} {games.get(n2, 0)}")
    left, right = st.columns(2)
    with left, ui.card("Per-game averages vs the league",
                       f"Percentile among {len(pool):,} players with 10+ games in the selected seasons; hover for "
                       "the raw per-game average."):
        c.chart(charts.head_to_head(avg, n1, n2))
    with right:
        with ui.card("Form over time"):
            stat = st.selectbox("Statistic", FORM_STATS, format_func=lambda s: STAT_LABELS[s], key="h2h_stat")
            df = df.sort_values("date")
            df["rolling"] = df.groupby("player")[stat].transform(lambda s: s.rolling(5, min_periods=1).mean())
            c.chart(charts.overlay_trend(df, stat, n1, n2))
        with ui.card("Consistency", "Spread of single-match output: box = middle 50% of games, line = median."):
            c.chart(charts.distribution_compare(df, stat, [n1, n2]))
    meet = c.sql("""
        SELECT a.date, a.season, a.round_label, a.team AS team_a, a.disposals AS disp_a, a.goals AS goals_a,
               a.fantasy_points AS fantasy_a, b.team AS team_b, b.disposals AS disp_b, b.goals AS goals_b,
               b.fantasy_points AS fantasy_b
        FROM pm a JOIN pm b ON a.match_id = b.match_id AND a.team <> b.team
        WHERE a.player_id = ? AND b.player_id = ? ORDER BY a.date DESC""", (p1, p2))
    with ui.card(f"Direct meetings ({len(meet)}, all seasons)"):
        if meet.empty:
            st.caption("These players have not played against each other in 2021–2026.")
        else:
            st.caption(f"Disposals in meetings: {n1} averaged {meet.disp_a.mean():.1f}, {n2} {meet.disp_b.mean():.1f}.")
            st.dataframe(meet, width="stretch", hide_index=True)
    with st.expander("Table view"):
        st.dataframe(avg.pivot(index="stat", columns="player", values=["value", "pct"]).round(2), width="stretch")
