"""AFL Player Performance Analytics — Streamlit dashboard."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import duckdb
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afl import charts  # noqa: E402
from afl.charts import STAT_LABELS, pretty_feature  # noqa: E402

PROC, RESULTS, REPORTS = ROOT / "data" / "processed", ROOT / "results", ROOT / "reports"
st.set_page_config(page_title="AFL Player Analytics", page_icon="🏉", layout="wide")


@st.cache_resource
def db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW pm AS SELECT * FROM read_parquet('{(PROC / 'app_player_matches.parquet').as_posix()}')")
    con.execute(f"CREATE VIEW matches AS SELECT * FROM read_parquet('{(PROC / 'matches.parquet').as_posix()}')")
    con.execute(f"CREATE VIEW preds AS SELECT * FROM read_parquet('{(PROC / 'predictions.parquet').as_posix()}')")
    return con


@st.cache_data
def sql(query: str, params: tuple = ()) -> pd.DataFrame:
    return db().execute(query, list(params)).df()


@st.cache_data
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


SEASONS = sql("SELECT DISTINCT season FROM matches ORDER BY 1")["season"].tolist()
TEAMS = sql("SELECT DISTINCT team FROM pm ORDER BY 1")["team"].tolist()
FORM_STATS = ["disposals", "fantasy_points", "kicks", "handballs", "marks", "tackles", "goals", "clearances",
              "contested_possessions", "inside_50s", "rebound_50s", "hitouts", "pct_time_played"]
H2H_STATS = ["disposals", "kicks", "handballs", "marks", "tackles", "goals", "clearances",
             "contested_possessions", "uncontested_possessions", "inside_50s", "rebound_50s",
             "one_percenters", "goal_assists", "fantasy_points"]


# --------------------------------------------------------------------------- pages
def page_teams() -> None:
    st.title("Team insights")
    c1, c2 = st.columns([1, 2])
    season = c1.selectbox("Season", SEASONS[::-1])
    highlight = c2.selectbox("Highlight team", ["(none)"] + TEAMS)
    hl = None if highlight == "(none)" else highlight

    ladder = sql("""
        WITH t AS (
          SELECT home_team team, home_score pf, away_score pa FROM matches
          WHERE season = ? AND regexp_matches(round_label, '^[0-9]+$')
          UNION ALL
          SELECT away_team, away_score, home_score FROM matches
          WHERE season = ? AND regexp_matches(round_label, '^[0-9]+$'))
        SELECT team AS Team, count(*) AS P, sum((pf > pa)::INT) AS W, sum((pf < pa)::INT) AS L,
               sum((pf = pa)::INT) AS D, sum(pf) AS PF, sum(pa) AS PA,
               round(100.0 * sum(pf) / sum(pa), 1) AS "%",
               sum(CASE WHEN pf > pa THEN 4 WHEN pf = pa THEN 2 ELSE 0 END) AS Pts
        FROM t GROUP BY 1 ORDER BY Pts DESC, "%" DESC""", (season, season))
    ladder.index = range(1, len(ladder) + 1)

    profile = sql("""
        SELECT team, count(DISTINCT match_id) games,
               sum(disposals) / count(DISTINCT match_id) disposals,
               sum(contested_possessions) / count(DISTINCT match_id) contested_possessions,
               sum(clearances) / count(DISTINCT match_id) clearances,
               sum(inside_50s) / count(DISTINCT match_id) inside_50s,
               sum(tackles) / count(DISTINCT match_id) tackles,
               sum(marks) / count(DISTINCT match_id) marks,
               sum(goals) / count(DISTINCT match_id) goals,
               sum(rebound_50s) / count(DISTINCT match_id) rebound_50s
        FROM pm WHERE season = ? GROUP BY team""", (season,))
    against = sql("""
        SELECT opponent AS team, sum(inside_50s) / count(DISTINCT match_id) inside_50s_against,
               sum(disposals) / count(DISTINCT match_id) disposals_against
        FROM pm WHERE season = ? GROUP BY opponent""", (season,))
    prof = profile.merge(against, on="team")

    left, right = st.columns([5, 6])
    with left:
        st.subheader(f"{season} home-and-away ladder")
        st.caption("Computed from match results: 4 points per win, 2 per draw; ties broken by percentage (PF/PA).")
        st.dataframe(ladder, width="stretch", height=min(670, 36 * len(ladder) + 40))
    with right:
        st.subheader("Team stat comparison (per game, all matches incl. finals)")
        metric = st.selectbox("Statistic", ["disposals", "contested_possessions", "clearances", "inside_50s",
                                            "tackles", "marks", "goals", "rebound_50s"],
                              format_func=lambda s: STAT_LABELS[s])
        st.altair_chart(charts.team_bars(prof, metric, f"{STAT_LABELS[metric]} per game", hl), width="stretch")

    st.subheader("Territory: inside 50s for vs against (per game)")
    st.caption("Bottom-right is best: teams that win the territory battle. Hover for exact values.")
    st.altair_chart(charts.team_scatter(prof, "inside_50s", "inside_50s_against", "Inside 50s per game",
                                        "Inside 50s conceded per game", hl), width="stretch")
    with st.expander("Table view"):
        st.dataframe(prof.round(1).sort_values("inside_50s", ascending=False), width="stretch", hide_index=True)


def player_options() -> pd.DataFrame:
    return sql("""
        SELECT player_id, any_value(player_name) AS player_name, arg_max(team, date) AS team, count(*) AS games,
               max(season) AS last_season
        FROM pm GROUP BY player_id ORDER BY games DESC""")


def label_for(players: pd.DataFrame) -> dict:
    dup = players["player_name"].duplicated(keep=False)
    return {r.player_id: f"{r.player_name} ({r.team}{', ' + r.player_id.split('/')[-1] if d else ''})"
            for r, d in zip(players.itertuples(), dup)}


def page_player() -> None:
    st.title("Player form trends")
    players = player_options()
    labels = label_for(players)
    c1, c2, c3 = st.columns([3, 2, 2])
    pid = c1.selectbox("Player", players["player_id"], format_func=labels.get,
                       index=int(players.index[players["player_name"] == "Daicos, Nick"][0])
                       if (players["player_name"] == "Daicos, Nick").any() else 0)
    stat = c2.selectbox("Statistic", FORM_STATS, format_func=lambda s: STAT_LABELS[s])
    seasons = c3.multiselect("Seasons", SEASONS, default=SEASONS)
    df = sql("SELECT * FROM pm WHERE player_id = ? ORDER BY date", (pid,))
    df = df[df["season"].isin(seasons)]
    if df.empty:
        st.info("No matches for this selection.")
        return

    last = df[df["season"] == df["season"].max()]
    k = st.columns(5)
    k[0].metric("Matches", len(df))
    k[1].metric(f"Disposals / game ({last['season'].iat[0]})", f"{last['disposals'].mean():.1f}",
                f"{last['disposals'].mean() - df['disposals'].mean():+.1f} vs selection avg")
    k[2].metric(f"Fantasy / game ({last['season'].iat[0]})", f"{last['fantasy_points'].mean():.1f}")
    k[3].metric("Goals (selection)", int(df["goals"].sum()))
    k[4].metric("Latest inferred role", df["role"].iat[-1])

    st.altair_chart(charts.form_trend(df, stat), width="stretch")

    preds = sql("SELECT * FROM preds WHERE player_id = ? ORDER BY date", (pid,))
    if not preds.empty:
        st.subheader("2026 out-of-sample predictions")
        st.caption("Each prediction was made using only matches played before that game (models trained on 2021–2025).")
        tgt = st.radio("Target", ["disposals", "fantasy_points"], horizontal=True, format_func=lambda s: STAT_LABELS[s])
        p = preds[preds["target"] == tgt][["date", "round_label", "opponent", "actual", "Ensemble (mean of 3)",
                                           "Baseline: last-5 average"]]
        long = p.melt(id_vars=["date", "round_label", "opponent"], var_name="series", value_name="value")

        chart = alt.Chart(long).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=40)).encode(
            x=alt.X("date:T", title="Match date"), y=alt.Y("value:Q", title=STAT_LABELS[tgt]),
            color=alt.Color("series:N", scale=alt.Scale(domain=["actual", "Ensemble (mean of 3)", "Baseline: last-5 average"],
                                                        range=["#52514e", charts.BLUE, charts.NEUTRAL]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["round_label:N", "opponent:N", "series:N", alt.Tooltip("value:Q", format=".1f")]
        ).properties(height=300)
        st.altair_chart(chart, width="stretch")

    st.subheader("Season averages")
    avg = df.groupby("season")[["disposals", "kicks", "handballs", "marks", "tackles", "goals", "clearances",
                                "contested_possessions", "inside_50s", "fantasy_points", "pct_time_played"]].mean()
    avg.insert(0, "games", df.groupby("season").size())
    st.dataframe(avg.round(1).rename(columns=STAT_LABELS), width="stretch")
    with st.expander("Match log"):
        st.dataframe(df[["date", "season", "round_label", "team", "opponent", "disposals", "kicks", "handballs",
                         "marks", "tackles", "goals", "behinds", "clearances", "fantasy_points", "pct_time_played",
                         "sub_status"]].sort_values("date", ascending=False), width="stretch", hide_index=True)


def page_h2h() -> None:
    st.title("Head-to-head")
    tab_p, tab_t = st.tabs(["Player vs player", "Team vs team"])
    with tab_p:
        players = player_options()
        labels = label_for(players)
        ids = players["player_id"].tolist()
        default2 = ids.index("Z/Zach_Merrett") if "Z/Zach_Merrett" in ids else 1
        default1 = ids.index("N/Nick_Daicos") if "N/Nick_Daicos" in ids else 0
        c1, c2, c3 = st.columns([3, 3, 2])
        p1 = c1.selectbox("Player A", ids, index=default1, format_func=labels.get)
        p2 = c2.selectbox("Player B", ids, index=default2, format_func=labels.get)
        seasons = c3.multiselect("Seasons", SEASONS, default=SEASONS[-2:], key="h2h_seasons")
        if p1 == p2 or not seasons:
            st.info("Pick two different players and at least one season.")
        else:
            df = sql(f"SELECT * FROM pm WHERE player_id IN (?, ?) AND season IN ({','.join('?' * len(seasons))})",
                     (p1, p2, *seasons))
            n1, n2 = labels[p1], labels[p2]
            df["player"] = df["player_id"].map({p1: n1, p2: n2})
            pool = sql(f"""SELECT player_id, count(*) AS games, {', '.join(f'avg({c}) AS {c}' for c in H2H_STATS)}
                           FROM pm WHERE season IN ({','.join('?' * len(seasons))}) GROUP BY player_id""",
                       tuple(seasons))
            pool = pool[(pool["games"] >= 10) | pool["player_id"].isin([p1, p2])].set_index("player_id")
            pct = pool[H2H_STATS].rank(pct=True) * 100
            rows = []
            for pid_, name in ((p1, n1), (p2, n2)):
                for stat in H2H_STATS:
                    rows.append({"player": name, "stat": STAT_LABELS[stat],
                                 "value": pool.at[pid_, stat], "pct": pct.at[pid_, stat]})
            avg = pd.DataFrame(rows)
            games = df.groupby("player").size()
            st.caption(f"Games in selection — {n1}: {games.get(n1, 0)}, {n2}: {games.get(n2, 0)}")
            l, r = st.columns([1, 1])
            with l:
                st.subheader("Per-game averages vs the league")
                st.caption(f"Percentile among {len(pool):,} players with 10+ games in the selected seasons; "
                           "hover for the raw per-game average.")
                st.altair_chart(charts.head_to_head(avg, n1, n2), width="stretch")
            with r:
                st.subheader("Form over time")
                stat = st.selectbox("Statistic", FORM_STATS, format_func=lambda s: STAT_LABELS[s], key="h2h_stat")
                df = df.sort_values("date")
                df["rolling"] = df.groupby("player")[stat].transform(lambda s: s.rolling(5, min_periods=1).mean())
                st.altair_chart(charts.overlay_trend(df, stat, n1, n2), width="stretch")
            with st.expander("Table view"):
                st.dataframe(avg.pivot(index="stat", columns="player", values=["value", "pct"]).round(2), width="stretch")

    with tab_t:
        c1, c2 = st.columns(2)
        t1 = c1.selectbox("Team A", TEAMS, index=TEAMS.index("Collingwood") if "Collingwood" in TEAMS else 0)
        t2 = c2.selectbox("Team B", TEAMS, index=TEAMS.index("Brisbane Lions") if "Brisbane Lions" in TEAMS else 1)
        games = sql("""
            SELECT date, season, round_label, venue, home_team, home_score, away_team, away_score
            FROM matches WHERE (home_team = ? AND away_team = ?) OR (home_team = ? AND away_team = ?)
            ORDER BY date DESC""", (t1, t2, t2, t1))
        if games.empty or t1 == t2:
            st.info("No matches between these teams in 2021–2026.")
        else:
            a_score = games.apply(lambda r: r.home_score if r.home_team == t1 else r.away_score, axis=1)
            b_score = games.apply(lambda r: r.away_score if r.home_team == t1 else r.home_score, axis=1)
            k = st.columns(4)
            k[0].metric("Meetings", len(games))
            k[1].metric(f"{t1} wins", int((a_score > b_score).sum()))
            k[2].metric(f"{t2} wins", int((b_score > a_score).sum()))
            k[3].metric(f"Avg margin ({t1})", f"{(a_score - b_score).mean():+.1f}")
            st.dataframe(games, width="stretch", hide_index=True)


def page_models() -> None:
    st.title("Prediction models")
    m = load_json(str(RESULTS / "metrics.json"))
    st.markdown(
        "Predicting each player's **next-match disposals** and **AFL Fantasy points** from information available "
        "before the bounce. **Train** 2021–2024 · **validate/tune** 2025 · **test** 2026 (held out; not used for tuning or model selection).")
    tgt = st.radio("Target", list(m["targets"]), horizontal=True, format_func=lambda s: STAT_LABELS[s])
    res = m["targets"][tgt]
    tab = pd.DataFrame(res["test_2026"]).T.reset_index(names="model")
    base = tab.loc[tab["model"] == "Baseline: last-5 average", "rmse"].iat[0]
    tab["RMSE vs last-5 baseline"] = (tab["rmse"] / base - 1).map("{:+.1%}".format)
    c1, c2 = st.columns([3, 2])
    with c1:
        metric = st.radio("Metric", ["rmse", "mae"], horizontal=True, format_func=str.upper)
        st.altair_chart(charts.model_metric_bars(tab, metric), width="stretch")
    with c2:
        st.dataframe(tab.drop(columns="n").rename(columns={"rmse": "RMSE", "mae": "MAE", "r2": "R²"})
                     .set_index("model").round(3), width="stretch")
        st.caption(f"{int(tab['n'].iat[0]):,} player-matches in the 2026 test season.")

    preds = sql("SELECT * FROM preds WHERE target = ?", (tgt,))
    best = res["selected_model"]
    st.caption(f"Selected model (lowest RMSE on the 2025 validation season, chosen before looking at 2026): **{best}**.")
    model = st.selectbox("Model for diagnostics", [c for c in charts.MODEL_ORDER if c in preds.columns],
                         index=charts.MODEL_ORDER.index(best))
    l, r = st.columns(2)
    with l:
        st.altair_chart(charts.predicted_vs_actual(preds, model, STAT_LABELS[tgt].lower()), width="stretch")
    with r:
        rows = []
        for role, d in res["test_2026_by_role"].items():
            for name in (model, "Baseline: last-5 average"):
                rows.append({"role": role, "model": name, "mae": d[name]["mae"], "n": d[name]["n"]})
        st.altair_chart(charts.error_by_group(pd.DataFrame(rows), "role", "Mean absolute error by inferred role"),
                        width="stretch")
        full = res["test_2026_full_match_established_players"]
        st.caption(f"Excluding substitutes and players with fewer than 5 prior games ({full[model]['n']:,} rows): "
                   f"{model} RMSE {full[model]['rmse']:.2f} vs baseline {full['Baseline: last-5 average']['rmse']:.2f}.")
    with st.expander("Validation (2025) scores used for tuning"):
        st.dataframe(pd.DataFrame(res["validation_2025"]).T.round(3), width="stretch")
        st.json(res["tuning"])


def page_importance() -> None:
    st.title("Feature importance")
    imp = pd.read_parquet(RESULTS / "feature_importance.parquet")
    tgt = st.radio("Target", imp["target"].unique().tolist(), horizontal=True, format_func=lambda s: STAT_LABELS[s])
    d = imp[imp["target"] == tgt].copy()
    d["feature"] = d["feature"].map(pretty_feature)
    st.caption("Permutation importance of the final LightGBM model on the 2026 test set: how much RMSE worsens when "
               "a feature's values are shuffled (mean ± std over 5 repeats). Model-agnostic and measured on unseen data.")
    st.altair_chart(charts.importance_bars(d), width="stretch")
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
    st.Page(page_teams, title="Team insights", icon="📊", url_path="teams", default=True),
    st.Page(page_player, title="Player form", icon="📈", url_path="player"),
    st.Page(page_h2h, title="Head-to-head", icon="⚖️", url_path="head-to-head"),
    st.Page(page_models, title="Prediction models", icon="🎯", url_path="models"),
    st.Page(page_importance, title="Feature importance", icon="🔍", url_path="importance"),
    st.Page(page_about, title="Data & methodology", icon="📚", url_path="about"),
])
st.sidebar.caption("Data: AFL Tables (player stats) · Squiggle API (results cross-check). Seasons 2021–2026.")
pg.run()
