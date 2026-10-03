"""Altair chart builders shared by the dashboard and the README figure export."""
from __future__ import annotations

import altair as alt
import pandas as pd

# Validated categorical palette (fixed order, never cycled) + neutral for context marks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#b8b6ae"
MODEL_ORDER = ["Ensemble (mean of 3)", "CatBoost", "LightGBM", "Ridge regression",
               "Baseline: season-to-date average", "Baseline: last-5 average"]

STAT_LABELS = {
    "disposals": "Disposals", "kicks": "Kicks", "handballs": "Handballs", "marks": "Marks",
    "goals": "Goals", "behinds": "Behinds", "tackles": "Tackles", "hitouts": "Hitouts",
    "clearances": "Clearances", "inside_50s": "Inside 50s", "rebound_50s": "Rebound 50s",
    "contested_possessions": "Contested possessions", "uncontested_possessions": "Uncontested possessions",
    "contested_marks": "Contested marks", "marks_inside_50": "Marks inside 50",
    "one_percenters": "One percenters", "goal_assists": "Goal assists", "clangers": "Clangers",
    "frees_for": "Free kicks for", "frees_against": "Free kicks against", "bounces": "Bounces",
    "fantasy_points": "AFL Fantasy points", "pct_time_played": "% time on ground",
}


def pretty_feature(name: str) -> str:
    for w in (3, 5, 10):
        if name.endswith(f"_avg{w}"):
            return f"{STAT_LABELS.get(name[:-len(f'_avg{w}')], name)}: last {w} avg"
    special = {
        "disposals_std10": "Disposals: last 10 std dev", "fantasy_points_std10": "Fantasy: last 10 std dev",
        "disposals_last": "Disposals: last match", "fantasy_points_last": "Fantasy: last match",
        "disposals_season_avg": "Disposals: season-to-date avg", "fantasy_points_season_avg": "Fantasy: season-to-date avg",
        "games_in_data": "Games played (since 2021)", "career_games_before": "Career games before match",
        "age_years": "Age (years)", "days_since_last": "Days since last match", "sub_last_match": "Was sub last match",
        "team_disp_for5": "Team disposals: last 5 avg", "opp_disp_conceded5": "Opponent disposals conceded: last 5",
        "opp_fp_conceded5": "Opponent fantasy conceded: last 5", "is_home": "Home game", "is_final": "Final",
        "role": "Inferred role", "team": "Team", "opponent": "Opponent", "venue": "Venue",
    }
    return special.get(name, name)


def form_trend(df: pd.DataFrame, stat: str, window: int = 5) -> alt.Chart:
    """Per-match values (dots) with a trailing rolling average (line)."""
    label = STAT_LABELS.get(stat, stat)
    d = df.sort_values("date").assign(
        rolling=lambda x: x[stat].rolling(window, min_periods=1).mean(),
        match=lambda x: x["season"].astype(str) + " R" + x["round_label"] + " v " + x["opponent"])
    base = alt.Chart(d).encode(x=alt.X("date:T", title="Match date"))
    dots = base.mark_circle(size=60, color=NEUTRAL, opacity=0.9).encode(
        y=alt.Y(f"{stat}:Q", title=label),
        tooltip=[alt.Tooltip("match:N", title="Match"), alt.Tooltip(f"{stat}:Q", title=label),
                 alt.Tooltip("rolling:Q", title=f"{window}-match average", format=".1f")])
    line = base.mark_line(color=BLUE, strokeWidth=2).encode(y="rolling:Q", detail="season:N")
    return (dots + line).properties(height=320, title=f"{label} per match, with {window}-match rolling average (blue)")


def team_bars(df: pd.DataFrame, metric: str, title: str, highlight: str | None = None, fmt: str = ".1f") -> alt.Chart:
    color = (alt.condition(alt.datum.team == highlight, alt.value(BLUE), alt.value(NEUTRAL))
             if highlight else alt.value(BLUE))
    return alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("team:N", sort="-x", title=None),
        x=alt.X(f"{metric}:Q", title=title),
        color=color,
        tooltip=["team:N", alt.Tooltip(f"{metric}:Q", title=title, format=fmt)],
    ).properties(height=alt.Step(20))


def team_scatter(df: pd.DataFrame, x: str, y: str, xt: str, yt: str, highlight: str | None = None) -> alt.Chart:
    color = (alt.condition(alt.datum.team == highlight, alt.value(BLUE), alt.value(NEUTRAL))
             if highlight else alt.value(BLUE))
    pts = alt.Chart(df).mark_circle(size=120, stroke="white", strokeWidth=2, opacity=1).encode(
        x=alt.X(f"{x}:Q", title=xt, scale=alt.Scale(zero=False)),
        y=alt.Y(f"{y}:Q", title=yt, scale=alt.Scale(zero=False)),
        color=color,
        tooltip=["team:N", alt.Tooltip(f"{x}:Q", title=xt, format=".1f"), alt.Tooltip(f"{y}:Q", title=yt, format=".1f")])
    # Put a label on the left when another team sits just to its right, to avoid collisions.
    xr, yr = df[x].max() - df[x].min(), df[y].max() - df[y].min()
    crowded = [((df[x] > r[x]) & (df[x] - r[x] < 0.12 * xr) & ((df[y] - r[y]).abs() < 0.04 * yr)).any()
               for _, r in df.iterrows()]
    df = df.assign(side=["left" if c else "right" for c in crowded])
    text = alt.Chart(df).encode(x=f"{x}:Q", y=f"{y}:Q", text="team:N", color=alt.value("#52514e"))
    right = text.transform_filter(alt.datum.side == "right").mark_text(align="left", dx=9, fontSize=11)
    left = text.transform_filter(alt.datum.side == "left").mark_text(align="right", dx=-9, fontSize=11)
    return (pts + right + left).properties(height=420)


def head_to_head(df: pd.DataFrame, p1: str, p2: str) -> alt.Chart:
    """df columns: stat, player, value (per-game average), pct (percentile among qualifying players)."""
    return alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("stat:N", title=None, sort=None, axis=alt.Axis(labelLimit=220)),
        yOffset=alt.YOffset("player:N", sort=[p1, p2]),
        x=alt.X("pct:Q", title="Percentile rank among players (per-game average)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("player:N", scale=alt.Scale(domain=[p1, p2], range=[BLUE, ORANGE]),
                        legend=alt.Legend(orient="top", title=None, labelLimit=320)),
        tooltip=["player:N", "stat:N", alt.Tooltip("value:Q", format=".2f", title="Per-game average"),
                 alt.Tooltip("pct:Q", format=".0f", title="Percentile")],
    ).properties(height=alt.Step(14))


def overlay_trend(df: pd.DataFrame, stat: str, p1: str, p2: str, window: int = 5) -> alt.Chart:
    label = STAT_LABELS.get(stat, stat)
    return alt.Chart(df).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=30)).encode(
        x=alt.X("date:T", title="Match date"),
        detail="season:N",  # break the line over the off-season
        y=alt.Y("rolling:Q", title=f"{label} ({window}-match rolling average)"),
        color=alt.Color("player:N", scale=alt.Scale(domain=[p1, p2], range=[BLUE, ORANGE]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["player:N", alt.Tooltip("date:T"), alt.Tooltip(f"{stat}:Q", title=f"{label} (match)"),
                 alt.Tooltip("rolling:Q", title="Rolling average", format=".1f")],
    ).properties(height=320)


def model_metric_bars(df: pd.DataFrame, metric: str) -> alt.Chart:
    """df columns: model, rmse, mae, r2."""
    d = df.assign(kind=lambda x: x["model"].str.startswith("Baseline").map({True: "Baseline", False: "Model"}))
    return alt.Chart(d).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("model:N", sort=MODEL_ORDER, title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X(f"{metric}:Q", title=metric.upper() + " on 2026 test matches (lower is better)"),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Model", "Baseline"], range=[BLUE, NEUTRAL]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["model:N", alt.Tooltip("rmse:Q", format=".3f", title="RMSE"),
                 alt.Tooltip("mae:Q", format=".3f", title="MAE"), alt.Tooltip("r2:Q", format=".3f", title="R²")],
    ).properties(height=alt.Step(26))


def predicted_vs_actual(df: pd.DataFrame, model: str, label: str) -> alt.Chart:
    """Binned density (handles 10k+ overlapping points) with the perfect-prediction diagonal."""
    step = 2 if df["actual"].max() < 60 else 5
    lo = min(0.0, float(df[model].min()), float(df["actual"].min()))
    lo = step * (lo // step)
    hi = step * (float(max(df["actual"].max(), df[model].max())) // step + 1)
    scale = alt.Scale(domain=[lo, hi], nice=False)
    heat = alt.Chart(df).mark_rect().encode(
        x=alt.X(f"{model}:Q", bin=alt.Bin(step=step, extent=[lo, hi]), title=f"Predicted {label}", scale=scale),
        y=alt.Y("actual:Q", bin=alt.Bin(step=step, extent=[lo, hi]), title=f"Actual {label}", scale=scale),
        color=alt.Color("count():Q", scale=alt.Scale(scheme="blues"), title="Player-matches"),
        tooltip=[alt.Tooltip("count():Q", title="Player-matches")])
    diag = alt.Chart(pd.DataFrame({"v": [lo, hi]})).mark_line(color=ORANGE, strokeDash=[6, 4], strokeWidth=2).encode(
        x=alt.X("v:Q", scale=scale), y=alt.Y("v:Q", scale=scale))
    return (heat + diag).properties(height=420, title=f"{model}: predicted vs actual (dashed line = perfect prediction)")


def error_by_group(df: pd.DataFrame, group: str, title: str) -> alt.Chart:
    """df columns: <group>, model, mae."""
    return alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y(f"{group}:N", title=None, axis=alt.Axis(labelLimit=200)),
        yOffset=alt.YOffset("model:N"),
        x=alt.X("mae:Q", title=title),
        color=alt.Color("model:N", scale=alt.Scale(range=[BLUE, NEUTRAL]), legend=alt.Legend(orient="top", title=None)),
        tooltip=[f"{group}:N", "model:N", alt.Tooltip("mae:Q", format=".2f", title="MAE"), "n:Q"],
    ).properties(height=alt.Step(14))


def importance_bars(df: pd.DataFrame, top: int = 20) -> alt.Chart:
    d = df.nlargest(top, "permutation_rmse_increase").assign(
        lo=lambda x: x["permutation_rmse_increase"] - x["permutation_std"],
        hi=lambda x: x["permutation_rmse_increase"] + x["permutation_std"])
    base = alt.Chart(d).encode(y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=280)))
    bars = base.mark_bar(color=BLUE, cornerRadiusEnd=4).encode(
        x=alt.X("permutation_rmse_increase:Q", title="Increase in test RMSE when feature is shuffled"),
        tooltip=["feature:N", alt.Tooltip("permutation_rmse_increase:Q", format=".3f", title="RMSE increase"),
                 alt.Tooltip("permutation_std:Q", format=".3f", title="Std (5 repeats)"),
                 alt.Tooltip("lgbm_gain_share:Q", format=".1%", title="LightGBM gain share")])
    err = base.mark_rule(color="#52514e").encode(x="lo:Q", x2="hi:Q")
    return (bars + err).properties(height=alt.Step(20))
