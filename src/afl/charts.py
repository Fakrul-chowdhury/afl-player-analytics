"""Altair chart builders shared by the dashboard and the README figure export."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import altair as alt
import pandas as pd

# Categorical palette adapted from the dashboard design (fixed order, never cycled). Validated for
# colour-vision-deficiency separation (all pairs) and >= 3:1 contrast on both the light (#ffffff) and
# dark (#1c2030) card surfaces; see docs/methodology.md. The same hues are used in both modes.
BLUE, ORANGE, AQUA = "#3d6be0", "#e8603c", "#17a673"

# Mode-dependent ink, surface and context colours. The dashboard calls set_mode() at the start of every
# script run (Streamlit runs each session in its own thread, hence thread-local); exports default to light.
_LIGHT = SimpleNamespace(ink="#1d2433", muted="#5b6474", surface="#ffffff", neutral="#bcc3d0",
                         diverging=["#e8603c", "#eef0f4", "#3d6be0"], seq=["#dfe7fb", "#1f3f99"])
_DARK = SimpleNamespace(ink="#e6e9f2", muted="#a3abbd", surface="#1c2030", neutral="#4a5266",
                        diverging=["#e8603c", "#2e3446", "#6b8ff0"], seq=["#26304a", "#9db8ff"])
_state = threading.local()


def set_mode(dark: bool) -> None:
    _state.dark = dark


def T() -> SimpleNamespace:
    return _DARK if getattr(_state, "dark", False) else _LIGHT
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
    dots = base.mark_circle(size=60, color=T().neutral, opacity=0.9).encode(
        y=alt.Y(f"{stat}:Q", title=label),
        tooltip=[alt.Tooltip("match:N", title="Match"), alt.Tooltip(f"{stat}:Q", title=label),
                 alt.Tooltip("rolling:Q", title=f"{window}-match average", format=".1f")])
    line = base.mark_line(color=BLUE, strokeWidth=2).encode(y="rolling:Q", detail="season:N")
    # Annotate the best game in the selection.
    best = d.loc[[d[stat].idxmax()]].assign(note=lambda x: "Best: " + x[stat].astype(str) + " (" + x["match"] + ")")
    best_pt = alt.Chart(best).mark_point(size=140, color=ORANGE, filled=False, strokeWidth=2).encode(
        x="date:T", y=f"{stat}:Q")
    best_lbl = alt.Chart(best).mark_text(dy=-14, fontSize=11, color=T().muted).encode(
        x="date:T", y=f"{stat}:Q", text="note:N")
    return (dots + line + best_pt + best_lbl).properties(
        height=340, title=f"{label} per match (grey), {window}-match rolling average (blue), best game circled")


def team_bars(df: pd.DataFrame, metric: str, title: str, highlight: str | None = None, fmt: str = ".1f") -> alt.Chart:
    color = (alt.condition(alt.datum.team == highlight, alt.value(BLUE), alt.value(T().neutral))
             if highlight else alt.value(BLUE))
    return alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("team:N", sort="-x", title=None),
        x=alt.X(f"{metric}:Q", title=title),
        color=color,
        tooltip=["team:N", alt.Tooltip(f"{metric}:Q", title=title, format=fmt)],
    ).properties(height=alt.Step(20))


def team_scatter(df: pd.DataFrame, x: str, y: str, xt: str, yt: str, highlight: str | None = None) -> alt.Chart:
    color = (alt.condition(alt.datum.team == highlight, alt.value(BLUE), alt.value(T().neutral))
             if highlight else alt.value(BLUE))
    pts = alt.Chart(df).mark_circle(size=120, stroke=T().surface, strokeWidth=2, opacity=1).encode(
        x=alt.X(f"{x}:Q", title=xt, scale=alt.Scale(zero=False)),
        y=alt.Y(f"{y}:Q", title=yt, scale=alt.Scale(zero=False)),
        color=color,
        tooltip=["team:N", alt.Tooltip(f"{x}:Q", title=xt, format=".1f"), alt.Tooltip(f"{y}:Q", title=yt, format=".1f")])
    # Put a label on the left when another team sits just to its right, to avoid collisions.
    xr, yr = df[x].max() - df[x].min(), df[y].max() - df[y].min()
    crowded = [((df[x] > r[x]) & (df[x] - r[x] < 0.12 * xr) & ((df[y] - r[y]).abs() < 0.04 * yr)).any()
               for _, r in df.iterrows()]
    df = df.assign(side=["left" if c else "right" for c in crowded])
    text = alt.Chart(df).encode(x=f"{x}:Q", y=f"{y}:Q", text="team:N", color=alt.value(T().muted))
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
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Model", "Baseline"], range=[BLUE, T().neutral]),
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
        color=alt.Color("count():Q", scale=alt.Scale(range=T().seq), title="Player-matches"),
        tooltip=[alt.Tooltip("count():Q", title="Player-matches")])
    diag = alt.Chart(pd.DataFrame({"v": [lo, hi]})).mark_line(color=ORANGE, strokeDash=[6, 4], strokeWidth=2).encode(
        x=alt.X("v:Q", scale=scale), y=alt.Y("v:Q", scale=scale))
    return (heat + diag).properties(height=420, title="Predicted vs actual (dashed = perfect)")


def error_by_group(df: pd.DataFrame, group: str, title: str) -> alt.Chart:
    """df columns: <group>, model, mae."""
    return alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y(f"{group}:N", title=None, axis=alt.Axis(labelLimit=200)),
        yOffset=alt.YOffset("model:N"),
        x=alt.X("mae:Q", title=title),
        # First model listed in df is the one being evaluated (blue); the second is the comparison baseline (grey).
        color=alt.Color("model:N", scale=alt.Scale(domain=list(dict.fromkeys(df["model"])), range=[BLUE, T().neutral]),
                        legend=alt.Legend(orient="top", title=None, labelLimit=260)),
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
    err = base.mark_rule(color=T().muted).encode(x="lo:Q", x2="hi:Q")
    return (bars + err).properties(height=alt.Step(20))


# --------------------------------------------------------------------------- season & team views


def ladder_bump(df: pd.DataFrame, highlight: str | None = None) -> alt.Chart:
    """df columns: round, team, position, pts, pct. Position 1 at the top; hover a line to highlight it."""
    sel = alt.selection_point(fields=["team"], on="pointerover", clear="pointerout", empty=False)
    hl = (alt.datum.team == highlight) if highlight else (alt.datum.team == "__none__")
    base = alt.Chart(df).encode(
        x=alt.X("round:O", title="Round (home-and-away)"),
        y=alt.Y("position:Q", title="Ladder position", scale=alt.Scale(domain=[18.5, 0.5], nice=False),
                axis=alt.Axis(values=list(range(1, 19)), grid=False)),
        detail="team:N")
    lines = base.mark_line(interpolate="monotone").encode(
        color=alt.when(sel).then(alt.value(BLUE)).when(hl).then(alt.value(BLUE)).otherwise(alt.value(T().neutral)),
        opacity=alt.when(sel).then(alt.value(1)).when(hl).then(alt.value(1)).otherwise(alt.value(0.45)),
        strokeWidth=alt.when(sel).then(alt.value(3)).when(hl).then(alt.value(3)).otherwise(alt.value(1.5)),
        tooltip=["team:N", "round:O", "position:Q", "pts:Q", alt.Tooltip("pct:Q", format=".1f", title="%")],
    ).add_params(sel)
    last = df[df["round"] == df["round"].max()]
    labels = alt.Chart(last).mark_text(align="left", dx=6, fontSize=10).encode(
        x="round:O", y="position:Q", text="team:N",
        color=alt.condition(hl, alt.value(BLUE), alt.value(T().muted)))
    return (lines + labels).properties(height=480)


def style_heatmap(df: pd.DataFrame, order: list[str]) -> alt.Chart:
    """df columns: team, stat, value, z. Diverging colour = standard deviations from the league average."""
    heat = alt.Chart(df).mark_rect(stroke=T().surface, strokeWidth=2).encode(
        x=alt.X("stat:N", title=None, sort=None, axis=alt.Axis(labelAngle=-35, labelLimit=120, orient="top",
                                                               labelOverlap=False, labelFontSize=11)),
        y=alt.Y("team:N", title=None, sort=order),
        color=alt.Color("z:Q", title="SDs vs league",
                        scale=alt.Scale(domain=[-2.5, 0, 2.5], range=T().diverging, clamp=True, interpolate="rgb")),
        tooltip=["team:N", "stat:N", alt.Tooltip("value:Q", format=".1f", title="Per game"),
                 alt.Tooltip("z:Q", format="+.2f", title="SDs vs league average")])
    text = alt.Chart(df).mark_text(fontSize=9, color=T().ink).encode(
        x=alt.X("stat:N", sort=None), y=alt.Y("team:N", sort=order), text=alt.Text("value:Q", format=".0f"))
    return (heat + text).properties(height=alt.Step(26), width=alt.Step(64))


def margin_bars(df: pd.DataFrame, team: str) -> alt.Chart:
    """df columns: n, round_label, opponent, margin, result, score, venue."""
    return alt.Chart(df).mark_bar(cornerRadius=3).encode(
        x=alt.X("n:O", title="Match (in date order)", axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("margin:Q", title=f"{team} margin (points)"),
        color=alt.Color("result:N", scale=alt.Scale(domain=["Win", "Loss", "Draw"], range=[BLUE, ORANGE, T().neutral]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("round_label:N", title="Round"), "opponent:N", "score:N", "margin:Q", "venue:N"],
    ).properties(height=260)


def season_trend(df: pd.DataFrame, y: str, title: str, fmt: str = ".1f") -> alt.Chart:
    base = alt.Chart(df).encode(x=alt.X("season:O", title="Season"))
    line = base.mark_line(color=BLUE, strokeWidth=2, point=alt.OverlayMarkDef(size=70, color=BLUE)).encode(
        y=alt.Y(f"{y}:Q", title=title, scale=alt.Scale(zero=False)),
        tooltip=["season:O", alt.Tooltip(f"{y}:Q", format=fmt, title=title)])
    text = base.mark_text(dy=-12, fontSize=11, color=T().muted).encode(
        y=f"{y}:Q", text=alt.Text(f"{y}:Q", format=fmt))
    return (line + text).properties(height=240)


def histogram(df: pd.DataFrame, col: str, title: str, step: float, color: str = BLUE) -> alt.Chart:
    return alt.Chart(df).mark_bar(color=color, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X(f"{col}:Q", bin=alt.Bin(step=step), title=title),
        y=alt.Y("count():Q", title="Matches"),
        tooltip=[alt.Tooltip(f"{col}:Q", bin=alt.Bin(step=step), title=title),
                 alt.Tooltip("count():Q", title="Matches")],
    ).properties(height=240)


def crowd_bars(df: pd.DataFrame) -> alt.Chart:
    """df columns: venue, avg_crowd, games."""
    return alt.Chart(df).mark_bar(color=BLUE, cornerRadiusEnd=4).encode(
        y=alt.Y("venue:N", sort="-x", title=None),
        x=alt.X("avg_crowd:Q", title="Average attendance"),
        tooltip=["venue:N", alt.Tooltip("avg_crowd:Q", format=",.0f", title="Average crowd"),
                 alt.Tooltip("games:Q", title="Matches")],
    ).properties(height=alt.Step(20))


# --------------------------------------------------------------------------- player views
def percentile_bars(df: pd.DataFrame, title: str) -> alt.Chart:
    """df columns: stat, value, pct. Dashed reference line at the median (50th percentile)."""
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("stat:N", sort=None, title=None, axis=alt.Axis(labelLimit=200)),
        x=alt.X("pct:Q", title=title, scale=alt.Scale(domain=[0, 105])),
        color=alt.condition(alt.datum.pct >= 50, alt.value(BLUE), alt.value(T().neutral)),
        tooltip=["stat:N", alt.Tooltip("value:Q", format=".2f", title="Per-game average"),
                 alt.Tooltip("pct:Q", format=".0f", title="Percentile")])
    text = alt.Chart(df).mark_text(align="left", dx=4, fontSize=10, color=T().muted).encode(
        y=alt.Y("stat:N", sort=None), x="pct:Q", text=alt.Text("pct:Q", format=".0f"))
    rule = alt.Chart(pd.DataFrame({"x": [50]})).mark_rule(strokeDash=[4, 4], color=T().muted).encode(x="x:Q")
    return (bars + text + rule).properties(height=alt.Step(22))


def split_bars(df: pd.DataFrame, group: str, stat_label: str, overall: float) -> alt.Chart:
    """df columns: <group>, value, games. Dashed orange rule = the player's overall average."""
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y(f"{group}:N", sort="-x", title=None),
        x=alt.X("value:Q", title=f"Average {stat_label.lower()}"),
        color=alt.condition(alt.datum.value >= overall, alt.value(BLUE), alt.value(T().neutral)),
        tooltip=[f"{group}:N", alt.Tooltip("value:Q", format=".1f", title="Average"),
                 alt.Tooltip("games:Q", title="Games")])
    rule = alt.Chart(pd.DataFrame({"x": [overall]})).mark_rule(
        strokeDash=[4, 4], color=ORANGE, strokeWidth=2).encode(
        x="x:Q", tooltip=[alt.Tooltip("x:Q", format=".1f", title="Overall average")])
    return (bars + rule).properties(height=alt.Step(22))


def distribution_compare(df: pd.DataFrame, stat: str, players: list[str]) -> alt.Chart:
    """Box plots (median, IQR, whiskers at 1.5 IQR) per player for one stat."""
    label = STAT_LABELS.get(stat, stat)
    colors = [BLUE, ORANGE, AQUA][: len(players)]
    return alt.Chart(df).mark_boxplot(size=34, median={"color": T().surface}, ticks=True).encode(
        y=alt.Y("player:N", title=None, sort=players, axis=alt.Axis(labelLimit=240)),
        x=alt.X(f"{stat}:Q", title=f"{label} per match"),
        color=alt.Color("player:N", scale=alt.Scale(domain=players, range=colors), legend=None),
    ).properties(height=alt.Step(56))


def prediction_band(df: pd.DataFrame, label: str, model: str) -> alt.Chart:
    """df columns: date, round_label, opponent, actual, <model>, pi_lower, pi_upper."""
    base = alt.Chart(df).encode(x=alt.X("date:T", title="Match date (2026)"))
    band = base.mark_area(color=BLUE, opacity=0.15).encode(y=alt.Y("pi_lower:Q", title=label), y2="pi_upper:Q")
    pred = base.mark_line(color=BLUE, strokeWidth=2).encode(y=f"{model}:Q")
    act = base.mark_circle(size=70, color=T().ink, opacity=1).encode(
        y="actual:Q",
        tooltip=[alt.Tooltip("round_label:N", title="Round"), "opponent:N",
                 alt.Tooltip("actual:Q", title="Actual"),
                 alt.Tooltip(f"{model}:Q", format=".1f", title="Predicted"),
                 alt.Tooltip("pi_lower:Q", format=".1f", title="80% interval low"),
                 alt.Tooltip("pi_upper:Q", format=".1f", title="80% interval high")])
    return (band + pred + act).properties(
        height=300, title="Actual (black dots) vs predicted (blue line), shaded band = 80% prediction interval")


# --------------------------------------------------------------------------- league views
def leader_bars(df: pd.DataFrame, value: str, title: str, fmt: str = ".1f") -> alt.Chart:
    """df columns: label, <value>, games."""
    bars = alt.Chart(df).mark_bar(color=BLUE, cornerRadiusEnd=4).encode(
        y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X(f"{value}:Q", title=title),
        tooltip=["label:N", alt.Tooltip(f"{value}:Q", format=fmt, title=title), alt.Tooltip("games:Q", title="Games")])
    text = alt.Chart(df).mark_text(align="left", dx=4, fontSize=10, color=T().muted).encode(
        y=alt.Y("label:N", sort="-x"), x=f"{value}:Q", text=alt.Text(f"{value}:Q", format=fmt))
    return (bars + text).properties(height=alt.Step(20))


def role_scatter(df: pd.DataFrame, x: str, y: str, role: str) -> alt.Chart:
    """All qualifying players; the selected role is blue, the rest grey."""
    sel = alt.datum.role == role
    pts = alt.Chart(df).mark_circle(size=55, stroke=T().surface, strokeWidth=1).encode(
        x=alt.X(f"{x}:Q", title=f"{STAT_LABELS[x]} per game"),
        y=alt.Y(f"{y}:Q", title=f"{STAT_LABELS[y]} per game"),
        color=alt.condition(sel, alt.value(BLUE), alt.value(T().neutral)),
        opacity=alt.condition(sel, alt.value(0.95), alt.value(0.4)),
        order=alt.condition(sel, alt.value(1), alt.value(0)),
        tooltip=["player_name:N", "team:N", "role:N", "games:Q",
                 alt.Tooltip(f"{x}:Q", format=".1f", title=STAT_LABELS[x]),
                 alt.Tooltip(f"{y}:Q", format=".1f", title=STAT_LABELS[y])])
    top = df[df["role"] == role].nlargest(6, y)
    lbl = alt.Chart(top).mark_text(align="left", dx=7, fontSize=10, color=T().muted).encode(
        x=f"{x}:Q", y=f"{y}:Q", text="player_name:N")
    return (pts + lbl).properties(height=460)


# --------------------------------------------------------------------------- model views
def calibration(df: pd.DataFrame, model: str, label: str) -> alt.Chart:
    """df columns: decile, pred_mean, actual_mean, lo, hi, n (actual mean ± 1.96 SE per predicted decile)."""
    lo_v = float(min(df["pred_mean"].min(), df["lo"].min()))
    hi_v = float(max(df["pred_mean"].max(), df["hi"].max()))
    pad = (hi_v - lo_v) * 0.05
    scale = alt.Scale(domain=[lo_v - pad, hi_v + pad], nice=False)
    diag = alt.Chart(pd.DataFrame({"v": [lo_v - pad, hi_v + pad]})).mark_line(
        color=T().neutral, strokeDash=[6, 4], strokeWidth=2).encode(
        x=alt.X("v:Q", scale=scale), y=alt.Y("v:Q", scale=scale))
    err = alt.Chart(df).mark_rule(color=BLUE, strokeWidth=2).encode(
        x=alt.X("pred_mean:Q", scale=scale), y=alt.Y("lo:Q", scale=scale), y2="hi:Q")
    pts = alt.Chart(df).mark_circle(size=90, color=BLUE, stroke=T().surface, strokeWidth=2, opacity=1).encode(
        x=alt.X("pred_mean:Q", title=f"Mean predicted {label} (per decile of predictions)", scale=scale),
        y=alt.Y("actual_mean:Q", title=f"Mean actual {label}", scale=scale),
        tooltip=["decile:O", alt.Tooltip("pred_mean:Q", format=".2f", title="Mean predicted"),
                 alt.Tooltip("actual_mean:Q", format=".2f", title="Mean actual"), "n:Q"])
    return (diag + err + pts).properties(
        height=420, title="Calibration by prediction decile (dashed = unbiased)")


def error_by_round(df: pd.DataFrame, model: str) -> alt.Chart:
    """df columns: round, model, mae, n."""
    return alt.Chart(df).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=35)).encode(
        x=alt.X("round:O", title="2026 round", sort=None),
        y=alt.Y("mae:Q", title="Mean absolute error", scale=alt.Scale(zero=False)),
        color=alt.Color("model:N", scale=alt.Scale(domain=[model, "Baseline: last-5 average"], range=[BLUE, T().neutral]),
                        legend=alt.Legend(orient="top", title=None, labelLimit=260)),
        tooltip=["round:O", "model:N", alt.Tooltip("mae:Q", format=".2f", title="MAE"), "n:Q"],
    ).properties(height=280)


def residual_hist(df: pd.DataFrame, label: str, step: float) -> alt.Chart:
    bars = alt.Chart(df).mark_bar(color=BLUE, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("residual:Q", bin=alt.Bin(step=step), title=f"Actual minus predicted {label}"),
        y=alt.Y("count():Q", title="Player-matches"),
        tooltip=[alt.Tooltip("residual:Q", bin=alt.Bin(step=step), title="Residual"),
                 alt.Tooltip("count():Q", title="Rows")])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        color=ORANGE, strokeDash=[4, 4], strokeWidth=2).encode(x="x:Q")
    return (bars + zero).properties(height=280)


FEATURE_FAMILIES = {
    "Recent form (rolling averages)": lambda f: "_avg" in f,
    "Season-to-date & last match": lambda f: f.endswith(("_season_avg", "_last")),
    "Form volatility": lambda f: f.endswith("_std10"),
    "Player profile (age, experience, rest)": lambda f: f in (
        "games_in_data", "career_games_before", "age_years", "days_since_last", "sub_last_match"),
    "Team & opponent context": lambda f: f in ("team_disp_for5", "opp_disp_conceded5", "opp_fp_conceded5"),
    "Match setting (home, final, venue)": lambda f: f in ("is_home", "is_final", "venue"),
    "Identity (team, opponent, role)": lambda f: f in ("team", "opponent", "role"),
}


def family_of(feature: str) -> str:
    return next((k for k, fn in FEATURE_FAMILIES.items() if fn(feature)), "Other")


def family_bars(df: pd.DataFrame) -> alt.Chart:
    """df columns: family, gain_share (LightGBM gain share summed per family), features."""
    bars = alt.Chart(df).mark_bar(color=BLUE, cornerRadiusEnd=4).encode(
        y=alt.Y("family:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("gain_share:Q", title="Share of LightGBM split gain", axis=alt.Axis(format="%")),
        tooltip=["family:N", alt.Tooltip("gain_share:Q", format=".1%", title="Gain share"), "features:Q"])
    text = alt.Chart(df).mark_text(align="left", dx=4, fontSize=10, color=T().muted).encode(
        y=alt.Y("family:N", sort="-x"), x="gain_share:Q", text=alt.Text("gain_share:Q", format=".1%"))
    return (bars + text).properties(height=alt.Step(26))


# --------------------------------------------------------------------------- new visualisations
def contribution_waterfall(df: pd.DataFrame, label: str) -> alt.Chart:
    """Waterfall: base value, then each feature's contribution in order, ending at the prediction.

    Expects columns: step (display label), start, end, right, delta, kind ('Base / total' | 'Raises prediction'
    | 'Lowers prediction'), value_text, delta_text. Base and total are drawn as markers, not bars from zero,
    so the axis can zoom to the range where the contributions happen.
    """
    order = df["step"].tolist()
    y = alt.Y("step:N", sort=order, title=None, axis=alt.Axis(labelLimit=320))
    tips = [alt.Tooltip("step:N", title="Step"), alt.Tooltip("value_text:N", title="Feature value"),
            alt.Tooltip("delta:Q", title="Contribution", format="+.2f"),
            alt.Tooltip("end:Q", title="Running total", format=".2f")]
    steps, ends = df[df["kind"] != "Base / total"], df[df["kind"] == "Base / total"]
    bars = alt.Chart(steps).mark_bar(cornerRadius=3, height=16).encode(
        y=y, x=alt.X("start:Q", title=label, scale=alt.Scale(zero=False)), x2="end:Q",
        color=alt.Color("kind:N", title=None, scale=alt.Scale(domain=["Raises prediction", "Lowers prediction"],
                                                              range=[BLUE, ORANGE]), legend=alt.Legend(orient="top")),
        tooltip=tips)
    marks = alt.Chart(ends).mark_tick(thickness=4, size=22, color=T().ink).encode(y=y, x="end:Q", tooltip=tips)
    text = alt.Chart(df).mark_text(align="left", dx=6, fontSize=10, color=T().muted).encode(
        y=y, x="right:Q", text="delta_text:N")
    return (bars + marks + text).properties(height=alt.Step(24))


def similarity_map(df: pd.DataFrame, xt: str, yt: str) -> alt.Chart:
    """2-D projection; df has x, y, label, group ('Selected' | 'Most similar' | 'Other')."""
    groups = ["Selected", "Most similar", "Other players"]
    pts = alt.Chart(df).mark_circle(stroke=T().surface, strokeWidth=1).encode(
        x=alt.X("x:Q", title=xt), y=alt.Y("y:Q", title=yt),
        color=alt.Color("group:N", title=None, scale=alt.Scale(domain=groups, range=[BLUE, ORANGE, T().neutral]),
                        legend=alt.Legend(orient="top")),
        size=alt.Size("group:N", scale=alt.Scale(domain=groups, range=[220, 110, 40]), legend=None),
        opacity=alt.Opacity("group:N", scale=alt.Scale(domain=groups, range=[1, 1, 0.55]), legend=None),
        order=alt.Order("rank:Q", sort="descending"),
        tooltip=[alt.Tooltip("label:N", title="Player"), alt.Tooltip("role:N", title="Inferred role"),
                 alt.Tooltip("games:Q", title="Games"), alt.Tooltip("disposals:Q", title="Disposals/g", format=".1f"),
                 alt.Tooltip("goals:Q", title="Goals/g", format=".2f")])
    lbl = alt.Chart(df[df["group"] != "Other players"]).mark_text(align="left", dx=8, fontSize=10,
                                                                  color=T().ink).encode(
        x="x:Q", y="y:Q", text="name:N")
    return (pts + lbl).properties(height=520)


def league_spread(df: pd.DataFrame, player: str) -> alt.Chart:
    """One row per stat; every qualifying player is a dot, the selected player a large blue dot.

    df: stat, value, label, is_player, jitter, pct.
    """
    base = alt.Chart(df)
    others = base.transform_filter(~alt.datum.is_player).mark_circle(size=22, opacity=0.45,
                                                                       color=T().neutral).encode(
        x=alt.X("value:Q", title=None, axis=alt.Axis(tickCount=5)),
        y=alt.Y("jitter:Q", axis=None, scale=alt.Scale(domain=[-1.4, 1.4])),
        tooltip=[alt.Tooltip("label:N", title="Player"), alt.Tooltip("value:Q", title="Per game", format=".2f"),
                 alt.Tooltip("pct:Q", title="Percentile", format=".0f")])
    me = base.transform_filter(alt.datum.is_player).mark_circle(size=170, color=BLUE, opacity=1,
                                                                 stroke=T().surface, strokeWidth=2).encode(
        x="value:Q", y=alt.Y("jitter:Q", axis=None),
        tooltip=[alt.Tooltip("label:N", title="Player"), alt.Tooltip("value:Q", title="Per game", format=".2f"),
                 alt.Tooltip("pct:Q", title="Percentile", format=".0f")])
    txt = base.transform_filter(alt.datum.is_player).mark_text(dy=-15, fontSize=10, fontWeight=600,
                                                                color=T().ink).encode(
        x="value:Q", y=alt.Y("jitter:Q", axis=None), text=alt.Text("pct_text:N"))
    return (others + me + txt).properties(height=46, width=780).facet(
        row=alt.Row("stat:N", title=None, sort=list(dict.fromkeys(df["stat"])),
                    header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=11, labelLimit=170)),
        spacing=4,
    ).resolve_scale(x="independent")


def age_curve(league: pd.DataFrame, player: pd.DataFrame, label: str, name: str) -> alt.Chart:
    """League mean by age with an interquartile band; the player's per-age averages overlaid."""
    base = alt.Chart(league).encode(x=alt.X("age:Q", title="Age at match (years)", scale=alt.Scale(zero=False),
                                            axis=alt.Axis(tickMinStep=1, format="d")))
    band = base.mark_area(color=T().neutral, opacity=0.35).encode(
        y=alt.Y("q25:Q", title=f"{label} per game"), y2="q75:Q")
    mean = base.mark_line(color=T().muted, strokeWidth=2, strokeDash=[5, 3]).encode(
        y="mean:Q", tooltip=[alt.Tooltip("age:Q", title="Age"), alt.Tooltip("mean:Q", title="League mean", format=".1f"),
                             alt.Tooltip("q25:Q", title="25th pct", format=".1f"),
                             alt.Tooltip("q75:Q", title="75th pct", format=".1f"),
                             alt.Tooltip("n:Q", title="Player-matches", format=",")])
    pl_ = alt.Chart(player).encode(x="age:Q", y="mean:Q")
    line = pl_.mark_line(color=BLUE, strokeWidth=2.5)
    pts = pl_.mark_circle(size=90, color=BLUE, opacity=1, stroke=T().surface, strokeWidth=2).encode(
        tooltip=[alt.Tooltip("age:Q", title="Age"), alt.Tooltip("mean:Q", title=name, format=".1f"),
                 alt.Tooltip("games:Q", title="Games at this age")])
    return (band + mean + line + pts).properties(height=360)


def elo_lines(df: pd.DataFrame, highlight: list[str]) -> alt.Chart:
    """Elo rating after each match; highlighted teams in colour, the rest as faint context lines."""
    hl = highlight[:3]
    ctx = alt.Chart(df[~df["team"].isin(hl)]).mark_line(strokeWidth=1, opacity=0.5, color=T().neutral,
                                                        interpolate="step-after").encode(
        x=alt.X("date:T", title=None), y=alt.Y("elo:Q", title="Elo rating", scale=alt.Scale(zero=False)),
        detail="team:N", tooltip=["team:N", alt.Tooltip("date:T", title="After match on"),
                                  alt.Tooltip("elo:Q", format=".0f")])
    main = alt.Chart(df[df["team"].isin(hl)]).mark_line(strokeWidth=2.5, interpolate="step-after").encode(
        x="date:T", y="elo:Q",
        color=alt.Color("team:N", title=None, scale=alt.Scale(domain=hl, range=[BLUE, ORANGE, AQUA][: len(hl)]),
                        legend=alt.Legend(orient="top")),
        tooltip=["team:N", alt.Tooltip("date:T", title="After match on"), alt.Tooltip("opponent:N"),
                 alt.Tooltip("result:N"), alt.Tooltip("elo:Q", title="Elo", format=".0f")])
    mid = alt.Chart(pd.DataFrame({"y": [1500]})).mark_rule(strokeDash=[4, 4], color=T().muted).encode(y="y:Q")
    return (ctx + mid + main).properties(height=400)


def hga_bars(df: pd.DataFrame) -> alt.Chart:
    """Home-ground advantage estimate per team with a 95% confidence interval."""
    base = alt.Chart(df).encode(y=alt.Y("team:N", sort="-x", title=None))
    bars = base.mark_bar(cornerRadiusEnd=4, height=14).encode(
        x=alt.X("hga:Q", title="Home advantage (points per game)"),
        color=alt.condition(alt.datum.hga >= 0, alt.value(BLUE), alt.value(ORANGE)),
        tooltip=[alt.Tooltip("team:N"), alt.Tooltip("hga:Q", title="Estimate", format="+.1f"),
                 alt.Tooltip("lo:Q", title="95% CI low", format="+.1f"),
                 alt.Tooltip("hi:Q", title="95% CI high", format="+.1f"),
                 alt.Tooltip("home_n:Q", title="Home games"), alt.Tooltip("away_n:Q", title="Away games")])
    err = base.mark_rule(color=T().muted).encode(x="lo:Q", x2="hi:Q")
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=T().muted).encode(x="x:Q")
    return (bars + err + zero).properties(height=alt.Step(22))


def stat_correlations(df: pd.DataFrame) -> alt.Chart:
    """Correlation between a team-stat difference and the final margin, one bar per stat."""
    base = alt.Chart(df).encode(y=alt.Y("label:N", sort="-x", title=None))
    bars = base.mark_bar(cornerRadiusEnd=4, height=14).encode(
        x=alt.X("r:Q", title="Correlation with final margin (r)", scale=alt.Scale(domain=[-1, 1])),
        color=alt.condition(alt.datum.r >= 0, alt.value(BLUE), alt.value(ORANGE)),
        tooltip=[alt.Tooltip("label:N", title="Stat difference"), alt.Tooltip("r:Q", format="+.2f"),
                 alt.Tooltip("slope:Q", title="Margin per +1 difference", format="+.2f"),
                 alt.Tooltip("win_pct:Q", title="Win % when ahead on stat", format=".0%"),
                 alt.Tooltip("n:Q", title="Matches")])
    txt = base.mark_text(align="left", dx=4, fontSize=10, color=T().muted).encode(
        x=alt.X("r_pos:Q"), text=alt.Text("r:Q", format="+.2f"))
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=T().muted).encode(x="x:Q")
    return (bars + txt + zero).properties(height=alt.Step(22))


def diff_scatter(df: pd.DataFrame, label: str) -> alt.Chart:
    """Each match (home-team perspective): stat difference vs final margin, with least-squares line."""
    pts = alt.Chart(df).mark_circle(size=26, opacity=0.5, color=BLUE).encode(
        x=alt.X("diff:Q", title=f"{label} difference (home − away)"),
        y=alt.Y("margin:Q", title="Final margin (home − away, points)"),
        tooltip=[alt.Tooltip("match:N", title="Match"), alt.Tooltip("diff:Q", title=f"{label} diff"),
                 alt.Tooltip("margin:Q", title="Margin")])
    fit = alt.Chart(df).transform_regression("diff", "margin").mark_line(color=ORANGE, strokeWidth=2.5).encode(
        x="diff:Q", y="margin:Q")
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=T().muted, strokeDash=[4, 4]).encode(y="y:Q")
    return (zero + pts + fit).properties(height=360)


def form_calendar(df: pd.DataFrame, label: str, rounds: list[str]) -> alt.Chart:
    """Season x round grid coloured by a player's output; blank cells = did not play."""
    rect = alt.Chart(df).mark_rect(cornerRadius=3, stroke=T().surface, strokeWidth=2).encode(
        x=alt.X("col:N", sort=rounds, title="Round (finals at right)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("season:O", title=None, sort="descending"),
        color=alt.Color("value:Q", title=label, scale=alt.Scale(range=T().seq)),
        tooltip=[alt.Tooltip("season:O"), alt.Tooltip("round_label:N", title="Round"),
                 alt.Tooltip("opponent:N"), alt.Tooltip("result:N"), alt.Tooltip("value:Q", title=label)])
    return rect.properties(height=alt.Step(34))


def with_without(df: pd.DataFrame, metric: str, title: str, fmt: str) -> alt.Chart:
    """Grouped bars by season: team result with vs without the player."""
    groups = ["With player", "Without player"]
    base = alt.Chart(df).encode(x=alt.X("season:N", title=None, axis=alt.Axis(labelAngle=0)),
                                xOffset=alt.XOffset("group:N", sort=groups))
    bars = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        y=alt.Y(f"{metric}:Q", title=title, axis=alt.Axis(format=fmt)),
        color=alt.Color("group:N", title=None, scale=alt.Scale(domain=groups, range=[BLUE, T().neutral]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("season:N"), alt.Tooltip("group:N", title=""),
                 alt.Tooltip(f"{metric}:Q", title=title, format=fmt), alt.Tooltip("games:Q", title="Team games")])
    txt = base.mark_text(dy=-6, fontSize=10, color=T().muted).encode(
        y=f"{metric}:Q", text=alt.Text("games:Q", format="d"))
    return (bars + txt).properties(height=280)
