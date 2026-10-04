"""Model pages: model comparison, per-prediction explanations, feature importance, methodology."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import common as c
import ui
from common import BASELINE, STAT_LABELS, charts, pretty_feature

BINARY = {"is_home", "is_final", "sub_last_match"}  # stored as 0/1 model inputs


def page_models() -> None:
    ui.header("Prediction models", "Next-match disposals and AFL Fantasy points, predicted before the bounce. "
              "Train 2021–2024 · tune 2025 · test 2026 (held out).")
    m = c.load_json(str(c.RESULTS / "metrics.json"))
    with ui.filter_bar():
        f = st.columns([2, 2, 3])
        tgt = f[0].radio("Target", list(m["targets"]), horizontal=True, format_func=lambda s: STAT_LABELS[s])
        res = m["targets"][tgt]
        best = res["selected_model"]
        metric = f[1].radio("Metric", ["rmse", "mae"], horizontal=True, format_func=str.upper)
    tab = pd.DataFrame(res["test_2026"]).T.reset_index(names="model")
    base_rmse = tab.loc[tab["model"] == BASELINE, "rmse"].iat[0]
    sel_row = tab[tab["model"] == best].iloc[0]
    iv = res["interval_80"]
    k = st.columns(4)
    ui.kpi(k[0], "Selected model", best)
    ui.kpi(k[1], "Test RMSE", f"{sel_row.rmse:.2f}", f"{sel_row.rmse / base_rmse - 1:+.1%} vs last-5 baseline",
           inverse=True)
    ui.kpi(k[2], "Test MAE", f"{sel_row.mae:.2f}")
    ui.kpi(k[3], "80% interval coverage", f"{iv['test_2026_coverage']:.1%}", f"avg width {iv['mean_width']:.1f}",
           neutral=True)

    left, right = st.columns([3, 2])
    with left, ui.card(f"Test {metric.upper()} by model (2026)"):
        c.chart(charts.model_metric_bars(tab, metric))
    with right, ui.card("Scores", f"{int(tab['n'].iat[0]):,} player-matches in the 2026 test season. Selected model "
                                  "= lowest RMSE on the 2025 validation season."):
        tab["RMSE vs last-5 baseline"] = (tab["rmse"] / base_rmse - 1).map("{:+.1%}".format)
        st.dataframe(tab.drop(columns="n").rename(columns={"rmse": "RMSE", "mae": "MAE", "r2": "R²"})
                     .set_index("model").round(3), width="stretch")

    preds = c.sql("SELECT * FROM preds WHERE target = ? AND NOT source_conflict", (tgt,))
    label = STAT_LABELS[tgt].lower()
    model = f[2].selectbox("Model for diagnostics", [x for x in charts.MODEL_ORDER if x in preds.columns],
                           index=charts.MODEL_ORDER.index(best))
    preds["residual"] = preds["actual"] - preds[model]
    left, right = st.columns(2)
    with left, ui.card("Predicted vs actual", "Density of every 2026 prediction against what happened."):
        c.chart(charts.predicted_vs_actual(preds, model, label))
    with right, ui.card("Calibration", "Predictions grouped into ten equal-sized bins; each dot is the bin's mean "
                                       "prediction vs mean actual, with a 95% confidence bar. Dots on the dashed line "
                                       "mean the model is unbiased across the whole range."):
        preds["decile"] = pd.qcut(preds[model], 10, labels=range(1, 11))
        cal = preds.groupby("decile", observed=True).agg(
            pred_mean=(model, "mean"), actual_mean=("actual", "mean"), sd=("actual", "std"), n=("actual", "size"))
        cal["lo"] = cal.actual_mean - 1.96 * cal.sd / np.sqrt(cal.n)
        cal["hi"] = cal.actual_mean + 1.96 * cal.sd / np.sqrt(cal.n)
        c.chart(charts.calibration(cal.reset_index(), model, label))

    left, right = st.columns(2)
    rounds = preds[preds["round_label"].str.isdigit()].assign(round=lambda d: d.round_label.astype(int))
    er = pd.concat([rounds.groupby("round").apply(
        lambda d, col=col: pd.Series({"mae": np.abs(d.actual - d[col]).mean(), "n": len(d)}), include_groups=False)
        .reset_index().assign(model=col) for col in (model, BASELINE)])
    with left, ui.card("Error across the 2026 season", "Mean absolute error per home-and-away round: the model's "
                                                       "edge over the baseline is consistent, not driven by a few rounds."):
        c.chart(charts.error_by_round(er, model))
    tol = 5 if tgt == "disposals" else 20
    with right, ui.card("Residuals", f"Mean residual {preds.residual.mean():+.2f} (bias); "
                                     f"{(preds.residual.abs() <= tol).mean():.0%} of predictions within {tol} {label}."):
        c.chart(charts.residual_hist(preds, label, 1 if tgt == "disposals" else 5))

    rows = [{"role": role, "model": name, "mae": d[name]["mae"], "n": d[name]["n"]}
            for role, d in res["test_2026_by_role"].items() for name in (model, BASELINE)]
    left, right = st.columns(2)
    with left, ui.card("Error by inferred role"):
        c.chart(charts.error_by_group(pd.DataFrame(rows), "role", "Mean absolute error"))
    with right, ui.card("Biggest surprises of 2026", "Largest gaps between actual output and the pre-match prediction."):
        cols = ["player_name", "team", "opponent", "round_label", "actual", model]
        tabs = st.tabs(["Outperformed", "Underperformed"])
        tabs[0].dataframe(preds.nlargest(10, "residual")[cols].round(1), width="stretch", hide_index=True)
        tabs[1].dataframe(preds.nsmallest(10, "residual")[cols].round(1), width="stretch", hide_index=True)

    with st.expander("Validation (2025) scores used for tuning, and interval method"):
        st.dataframe(pd.DataFrame(res["validation_2025"]).T.round(3), width="stretch")
        st.json({"tuning": res["tuning"], "interval_80": iv})


def page_explain() -> None:
    ui.header("Why this prediction?", "A feature-by-feature breakdown of any 2026 prediction, from LightGBM's "
              "built-in contribution values.")
    m = c.load_json(str(c.RESULTS / "metrics.json"))
    players = c.sql("""SELECT player_id, any_value(player_name) AS player_name, arg_max(team, date) AS team,
                              count(*) AS games
                       FROM preds WHERE target = 'disposals' GROUP BY 1 ORDER BY games DESC""")
    labels = c.label_for(players)
    ids = players["player_id"].tolist()
    with ui.filter_bar():
        f = st.columns([3, 3, 2])
        pid = f[0].selectbox("Player (2026)", ids, format_func=labels.get, index=c.default_index(ids, "N/Nick_Daicos"),
                             key="ex_player")
        tgt = f[2].radio("Target", list(m["targets"]), horizontal=True, format_func=lambda s: STAT_LABELS[s],
                         key="ex_tgt")
        p = c.sql("SELECT * FROM preds WHERE player_id = ? AND target = ? ORDER BY date", (pid, tgt))
        rnd = p["round_label"].where(~p["round_label"].str.isdigit(), "Round " + p["round_label"])
        p["match"] = rnd + " v " + p["opponent"] + " (" + p["date"].dt.strftime("%d %b") + ")"
        match = f[1].selectbox("Match", p["match_id"].tolist()[::-1], format_func=dict(zip(p.match_id, p.match)).get,
                               key="ex_match")
    row = p[p["match_id"] == match].iloc[0]
    sel = m["targets"][tgt]["selected_model"]
    ex = c.sql("SELECT * FROM explain WHERE player_id = ? AND match_id = ? AND target = ?", (pid, match, tgt))
    base = float(ex["base_value"].iat[0])
    k = st.columns(5)
    ui.kpi(k[0], "Actual", f"{row.actual:.0f}", neutral=True)
    ui.kpi(k[1], "LightGBM prediction", f"{row.LightGBM:.1f}", f"{row.actual - row.LightGBM:+.1f} actual − predicted",
           neutral=True)
    ui.kpi(k[2], f"{sel}", f"{row[sel]:.1f}", "the dashboard's headline model", neutral=True)
    ui.kpi(k[3], "80% interval", f"{row.pi_lower:.0f}–{row.pi_upper:.0f}",
           "inside" if row.pi_lower <= row.actual <= row.pi_upper else "outside", neutral=True)
    ui.kpi(k[4], "Starting point", f"{base:.1f}", "average LightGBM prediction (training data)", neutral=True)

    ex["abs"] = ex["contribution"].abs()
    flags = ex["feature"].isin(BINARY)
    ex.loc[flags, "feature_value"] = ex.loc[flags, "feature_value"].map({"1": "yes", "0": "no"}).fillna(
        ex.loc[flags, "feature_value"])
    other = ex[ex["feature"] == "__other__"]
    top = ex[ex["feature"] != "__other__"].sort_values("abs", ascending=False)
    steps = [("Starting point (average prediction)", base, "", "Base / total")]
    steps += [(f"{pretty_feature(r.feature)} = {r.feature_value}", r.contribution, r.feature_value,
               "Raises prediction" if r.contribution >= 0 else "Lowers prediction") for r in top.itertuples()]
    if not other.empty:
        o = other.iloc[0]
        steps.append((f"All other features ({o.feature_value})", o.contribution, o.feature_value,
                      "Raises prediction" if o.contribution >= 0 else "Lowers prediction"))
    rows, run = [], 0.0
    for i, (step, delta, val, kind) in enumerate(steps):
        start = 0.0 if i == 0 else run
        run = delta if i == 0 else run + delta
        rows.append({"step": step, "start": start if i else run, "end": run, "delta": delta if i else np.nan,
                     "value_text": val, "kind": kind})
    rows.append({"step": "LightGBM prediction", "start": run, "end": run, "delta": np.nan, "value_text": "",
                 "kind": "Base / total"})
    w = pd.DataFrame(rows)
    w["right"] = w[["start", "end"]].max(axis=1)
    w["delta_text"] = np.where(w["delta"].isna(), w["end"].map("{:.1f}".format), w["delta"].map("{:+.2f}".format))
    label = STAT_LABELS[tgt]
    with ui.card(f"{row.player_name}: predicted {label.lower()} v {row.opponent}, {row.date:%d %b %Y}",
                 "Read top to bottom: start from the average prediction, then each feature pushes the prediction up "
                 "(blue) or down (coral) until it lands on the final LightGBM prediction. Values are LightGBM's exact "
                 "per-feature contributions (TreeSHAP via pred_contrib); the 12 largest are shown and the rest are "
                 "summed in one row. Only information available before the match is used."):
        c.chart(charts.contribution_waterfall(w, label))
    st.caption(f"The headline model is the {sel}; LightGBM is one of its three members, and the only one explained "
               f"here. Here the ensemble predicted {row[sel]:.1f} vs LightGBM's {row.LightGBM:.1f}. Contributions show "
               "what the model relied on, not what caused the result.")

    allp = c.sql("SELECT feature, avg(abs(contribution)) AS mean_abs FROM explain WHERE player_id = ? AND target = ? "
                 "AND feature <> '__other__' GROUP BY 1 ORDER BY 2 DESC LIMIT 10", (pid, tgt))
    allp["feature"] = allp["feature"].map(pretty_feature)
    with ui.card(f"What drives {row.player_name}'s predictions across 2026",
                 "Average size of each feature's contribution over all of this player's 2026 predictions (features "
                 "outside a match's top 12 count as zero for that match)."):
        c.chart(charts.leader_bars(allp.rename(columns={"feature": "label"}), "mean_abs",
                                   f"Mean |contribution| ({label.lower()})", ".2f"))


def page_importance() -> None:
    ui.header("Feature importance", "What the models rely on, measured on unseen 2026 data.")
    imp = pd.read_parquet(c.RESULTS / "feature_importance.parquet")
    with ui.filter_bar():
        f = st.columns([2, 3])
        tgt = f[0].radio("Target", imp["target"].unique().tolist(), horizontal=True,
                         format_func=lambda s: STAT_LABELS[s])
    d = imp[imp["target"] == tgt].copy()
    d["family"] = d["feature"].map(charts.family_of)
    fam = d.groupby("family").agg(gain_share=("lgbm_gain_share", "sum"), features=("feature", "size")).reset_index()
    d["feature"] = d["feature"].map(pretty_feature)
    top = d.nlargest(1, "permutation_rmse_increase").iloc[0]
    zoom = f[1].toggle(f"Zoom in: hide the dominant feature ({top.feature}, +{top.permutation_rmse_increase:.2f} RMSE)")
    left, right = st.columns([4, 5])
    with left, ui.card("What kind of information matters",
                       "LightGBM split gain summed by feature family. Recent form carries most of the signal; "
                       "context features add a smaller, real increment."):
        c.chart(charts.family_bars(fam))
    with right, ui.card("Top individual features",
                        "Permutation importance on the 2026 test set: how much RMSE worsens when a feature is shuffled "
                        "(mean ± std over 5 repeats). Model-agnostic and measured on unseen data."):
        c.chart(charts.importance_bars(d[d["feature"] != top.feature] if zoom else d))
    with st.expander("All features (permutation, LightGBM gain share, CatBoost importance)"):
        st.dataframe(d.drop(columns="target").sort_values("permutation_rmse_increase", ascending=False).round(4),
                     width="stretch", hide_index=True)


def page_about() -> None:
    ui.header("Data & methodology", "Sources, validation, modelling and design notes.")
    with ui.card():
        st.markdown((c.ROOT / "docs" / "methodology.md").read_text(encoding="utf-8"))
    v = c.load_json(str(c.REPORTS / "validation.json"))
    with ui.card("Validation summary", "Full report: reports/VALIDATION.md in the repository."):
        st.json({k: v[k] for k in ("rows", "squiggle_match", "score_rebuild_mismatches",
                                   "duplicate_player_match_rows", "unused_substitutes")})
