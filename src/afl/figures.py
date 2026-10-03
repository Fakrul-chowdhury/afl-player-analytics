"""Export static PNG figures for the README using the same chart code as the dashboard."""
from __future__ import annotations

import json

import altair as alt
import pandas as pd

from afl import charts
from afl.config import PROCESSED, RESULTS, ROOT

OUT = ROOT / "docs" / "figures"
WIDTH = 760


def _save(chart: alt.TopLevelMixin, name: str, title: str | None = None) -> None:
    if title:
        chart = chart.properties(title=title)
    chart.configure(background="white").configure_view(strokeWidth=0).save(OUT / f"{name}.png", scale_factor=2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = json.loads((RESULTS / "metrics.json").read_text())
    preds = pd.read_parquet(PROCESSED / "predictions.parquet")
    imp = pd.read_parquet(RESULTS / "feature_importance.parquet")
    pm = pd.read_parquet(PROCESSED / "app_player_matches.parquet")
    matches = pd.read_parquet(PROCESSED / "matches.parquet")

    for tgt, label in (("disposals", "disposals"), ("fantasy_points", "AFL Fantasy points")):
        tab = pd.DataFrame(m["targets"][tgt]["test_2026"]).T.reset_index(names="model")
        _save(charts.model_metric_bars(tab, "rmse").properties(width=WIDTH - 260),
              f"model_rmse_{tgt}", f"Next-match {label}: test RMSE by model (2026)")
        best = m["targets"][tgt]["selected_model"]
        _save(charts.predicted_vs_actual(preds[preds["target"] == tgt], best, label).properties(width=WIDTH - 200),
              f"pred_vs_actual_{tgt}")
        d = imp[imp["target"] == tgt].copy()
        d["feature"] = d["feature"].map(charts.pretty_feature)
        _save(charts.importance_bars(d, top=15).properties(width=WIDTH - 300),
              f"importance_{tgt}", f"Top 15 features for {label} (LightGBM permutation importance on 2026 test)")

    season = int(matches["season"].max())
    s = pm[pm["season"] == season]
    games = s.groupby("team")["match_id"].nunique()
    prof = pd.DataFrame({"inside_50s": s.groupby("team")["inside_50s"].sum() / games,
                         "inside_50s_against": s.groupby("opponent")["inside_50s"].sum()
                         / s.groupby("opponent")["match_id"].nunique()}).reset_index(names="team")
    _save(charts.team_scatter(prof, "inside_50s", "inside_50s_against", "Inside 50s per game",
                              "Inside 50s conceded per game").properties(width=WIDTH),
          "team_territory", f"{season} territory battle: inside 50s for vs against (per game)")

    star = pm[pm["player_id"] == "N/Nick_Daicos"]
    if not star.empty:
        _save(charts.form_trend(star, "disposals").properties(width=WIDTH, title="Nick Daicos (Collingwood): "
              "disposals per match with 5-match rolling average"), "form_daicos")
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
