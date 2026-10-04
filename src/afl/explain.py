"""Per-prediction explanations for the 2026 test season ("why this prediction?").

Refits the final LightGBM for each target with exactly the hyperparameters recorded in
results/metrics.json (same features, same 2021-2025 training rows, same seed), checks that
it reproduces the LightGBM predictions already stored in predictions.parquet, then saves
LightGBM's built-in per-feature contributions (``pred_contrib=True``, TreeSHAP).

For every row: base value + sum(contributions) == LightGBM prediction (exactly).
Model results are not changed; this only reads them.

Output: data/processed/explanations.parquet (long format). To keep the deployed app small, each
prediction keeps its TOP_K largest contributions plus one "other features" row holding the exact sum
of the rest, so the breakdown still adds up to the prediction.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from afl.config import PROCESSED, RESULTS
from afl.features import TARGETS
from afl.train import FEATURES, TEST_SEASON, VALID_SEASON, align_categories, to_pandas

TOLERANCE = 1e-6  # max allowed gap between refitted and stored LightGBM predictions
TOP_K = 12
OTHER = "__other__"


def explain_target(df: pl.DataFrame, target: str, params: dict, stored: pd.DataFrame) -> pd.DataFrame:
    tr, te = df.filter(pl.col("season") <= VALID_SEASON), df.filter(pl.col("season") == TEST_SEASON)
    # Same category alignment as training (which also aligned the 2021-2024 and 2025 frames).
    frames = [to_pandas(d) for d in (df.filter(pl.col("season") <= 2024), df.filter(pl.col("season") == VALID_SEASON),
                                    te, tr)]
    align_categories(*frames)
    Xte, Xtr = frames[2], frames[3]
    model = lgb.LGBMRegressor(**params).fit(Xtr, tr[target].to_numpy())
    pred = model.predict(Xte)

    ref = te.select("match_id", "player_id").to_pandas().merge(
        stored[["match_id", "player_id", "LightGBM"]], on=["match_id", "player_id"], how="left")
    gap = float(np.max(np.abs(ref["LightGBM"].to_numpy() - pred)))
    print(f"  [{target}] max |refit - stored| = {gap:.2e} over {len(pred)} predictions")
    if not gap <= TOLERANCE:
        raise SystemExit(f"Refitted LightGBM does not reproduce stored predictions for {target} (gap {gap}).")

    contrib = model.predict(Xte, pred_contrib=True)  # n x (features + 1); last column = base value
    assert np.allclose(contrib.sum(axis=1), pred)
    keys = te.select("match_id", "player_id").to_pandas()
    wide = pd.DataFrame(contrib[:, :-1], columns=FEATURES)
    wide[["match_id", "player_id"]] = keys
    long = wide.melt(id_vars=["match_id", "player_id"], var_name="feature", value_name="contribution")
    values = pd.DataFrame({c: (Xte[c].astype(str) if c in Xte.select_dtypes("category")
                               else Xte[c].map(lambda v: f"{v:.4g}")) for c in FEATURES})
    values = values.where(Xte.notna(), "missing")
    values[["match_id", "player_id"]] = keys
    vals = values.melt(id_vars=["match_id", "player_id"], var_name="feature", value_name="feature_value")
    out = long.merge(vals, on=["match_id", "player_id", "feature"])
    out["rank"] = out.groupby(["match_id", "player_id"])["contribution"].transform(
        lambda c: c.abs().rank(method="first", ascending=False))
    rest = (out[out["rank"] > TOP_K].groupby(["match_id", "player_id"], as_index=False)["contribution"].sum()
            .assign(feature=OTHER, feature_value=f"{len(FEATURES) - TOP_K} features", rank=TOP_K + 1))
    out = pd.concat([out[out["rank"] <= TOP_K], rest], ignore_index=True).drop(columns="rank")
    out["target"], out["base_value"] = target, float(contrib[0, -1])
    out["contribution"] = out["contribution"].astype("float32")
    return out


def main() -> None:
    df = pl.read_parquet(PROCESSED / "features.parquet")
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    preds = pd.read_parquet(PROCESSED / "predictions.parquet")
    parts = []
    for target in TARGETS:
        params = dict(metrics["targets"][target]["tuning"]["LightGBM"])
        parts.append(explain_target(df, target, params, preds[preds["target"] == target]))
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(PROCESSED / "explanations.parquet", index=False, compression="zstd")
    print(f"explanations: {len(out):,} rows")


if __name__ == "__main__":
    main()
