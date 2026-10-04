"""Train and compare models for next-match disposals and AFL Fantasy points.

Time-based split: train 2021-2024, validate 2025 (tuning / early stopping),
test 2026 (scored after the final refit on 2021-2025; never used for tuning or selection).
"""
from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from afl.config import PROCESSED, RESULTS
from afl.features import FORM_STATS, TARGETS, WINDOWS

TRAIN_END, VALID_SEASON, TEST_SEASON = 2024, 2025, 2026
CAT_FEATURES = ["role", "team", "opponent", "venue"]
NUM_FEATURES = (
    [f"{s}_avg{w}" for s in FORM_STATS for w in WINDOWS]
    + [f"{t}_{k}" for t in TARGETS for k in ("std10", "last", "season_avg")]
    + ["games_in_data", "career_games_before", "age_years", "days_since_last", "sub_last_match",
       "team_disp_for5", "opp_disp_conceded5", "opp_fp_conceded5", "is_home", "is_final"]
)
FEATURES = NUM_FEATURES + CAT_FEATURES
SEED = 42


def to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    X = df.select(FEATURES).with_columns(
        pl.col("sub_last_match", "is_home", "is_final").cast(pl.Int8)).to_pandas()
    for c in CAT_FEATURES:
        X[c] = X[c].astype("category")
    return X


def align_categories(*frames: pd.DataFrame) -> None:
    for c in CAT_FEATURES:
        cats = sorted(set().union(*(f[c].dropna().unique() for f in frames)))
        for f in frames:
            f[c] = pd.Categorical(f[c], categories=cats)


def metrics(y, p) -> dict:
    return {"rmse": float(np.sqrt(mean_squared_error(y, p))), "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p)), "n": int(len(y))}


def baselines(df: pl.DataFrame, target: str, fallback: float) -> dict[str, np.ndarray]:
    last5 = df[f"{target}_avg5"].fill_null(fallback).to_numpy()
    season = df[f"{target}_season_avg"].fill_null(df[f"{target}_avg5"]).fill_null(fallback).to_numpy()
    return {"Baseline: last-5 average": last5, "Baseline: season-to-date average": season}


def ridge():
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT_FEATURES),
    ])
    return lambda alpha: make_pipeline(pre, Ridge(alpha=alpha))


def fit_target(df: pl.DataFrame, target: str) -> tuple[dict, pl.DataFrame, pd.DataFrame]:
    tr = df.filter(pl.col("season") <= TRAIN_END)
    va = df.filter(pl.col("season") == VALID_SEASON)
    te = df.filter(pl.col("season") == TEST_SEASON)
    trva = df.filter(pl.col("season") <= VALID_SEASON)
    Xtr, Xva, Xte, Xtrva = (to_pandas(d) for d in (tr, va, te, trva))
    align_categories(Xtr, Xva, Xte, Xtrva)
    ytr, yva, yte, ytrva = (d[target].to_numpy() for d in (tr, va, te, trva))
    tuning, preds_va, preds_te = {}, {}, {}

    t0 = time.monotonic()
    log = lambda msg: print(f"  [{target}] {msg} ({time.monotonic() - t0:.0f}s)", flush=True)  # noqa: E731

    # Ridge: choose alpha on 2025.
    make = ridge()
    scores = {a: np.sqrt(mean_squared_error(yva, make(a).fit(Xtr, ytr).predict(Xva))) for a in (1, 10, 100, 1000, 3000, 10000, 30000)}
    alpha = min(scores, key=scores.get)
    preds_va["Ridge regression"] = make(alpha).fit(Xtr, ytr).predict(Xva)
    preds_te["Ridge regression"] = make(alpha).fit(Xtrva, ytrva).predict(Xte)
    tuning["Ridge regression"] = {"alpha": alpha}
    log(f"ridge done, alpha={alpha}")

    # LightGBM: early stopping on 2025, then refit on 2021-2025 with the chosen rounds.
    # Small grid over tree complexity, selected on 2025 RMSE.
    base_params = dict(n_estimators=5000, learning_rate=0.03, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1)
    fits = {}
    for leaves in (15, 31, 63):
        for min_child in (50, 200):
            p = {**base_params, "num_leaves": leaves, "min_child_samples": min_child}
            mm = lgb.LGBMRegressor(**p).fit(Xtr, ytr, eval_X=Xva, eval_y=yva,
                                            callbacks=[lgb.early_stopping(200, verbose=False)])
            rmse = np.sqrt(mean_squared_error(yva, mm.predict(Xva, num_iteration=mm.best_iteration_)))
            fits[(leaves, min_child)] = (rmse, mm, p)
            log(f"lgbm grid leaves={leaves} min_child={min_child} rmse={rmse:.3f}")
    _, m, lgb_params = min(fits.values(), key=lambda f: f[0])
    best = m.best_iteration_
    preds_va["LightGBM"] = m.predict(Xva, num_iteration=best)
    lgbm = lgb.LGBMRegressor(**{**lgb_params, "n_estimators": best}).fit(Xtrva, ytrva)
    preds_te["LightGBM"] = lgbm.predict(Xte)
    tuning["LightGBM"] = {"n_estimators": int(best), **{k: v for k, v in lgb_params.items() if k != "n_estimators"}}

    # CatBoost: same protocol.
    cb_params = dict(iterations=5000, learning_rate=0.05, depth=6, loss_function="RMSE",
                     random_seed=SEED, verbose=0, cat_features=CAT_FEATURES)
    to_cb = lambda X: X.assign(**{c: X[c].astype(str) for c in CAT_FEATURES})  # noqa: E731
    cb = CatBoostRegressor(**cb_params, od_type="Iter", od_wait=200).fit(
        to_cb(Xtr), ytr, eval_set=(to_cb(Xva), yva), use_best_model=True)
    best_cb = cb.get_best_iteration() + 1
    preds_va["CatBoost"] = cb.predict(to_cb(Xva))
    cb_final = CatBoostRegressor(**{**cb_params, "iterations": best_cb}).fit(to_cb(Xtrva), ytrva)
    preds_te["CatBoost"] = cb_final.predict(to_cb(Xte))
    log("catboost done")
    tuning["CatBoost"] = {"iterations": int(best_cb), "learning_rate": 0.05, "depth": 6}

    ens = ["Ridge regression", "LightGBM", "CatBoost"]
    preds_va["Ensemble (mean of 3)"] = np.mean([preds_va[k] for k in ens], axis=0)
    preds_te["Ensemble (mean of 3)"] = np.mean([preds_te[k] for k in ens], axis=0)

    fallback_tr = float(tr.filter(pl.col("games_in_data") == 0)[target].mean())
    fallback_trva = float(trva.filter(pl.col("games_in_data") == 0)[target].mean())
    for name, p in baselines(va, target, fallback_tr).items():
        preds_va[name] = p
    for name, p in baselines(te, target, fallback_trva).items():
        preds_te[name] = p

    # Rows from matches whose score conflicts between sources are predicted but not scored.
    ok_va = ~va["source_conflict"].to_numpy()
    ok_te = ~te["source_conflict"].to_numpy()
    full = ok_te & (te["sub_status"].is_null() & (te["games_in_data"] >= 5)).to_numpy()
    res = {
        "validation_2025": {k: metrics(yva[ok_va], v[ok_va]) for k, v in preds_va.items()},
        "test_2026": {k: metrics(yte[ok_te], v[ok_te]) for k, v in preds_te.items()},
        "test_2026_full_match_established_players": {k: metrics(yte[full], v[full]) for k, v in preds_te.items()},
        "test_2026_by_role": {
            role: {k: metrics(yte[mask], v[mask]) for k, v in preds_te.items()}
            for role in sorted(te["role"].unique())
            if (mask := ok_te & (te["role"] == role).to_numpy()).sum() > 0
        },
        "excluded_source_conflict_rows": {"validation_2025": int((~ok_va).sum()), "test_2026": int((~ok_te).sum())},
        "tuning": tuning,
        "selected_model": min((k for k in preds_va if not k.startswith("Baseline")),
                              key=lambda k: metrics(yva[ok_va], preds_va[k][ok_va])["rmse"]),
    }

    # 80% prediction intervals from empirical 2025 residual quantiles of the selected model,
    # conditioned on the size of the prediction (quintile bins fitted on 2025 predictions).
    sel = res["selected_model"]
    pva, pte = preds_va[sel][ok_va], preds_te[sel]
    edges = np.quantile(pva, [0.2, 0.4, 0.6, 0.8])
    resid = yva[ok_va] - pva
    bins_va, bins_te = np.digitize(pva, edges), np.digitize(pte, edges)
    q = {b: np.quantile(resid[bins_va == b], [0.1, 0.9]) for b in range(5)}
    lower = pte + np.array([q[b][0] for b in bins_te])
    upper = pte + np.array([q[b][1] for b in bins_te])
    inside = (yte >= lower) & (yte <= upper)
    res["interval_80"] = {
        "method": "empirical 10th/90th percentile of 2025 validation residuals, by predicted-value quintile",
        "test_2026_coverage": float(inside[ok_te].mean()),
        "mean_width": float((upper - lower)[ok_te].mean()),
        "bin_edges": [float(e) for e in edges],
        "residual_quantiles": {int(b): [float(v) for v in q[b]] for b in q},
    }

    pred_df = te.select("match_id", "date", "season", "round_label", "player_id", "player_name", "team",
                        "opponent", "role", "sub_status", "games_in_data", "source_conflict",
                        pl.col(target).alias("actual")).with_columns(
        [pl.Series(k, v.astype(float)) for k, v in preds_te.items()]).with_columns(
        target=pl.lit(target), pi_lower=pl.Series(lower.astype(float)), pi_upper=pl.Series(upper.astype(float)))

    log("predictions and intervals done")
    gain = pd.Series(lgbm.booster_.feature_importance("gain"), index=FEATURES)
    perm = permutation_importance(lgbm, Xte[ok_te], yte[ok_te], n_repeats=5, random_state=SEED,
                                  scoring="neg_root_mean_squared_error")
    imp = pd.DataFrame({"feature": FEATURES, "lgbm_gain_share": (gain / gain.sum()).values,
                        "permutation_rmse_increase": perm.importances_mean,
                        "permutation_std": perm.importances_std,
                        "catboost_importance": cb_final.get_feature_importance() / 100})
    imp["target"] = target
    log("permutation importance done")
    return res, pred_df, imp


def main() -> None:
    df = pl.read_parquet(PROCESSED / "features.parquet")
    out = {"split": {"train": "2021-2024", "validation": "2025", "test": "2026"}, "targets": {}}
    preds, imps = [], []
    for target in TARGETS:
        res, p, imp = fit_target(df, target)
        out["targets"][target] = res
        preds.append(p)
        imps.append(imp)
        print(target, {k: round(v["rmse"], 3) for k, v in res["test_2026"].items()})
    out["rows"] = {s: int(n) for s, n in df.group_by("season").len().sort("season").iter_rows()}
    out["n_features"] = len(FEATURES)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2))
    pl.concat(preds).write_parquet(PROCESSED / "predictions.parquet")
    pl.from_pandas(pd.concat(imps)).write_parquet(RESULTS / "feature_importance.parquet")


if __name__ == "__main__":
    main()
