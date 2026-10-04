"""Derived analyses for the dashboard, computed from the processed tables only (no new data).

  elo_ratings          Elo team ratings after every match (parameters tuned on 2022-2024 results)
  home_advantage       per-team home-ground advantage estimate with a 95% confidence interval
  stat_vs_margin       how each team-stat difference relates to the final margin
  similarity           2-D projection of player stat profiles + nearest neighbours
  age_profile          league output by age, for age curves
  with_without         team results with vs without a given player
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

ELO_START = 1500.0
ELO_GRID = {"k": (20, 30, 40, 50, 60), "hga": (0, 20, 40, 60, 80, 100, 120), "carry": (0.5, 0.7, 0.85, 1.0)}
ELO_TUNE_SEASONS = (2022, 2023, 2024)  # 2021 is burn-in (everyone starts at 1500); 2025-26 kept out of sample


def _elo_run(m: pd.DataFrame, k: float, hga: float, carry: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Walk through matches in date order. Returns per-team rating history and pre-match home win probs."""
    rating: dict[str, float] = {}
    season, rows, probs = None, [], np.empty(len(m))
    for i, g in enumerate(m.itertuples(index=False)):
        if g.season != season:  # regress every team part-way back to the mean between seasons
            rating = {t: ELO_START + carry * (r - ELO_START) for t, r in rating.items()}
            season = g.season
        rh, ra = rating.get(g.home_team, ELO_START), rating.get(g.away_team, ELO_START)
        p = 1 / (1 + 10 ** (-(rh + hga - ra) / 400))
        s = 1.0 if g.home_score > g.away_score else 0.0 if g.home_score < g.away_score else 0.5
        rating[g.home_team], rating[g.away_team] = rh + k * (s - p), ra - k * (s - p)
        probs[i] = p
        res = {1.0: "Win", 0.0: "Loss", 0.5: "Draw"}
        rows.append((g.match_id, g.date, g.season, g.round_label, g.home_team, g.away_team, rating[g.home_team],
                     rh, res[s]))
        rows.append((g.match_id, g.date, g.season, g.round_label, g.away_team, g.home_team, rating[g.away_team],
                     ra, res[1 - s]))
    hist = pd.DataFrame(rows, columns=["match_id", "date", "season", "round_label", "team", "opponent", "elo",
                                       "elo_before", "result"])
    return hist, probs


def elo_ratings(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Tune K, home advantage and season carry-over on 2022-2024 (Brier score), report 2025-2026 out of sample.

    Only win/loss/draw is used, so the one match with a cross-source score conflict (same winner in both
    sources) is unaffected.
    """
    m = matches.sort_values(["date", "match_id"]).reset_index(drop=True)
    outcome = np.where(m.home_score > m.away_score, 1.0, np.where(m.home_score < m.away_score, 0.0, 0.5))
    tune, test = m.season.isin(ELO_TUNE_SEASONS).to_numpy(), (m.season >= 2025).to_numpy()
    scores = {}
    for k, hga, carry in itertools.product(*ELO_GRID.values()):
        _, p = _elo_run(m, k, hga, carry)
        scores[(k, hga, carry)] = float(np.mean((p[tune] - outcome[tune]) ** 2))
    k, hga, carry = min(scores, key=scores.get)
    hist, p = _elo_run(m, k, hga, carry)
    decided = test & (outcome != 0.5)
    always_home = float(np.mean(outcome[decided] == 1))
    info = {"k": k, "hga": hga, "carry": carry, "tune_brier": scores[(k, hga, carry)],
            "test_brier": float(np.mean((p[test] - outcome[test]) ** 2)),
            "test_accuracy": float(np.mean((p[decided] > 0.5) == (outcome[decided] == 1))),
            "test_home_team_win_rate": always_home, "test_matches": int(test.sum())}
    return hist, info


def home_advantage(matches: pd.DataFrame) -> pd.DataFrame:
    """(mean margin as designated home team - mean margin as away team) / 2, home-and-away rounds only.

    Halving turns the home-vs-away gap into a per-game advantage relative to a neutral venue. The CI uses
    the standard error of the difference of two independent means.
    """
    m = matches[matches.round_label.str.fullmatch(r"\d+") & ~matches.source_conflict]
    margin = m.home_score - m.away_score
    home = pd.DataFrame({"team": m.home_team, "margin": margin})
    away = pd.DataFrame({"team": m.away_team, "margin": -margin})
    h = home.groupby("team")["margin"].agg(["mean", "var", "count"])
    a = away.groupby("team")["margin"].agg(["mean", "var", "count"])
    out = pd.DataFrame({"home_margin": h["mean"], "away_margin": a["mean"], "home_n": h["count"],
                        "away_n": a["count"]})
    out["hga"] = (out.home_margin - out.away_margin) / 2
    se = np.sqrt(h["var"] / h["count"] + a["var"] / a["count"]) / 2
    out["lo"], out["hi"] = out.hga - 1.96 * se, out.hga + 1.96 * se
    return out.reset_index()


# Stats excluded from "what wins games": goals and behinds *are* the score, Brownlow votes are awarded
# after the match, and rushed behinds are part of the score too.
SCORE_STATS = {"goals", "behinds", "rushed_behinds", "brownlow_votes"}


def stat_vs_margin(matches: pd.DataFrame, totals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per match from the home team's perspective (each match counted once)."""
    m = matches[~matches.source_conflict]
    stats = [c for c in totals.columns if c not in ("match_id", "team") and c not in SCORE_STATS]
    h = totals.rename(columns={c: f"h_{c}" for c in stats}).rename(columns={"team": "home_team"})
    a = totals.rename(columns={c: f"a_{c}" for c in stats}).rename(columns={"team": "away_team"})
    d = (m[["match_id", "season", "round_label", "home_team", "away_team", "home_score", "away_score"]]
         .merge(h[["match_id", "home_team", *[f"h_{c}" for c in stats]]], on=["match_id", "home_team"])
         .merge(a[["match_id", "away_team", *[f"a_{c}" for c in stats]]], on=["match_id", "away_team"]))
    d["margin"] = d.home_score - d.away_score
    for c in stats:
        d[c] = d[f"h_{c}"] - d[f"a_{c}"]
    rows = []
    for c in stats:
        x, y = d[c].to_numpy(float), d["margin"].to_numpy(float)
        ahead = x != 0
        rows.append({"stat": c, "r": float(np.corrcoef(x, y)[0, 1]), "slope": float(np.polyfit(x, y, 1)[0]),
                     "win_pct": float(np.mean((np.sign(x) == np.sign(y))[ahead & (y != 0)])), "n": len(d)})
    return pd.DataFrame(rows), d[["match_id", "season", "round_label", "home_team", "away_team", "margin", *stats]]


def similarity(pool: pd.DataFrame, stats: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Standardise per-game averages, project onto the first two principal components (numpy SVD).

    Returns the 2-D coordinates, the explained-variance ratio of each component, and the standardised
    matrix (used for nearest neighbours in the full space, not just the 2-D picture).
    """
    X = pool[stats].to_numpy(float)
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    coords = pd.DataFrame(U[:, :2] * S[:2], index=pool.index, columns=["x", "y"])
    return coords, (S ** 2 / np.sum(S ** 2))[:2], Z


def nearest(Z: np.ndarray, index: pd.Index, pid: str, n: int = 10) -> pd.Series:
    i = index.get_loc(pid)
    dist = np.sqrt(((Z - Z[i]) ** 2).sum(axis=1))
    s = pd.Series(dist, index=index).drop(pid).nsmallest(n)
    return s


def age_profile(pm: pd.DataFrame, stat: str, min_n: int = 150) -> pd.DataFrame:
    """League per-match output by whole year of age (ages with fewer than min_n player-matches dropped)."""
    d = pm.dropna(subset=["age_days"]).assign(age=lambda x: (x.age_days // 365.25).astype(int))
    g = d.groupby("age")[stat]
    out = pd.DataFrame({"mean": g.mean(), "q25": g.quantile(0.25), "q75": g.quantile(0.75), "n": g.size()})
    return out[out.n >= min_n].reset_index()


def with_without(team_games: pd.DataFrame, played: set[str], team: str, seasons: list[int]) -> pd.DataFrame:
    """Team results per season split by whether the player took the field.

    team_games: one row per team per match (team, season, match_id, pf, pa). Only seasons in which the
    player played at least once for `team` are considered.
    """
    g = team_games[(team_games.team == team) & team_games.season.isin(seasons)].copy()
    g["group"] = np.where(g.match_id.isin(played), "With player", "Without player")
    g["win"] = np.select([g.pf > g.pa, g.pf < g.pa], [1.0, 0.0], 0.5)
    g["margin"] = g.pf - g.pa
    return g
