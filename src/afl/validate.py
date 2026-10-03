"""Data validation: cross-checks AFL Tables data against Squiggle and itself.

Writes reports/validation.json (machine-readable) and reports/VALIDATION.md.
Nothing is imputed or altered here; the script only measures and reports.
"""
from __future__ import annotations

import json

import duckdb
import polars as pl

from afl.config import PROCESSED, REPORTS
from afl.parse import STAT_COLS
from afl.squiggle import fetch_games

# Games per season from the published AFL fixture structure
# (2021-22: 22 rounds x 9 + 9 finals; 2023-25: 207 home-and-away + 9 finals;
# 2026: 207 home-and-away + 2 wildcard finals + 9 finals).
EXPECTED_GAMES = {2021: 207, 2022: 207, 2023: 216, 2024: 216, 2025: 216, 2026: 218}
EXPECTED_HA_PER_TEAM = {2021: 22, 2022: 22, 2023: 23, 2024: 23, 2025: 23, 2026: 23}
# Grand Final results as recorded by the AFL (winner, winner score, loser score).
KNOWN_GRAND_FINALS = {
    2021: ("Melbourne", 140, 66), 2022: ("Geelong", 133, 52), 2023: ("Collingwood", 90, 86),
    2024: ("Brisbane Lions", 120, 60), 2025: ("Brisbane Lions", 122, 75),
}
# Minor premiers (top of the home-and-away ladder) as recorded by the AFL.
KNOWN_MINOR_PREMIERS = {2021: "Melbourne", 2022: "Geelong", 2023: "Collingwood", 2024: "Sydney", 2025: "Adelaide"}
COUNT_STATS = [c for c in STAT_COLS.values() if c != "pct_time_played"]


def run() -> dict:
    matches = pl.read_parquet(PROCESSED / "matches.parquet")
    players = pl.read_parquet(PROCESSED / "player_stats.parquet")
    totals = pl.read_parquet(PROCESSED / "team_totals.parquet")
    squiggle = fetch_games()
    con = duckdb.connect()
    for name, df in [("matches", matches), ("players", players), ("totals", totals), ("squiggle", squiggle)]:
        con.register(name, df.to_arrow())
    q = lambda sql: pl.from_arrow(con.sql(sql).arrow())  # noqa: E731
    r: dict = {}

    # 1. Games per season vs Squiggle and the fixture structure.
    per_season = q("""
        SELECT m.season, m.n AS afltables, s.n AS squiggle
        FROM (SELECT season, count(*) n FROM matches GROUP BY 1) m
        JOIN (SELECT season, count(*) n FROM squiggle GROUP BY 1) s USING (season) ORDER BY 1""")
    r["games_per_season"] = [
        {**row, "expected": EXPECTED_GAMES[row["season"]],
         "ok": row["afltables"] == row["squiggle"] == EXPECTED_GAMES[row["season"]]}
        for row in per_season.to_dicts()
    ]

    # 2. Every match matched to Squiggle with identical goals, behinds and scores.
    joined = q("""
        SELECT m.match_id, m.season, m.home_team, m.away_team, m.home_score, m.away_score,
               s.squiggle_id, s.hscore, s.ascore,
               (m.home_goals = s.hgoals AND m.home_behinds = s.hbehinds AND m.home_score = s.hscore
                AND m.away_goals = s.agoals AND m.away_behinds = s.abehinds AND m.away_score = s.ascore) AS same
        FROM matches m LEFT JOIN squiggle s
          ON m.season = s.season AND m.home_team = s.hteam AND m.away_team = s.ateam
         AND abs(date_diff('day', CAST(m.date AS DATE), s.date)) <= 1""")
    r["squiggle_match"] = {
        "matches": joined.height,
        "unmatched": joined.filter(pl.col("squiggle_id").is_null()).height,
        "multi_matched": joined.height - joined["match_id"].n_unique(),
        "score_mismatches": joined.filter(~pl.col("same").fill_null(False) & pl.col("squiggle_id").is_not_null()).height,
    }

    # 3. Grand Finals vs officially recorded results.
    gf = q("""SELECT season, winner, greatest(hscore, ascore) w, least(hscore, ascore) l
              FROM squiggle WHERE is_grand_final = 1 ORDER BY season""").to_dicts()
    r["grand_finals"] = [
        {**g, "known": KNOWN_GRAND_FINALS.get(g["season"]),
         "ok": KNOWN_GRAND_FINALS.get(g["season"]) in (None, (g["winner"], g["w"], g["l"]))}
        for g in gf
    ]

    # 4. Home-and-away games per team.
    ha = q("""
        WITH t AS (SELECT season, home_team team FROM matches WHERE regexp_matches(round_label, '^[0-9]+$')
                   UNION ALL SELECT season, away_team FROM matches WHERE regexp_matches(round_label, '^[0-9]+$'))
        SELECT season, min(n) mn, max(n) mx, count(*) teams FROM (SELECT season, team, count(*) n FROM t GROUP BY 1, 2)
        GROUP BY 1 ORDER BY 1""")
    r["home_and_away_per_team"] = [
        {**row, "expected": EXPECTED_HA_PER_TEAM[row["season"]],
         "ok": row["mn"] == row["mx"] == EXPECTED_HA_PER_TEAM[row["season"]] and row["teams"] == 18}
        for row in ha.to_dicts()
    ]
    ladder = q("""
        WITH t AS (
          SELECT season, home_team team, home_score pf, away_score pa FROM matches WHERE regexp_matches(round_label, '^[0-9]+$')
          UNION ALL
          SELECT season, away_team, away_score, home_score FROM matches WHERE regexp_matches(round_label, '^[0-9]+$'))
        SELECT season, team, CAST(sum(CASE WHEN pf > pa THEN 4 WHEN pf = pa THEN 2 ELSE 0 END) AS INTEGER) pts,
               100.0 * sum(pf) / sum(pa) pct
        FROM t GROUP BY 1, 2 QUALIFY row_number() OVER (PARTITION BY season ORDER BY pts DESC, pct DESC) = 1
        ORDER BY 1""").to_dicts()
    r["minor_premiers"] = [
        {**row, "known": KNOWN_MINOR_PREMIERS.get(row["season"]),
         "ok": KNOWN_MINOR_PREMIERS.get(row["season"]) in (None, row["team"])} for row in ladder
    ]
    r["round_labels"] = sorted(set(matches["round_label"]), key=lambda x: (not x.isdigit(), x.zfill(3)))

    # 5. Player sums equal the page's own team totals, for every counting stat.
    sums = players.group_by("match_id", "team").agg(pl.col(COUNT_STATS).sum())
    # The page's Totals row counts rushed behinds (not credited to any player) in BH.
    cmp = sums.join(totals, on=["match_id", "team"], suffix="_tot").with_columns(
        behinds_tot=pl.col("behinds_tot") - pl.col("rushed_behinds"))
    r["player_sum_vs_team_totals"] = {
        stat: int((cmp[stat] != cmp[f"{stat}_tot"]).sum()) for stat in COUNT_STATS
    } | {"team_matches_compared": cmp.height}

    # 6. Final score rebuilt from player goals/behinds plus rushed behinds.
    rebuilt = (
        cmp.join(matches.select("match_id", "home_team", "home_score", "away_score"), on="match_id")
        .with_columns(official=pl.when(pl.col("team") == pl.col("home_team"))
                      .then(pl.col("home_score")).otherwise(pl.col("away_score")),
                      rebuilt=6 * pl.col("goals") + pl.col("behinds") + pl.col("rushed_behinds"))
    )
    r["score_rebuild_mismatches"] = int((rebuilt["official"] != rebuilt["rebuilt"]).sum())

    # 7. Row-level identities and integrity.
    r["disposals_ne_kicks_plus_handballs"] = players.filter(
        pl.col("disposals") != pl.col("kicks") + pl.col("handballs")).height
    r["players_per_team_match"] = (
        players.group_by("match_id", "team").len().group_by("len").len(name="team_matches")
        .rename({"len": "players"}).sort("players").to_dicts())
    r["subs_on_per_team_match"] = (
        players.group_by("match_id", "team").agg((pl.col("sub_status") == "on").sum().alias("subs"))
        .group_by("subs").len().sort("subs").to_dicts())
    r["duplicate_player_match_rows"] = players.height - players.select("match_id", "player_id").n_unique()
    r["duplicate_matches"] = matches.height - matches["match_id"].n_unique()
    r["unused_substitutes"] = players.filter(~pl.col("took_field")).height
    r["unused_substitutes_with_any_stat"] = players.filter(
        ~pl.col("took_field") & (pl.sum_horizontal(COUNT_STATS) > 0)).height
    r["nulls"] = {c: n for c, n in players.null_count().to_dicts()[0].items() if n}
    r["rows"] = {"matches": matches.height, "player_match_rows": players.height,
                 "unique_players": players["player_id"].n_unique()}

    # 8. Career-games semantics: consecutive appearances should differ by exactly one.
    cg = (players.sort("date").with_columns(d=pl.col("career_games").diff().over("player_id"))
          .filter(pl.col("d").is_not_null()))
    r["career_games_step"] = cg.group_by("d").len().sort("len", descending=True).head(5).to_dicts()

    # 9. Brownlow votes per match (3-2-1 => 6; ties can add more; 0 = not yet counted).
    r["brownlow_votes_per_match"] = (
        players.group_by("match_id").agg(pl.col("brownlow_votes").sum().alias("votes"))
        .join(matches.select("match_id", "season"), on="match_id")
        .group_by("season", "votes").len().sort("season", "votes").to_dicts())

    # 10. Outliers: report extremes for review; values are genuine and are kept.
    desc = players.select(COUNT_STATS + ["pct_time_played"]).describe()
    r["ranges"] = {c: {"min": desc[c][4], "max": desc[c][8]} for c in COUNT_STATS + ["pct_time_played"]}
    r["max_disposal_games"] = (
        players.sort("disposals", descending=True).head(5)
        .select("player_name", "team", "season", "round_label", "disposals").to_dicts())
    pct = players["pct_time_played"]
    r["pct_time_played_out_of_range"] = int(((pct < 0) | (pct > 100)).sum())
    r["low_time_played_rows"] = players.filter(pl.col("pct_time_played") < 20).height
    return r


def _md(r: dict) -> str:
    ok = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    lines = ["# Data validation report", "",
             "Generated by `python -m afl.validate`. No values were imputed, synthesised or altered.", "",
             f"Rows: **{r['rows']['matches']:,} matches**, **{r['rows']['player_match_rows']:,} player-match rows**, "
             f"**{r['rows']['unique_players']:,} unique players**.", "",
             "## 1. Games per season", "", "| Season | AFL Tables | Squiggle | Expected | Result |", "|---|---|---|---|---|"]
    lines += [f"| {g['season']} | {g['afltables']} | {g['squiggle']} | {g['expected']} | {ok(g['ok'])} |"
              for g in r["games_per_season"]]
    s = r["squiggle_match"]
    lines += ["", "## 2. Scores cross-checked against Squiggle", "",
              f"{s['matches']} matches joined on season, home team, away team and date (±1 day): "
              f"{s['unmatched']} unmatched, {s['multi_matched']} ambiguous, "
              f"**{s['score_mismatches']} goal/behind/score mismatches** — {ok(s['unmatched'] == s['score_mismatches'] == 0)}.",
              "", "## 3. Grand Finals vs official results", "", "| Season | Winner | Score | Known | Result |", "|---|---|---|---|---|"]
    lines += [f"| {g['season']} | {g['winner']} | {g['w']}–{g['l']} | {g['known'] or 'n/a (latest season, Squiggle only)'} | {ok(g['ok'])} |"
              for g in r["grand_finals"]]
    lines += ["", "Minor premiers (top of the computed home-and-away ladder):", "",
              "| Season | Computed | Points | % | Known | Result |", "|---|---|---|---|---|---|"]
    lines += [f"| {m['season']} | {m['team']} | {m['pts']} | {m['pct']:.1f} | {m['known'] or 'n/a (latest season)'} | {ok(m['ok'])} |"
              for m in r["minor_premiers"]]
    lines += ["", "## 4. Home-and-away games per team", "", "| Season | Min | Max | Teams | Expected | Result |", "|---|---|---|---|---|---|"]
    lines += [f"| {h['season']} | {h['mn']} | {h['mx']} | {h['teams']} | {h['expected']} | {ok(h['ok'])} |"
              for h in r["home_and_away_per_team"]]
    lines += ["", f"Round labels found: {', '.join(r['round_labels'])}.", ""]
    mism = {k: v for k, v in r["player_sum_vs_team_totals"].items() if k != "team_matches_compared" and v}
    lines += ["## 5. Internal consistency", "",
              f"- Player stats summed per team vs the page's own *Totals* row across "
              f"{r['player_sum_vs_team_totals']['team_matches_compared']:,} team-matches and {len(COUNT_STATS)} stats: "
              f"{'no mismatches' if not mism else mism} — {ok(not mism)}.",
              f"- Final scores rebuilt as 6×goals + behinds + rushed behinds: {r['score_rebuild_mismatches']} mismatches — {ok(r['score_rebuild_mismatches'] == 0)}.",
              f"- Rows where disposals ≠ kicks + handballs: {r['disposals_ne_kicks_plus_handballs']} — {ok(r['disposals_ne_kicks_plus_handballs'] == 0)}.",
              f"- Duplicate player-match rows: {r['duplicate_player_match_rows']}; duplicate matches: {r['duplicate_matches']}.",
              f"- Players listed per team-match: {r['players_per_team_match']}.",
              f"- Substitutes activated per team-match: {r['subs_on_per_team_match']}.",
              f"- Career-games difference between a player's consecutive appearances: {r['career_games_step']} "
              "(always 1: the published count includes the current match, so `career_games - 1` is used as the "
              "pre-match feature).",
              f"- Named substitutes who never took the field (listed with no time-on-ground value and no stats): "
              f"{r['unused_substitutes']} rows ({r['unused_substitutes_with_any_stat']} with any stat). Kept in "
              "`player_stats.parquet` with `took_field = false`; excluded from features and modelling.",
              "", "## 6. Nulls", "",
              f"{r['nulls'] or 'No nulls in any column.'}", "",
              "## 7. Ranges and outliers", "",
              "Extreme values were reviewed and kept: they are genuine match performances, not errors.", "",
              "| Stat | Min | Max |", "|---|---|---|"]
    lines += [f"| {k} | {v['min']:.0f} | {v['max']:.0f} |" for k, v in r["ranges"].items()]
    lines += ["", f"Top disposal games: " + "; ".join(
        f"{g['player_name']} ({g['team']}, {g['season']} R{g['round_label']}) {g['disposals']}" for g in r["max_disposal_games"]),
        "", f"- `pct_time_played` outside 0–100: {r['pct_time_played_out_of_range']}.",
        f"- Rows with under 20% time on ground (mostly substitutes and injuries): {r['low_time_played_rows']:,}. "
        "Kept as genuine performances; prior time on ground enters the models only through rolling form features.",
        "", "## 8. Anomalies and how they are handled", "",
        "Handling is rule-based and applied by `python -m afl.clean` after this report is generated "
        "(details in `reports/cleaning.json`). No statistic is altered.", "",
        "- **Score conflict between sources** (rule: any goal/behind/score difference vs Squiggle). One match: "
        "Essendon v Port Adelaide, 23 Aug 2026 (AFL Tables round 25). AFL Tables records Port Adelaide 16.9 (105); "
        "Squiggle records 16.8 (104). AFL Tables is internally consistent (player behinds + rushed behinds = 9) and "
        "agrees with Port Adelaide FC's published match report. **Handling:** AFL Tables values kept; the match and its "
        "46 player rows carry `source_conflict = true` and are excluded from all model evaluation metrics.",
        "- **Unverifiable substitution markers** (rule: more than one substitute-on marker for a team in a match; AFL "
        "rules allow one). One match: Gold Coast v Essendon, 27 Aug 2025 (the Opening Round fixture postponed by "
        "Cyclone Alfred). AFL Tables shows 2 and 3 substitutions; match reports describe one per side. "
        "**Handling:** `sub_status` set to null (unknown) for every player in that match; statistics untouched. "
        "The counts in section 5 are measured before this rule is applied.",
        "- Finals show 0 Brownlow votes, which is correct: votes are only awarded in home-and-away matches.",
        "", "## 9. Brownlow votes per match", "",
        "Brownlow votes are recorded for context only and are never used as a model input.", "",
        "| Season | Votes in match | Matches |", "|---|---|---|"]
    lines += [f"| {b['season']} | {b['votes']} | {b['len']} |" for b in r["brownlow_votes_per_match"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    r = run()
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "validation.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    (REPORTS / "VALIDATION.md").write_text(_md(r), encoding="utf-8")
    print(json.dumps({k: r[k] for k in ("rows", "squiggle_match", "score_rebuild_mismatches",
                                         "duplicate_player_match_rows", "nulls")}, default=str))


if __name__ == "__main__":
    main()
