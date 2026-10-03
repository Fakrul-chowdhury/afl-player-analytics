"""Parse cached AFL Tables match pages into tidy Parquet tables.

Outputs (data/processed/):
  matches.parquet        one row per match (header and final scores)
  player_stats.parquet   one row per player per match (23 stats + player details)
  team_totals.parquet    the page's own team "Totals" rows, used for validation
"""
from __future__ import annotations

import re
from datetime import datetime

import polars as pl
from selectolax.parser import HTMLParser

from afl.config import HTML_CACHE, PROCESSED, SEASONS

STAT_COLS = {
    "KI": "kicks", "MK": "marks", "HB": "handballs", "DI": "disposals", "GL": "goals",
    "BH": "behinds", "HO": "hitouts", "TK": "tackles", "RB": "rebound_50s", "IF": "inside_50s",
    "CL": "clearances", "CG": "clangers", "FF": "frees_for", "FA": "frees_against",
    "BR": "brownlow_votes", "CP": "contested_possessions", "UP": "uncontested_possessions",
    "CM": "contested_marks", "MI": "marks_inside_50", "1%": "one_percenters", "BO": "bounces",
    "GA": "goal_assists", "%P": "pct_time_played",
}
HEADER_RE = re.compile(
    r"Round:\s*(?P<round>.*?)\s*Venue:\s*(?P<venue>.*?)\s*"
    r"Date:\s*(?P<date>\w{3}, \d{1,2}-\w{3}-\d{4} \d{1,2}:\d{2} [AP]M)"
    r"(?:.*?Attendance:\s*(?P<att>\d+))?"
)
SCORE_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")
AGE_RE = re.compile(r"(\d+)y(?:\s*(\d+)d)?")
LEADING_INT = re.compile(r"^(\d+)")
SUB_ON, SUB_OFF = "↑", "↓"


def decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _num(text: str) -> int | None:
    """AFL Tables leaves zero counts blank; anything non-numeric becomes null."""
    text = text.strip()
    if text == "":
        return 0
    return int(text) if text.isdigit() else None


def _player_id(href: str) -> str:
    return href.split("players/")[-1].removesuffix(".html")


def parse_match(html: str, season: int, match_id: str):
    tree = HTMLParser(html)
    rows = tree.css_first("table").css("tr")
    m = HEADER_RE.search(rows[0].css_first("td[colspan]").text(separator=" ", strip=True))
    if m is None:
        raise ValueError(f"Unparsed header in {match_id}")
    teams = []
    for r in rows[1:3]:
        cells = r.css("td")
        periods = [q for q in (SCORE_RE.search(c.text(strip=True)) for c in cells[1:]) if q]
        g, b, p = (int(x) for x in periods[-1].groups())
        teams.append({"team": cells[0].text(strip=True), "goals": g, "behinds": b, "score": p,
                      "n_periods": len(periods)})
    home, away = teams
    match = {
        "match_id": match_id, "season": season, "round_label": m["round"],
        "date": datetime.strptime(m["date"], "%a, %d-%b-%Y %I:%M %p"),
        "venue": m["venue"], "attendance": int(m["att"]) if m["att"] else None,
        "home_team": home["team"], "away_team": away["team"],
        "home_goals": home["goals"], "home_behinds": home["behinds"], "home_score": home["score"],
        "away_goals": away["goals"], "away_behinds": away["behinds"], "away_score": away["score"],
        "n_periods": home["n_periods"],
    }

    players, totals, rushed, details = [], [], {}, {}
    for tb in tree.css("table.sortable"):
        title = tb.css_first("thead tr").text(separator=" ", strip=True)
        if "Match Statistics" in title:
            team = title.split(" Match Statistics")[0].strip()
            heads = [th.text(strip=True) for th in tb.css("thead tr")[-1].css("th")]
            for r in tb.css("tbody tr"):
                cells = [c.text(strip=True) for c in r.css("td")]
                digits = re.sub(r"\D", "", cells[0])
                rec = {
                    "match_id": match_id, "team": team,
                    "player_id": _player_id(r.css_first("a").attributes["href"]),
                    "player_name": cells[1],
                    "jumper": int(digits) if digits else None,
                    "sub_status": "on" if SUB_ON in cells[0] else ("off" if SUB_OFF in cells[0] else None),
                }
                for h, v in zip(heads[2:], cells[2:]):
                    rec[STAT_COLS[h]] = _num(v) if h != "%P" else (int(v) if v.isdigit() else None)
                players.append(rec)
            for r in tb.css("tfoot tr"):
                cells = [c.text(strip=True) for c in r.css("td")]
                if cells[0] == "Totals":
                    rec = {"match_id": match_id, "team": team}
                    for h, v in zip(heads[2:], cells[1:]):
                        if h != "%P":
                            rec[STAT_COLS[h]] = _num(v)
                    totals.append(rec)
                elif cells[0] == "Rushed":
                    rushed[team] = next((int(c) for c in cells[1:] if c.isdigit()), 0)
        elif "Player Details" in title:
            for r in tb.css("tbody tr"):
                link = r.css_first("a")
                if link is None:
                    continue
                cells = [c.text(strip=True) for c in r.css("td")]
                age = AGE_RE.search(cells[2])
                career = LEADING_INT.search(cells[3])
                goals = LEADING_INT.search(cells[4])
                details[_player_id(link.attributes["href"])] = {
                    "age_days": round(int(age.group(1)) * 365.25 + int(age.group(2) or 0)) if age else None,
                    "career_games": int(career.group(1)) if career else None,
                    "career_goals": int(goals.group(1)) if goals else 0,
                }
    for t in totals:
        t["rushed_behinds"] = rushed.get(t["team"]) or 0
    empty = {"age_days": None, "career_games": None, "career_goals": None}
    for p in players:
        p.update(details.get(p["player_id"], empty))
    return match, players, totals


def main() -> None:
    matches, players, totals = [], [], []
    for season in SEASONS:
        files = sorted((HTML_CACHE / "games" / str(season)).glob("*.html"))
        for f in files:
            m, p, t = parse_match(decode(f.read_bytes()), season, f.stem)
            matches.append(m)
            players.extend(p)
            totals.extend(t)
        print(f"{season}: parsed {len(files)} matches")

    mdf = pl.DataFrame(matches).sort("date")
    pdf = (
        pl.DataFrame(players, infer_schema_length=None)
        .join(mdf.select("match_id", "season", "date", "round_label", "venue", "home_team", "away_team"),
              on="match_id")
        .with_columns(
            is_home=pl.col("team") == pl.col("home_team"),
            opponent=pl.when(pl.col("team") == pl.col("home_team"))
            .then(pl.col("away_team")).otherwise(pl.col("home_team")),
            # Named substitutes who never came on are listed with no time-on-ground figure.
            took_field=pl.col("pct_time_played").is_not_null(),
        )
        .drop("home_team", "away_team")
        .sort("date", "match_id", "team", "player_name")
    )
    PROCESSED.mkdir(parents=True, exist_ok=True)
    mdf.write_parquet(PROCESSED / "matches.parquet")
    pdf.write_parquet(PROCESSED / "player_stats.parquet")
    pl.DataFrame(totals).write_parquet(PROCESSED / "team_totals.parquet")
    print(f"matches={mdf.height} player_rows={pdf.height} players={pdf['player_id'].n_unique()}")


if __name__ == "__main__":
    main()
