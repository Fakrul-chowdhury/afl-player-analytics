"""Fetch official match results from the Squiggle API for cross-validation.

Squiggle terms: identify your client with a descriptive User-Agent, cache
responses and avoid unnecessary requests. One request per season, cached.
"""
from __future__ import annotations

import json
import time

import httpx
import polars as pl

from afl.config import REQUEST_DELAY, SEASONS, SQUIGGLE_CACHE, USER_AGENT

API = "https://api.squiggle.com.au/"


def fetch_games(seasons: list[int] = SEASONS, refresh: bool = False) -> pl.DataFrame:
    SQUIGGLE_CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30) as client:
        for season in seasons:
            path = SQUIGGLE_CACHE / f"games_{season}.json"
            if refresh or not path.exists():
                r = client.get(API, params={"q": "games", "year": season})
                r.raise_for_status()
                path.write_text(r.text, encoding="utf-8")
                time.sleep(REQUEST_DELAY)
            games = json.loads(path.read_text(encoding="utf-8"))["games"]
            frames.append(pl.DataFrame(games, infer_schema_length=None))
    df = pl.concat(frames, how="diagonal_relaxed")
    return df.select(
        pl.col("id").alias("squiggle_id"),
        pl.col("year").alias("season"),
        "round", "roundname", "is_final", "is_grand_final", "complete",
        pl.col("date").str.slice(0, 10).str.to_date().alias("date"),
        "venue", "hteam", "ateam", "hscore", "ascore", "hgoals", "hbehinds", "agoals", "abehinds", "winner",
    )


if __name__ == "__main__":
    print(fetch_games().shape)
