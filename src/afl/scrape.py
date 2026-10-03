"""Polite, cached downloader for AFL Tables season and match pages.

AFL Tables publishes no robots.txt and no terms-of-use page; we still rate-limit
every request, identify ourselves with a descriptive User-Agent and cache every
page to disk so each URL is fetched at most once.
"""
from __future__ import annotations

import re
import sys
import time

import httpx

from afl.config import HTML_CACHE, REQUEST_DELAY, SEASONS, USER_AGENT

BASE = "https://afltables.com/afl"
GAME_LINK = re.compile(r"stats/games/(\d{4})/(\d+)\.html")

_last_request = 0.0


def _get(client: httpx.Client, url: str) -> bytes:
    global _last_request
    wait = REQUEST_DELAY - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    for attempt in range(4):
        _last_request = time.monotonic()
        try:
            r = client.get(url)
            if r.status_code == 200:
                return r.content
            print(f"  HTTP {r.status_code} for {url}", file=sys.stderr)
        except httpx.HTTPError as exc:
            print(f"  {type(exc).__name__} for {url}", file=sys.stderr)
        time.sleep(REQUEST_DELAY * 2 ** (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}")


def fetch_cached(client: httpx.Client, url: str, path, refresh: bool = False) -> bytes:
    if path.exists() and not refresh:
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _get(client, url)
    path.write_bytes(content)
    return content


def game_ids(season_html: bytes, season: int) -> list[str]:
    ids = {g for y, g in GAME_LINK.findall(season_html.decode("latin-1")) if int(y) == season}
    return sorted(ids)


def main(seasons: list[int] = SEASONS) -> None:
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True) as client:
        for season in seasons:
            season_html = fetch_cached(client, f"{BASE}/seas/{season}.html", HTML_CACHE / "seasons" / f"{season}.html")
            ids = game_ids(season_html, season)
            print(f"{season}: {len(ids)} games", flush=True)
            for i, gid in enumerate(ids, 1):
                fetch_cached(client, f"{BASE}/stats/games/{season}/{gid}.html",
                             HTML_CACHE / "games" / str(season) / f"{gid}.html")
                if i % 50 == 0:
                    print(f"  {season}: {i}/{len(ids)}", flush=True)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or SEASONS)
