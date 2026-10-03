"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
HTML_CACHE = RAW / "afltables"
SQUIGGLE_CACHE = RAW / "squiggle"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
RESULTS = ROOT / "results"

SEASONS = list(range(2021, 2027))

REPO_URL = "https://github.com/Fakrul-chowdhury/afl-player-analytics"
USER_AGENT = f"afl-player-analytics/1.0 (portfolio research project; contact via {REPO_URL})"

# Seconds between requests to the same host (polite scraping).
REQUEST_DELAY = 2.5
