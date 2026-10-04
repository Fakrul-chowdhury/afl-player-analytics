"""Capture dashboard screenshots for the README (requires the app running on localhost:8501)."""
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
TALL = "--tall" in sys.argv  # review mode: capture whole pages into a separate folder
args = [a for a in sys.argv[1:] if a != "--tall"]
BASE = args[0] if args else "http://localhost:8501"
if TALL:
    OUT = OUT.parent / "screenshots_full"
PAGES = {"": "overview", "leaders": "leaders", "teams": "teams", "elo": "team_strength", "what-wins": "what_wins",
         "player": "player", "similar": "similar", "head-to-head": "head_to_head", "models": "models",
         "explain": "explain", "importance": "importance", "about": "about"}
# Optional: a pre-installed Chromium (e.g. CHROMIUM=/opt/pw-browsers/chromium) instead of `playwright install`.
CHROMIUM = os.environ.get("CHROMIUM")
SCHEME = os.environ.get("SCHEME", "light")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    problems = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROMIUM)
        page = browser.new_page(viewport={"width": 1440, "height": 4200 if TALL else 1000}, color_scheme=SCHEME)
        page.on("console", lambda m: m.type == "error" and problems.append(f"console: {m.text}"))
        for path, name in PAGES.items():
            page.goto(f"{BASE}/{path}")
            page.wait_for_selector("[data-testid='stMainBlockContainer']", timeout=60000)
            page.wait_for_timeout(9000)
            errors = page.locator("[data-testid='stException']").all_inner_texts()
            problems += [f"{path}: {e[:300]}" for e in errors]
            suffix = "" if SCHEME == "light" else f"_{SCHEME}"
            page.screenshot(path=OUT / f"{name}{suffix}.png", full_page=True)
        browser.close()
    print("\n".join(problems) or "no errors")


if __name__ == "__main__":
    main()
