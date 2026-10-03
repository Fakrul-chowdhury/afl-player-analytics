"""Capture dashboard screenshots for the README (requires the app running on localhost:8501)."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
PAGES = {"": "teams", "player": "player", "head-to-head": "head_to_head",
         "models": "models", "importance": "importance", "about": "about"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    problems = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, color_scheme="light")
        page.on("console", lambda m: m.type == "error" and problems.append(f"console: {m.text}"))
        for path, name in PAGES.items():
            page.goto(f"{BASE}/{path}")
            page.wait_for_selector("[data-testid='stMainBlockContainer']", timeout=60000)
            page.wait_for_timeout(7000)
            errors = page.locator("[data-testid='stException']").all_inner_texts()
            problems += [f"{path}: {e[:300]}" for e in errors]
            page.screenshot(path=OUT / f"{name}.png", full_page=True)
        browser.close()
    print("\n".join(problems) or "no errors")


if __name__ == "__main__":
    main()
