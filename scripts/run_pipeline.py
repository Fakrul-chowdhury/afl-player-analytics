"""Run the full pipeline end to end: scrape -> parse -> validate -> clean -> features -> train -> explain -> figures."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afl import clean, explain, features, figures, parse, scrape, train, validate  # noqa: E402

if __name__ == "__main__":
    for step in (scrape, parse, validate, clean, features, train, explain, figures):
        print(f"\n=== {step.__name__} ===", flush=True)
        step.main()
