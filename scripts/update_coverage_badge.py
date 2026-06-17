"""
scripts/update_coverage_badge.py

Reads coverage.json (written by `pytest --cov-report=json`) and rewrites
the coverage badge in README.md between the coverage-badge marker
comments. Run only by CI, after the test step, on pushes to main.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

COVERAGE_FILE = Path("coverage.json")
README_FILE = Path("README.md")
MARKER_PATTERN = re.compile(
    r"<!-- coverage-badge-start -->.*<!-- coverage-badge-end -->", re.DOTALL
)


def badge_colour(percent: int) -> str:
    if percent >= 90:
        return "4ade80"  # brand green: status good
    if percent >= 75:
        return "f5a623"  # brand amber: acceptable, room to improve
    return "e24b4a"  # brand red: needs attention


def main() -> None:
    data = json.loads(COVERAGE_FILE.read_text())
    percent = round(data["totals"]["percent_covered"])
    colour = badge_colour(percent)
    badge = (
        f"![Coverage](https://img.shields.io/badge/coverage-{percent}%25-"
        f"{colour}?style=flat-square&labelColor=0a0a0f)"
    )
    replacement = f"<!-- coverage-badge-start -->\n{badge}\n<!-- coverage-badge-end -->"

    readme_text = README_FILE.read_text()
    if not MARKER_PATTERN.search(readme_text):
        raise SystemExit("README.md is missing the coverage-badge markers")

    README_FILE.write_text(MARKER_PATTERN.sub(replacement, readme_text))
    print(f"Coverage badge updated: {percent}%")


if __name__ == "__main__":
    main()
