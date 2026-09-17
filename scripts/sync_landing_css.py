#!/usr/bin/env python3
"""Regenerate docs/style.css from the canonical package stylesheet.

The source of truth is src/ragaudit/report/templates/style.css (shipped
as package data and used by the HTML report). docs/style.css is a plain
copy for the GitHub Pages landing page, which has no build step of its
own — run this script after editing the canonical file.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "src" / "ragaudit" / "report" / "templates" / "style.css"
DEST = REPO_ROOT / "docs" / "style.css"

GENERATED_HEADER = (
    "/* GENERATED FILE — do not edit directly.\n"
    "   Source of truth: src/ragaudit/report/templates/style.css\n"
    "   Regenerate with: python scripts/sync_landing_css.py */\n\n"
)


def main() -> None:
    css = SOURCE.read_text(encoding="utf-8")
    DEST.write_text(GENERATED_HEADER + css, encoding="utf-8")
    print(f"Wrote {DEST} from {SOURCE}")


if __name__ == "__main__":
    main()
