"""Load RAGAUDIT_* settings from a .env file in the working directory.

Only KEY=VALUE lines are read; comments and blank lines are skipped, and
surrounding quotes are stripped. Variables already set in the environment
always win, so an exported value is never replaced by the file.
"""

import os
from pathlib import Path

DOTENV_PATH = Path(".env")


def parse_dotenv(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key:
            values[key] = value
    return values


def load_dotenv(path: Path = DOTENV_PATH) -> list[str]:
    """Set variables from `path` that aren't already set. Returns the names it set (never the values)."""
    if not path.is_file():
        return []
    try:
        values = parse_dotenv(path.read_text())
    except (OSError, UnicodeDecodeError):
        return []
    loaded = []
    for key, value in values.items():
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
