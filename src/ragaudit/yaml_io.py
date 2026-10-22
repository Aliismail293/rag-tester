"""Read and write the small YAML files v2 uses (pipeline configs, tuning search spaces).

Uses PyYAML when it's installed (`pip install ragaudit[yaml]`); otherwise
falls back to a built-in parser for the subset these files need, so YAML
support adds no required dependency:

- nested mappings by indentation, `# comments`, blank lines;
- scalars: integers, floats, true/false, null/~, quoted or plain strings;
- lists, either inline (`[1, 2, "a"]`) or as `- item` blocks of scalars;
- inline mappings (`{min: 0, max: 1}`).

Files written here use only that subset, so they round-trip either way.
"""

import json
import re
from pathlib import Path

try:  # pragma: no cover - exercised only when PyYAML is installed
    import yaml as _pyyaml
except ImportError:  # pragma: no cover
    _pyyaml = None


class YAMLError(ValueError):
    """Raised for YAML the reader can't parse."""


_INT_RE = re.compile(r"^[-+]?\d+$")
_FLOAT_RE = re.compile(r"^[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?$")
_PLAIN_SAFE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_./-]*$")


def _strip_comment(line: str) -> str:
    in_quote = None
    for i, char in enumerate(line):
        if char in "\"'" and in_quote in (None, char):
            in_quote = None if in_quote == char else char
        elif char == "#" and in_quote is None and (i == 0 or line[i - 1].isspace()):
            return line[:i].rstrip()
    return line.rstrip()


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    parts, depth, quote, current = [], 0, None, ""
    for char in text:
        if quote:
            current += char
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        if char == sep and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current.strip())
    return parts


def _parse_scalar(text: str):
    text = text.strip()
    if text == "" or text in {"null", "~", "Null", "NULL"}:
        return None
    if text in {"true", "True", "TRUE"}:
        return True
    if text in {"false", "False", "FALSE"}:
        return False
    if text.startswith("[") and text.endswith("]"):
        return [_parse_scalar(part) for part in _split_top_level(text[1:-1])]
    if text.startswith("{") and text.endswith("}"):
        mapping = {}
        for part in _split_top_level(text[1:-1]):
            key, _, value = part.partition(":")
            mapping[str(_parse_scalar(key))] = _parse_scalar(value)
        return mapping
    if text[0] == '"':
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise YAMLError(f"bad double-quoted string: {text}") from exc
    if text[0] == "'":
        if not text.endswith("'"):
            raise YAMLError(f"bad single-quoted string: {text}")
        return text[1:-1].replace("''", "'")
    if _INT_RE.match(text):
        return int(text)
    if _FLOAT_RE.match(text):
        return float(text)
    return text


def _parse_block(lines: list[tuple[int, str]], start: int, indent: int):
    """Parse lines[start:] at this indent into a dict or list. Returns (value, next index)."""
    if lines[start][1].startswith("- "):
        items = []
        index = start
        while index < len(lines) and lines[index][0] == indent and lines[index][1].startswith("- "):
            items.append(_parse_scalar(lines[index][1][2:]))
            index += 1
        return items, index

    mapping = {}
    index = start
    while index < len(lines):
        line_indent, text = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise YAMLError(f"unexpected indentation: {text!r}")
        key, sep, rest = text.partition(":")
        if not sep:
            raise YAMLError(f"expected 'key: value', got {text!r}")
        key = str(_parse_scalar(key))
        index += 1
        if rest.strip():
            mapping[key] = _parse_scalar(rest)
        elif index < len(lines) and lines[index][0] > indent:
            mapping[key], index = _parse_block(lines, index, lines[index][0])
        elif index < len(lines) and lines[index][0] == indent and lines[index][1].startswith("- "):
            mapping[key], index = _parse_block(lines, index, indent)
        else:
            mapping[key] = None
    return mapping, index


def loads(text: str):
    """Parse YAML text (the subset above, or anything PyYAML accepts when it's installed)."""
    if _pyyaml is not None:
        try:
            return _pyyaml.safe_load(text)
        except _pyyaml.YAMLError as exc:
            raise YAMLError(str(exc)) from exc

    lines = []
    for raw in text.splitlines():
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise YAMLError("tabs are not allowed for indentation")
        stripped = _strip_comment(raw)
        if stripped.strip() and stripped.strip() != "---":
            lines.append((len(stripped) - len(stripped.lstrip()), stripped.strip()))
    if not lines:
        return None
    value, index = _parse_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise YAMLError(f"could not parse line: {lines[index][1]!r}")
    return value


def load(path: str | Path):
    return loads(Path(path).read_text(encoding="utf-8"))


def _format_scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_scalar(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_format_scalar(str(k))}: {_format_scalar(v)}" for k, v in value.items()) + "}"
    text = str(value)
    if _PLAIN_SAFE_RE.match(text) and _parse_scalar(text) == text:
        return text
    return json.dumps(text)


def dumps(data: dict, header: str | None = None) -> str:
    """Serialize a mapping (nested mappings, lists and scalars) as block-style YAML."""
    lines = [f"# {line}" if line else "#" for line in header.splitlines()] if header else []

    def emit(mapping: dict, indent: int) -> None:
        for key, value in mapping.items():
            prefix = " " * indent + f"{_format_scalar(str(key))}:"
            if isinstance(value, dict) and value:
                lines.append(prefix)
                emit(value, indent + 2)
            else:
                lines.append(f"{prefix} {_format_scalar(value)}")

    emit(data, 0)
    return "\n".join(lines) + "\n"


def dump(data: dict, path: str | Path, header: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(data, header), encoding="utf-8")
    return path
