"""Pull one JSON object out of an LLM reply, tolerating the ways real models get it slightly wrong.

Every v2 prompt asks for strict JSON, but local models often wrap it in a
markdown fence, add a sentence before or after it, put a raw newline inside
a string, or drop the final closing brace. Each of those used to throw away
an otherwise good answer. This tries, in order:

1. the reply as is (inside a markdown fence, if there is one);
2. the span from the first "{" to the last "}";
3. the span from the first "{" with any unclosed brackets closed.

Strings are parsed with strict=False, so raw control characters inside a
string are accepted (callers that need one-line text check that
themselves). Anything still unparseable returns None.
"""

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _close_brackets(text: str) -> str | None:
    """Append the closers for any "{" / "[" left open, or None if the text ends inside a string."""
    stack: list[str] = []
    in_string = escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or stack.pop() != char:
                return None
    if in_string:
        return None
    return text + "".join(reversed(stack))


def _candidates(text: str) -> list[str]:
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    if start != -1:
        closed = _close_brackets(text[start:].rstrip())
        if closed is not None:
            candidates.append(closed)
    return candidates


def parse_json_object(raw: str) -> dict | None:
    """The first JSON object found in raw, or None."""
    text = raw.strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    for candidate in _candidates(text):
        try:
            data = json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None
