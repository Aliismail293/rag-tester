"""Pure text helpers for v2 generation: tokenizing, lexical overlap, relative-time detection."""

import re

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have having
    he her here hers herself him himself his how i if in into is it its itself just me more most my myself no
    nor not of off on once only or other our ours ourselves out over own same she should so some such than that
    the their theirs them themselves then there these they this those through to too under until up very was we
    were what when where which while who whom whose why will with would you your yours yourself yourselves
    s t don shall may might must per via vs etc many much
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")

RELATIVE_TIME_RE = re.compile(
    r"\b("
    r"yesterday|today|tonight|tomorrow|now|nowadays|currently|current|recent|recently|lately|latest|"
    r"these\s+days|at\s+present|at\s+the\s+moment|right\s+now|so\s+far|to\s+date|up\s+to\s+date|"
    r"(?:this|last|next|past|previous|coming|upcoming)\s+"
    r"(?:few\s+)?(?:day|week|weekend|month|quarter|year|decade|season|spring|summer|autumn|fall|winter)s?|"
    r"(?:a\s+few|several|some|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:day|week|month|year|decade)s?\s+(?:ago|back)|"
    r"ago|most\s+recent(?:ly)?"
    r")\b",
    re.IGNORECASE,
)


def _normalize(token: str) -> str:
    """A crude plural fold so "calibrations" and "calibration" count as the same word."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_normalize(t) for t in _TOKEN_RE.findall(text.lower())]


def content_tokens(text: str) -> set[str]:
    """Distinct non-stopword tokens (stopwords are checked before the plural fold, so "does" stays out)."""
    return {_normalize(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS}


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_SENTENCE_END_RE = re.compile(r"[.!?:;]['\")\]]*$")


def name_tokens(source_text: str) -> set[str]:
    """Tokens that are names in the source: capitalised mid-sentence and never written in lower case there.

    "Redwatch Protocol", "Thistlewick Robotics", "EL-7". A question about a
    named thing has to use its name, so names don't count as reused wording.
    Words at the start of a sentence, a line or a list item, and Markdown
    headings, are ignored, so ordinary capitalised words don't qualify.
    """
    capitalised, lower = set(), set()
    sentence_start = True
    for line in source_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            sentence_start = True  # a new paragraph or a heading
            continue
        item = re.sub(r"^([-*+>]|\d+[.)])\s+", "", stripped)
        if item != stripped:
            sentence_start = True  # a list item starts a sentence; a wrapped line continues one
        stripped = item
        for match in _WORD_RE.finditer(stripped):
            word = match.group(0)
            tokens = {_normalize(t) for t in _TOKEN_RE.findall(word.lower())}
            if word[0].isupper() and not sentence_start:
                capitalised |= tokens
            elif word.islower():
                lower |= tokens
            following = stripped[match.end():match.end() + 3]
            sentence_start = bool(_SENTENCE_END_RE.match(following.split(" ")[0])) if following else False
    return capitalised - lower


def lexical_overlap(question: str, source_text: str) -> float:
    """Fraction of the question's distinct content tokens that also appear in the source text.

    High overlap means the question reuses the chunk's wording, which makes
    keyword retrieval trivially easy and inflates retrieval scores. Names
    from the source (see name_tokens) are left out: they can't be reworded.
    A question with no other content tokens scores 0.0.
    """
    question_tokens = content_tokens(question) - name_tokens(source_text)
    if not question_tokens:
        return 0.0
    return len(question_tokens & content_tokens(source_text)) / len(question_tokens)


def find_relative_time(text: str) -> str | None:
    """Return the first relative-time expression in text ("last week", "recently", ...), or None."""
    match = RELATIVE_TIME_RE.search(text)
    return match.group(0) if match else None


def has_relative_time(text: str) -> bool:
    return find_relative_time(text) is not None


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
