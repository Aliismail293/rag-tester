"""Retrieval scoring by text overlap, for pipelines that don't carry ragaudit's chunk ids.

`--match-by id` (the default) counts a retrieval hit only when the adapter
returns the gold chunk's id, so the pipeline has to be indexed against
ragaudit's chunks. `--match-by text` instead compares the gold chunk's
*text* with the contexts the adapter already returns, so a pipeline that
chunks the corpus its own way works without re-indexing.

The score is the overlap coefficient of word bigrams: shared bigrams
divided by the bigram count of the smaller side. It is high when most of
the gold passage appears inside a returned context (the pipeline's chunks
are bigger) or most of a context appears inside the gold passage (they are
smaller), whatever the chunk boundaries. Bigrams rather than single words
keep two passages on the same topic from matching just because they share
vocabulary. The denominator never drops below MIN_BIGRAMS, so a context of
a few words can't match everything.
"""

import re

DEFAULT_MATCH_THRESHOLD = 0.5
MIN_BIGRAMS = 8
MATCH_MODES = ("id", "text")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _bigrams(text: str) -> set[tuple[str, str]]:
    tokens = _TOKEN_RE.findall(text.lower())
    return set(zip(tokens, tokens[1:]))


def text_match_score(gold_text: str, context: str) -> float:
    """Overlap of gold_text and context in [0, 1] (see the module docstring)."""
    gold, ctx = _bigrams(gold_text), _bigrams(context)
    if not gold or not ctx:
        return 0.0
    return len(gold & ctx) / max(min(len(gold), len(ctx)), MIN_BIGRAMS)


def first_matching_context(gold_text: str, contexts: list[str], threshold: float) -> tuple[int | None, float]:
    """(index of the first context scoring >= threshold or None, best score over all contexts)."""
    scores = [text_match_score(gold_text, context) for context in contexts]
    rank = next((i for i, score in enumerate(scores) if score >= threshold), None)
    return rank, max(scores, default=0.0)


def validate_match_by(match_by: str) -> str:
    if match_by not in MATCH_MODES:
        raise ValueError(f"--match-by must be id or text, got {match_by!r}")
    return match_by
