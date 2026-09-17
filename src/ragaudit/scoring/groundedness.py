"""LLM judge for whether an answer is grounded in the contexts it was given."""

import json
import logging
from typing import Protocol

from ragaudit.models.scoring import GroundednessResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are judging whether an answer is faithfully grounded in the "
    "provided contexts — i.e. every claim in the answer is supported by "
    "the contexts, regardless of whether the contexts are actually relevant "
    "to the question. Respond with strict JSON only, no prose, no markdown "
    'fences, in the form: {"is_grounded": bool, "score": float between 0 '
    'and 1, "rationale": str}'
)


class Completer(Protocol):
    """The subset of LLMClient this module depends on — easy to stub in tests."""

    model: str

    def complete(self, system: str, user: str) -> str: ...


def _build_user_prompt(answer: str, contexts: list[str]) -> str:
    joined_contexts = "\n---\n".join(contexts) if contexts else "(no contexts provided)"
    return f"Contexts:\n{joined_contexts}\n\nAnswer:\n{answer}\n\nIs the answer grounded in the contexts?"


def parse_groundedness_response(raw: str) -> dict | None:
    """Parse the judge's JSON response. Returns None on any malformed input."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    is_grounded = data.get("is_grounded")
    if not isinstance(is_grounded, bool):
        return None

    score = data.get("score")
    if score is not None and not isinstance(score, (int, float)):
        return None

    rationale = data.get("rationale")
    if rationale is not None and not isinstance(rationale, str):
        return None

    return {
        "is_grounded": is_grounded,
        "score": float(score) if score is not None else None,
        "rationale": rationale,
    }


def judge_groundedness(client: Completer, answer: str, contexts: list[str]) -> GroundednessResult:
    """Judge whether answer is faithfully supported by contexts.

    On unparseable judge output, conservatively returns is_grounded=False
    with a rationale noting the parse failure, rather than raising.
    """
    raw = client.complete(SYSTEM_PROMPT, _build_user_prompt(answer, contexts))
    parsed = parse_groundedness_response(raw)

    if parsed is None:
        logger.warning("Groundedness judge returned unparseable JSON for answer: %s", answer)
        return GroundednessResult(
            is_grounded=False,
            score=None,
            rationale="judge returned unparseable JSON",
            judge_model=client.model,
            judge_parse_failed=True,
        )

    return GroundednessResult(
        is_grounded=parsed["is_grounded"],
        score=parsed["score"],
        rationale=parsed["rationale"],
        judge_model=client.model,
    )
