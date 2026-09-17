"""LLM judge for answer correctness against the expected answer."""

import json
import logging
from typing import Protocol

from ragaudit.models.scoring import CorrectnessResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are judging whether a candidate answer correctly answers a question, "
    "given a reference (expected) answer. Respond with strict JSON only, no "
    "prose, no markdown fences, in the form: "
    '{"is_correct": bool, "score": float between 0 and 1, "rationale": str}'
)


class Completer(Protocol):
    """The subset of LLMClient this module depends on — easy to stub in tests."""

    model: str

    def complete(self, system: str, user: str) -> str: ...


def _build_user_prompt(question: str, expected_answer: str, candidate_answer: str) -> str:
    return (
        f"Question: {question}\n"
        f"Expected answer: {expected_answer}\n"
        f"Candidate answer: {candidate_answer}\n\n"
        "Is the candidate answer correct?"
    )


def parse_correctness_response(raw: str) -> dict | None:
    """Parse the judge's JSON response. Returns None on any malformed input."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    is_correct = data.get("is_correct")
    if not isinstance(is_correct, bool):
        return None

    score = data.get("score")
    if score is not None and not isinstance(score, (int, float)):
        return None

    rationale = data.get("rationale")
    if rationale is not None and not isinstance(rationale, str):
        return None

    return {
        "is_correct": is_correct,
        "score": float(score) if score is not None else None,
        "rationale": rationale,
    }


def judge_correctness(
    client: Completer,
    question: str,
    expected_answer: str,
    candidate_answer: str,
) -> CorrectnessResult:
    """Judge whether candidate_answer correctly answers question, vs expected_answer.

    On unparseable judge output, conservatively returns is_correct=False
    with a rationale noting the parse failure, rather than raising — one
    bad judge call must not crash a run.
    """
    raw = client.complete(SYSTEM_PROMPT, _build_user_prompt(question, expected_answer, candidate_answer))
    parsed = parse_correctness_response(raw)

    if parsed is None:
        logger.warning("Correctness judge returned unparseable JSON for question: %s", question)
        return CorrectnessResult(
            is_correct=False,
            score=None,
            rationale="judge returned unparseable JSON",
            judge_model=client.model,
            judge_parse_failed=True,
        )

    return CorrectnessResult(
        is_correct=parsed["is_correct"],
        score=parsed["score"],
        rationale=parsed["rationale"],
        judge_model=client.model,
    )
