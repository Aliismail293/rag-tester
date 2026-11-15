"""A plain-English summary of a run: what's wrong, in at most six lines.

Shared by `ragaudit check` (printed in the terminal) and the top of the
HTML run report, so both say the same thing.
"""

from dataclasses import dataclass

from ragaudit.models.run import RunSummary
from ragaudit.models.scoring import ScoredCase, Verdict

HEALTHY_TRUE_PASS_RATE = 0.8
"""A run is healthy when at least this share of answers is correct and backed by the right passage."""

# Tie-break order when two failure classes are equally common: a retrieval
# problem undermines everything downstream, and a lucky pass hides one.
_FAILURE_ORDER = (Verdict.RETRIEVAL_FAILURE, Verdict.LUCKY_PASS, Verdict.GENERATION_FAILURE)


@dataclass
class PlainSummary:
    status: str
    """healthy, needs_work, or empty."""
    verdict: str
    retrieval: str
    correctness: str
    lucky: str
    fix: str
    dominant: Verdict | None

    def lines(self) -> list[str]:
        return [self.verdict, self.retrieval, self.correctness, self.lucky, f"Most useful fix: {self.fix}"]


def _pct(rate: float) -> str:
    return f"{rate:.0%}"


def dominant_failure(summary: RunSummary) -> Verdict | None:
    counts = {verdict: summary.verdict_counts.get(verdict, 0) for verdict in _FAILURE_ORDER}
    worst = max(_FAILURE_ORDER, key=lambda verdict: (counts[verdict], -_FAILURE_ORDER.index(verdict)))
    return worst if counts[worst] else None


def _fix(dominant: Verdict | None, count: int, no_contexts: bool, match_by: str, no_id_hits: bool = False) -> str:
    if no_id_hits:
        return (
            "no retrieved chunk id matched ragaudit's, which usually means your pipeline uses its own chunks. "
            "Rerun with --match-by text, or return ragaudit's chunk ids in retrieved_chunk_ids."
        )
    if no_contexts:
        return (
            "your adapter returned no retrieved passages, so retrieval could not be scored. Return the text "
            "of each retrieved passage in RAGResponse.contexts."
        )
    if dominant is None:
        return "nothing stands out. Keep this check in CI to catch regressions."
    if dominant == Verdict.RETRIEVAL_FAILURE:
        hint = "retrieve more passages (raise top-k), add keyword/hybrid search, or try smaller chunks"
        if match_by == "id":
            hint += "; if your pipeline chunks the docs itself, rerun with --match-by text"
        return f"improve retrieval. For {count} question(s) the right passage was not found and the answer was wrong: {hint}."
    if dominant == Verdict.LUCKY_PASS:
        return (
            f"don't trust the correct answers yet. {count} were right without the right passage being retrieved, "
            "so the model answered from memory. Fix retrieval first, and tell the model to answer only from the context."
        )
    return (
        f"improve answer generation. For {count} question(s) the right passage was retrieved but the answer was "
        "still wrong: tighten the prompt to answer from the context, show it fewer but better passages, or use a stronger model."
    )


def build_plain_summary(summary: RunSummary, scored_cases: list[ScoredCase]) -> PlainSummary:
    total = summary.total_cases
    if total == 0:
        return PlainSummary(
            status="empty",
            verdict="Verdict: NO RESULTS. The run had no questions to score.",
            retrieval="Retrieval hit rate: n/a",
            correctness="Answer correctness: n/a",
            lucky="Lucky passes: n/a",
            fix="generate questions first (check that the docs folder has .txt or .md files).",
            dominant=None,
        )
    counts = summary.verdict_counts
    true_pass = counts.get(Verdict.TRUE_PASS, 0)
    lucky = counts.get(Verdict.LUCKY_PASS, 0)
    hits = sum(1 for case in scored_cases if case.retrieval.hit)
    correct = sum(1 for case in scored_cases if case.correctness.is_correct)
    match_by = next((case.retrieval.match_by for case in scored_cases), "id")
    no_contexts = match_by == "text" and not any(case.response.contexts for case in scored_cases)
    dominant = dominant_failure(summary)

    if true_pass / total >= HEALTHY_TRUE_PASS_RATE and not no_contexts:
        status = "healthy"
        verdict = (
            f"Verdict: HEALTHY. {true_pass} of {total} answers were correct and came from the right passage."
        )
    else:
        status = "needs_work"
        problem = {
            Verdict.RETRIEVAL_FAILURE: "retrieval is missing the right passage",
            Verdict.LUCKY_PASS: "correct answers are not coming from retrieval",
            Verdict.GENERATION_FAILURE: "answers are wrong even when retrieval is right",
            None: "too few answers are both correct and grounded",
        }[None if no_contexts else dominant]
        verdict = f"Verdict: NEEDS WORK. Only {true_pass} of {total} answers were correct and from the right passage; main problem: {problem}."

    confirmed = summary.ablation_confirmed_count
    lucky_line = f"Lucky passes: {lucky} (correct answer, but the right passage was not retrieved)"
    if lucky and confirmed is not None:
        lucky_line += f"; {confirmed} still correct with no context at all, so answered from memory"
    return PlainSummary(
        status=status,
        verdict=verdict,
        retrieval=f"Retrieval hit rate: {_pct(summary.retrieval_hit_rate)} ({hits} of {total} found the passage the question came from, matched by {match_by})",
        correctness=f"Answer correctness: {_pct(summary.answer_correctness_rate)} ({correct} of {total} judged correct)",
        lucky=lucky_line,
        fix=_fix(dominant, counts.get(dominant, 0) if dominant else 0, no_contexts, match_by, match_by == "id" and hits == 0),
        dominant=dominant,
    )
