"""Pure aggregation of ScoredCases into a RunSummary. No I/O, no LLM."""

from ragaudit.models.run import RunSummary
from ragaudit.models.scoring import ScoredCase, Verdict


def compute_run_summary(scored_cases: list[ScoredCase]) -> RunSummary:
    """Aggregate a list of ScoredCases into a RunSummary."""
    total = len(scored_cases)
    verdict_counts: dict[Verdict, int] = {verdict: 0 for verdict in Verdict}
    groundedness_scores: list[float] = []
    lucky_pass_cases: list[ScoredCase] = []
    retrieval_hits = 0
    correct = 0

    for case in scored_cases:
        verdict_counts[case.verdict] += 1
        if case.retrieval.hit:
            retrieval_hits += 1
        if case.correctness.is_correct:
            correct += 1
        if case.groundedness.score is not None:
            groundedness_scores.append(case.groundedness.score)
        if case.verdict == Verdict.LUCKY_PASS:
            lucky_pass_cases.append(case)

    ablation_confirmed_count = None
    if lucky_pass_cases:
        ablation_confirmed_count = sum(
            1 for case in lucky_pass_cases if case.ablation is not None and case.ablation.confirms_lucky_pass
        )

    return RunSummary(
        total_cases=total,
        verdict_counts=verdict_counts,
        retrieval_hit_rate=(retrieval_hits / total) if total else 0.0,
        answer_correctness_rate=(correct / total) if total else 0.0,
        lucky_pass_rate=(verdict_counts[Verdict.LUCKY_PASS] / total) if total else 0.0,
        mean_groundedness=(sum(groundedness_scores) / len(groundedness_scores)) if groundedness_scores else None,
        ablation_confirmed_count=ablation_confirmed_count,
    )
