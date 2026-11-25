"""The lucky-pass report: one row per question, plus the overstatement with a bootstrap interval.

A lucky pass is an answer judged correct while the gold passage was not retrieved. The closed-book
check (the run's ablation) splits them in two:

- memory lucky: still correct with no context at all, so the model knew it from training;
- guess lucky: wrong with no context, so the retrieved (non-gold) passages or a guess got it right.

Overstatement = answer accuracy - true-pass rate = lucky-pass rate. It is how much an answer-only
score credits retrieval for answers retrieval didn't earn. Pure: no I/O except write_lucky_report,
no LLM.
"""

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from ragaudit.models.run import Run
from ragaudit.models.scoring import ScoredCase, Verdict
from ragaudit.models.testset import TestSet

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0

MEMORY_LUCKY = "memory_lucky"
GUESS_LUCKY = "guess_lucky"
UNCHECKED = "unchecked"
"""A lucky pass whose closed-book check didn't run (`ragaudit run --no-ablation`)."""


@dataclass
class LuckyRow:
    test_id: str
    question: str
    expected_answer: str
    answer: str
    verdict: str
    gold_retrieved: bool
    correct: bool
    closed_book_answer: str | None
    closed_book_correct: bool | None
    lucky_kind: str | None
    """memory_lucky, guess_lucky or unchecked for lucky passes; None otherwise."""


@dataclass
class LuckySummary:
    n: int
    accuracy: float
    true_pass_rate: float
    overstatement: float
    overstatement_ci: tuple[float, float]
    lucky_passes: int
    memory_lucky: int
    guess_lucky: int
    unchecked: int


def lucky_kind(case: ScoredCase) -> str | None:
    if case.verdict != Verdict.LUCKY_PASS:
        return None
    if case.ablation is None:
        return UNCHECKED
    return MEMORY_LUCKY if case.ablation.is_correct else GUESS_LUCKY


def lucky_rows(run: Run, test_set: TestSet) -> list[LuckyRow]:
    cases = {tc.test_id: tc for tc in test_set.test_cases}
    rows = []
    for case in run.scored_cases:
        test_case = cases.get(case.test_id)
        rows.append(
            LuckyRow(
                test_id=case.test_id,
                question=test_case.question if test_case else "",
                expected_answer=test_case.expected_answer if test_case else "",
                answer=case.response.answer,
                verdict=case.verdict.value,
                gold_retrieved=case.retrieval.hit,
                correct=case.correctness.is_correct,
                closed_book_answer=case.ablation.answer if case.ablation else None,
                closed_book_correct=case.ablation.is_correct if case.ablation else None,
                lucky_kind=lucky_kind(case),
            )
        )
    return rows


def bootstrap_ci(flags: list[bool], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """95% percentile bootstrap interval for the share of True values. Seeded, so reports are reproducible."""
    if not flags:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(flags)
    means = sorted(sum(rng.choice(flags) for _ in range(n)) / n for _ in range(resamples))
    return (means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1])


def summarize(rows: list[LuckyRow]) -> LuckySummary:
    n = len(rows)
    lucky = [r.verdict == Verdict.LUCKY_PASS.value for r in rows]
    kinds = [r.lucky_kind for r in rows]
    return LuckySummary(
        n=n,
        accuracy=sum(r.correct for r in rows) / n if n else 0.0,
        true_pass_rate=sum(r.verdict == Verdict.TRUE_PASS.value for r in rows) / n if n else 0.0,
        overstatement=sum(lucky) / n if n else 0.0,
        overstatement_ci=bootstrap_ci(lucky),
        lucky_passes=sum(lucky),
        memory_lucky=kinds.count(MEMORY_LUCKY),
        guess_lucky=kinds.count(GUESS_LUCKY),
        unchecked=kinds.count(UNCHECKED),
    )


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _cell(text: str | None) -> str:
    return "" if text is None else str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(run: Run, rows: list[LuckyRow], summary: LuckySummary) -> str:
    lo, hi = summary.overstatement_ci
    lines = [
        f"# Lucky-pass report: run `{run.config.run_id}`",
        "",
        "A **lucky pass** is an answer judged correct although the gold passage was not retrieved.",
        "",
        f"- Questions: **{summary.n}**",
        f"- Answer accuracy: **{_pct(summary.accuracy)}**",
        f"- True-pass rate (correct and gold retrieved): **{_pct(summary.true_pass_rate)}**",
        f"- **Overstatement** (accuracy minus true-pass rate = lucky-pass rate): **{_pct(summary.overstatement)}** "
        f"(95% bootstrap CI {_pct(lo)} to {_pct(hi)}, n = {summary.n})",
        f"- Lucky passes: {summary.lucky_passes}: {summary.memory_lucky} memory lucky (correct with no context), "
        f"{summary.guess_lucky} guess lucky (wrong with no context), {summary.unchecked} not checked",
        "",
    ]
    if summary.unchecked:
        lines += [
            "Some lucky passes have no closed-book check. Re-run without `--no-ablation` to split them into "
            "memory and guess.",
            "",
        ]
    lucky = [r for r in rows if r.lucky_kind is not None]
    lines += ["## Lucky passes", ""]
    if not lucky:
        lines += ["None: every correct answer came with the gold passage retrieved.", ""]
    else:
        lines += [
            "| Question | Expected | RAG answer | Closed-book answer | Kind |",
            "|---|---|---|---|---|",
        ]
        for r in lucky:
            lines.append(
                f"| {_cell(r.question)} | {_cell(r.expected_answer)} | {_cell(r.answer)} | "
                f"{_cell(r.closed_book_answer)} | {r.lucky_kind} |"
            )
        lines.append("")
    lines += ["The full per-question table is in `lucky.csv` next to this file.", ""]
    return "\n".join(lines)


CSV_FIELDS = [
    "test_id", "question", "expected_answer", "answer", "verdict", "gold_retrieved", "correct",
    "closed_book_answer", "closed_book_correct", "lucky_kind",
]


def write_lucky_report(run: Run, test_set: TestSet, out: Path) -> Path:
    """Write <out> (Markdown) and lucky.csv beside it. Returns the Markdown path."""
    rows = lucky_rows(run, test_set)
    summary = summarize(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(run, rows, summary))
    with (out.parent / "lucky.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({field: getattr(r, field) for field in CSV_FIELDS})
    return out
