"""Write docs/sample-lucky-report.md (and docs/lucky.csv) from a small invented run. No LLM calls.

The questions and answers are made up to show the report's layout; they are not results.
Run: python scripts/make_sample_lucky_report.py
"""

from datetime import datetime
from pathlib import Path

from ragaudit.models.adapter import RAGResponse
from ragaudit.models.run import Run, RunConfig
from ragaudit.models.scoring import (
    AblationResult,
    CorrectnessResult,
    GroundednessResult,
    RetrievalResult,
    ScoredCase,
    Verdict,
)
from ragaudit.models.testset import TestCase, TestSet
from ragaudit.report.lucky import write_lucky_report

NOW = datetime(2026, 11, 25)

# (question, expected, rag answer, verdict, closed-book answer or None, closed-book correct)
CASES = [
    ("What is the capital of France?", "Paris", "Paris", Verdict.LUCKY_PASS, "Paris", True),
    ("Who wrote Pride and Prejudice?", "Jane Austen", "Jane Austen", Verdict.LUCKY_PASS, "Jane Austen", True),
    ("How many tide gauges does the Harrow Point office run?", "14", "14", Verdict.TRUE_PASS, None, None),
    ("Which boat surveys the north channel?", "the Wren", "the Wren", Verdict.TRUE_PASS, None, None),
    ("When is the survey office's annual report due?", "March 31", "March 31", Verdict.LUCKY_PASS, "June 30", False),
    ("Who approves overtime for field crews?", "the duty hydrographer", "the office manager", Verdict.GENERATION_FAILURE, None, None),
    ("What depth triggers a re-survey?", "0.5 m change", "I don't know", Verdict.RETRIEVAL_FAILURE, None, None),
    ("Which datum do the charts use?", "chart datum CD-2", "CD-2", Verdict.TRUE_PASS, None, None),
]


def main() -> None:
    cases, test_cases = [], []
    for i, (question, expected, answer, verdict, closed, closed_ok) in enumerate(CASES, start=1):
        test_id = f"q{i}"
        test_cases.append(
            TestCase(
                test_id=test_id, question=question, expected_answer=expected, source_chunk_id=f"c{i}",
                source_doc_id="sample", generation_model="sample", created_at=NOW,
            )
        )
        cases.append(
            ScoredCase(
                test_id=test_id,
                response=RAGResponse(answer=answer, retrieved_chunk_ids=[], contexts=[]),
                correctness=CorrectnessResult(
                    is_correct=verdict in (Verdict.TRUE_PASS, Verdict.LUCKY_PASS), judge_model="sample"
                ),
                retrieval=RetrievalResult(
                    hit=verdict in (Verdict.TRUE_PASS, Verdict.GENERATION_FAILURE), expected_chunk_id=f"c{i}",
                    retrieved_chunk_ids=[],
                ),
                groundedness=GroundednessResult(is_grounded=True, judge_model="sample"),
                verdict=verdict,
                ablation=(
                    AblationResult(answer=closed, is_correct=closed_ok, confirms_lucky_pass=closed_ok)
                    if closed is not None
                    else None
                ),
            )
        )
    run = Run(
        config=RunConfig(
            run_id="sample", testset_id="sample", adapter_path="ragaudit_adapter.py", correctness_judge_model="sample",
            groundedness_judge_model="sample", started_at=NOW,
        ),
        scored_cases=cases,
    )
    test_set = TestSet(testset_id="sample", corpus_hash="sample", created_at=NOW, generation_model="sample", test_cases=test_cases)
    out = write_lucky_report(run, test_set, Path(__file__).resolve().parent.parent / "docs" / "sample-lucky-report.md")
    out.write_text("> Sample with invented questions, to show the layout. Not a result.\n\n" + out.read_text())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
