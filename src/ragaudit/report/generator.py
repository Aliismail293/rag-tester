"""Renders a Run into a single, self-contained static HTML report.

The report needs question text for its per-case table, which lives on
TestCase, not ScoredCase — so this takes both the Run and the TestSet it
was run against, cross-referenced by test_id.
"""

from importlib import resources
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ragaudit.models.run import Run
from ragaudit.models.scoring import ScoredCase, Verdict
from ragaudit.models.testset import TestSet

TEMPLATES_DIR = Path(__file__).parent / "templates"

_VERDICT_LABELS = {
    Verdict.TRUE_PASS: "True pass",
    Verdict.LUCKY_PASS: "Lucky pass",
    Verdict.GENERATION_FAILURE: "Generation failure",
    Verdict.RETRIEVAL_FAILURE: "Retrieval failure",
}


def _build_case_row(case: ScoredCase, question: str, expected_answer: str) -> dict:
    return {
        "test_id": case.test_id,
        "question": question,
        "expected_answer": expected_answer,
        "answer": case.response.answer,
        "contexts": case.response.contexts,
        "verdict": case.verdict.value,
        "verdict_label": _VERDICT_LABELS[case.verdict],
        "retrieval_rank": case.retrieval.rank,
        "judge_parse_failed": case.correctness.judge_parse_failed or case.groundedness.judge_parse_failed,
        "ablation": (
            {"answer": case.ablation.answer, "confirms_lucky_pass": case.ablation.confirms_lucky_pass}
            if case.ablation is not None
            else None
        ),
    }


def _build_case_rows(run: Run, test_set: TestSet) -> list[dict]:
    test_case_by_id = {tc.test_id: tc for tc in test_set.test_cases}
    rows = []
    for case in run.scored_cases:
        test_case = test_case_by_id.get(case.test_id)
        question = test_case.question if test_case else "(question not found in test set)"
        expected_answer = test_case.expected_answer if test_case else ""
        rows.append(_build_case_row(case, question, expected_answer))
    return rows


def _load_style_css() -> str:
    """Load the canonical stylesheet from package data.

    A silently unstyled report is worse than a crash — if the stylesheet
    can't be loaded (e.g. it was left out of a packaging change), raise
    rather than falling back to an empty string.
    """
    try:
        css = resources.files("ragaudit.report").joinpath("templates", "style.css").read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise RuntimeError(
            "Could not load the report stylesheet from package data "
            "(ragaudit/report/templates/style.css). The report cannot be rendered without it."
        ) from exc

    if not css.strip():
        raise RuntimeError("The report stylesheet (ragaudit/report/templates/style.css) is empty.")

    return css


def generate_report(run: Run, test_set: TestSet, output_path: str | Path) -> Path:
    """Render `run` into a self-contained HTML file at `output_path`.

    The canonical stylesheet (ragaudit/report/templates/style.css, shipped
    as package data) is inlined into the output, so the resulting file has
    no external dependencies. Returns output_path.
    """
    if run.summary is None:
        raise ValueError("Run.summary must be computed (see runner.summary.compute_run_summary) before reporting")

    output_path = Path(output_path)
    summary = run.summary

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "jinja"]),
    )
    template = env.get_template("report.html.jinja")

    verdict_matrix = {
        "true_pass": summary.verdict_counts.get(Verdict.TRUE_PASS, 0),
        "lucky_pass": summary.verdict_counts.get(Verdict.LUCKY_PASS, 0),
        "generation_failure": summary.verdict_counts.get(Verdict.GENERATION_FAILURE, 0),
        "retrieval_failure": summary.verdict_counts.get(Verdict.RETRIEVAL_FAILURE, 0),
    }
    lucky_pass_count = verdict_matrix["lucky_pass"]

    html = template.render(
        run=run,
        summary=summary,
        style_css=_load_style_css(),
        rows=_build_case_rows(run, test_set),
        verdict_matrix=verdict_matrix,
        lucky_pass_count=lucky_pass_count,
        lucky_pass_pct=round(summary.lucky_pass_rate * 100, 1),
        total_cases=summary.total_cases,
        judge_failure_count=summary.judge_failure_count,
        ablation_confirmed_count=summary.ablation_confirmed_count,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
