"""Renders the v2 generate and tune reports as single, self-contained HTML files.

Same approach as the v1 report (report/generator.py): a Jinja template
with the canonical style.css inlined, plus `v2.css` for the few v2-only
components (charts, callouts). style.css itself is untouched, so v1
reports render byte-for-byte as before.

Charts are inline SVG built here, so the reports need no JavaScript
library; a small script adds hover/focus tooltips. Every number a chart
shows is also in a table, so nothing depends on hovering.
"""

import html
from collections import Counter
from importlib import resources
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ragaudit.generate.pipeline import GenerateSummary
from ragaudit.models.question import Question, QuestionStatus
from ragaudit.models.tune import TuneResult
from ragaudit.report.generator import TEMPLATES_DIR, _load_style_css

REJECT_REASON_LABELS = {
    "too_lexical": "Too lexical: still reused the passage's wording after one rewrite",
    "relative_time": "Relative time: used words like \"recently\" or \"last quarter\"",
    "unsupported": "Unsupported: the verifier couldn't find the fact or answer in the passage",
    "unanswerable": "Unanswerable: the question can't be answered from the passage, or gives the answer away",
    "ambiguous": "Ambiguous: more than one answer, or unclear what is asked",
    "verifier_unparseable": "Verifier output unparseable: rejected to be safe",
}
FAILURE_CLASSES = [
    ("pass", "Pass", "v2-fill-pass"),
    ("generation_miss", "Generation miss", "v2-fill-gen"),
    ("retrieval_miss", "Retrieval miss", "v2-fill-ret"),
]

_W = 720


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _load_v2_css() -> str:
    try:
        return resources.files("ragaudit.report").joinpath("templates", "v2.css").read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise RuntimeError("Could not load ragaudit/report/templates/v2.css from package data.") from exc


def _env() -> Environment:
    return Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=select_autoescape(["html", "jinja"]))


def _is_mock(*models: str) -> bool:
    return any(model.startswith("mock") for model in models if model)


def _bar(x: float, y: float, w: float, h: float, cls: str, tip: str, *, end: str = "top", r: float = 4) -> str:
    """A bar with a rounded data end and a square baseline end, carrying a tooltip."""
    r = max(0.0, min(r, w / 2, h / 2))
    if end == "top":
        d = (
            f"M{x:.1f},{y + h:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
            f"H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} V{y + h:.1f} Z"
        )
    elif end == "right":
        d = (
            f"M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
            f"V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z"
        )
    else:
        d = f"M{x:.1f},{y:.1f} H{x + w:.1f} V{y + h:.1f} H{x:.1f} Z"
    return f'<path class="v2-mark {cls}" d="{d}" data-tip="{_e(tip)}" tabindex="0"></path>'


def _svg(height: int, body: list[str], label: str) -> str:
    return (
        f'<svg class="v2-chart" viewBox="0 0 {_W} {height}" role="img" aria-label="{_e(label)}" '
        f'preserveAspectRatio="xMidYMid meet">{"".join(body)}</svg>'
    )


def _nice_step(raw: float) -> int:
    """The smallest 1/2/5 x 10^n step at or above raw (minimum 1)."""
    magnitude = 1
    while True:
        for factor in (1, 2, 5):
            if factor * magnitude >= raw:
                return factor * magnitude
        magnitude *= 10


def _y_grid(left: int, right: int, top: int, bottom: int, ticks: list[float], vmax: float, fmt=str) -> list[str]:
    parts = []
    for tick in ticks:
        y = bottom - (bottom - top) * tick / vmax
        parts.append(f'<line class="v2-grid" x1="{left}" x2="{right}" y1="{y:.1f}" y2="{y:.1f}"></line>')
        parts.append(f'<text class="v2-tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{_e(fmt(tick))}</text>')
    return parts


# --- generate report charts --------------------------------------------------


def overlap_histogram(values: list[float], threshold: float, bins: int = 10) -> tuple[str, list[dict]]:
    """Histogram of final lexical overlaps; bins at or below the threshold are the accent, above are muted."""
    counts = [0] * bins
    for value in values:
        index = 0 if value <= 1 / bins else min(bins - 1, int(-(-value * bins // 1)) - 1)
        counts[index] += 1
    rows = []
    for i, count in enumerate(counts):
        low, high = i / bins, (i + 1) / bins
        rows.append({"range": f"{'[' if i == 0 else '('}{low:.1f}, {high:.1f}]", "count": count, "kept": high <= threshold + 1e-9})

    left, right, top, bottom = 44, _W - 16, 28, 230
    peak = max(max(counts), 1)
    step = _nice_step(peak / 4)
    vmax = step * -(-peak // step)
    ticks = [step * i for i in range(int(vmax // step) + 1)]
    band = (right - left) / bins
    body = _y_grid(left, right, top, bottom, ticks, vmax)
    for i, row in enumerate(rows):
        h = (bottom - top) * row["count"] / vmax
        if h > 0:
            cls = "v2-fill-accent" if row["kept"] else "v2-fill-muted"
            verdict = "kept" if row["kept"] else "above threshold: rejected as too_lexical"
            body.append(_bar(left + i * band + 1, bottom - h, band - 2, h, cls, f"{row['count']} question(s)\noverlap {row['range']}, {verdict}"))
    for i in range(bins + 1):
        if i % 2 == 0:
            body.append(f'<text class="v2-tick" x="{left + i * band:.1f}" y="{bottom + 18}" text-anchor="middle">{i / bins:.1f}</text>')
    tx = left + (right - left) * threshold
    body.append(f'<line class="v2-rule" x1="{tx:.1f}" x2="{tx:.1f}" y1="{top - 10}" y2="{bottom}"></line>')
    body.append(f'<text class="v2-label" x="{tx + 6:.1f}" y="{top - 2}">threshold {threshold:g}</text>')
    body.append(f'<text class="v2-axis-title" x="{(left + right) / 2:.1f}" y="{bottom + 36}" text-anchor="middle">share of the question\'s content words found in its source chunk</text>')
    return _svg(272, body, "Histogram of lexical overlap"), rows


def reason_bars(reasons: dict[str, int]) -> str:
    items = sorted(reasons.items(), key=lambda item: (-item[1], item[0]))
    if not items:
        return ""
    row_h, top, left, right = 34, 8, 170, _W - 60
    vmax = max(count for _, count in items)
    body = []
    for i, (reason, count) in enumerate(items):
        y = top + i * row_h
        w = (right - left) * count / vmax
        body.append(f'<text class="v2-label" x="{left - 10}" y="{y + 16}" text-anchor="end">{_e(reason)}</text>')
        body.append(_bar(left, y + 4, w, 18, "v2-fill-accent", f"{count} rejected\n{REJECT_REASON_LABELS.get(reason, reason)}", end="right"))
        body.append(f'<text class="v2-value" x="{left + w + 8:.1f}" y="{y + 17}">{count}</text>')
    return _svg(top + row_h * len(items), body, "Rejected questions by reason")


# --- tune report charts ------------------------------------------------------


def trial_chart(result: TuneResult) -> str:
    trials = result.trials
    left, right, top, bottom = 44, _W - 24, 24, 230
    n = len(trials)
    span = max(n - 1, 1)

    def x_of(index: int) -> float:
        return left + 12 + (right - left - 24) * index / span if n > 1 else (left + right) / 2

    def y_of(score: float) -> float:
        return bottom - (bottom - top) * score / 100

    body = _y_grid(left, right, top, bottom, [0, 25, 50, 75, 100], 100)
    best_so_far, running = [], -1.0
    for trial in trials:
        running = max(running, trial.dev.score)
        best_so_far.append(running)
    if n > 1:
        points = []
        for i, score in enumerate(best_so_far):
            if i:
                points.append(f"{x_of(i):.1f},{y_of(best_so_far[i - 1]):.1f}")
            points.append(f"{x_of(i):.1f},{y_of(score):.1f}")
        body.append(f'<polyline class="v2-line-ref" points="{" ".join(points)}"></polyline>')
    for trial in trials:
        cx, cy = x_of(trial.index), y_of(trial.dev.score)
        changed = ", ".join(f"{k}={v}" for k, v in trial.config.items() if result.baseline.config.get(k) != v) or "baseline config"
        tip = f"{trial.dev.score:.1f} dev\ntrial {trial.index} ({trial.source})\n{changed}"
        cls = "v2-dot v2-dot-best" if trial.index == result.best_index else "v2-dot"
        body.append(f'<circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="5"></circle>')
        body.append(f'<circle class="v2-hit" cx="{cx:.1f}" cy="{cy:.1f}" r="12" data-tip="{_e(tip)}" tabindex="0"></circle>')
    for index, name in ((result.baseline_index, "baseline"), (result.best_index, "best")):
        if name == "best" and index == result.baseline_index:
            continue
        trial = trials[index]
        anchor = "start" if x_of(index) < (left + right) / 2 else "end"
        dx = 10 if anchor == "start" else -10
        body.append(
            f'<text class="v2-label" x="{x_of(index) + dx:.1f}" y="{y_of(trial.dev.score) - 10:.1f}" '
            f'text-anchor="{anchor}">{name} {trial.dev.score:.1f}</text>'
        )
    step = max(1, n // 10)
    for i in range(0, n, step):
        body.append(f'<text class="v2-tick" x="{x_of(i):.1f}" y="{bottom + 18}" text-anchor="middle">{i}</text>')
    body.append(f'<text class="v2-axis-title" x="{(left + right) / 2:.1f}" y="{bottom + 36}" text-anchor="middle">trial</text>')
    return _svg(272, body, "Dev score per trial")


def dev_test_columns(result: TuneResult) -> str:
    groups = [("Baseline", result.baseline.dev.score, result.baseline_test.score), ("Best", result.best.dev.score, result.best_test.score)]
    left, right, top, bottom = 44, _W - 16, 28, 210
    body = _y_grid(left, right, top, bottom, [0, 25, 50, 75, 100], 100)
    band = (right - left) / len(groups)
    for g, (name, dev, test) in enumerate(groups):
        center = left + band * (g + 0.5)
        for offset, value, cls, split in ((-32, dev, "v2-fill-dev", "dev"), (8, test, "v2-fill-test", "locked test")):
            h = (bottom - top) * value / 100
            x = center + offset
            if h > 0:
                body.append(_bar(x, bottom - h, 24, h, cls, f"{value:.1f} {split}\n{name.lower()} config"))
            body.append(f'<text class="v2-value" x="{x + 12:.1f}" y="{bottom - h - 6:.1f}" text-anchor="middle">{value:.1f}</text>')
        body.append(f'<text class="v2-label" x="{center:.1f}" y="{bottom + 20}" text-anchor="middle">{name}</text>')
    return _svg(232, body, "Dev and test score, baseline and best")


def failure_stack(rows: list[tuple[str, dict[str, int]]]) -> str:
    row_h, top, left, right = 40, 6, 130, _W - 16
    body = []
    for i, (label, counts) in enumerate(rows):
        y = top + i * row_h
        total = sum(counts.get(key, 0) for key, _, _ in FAILURE_CLASSES) or 1
        body.append(f'<text class="v2-label" x="{left - 10}" y="{y + 17}" text-anchor="end">{_e(label)}</text>')
        x = left
        segments = [(key, name, cls, counts.get(key, 0)) for key, name, cls in FAILURE_CLASSES if counts.get(key, 0)]
        for j, (key, name, cls, count) in enumerate(segments):
            w = (right - left) * count / total
            last = j == len(segments) - 1
            draw_w = w - (0 if last else 2)
            share = count / total * 100
            body.append(_bar(x, y + 2, max(draw_w, 0.5), 24, cls, f"{count} ({share:.0f}%)\n{name}, {label}", end="right" if last else "none"))
            text = f"{count}"
            if draw_w > len(text) * 7 + 12:
                body.append(f'<text class="v2-inside" x="{x + draw_w / 2:.1f}" y="{y + 18}" text-anchor="middle">{text}</text>')
            x += w
    return _svg(top + row_h * len(rows), body, "Failure classes before and after tuning")


# --- page assembly -----------------------------------------------------------


def _write(template_name: str, output_path: str | Path, **context) -> Path:
    output_path = Path(output_path)
    page = _env().get_template(template_name).render(style_css=_load_style_css(), v2_css=_load_v2_css(), **context)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return output_path


def _question_row(q: Question) -> dict:
    return {
        "question": q.question,
        "answer": q.gold_answer,
        "fact": q.fact,
        "chunks": ", ".join(q.chunk_ids),
        "type": q.question_type.value,
        "overlap": f"{q.lexical_overlap:.2f}" if q.lexical_overlap is not None else "n/a",
        "reason": q.reject_reason,
        "draft": q.metadata.get("draft_question"),
    }


def generate_generate_report(
    summary: GenerateSummary,
    totals: dict,
    accepted_examples: list[Question],
    rejected_examples: list[Question],
    output_path: str | Path,
) -> Path:
    """Render a generate run. Accepted examples must be dev-split only; the caller guarantees it."""
    if any(q.split.value != "dev" for q in accepted_examples):
        raise ValueError("generate report examples must come from the dev split only")
    histogram, histogram_rows = overlap_histogram(summary.lexical_overlaps, summary.lexical_threshold)
    reasons = sorted(summary.reject_reasons.items(), key=lambda item: (-item[1], item[0]))
    checked = summary.generated + summary.rejected
    rejected_by_reason: dict[str, list[dict]] = {}
    for q in rejected_examples:
        rejected_by_reason.setdefault(q.reject_reason or "unknown", []).append(_question_row(q))

    return _write(
        "generate_report.html.jinja",
        output_path,
        summary=summary,
        totals=totals["by_split_status"],
        is_mock=_is_mock(summary.generator_model, summary.verifier_model),
        checked=checked,
        pass_pct=round(summary.generated / checked * 100, 1) if checked else 0.0,
        top_reason=reasons[0] if reasons else None,
        reasons=[{"reason": r, "count": c, "label": REJECT_REASON_LABELS.get(r, r)} for r, c in reasons],
        reason_chart=reason_bars(summary.reject_reasons),
        histogram=histogram,
        histogram_rows=histogram_rows,
        overlap_median=sorted(summary.lexical_overlaps)[len(summary.lexical_overlaps) // 2] if summary.lexical_overlaps else None,
        accepted=[_question_row(q) for q in accepted_examples],
        rejected_by_reason=rejected_by_reason,
    )


def generate_tune_report(result: TuneResult, output_path: str | Path) -> Path:
    if result.baseline_test is None or result.best_test is None:
        raise ValueError("tune result has no final test evaluation; it can't be reported")
    baseline, best = result.baseline, result.best
    knobs = [
        {"knob": knob, "baseline": baseline.config.get(knob), "best": best.config.get(knob), "changed": baseline.config.get(knob) != best.config.get(knob)}
        for knob in result.search_space
    ]
    sources = Counter(t.source for t in result.trials)
    trial_rows = [
        {
            "index": t.index,
            "source": t.source,
            "score": t.dev.score,
            "counts": t.dev.failure_counts,
            "changed": ", ".join(f"{k}={v}" for k, v in t.config.items() if baseline.config.get(k) != v) or "(baseline)",
            "rationale": t.rationale or "",
            "is_best": t.index == result.best_index,
        }
        for t in result.trials
    ]
    stack_rows = [
        ("Baseline, dev", baseline.dev.failure_counts),
        ("Best, dev", best.dev.failure_counts),
        ("Baseline, test", result.baseline_test.failure_counts),
        ("Best, test", result.best_test.failure_counts),
    ]
    return _write(
        "tune_report.html.jinja",
        output_path,
        result=result,
        baseline=baseline,
        best=best,
        is_mock=_is_mock(result.judge_model),
        improvement=round(result.best_test.score - result.baseline_test.score, 1),
        dev_improvement=round(best.dev.score - baseline.dev.score, 1),
        trial_chart=trial_chart(result),
        dev_test_chart=dev_test_columns(result),
        failure_chart=failure_stack(stack_rows),
        stack_rows=stack_rows,
        failure_classes=FAILURE_CLASSES,
        knobs=knobs,
        trial_rows=trial_rows,
        sources=dict(sources),
    )


def select_generate_examples(store, summary: GenerateSummary, limit: int = 10) -> tuple[list[Question], list[Question]]:
    """Accepted examples from the dev split only, and rejected examples (never scored, so any split)."""
    accepted = [q for q in store.dev_questions() if q.created_at >= summary.created_at] or store.dev_questions()
    rejected_all = [
        q
        for split in ("dev", "test")
        for q in store.load_questions(split=_split(split), statuses=[QuestionStatus.REJECTED])
        if q.created_at >= summary.created_at
    ]
    by_reason: dict[str, list[Question]] = {}
    for q in sorted(rejected_all, key=lambda q: q.created_at):
        by_reason.setdefault(q.reject_reason or "unknown", []).append(q)
    rejected = [q for questions in by_reason.values() for q in questions[:3]]
    accepted = sorted(accepted, key=lambda q: (q.question_type.value != "multi_hop", q.created_at))
    picked = accepted[: limit // 2] + [q for q in accepted if q.question_type.value == "single"][limit // 2 : limit]
    seen, unique = set(), []
    for q in picked:
        if q.question_id not in seen:
            seen.add(q.question_id)
            unique.append(q)
    return unique[:limit], rejected


def _split(name: str):
    from ragaudit.models.question import Split

    return Split(name)
