"""The v3 report: one self-contained HTML file with a welcome popup and 8 tabs.

Built from a neutral view model, so a v1 run (`check`, `run`, `report`) and
v2 results (`improve`) render the same page:

- the per-question cases (question, answer key, answer, retrieved passages,
  verdict, why it failed, what to try);
- the project's run history (see ragaudit.history), drawn as the score chart;
- the latest tune or improve run, for the Improvement loop tab.

Everything is rendered server-side, so the page reads without JavaScript;
the inlined script only switches tabs, moves the chart selection, filters
the table, copies commands and runs the popup and the pictures. The only
external request is the web font, with a system-font fallback.

Verdict badge thresholds (true-pass rate): STRONG at 80% or more, GOOD at
60% or more, NEEDS WORK below 60%.
"""

import re
import shlex
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup, escape

from ragaudit.history import HistoryRow, dumps_for_html, history_for_report
from ragaudit.models.scoring import Verdict
from ragaudit.report.plain_summary import dominant_failure
from ragaudit.scaffold import DEFAULT_ADAPTER_PATH

TEMPLATES_DIR = Path(__file__).parent / "templates"
STRONG_TRUE_PASS = 80.0
GOOD_TRUE_PASS = 60.0
SMALL_SAMPLE = 20
MAX_CHART_RUNS = 30
PASSAGE_LIMIT = 700
FONT_URL = "https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap"

VERDICT_LABELS = {
    "true_pass": "True pass",
    "lucky_pass": "Lucky pass",
    "generation_failure": "Generation failure",
    "retrieval_failure": "Retrieval failure",
}
VERDICT_ORDER = ("true_pass", "lucky_pass", "generation_failure", "retrieval_failure")
KIND_LABELS = {"check": "Check", "run": "Run", "evaluate": "Evaluate", "tune": "Tune", "improve": "Improve"}

TRIES = {
    "generation_failure": ["Answer only the exact question asked", "Answer only from the passage", "Pass fewer, shorter passages"],
    "retrieval_failure": [
        "Retrieve more passages (raise top-k)",
        "Try smaller chunks, or add keyword or hybrid search",
        "Check this page is in your index at all",
    ],
    "lucky_pass": [
        "Fix retrieval first: this page should come back",
        "Tell the model to answer only from the context",
        "Raise top-k or add re-ranking",
    ],
}


def badge_for(true_pass_pct: float) -> dict:
    if true_pass_pct >= STRONG_TRUE_PASS:
        return {"label": "STRONG", "cls": "strong"}
    if true_pass_pct >= GOOD_TRUE_PASS:
        return {"label": "GOOD", "cls": "good"}
    return {"label": "NEEDS WORK", "cls": "needs_work"}


def _pct(count: int, total: int) -> float:
    return round(count / total * 100, 1) if total else 0.0


def _pct_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}%"


def _date(dt: datetime) -> str:
    local = dt.astimezone() if dt.tzinfo else dt.replace(tzinfo=timezone.utc).astimezone()
    return f"{local:%b} {local.day}, {local.year}"


def _short_date(dt: datetime) -> str:
    local = dt.astimezone() if dt.tzinfo else dt.replace(tzinfo=timezone.utc).astimezone()
    return f"{local:%b} {local.day}"


def _q(value: str) -> str:
    return shlex.quote(str(value))


# --- passages ----------------------------------------------------------------


def _excerpt(text: str, answer: str, limit: int = PASSAGE_LIMIT) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    at = text.lower().find(answer.lower()) if answer else -1
    start = 0 if at < 0 else max(0, at - limit // 3)
    piece = text[start : start + limit]
    return ("…" if start else "") + piece + ("…" if start + limit < len(text) else "")


def highlight(text: str, answer: str) -> Markup:
    """Escape the passage and bold each occurrence of the answer key in it."""
    safe = str(escape(_excerpt(text, answer)))
    needle = str(escape(answer.strip())) if answer else ""
    if not needle or len(needle) > 80:
        return Markup(safe)
    return Markup(re.sub(re.escape(needle), lambda m: f"<b>{m.group(0)}</b>", safe, flags=re.IGNORECASE))


# --- cases ---------------------------------------------------------------


def _why(verdict: str, rank: int | None, rationale: str | None, ablation: dict | None, error: str | None) -> str:
    if error:
        return f"Your adapter raised an error for this question, so it counts as a failure: {error}"
    if verdict == "generation_failure":
        text = (
            f"The right passage came back (rank #{rank}), but the answer did not match the key. "
            "The miss happened after the search: in the prompt or the model."
            if rank
            else "The right passage came back, but the answer did not match the key."
        )
    elif verdict == "retrieval_failure":
        text = "The search did not return the passage this question was written from, and the answer was wrong."
    elif verdict == "lucky_pass":
        text = (
            "The answer matched the key, but the passage it lives on was not retrieved, "
            "so the model answered from memory or from a look-alike page."
        )
        if ablation is not None:
            text += (
                " Re-asked with no passages at all, it was still right: confirmed from memory."
                if ablation["confirms"]
                else " Re-asked with no passages at all, it was wrong, so a retrieved page probably helped indirectly."
            )
    else:
        text = "Right page, right answer."
    if rationale and verdict != "true_pass":
        text += f" Judge: {rationale}"
    return text


def make_case(
    *,
    case_id: str,
    question: str,
    key: str,
    answer: str,
    verdict: str,
    contexts: list[str],
    rank: int | None,
    hit: bool,
    gold_text: str | None = None,
    rationale: str | None = None,
    ablation: dict | None = None,
    error: str | None = None,
    judge_failed: bool = False,
) -> dict:
    """One question for the report. rank is 1-based (the first retrieved passage that matched), or None."""
    passages = [
        {"n": i + 1, "html": highlight(text, key if (hit and rank == i + 1) else ""), "source": hit and rank == i + 1}
        for i, text in enumerate(contexts)
    ]
    return {
        "id": case_id,
        "question": question,
        "key": key,
        "answer": answer,
        "verdict": verdict,
        "label": VERDICT_LABELS[verdict],
        "rank": rank,
        "rank_text": f"#{rank}" if rank else "–",
        "hit": hit,
        "passages": passages,
        "gold_html": highlight(gold_text, key) if (gold_text and not hit) else None,
        "why": _why(verdict, rank, rationale, ablation, error),
        "tries": TRIES.get(verdict, []),
        "ablation": ablation,
        "judge_failed": judge_failed,
    }


def cases_from_v1(run, test_set, chunk_texts: dict[str, str] | None = None) -> list[dict]:
    by_id = {tc.test_id: tc for tc in test_set.test_cases}
    cases = []
    for scored in run.scored_cases:
        tc = by_id.get(scored.test_id)
        retrieval = scored.retrieval
        rank = None if retrieval.rank is None else retrieval.rank + 1
        error = None
        rationale = scored.correctness.rationale
        if rationale and rationale.startswith("adapter raised"):
            error, rationale = rationale.split(":", 1)[-1].strip(), None
        cases.append(
            make_case(
                case_id=scored.test_id,
                question=tc.question if tc else "(question not found in the test set)",
                key=tc.expected_answer if tc else "",
                answer=scored.response.answer,
                verdict=scored.verdict.value,
                contexts=list(scored.response.contexts),
                rank=rank if retrieval.hit else None,
                hit=retrieval.hit,
                gold_text=(chunk_texts or {}).get(tc.source_chunk_id) if tc else None,
                rationale=rationale,
                ablation=(
                    {"answer": scored.ablation.answer, "confirms": scored.ablation.confirms_lucky_pass}
                    if scored.ablation is not None
                    else None
                ),
                error=error,
                judge_failed=scored.correctness.judge_parse_failed or scored.groundedness.judge_parse_failed,
            )
        )
    return cases


# --- the overview -----------------------------------------------------------


class _Counts:
    def __init__(self, cases: list[dict]):
        self.total = len(cases)
        self.verdicts = {v: sum(c["verdict"] == v for c in cases) for v in VERDICT_ORDER}
        self.hits = sum(c["hit"] for c in cases)
        self.correct = self.verdicts["true_pass"] + self.verdicts["lucky_pass"]

    @property
    def true_pass_pct(self) -> float:
        return _pct(self.verdicts["true_pass"], self.total)


class _SummaryShim:
    """Just enough of a RunSummary for plain_summary.dominant_failure."""

    def __init__(self, counts: _Counts):
        self.verdict_counts = {Verdict(v): n for v, n in counts.verdicts.items()}


def headline(counts: _Counts) -> dict:
    n, v = counts.total, counts.verdicts
    if n == 0:
        return {
            "title": "No questions were scored.",
            "body": "The run had no questions. Check that the docs folder has .txt or .md files.",
            "next": "Add documents, then run the check again.",
        }
    dominant = dominant_failure(_SummaryShim(counts))
    hit_pct = _pct(counts.hits, n)
    if dominant is None:
        return {
            "title": "Every answer was right, and came from the right page.",
            "body": (
                f"All {n} questions pulled the passage their answer lives on and were judged correct. "
                "This is the result to protect: re-run after every change."
            ),
            "next": "Keep this check in CI and re-run it after every change.",
        }
    if dominant == Verdict.GENERATION_FAILURE:
        g = v["generation_failure"]
        found = "Every question" if counts.hits == n else f"{counts.hits} of {n} questions"
        return {
            "title": "Retrieval works. Answer generation is the weak link." if hit_pct >= 80 else "Answer generation is the weak link.",
            "body": (
                f"{found} pulled the page its answer lives on, but {g} answer{'s were' if g != 1 else ' was'} still wrong. "
                "That points at the prompt or the model, not at chunking or embeddings."
            ),
            "next": "Tighten the answer prompt, then re-run.",
        }
    if dominant == Verdict.RETRIEVAL_FAILURE:
        r = v["retrieval_failure"]
        return {
            "title": "Search is missing the right page.",
            "body": (
                f"For {r} of {n} questions the passage holding the answer never came back, and the answer was wrong. "
                "Fix retrieval first: chunking, embeddings or top-k."
            ),
            "next": "Improve retrieval (top-k, chunk size, hybrid search), then re-run.",
        }
    lucky = v["lucky_pass"]
    return {
        "title": "Some right answers didn't come from your documents.",
        "body": (
            f"{lucky} of {n} answers were right even though search missed the page they live on: the model answered "
            "from memory. Those wins won't hold for facts it doesn't already know."
        ),
        "next": "Fix retrieval, and tell the model to answer only from the context.",
    }


def reading_bullets(counts: _Counts, judge_failures: int = 0) -> list[str]:
    n = counts.total
    if n == 0:
        return []
    hit_pct, correct_pct = _pct(counts.hits, n), _pct(counts.correct, n)
    bullets = []
    if hit_pct > correct_pct:
        bullets.append("Retrieval hit higher than correctness means the miss happened after the search.")
    elif correct_pct > hit_pct:
        bullets.append("Correctness higher than retrieval hit means some right answers did not come from the right page.")
    else:
        bullets.append("Retrieval hit and correctness match: when the search found the page, the answer followed.")
    lucky = counts.verdicts["lucky_pass"]
    if lucky == 0:
        bullets.append("Lucky passes at 0 means your correct answers really came from your documents.")
    else:
        bullets.append(f"{lucky} lucky pass{'es' if lucky != 1 else ''}: right answers your search didn't earn. Don't count them as wins.")
    if n < SMALL_SAMPLE:
        bullets.append(f"{n} question{'s is' if n != 1 else ' is'} a small sample. Run with the default 20 (or more) before trusting a trend.")
    elif judge_failures:
        bullets.append(f"The judge's reply could not be read for {judge_failures} question(s); they count as failures.")
    return bullets[:3]


def stat_cards(counts: _Counts, groundedness_pct: float | None, grounded_note: str | None = None) -> list[dict]:
    n = counts.total
    lucky = counts.verdicts["lucky_pass"]
    return [
        {"label": "Retrieval hit", "value": _pct_text(_pct(counts.hits, n)), "note": f"{counts.hits} of {n} found the source page"},
        {"label": "Answer correctness", "value": _pct_text(_pct(counts.correct, n)), "note": f"{counts.correct} of {n} judged correct"},
        {
            "label": "Grounded",
            "value": _pct_text(groundedness_pct),
            "note": grounded_note or ("answers used the passages" if groundedness_pct is not None else "not measured in this run"),
        },
        {
            "label": "Lucky passes",
            "value": str(lucky),
            "note": "none answered from memory" if lucky == 0 else "right answer, wrong page",
        },
    ]


def ring(counts: _Counts) -> dict:
    circumference = 2 * 3.14159265 * 62
    share = counts.verdicts["true_pass"] / counts.total if counts.total else 0
    return {
        "dash": f"{share * circumference:.1f} {circumference:.1f}",
        "text": f"{counts.verdicts['true_pass']}/{counts.total}",
        "aria": f"{counts.verdicts['true_pass']} of {counts.total} true passes",
        "empty": share == 0,
    }


# --- score history -------------------------------------------------------------


def _row_note(row: HistoryRow, first_of_kind: bool) -> str:
    if row.note:
        return row.note
    label = KIND_LABELS.get(row.kind, row.kind.title())
    prefix = f"First {label.lower()}" if first_of_kind else label
    return f"{prefix} · {row.n_questions} question{'s' if row.n_questions != 1 else ''}"


def chart_view(rows: list[HistoryRow], current_run_id: str | None) -> dict:
    all_runs = history_for_report(rows)
    shown_rows = rows[-MAX_CHART_RUNS:]
    shown = all_runs[-MAX_CHART_RUNS:]
    older = len(rows) - len(shown_rows)
    n = len(shown)
    if n == 0:
        return {"runs": [], "older": 0}

    def x(k: int) -> float:
        return 335.0 if n == 1 else round(70 + k * (530 / (n - 1)), 1)

    def y(v: float) -> float:
        return round(250 - v * 2.1, 1)

    seen_kinds: set[str] = set()
    for row in rows[: len(rows) - n]:
        seen_kinds.add(row.kind)
    runs = []
    selected = n - 1
    for k, (row, data) in enumerate(zip(shown_rows, shown)):
        first = row.kind not in seen_kinds
        seen_kinds.add(row.kind)
        if row.run_id == current_run_id:
            selected = k
        tp, hit = row.true_pass_pct, row.retrieval_hit_pct
        if row.delta_true_pass is not None:
            delta_text = f"{row.delta_true_pass:+.0f} pts"
            delta_class = "delta-good" if row.delta_true_pass >= 0 else "delta-bad"
        elif row.prev_run_id is not None:
            delta_text, delta_class = "no shared questions", ""
        else:
            delta_text, delta_class = "first run", ""
        warn = ""
        if row.prev_run_id is not None and row.shared_questions and row.small_sample:
            warn = (
                f"Compared on only {row.shared_questions} shared question{'s' if row.shared_questions != 1 else ''}: "
                "a small sample, so treat the delta as a hint."
            )
        elif row.prev_run_id is not None and row.shared_questions:
            warn = ""
        tags = []
        if row.kind != "check":
            tags.append(KIND_LABELS.get(row.kind, row.kind))
        if row.corpus_changed:
            tags.append("docs changed")
        if row.run_id == current_run_id:
            tags.append("this report")
        shared_text = f" on {row.shared_questions} shared" if row.delta_true_pass is not None else ""
        runs.append({
            "number": data["number"],
            "name": f"Run {data['number']}",
            "date": _short_date(row.created_at),
            "kind": row.kind,
            "x": x(k),
            "y": y(tp),
            "y_hit": y(hit),
            "label_y": round(y(tp) - 18, 1),
            "tp_text": f"{tp:.0f}%",
            "hit_text": f"{hit:.0f}%",
            "delta_text": delta_text + shared_text,
            "delta_class": delta_class,
            "note": _row_note(row, first),
            "meta": f"Run {data['number']} · {_short_date(row.created_at)}" + "".join(f" · {t}" for t in tags),
            "warn": warn,
            "corpus_changed": row.corpus_changed,
        })
    sel = runs[selected]
    return {
        "runs": runs,
        "selected": selected,
        "sel": sel,
        "older": older,
        "tp_points": " ".join(f"{r['x']},{r['y']}" for r in runs),
        "hit_points": " ".join(f"{r['x']},{r['y_hit']}" for r in runs),
        "any_changed": any(r["corpus_changed"] for r in runs),
        "aria": f"True pass rate across {n} run{'s' if n != 1 else ''}: " + ", ".join(r["tp_text"] for r in runs),
        "single": n == 1,
    }


# --- improvement loop ------------------------------------------------------------


def loop_view(rows: list[HistoryRow]) -> dict | None:
    tuned = [row for row in rows if row.kind in ("tune", "improve") and row.extra.get("trials")]
    if not tuned:
        return None
    row = tuned[-1]
    e = row.extra
    gap = e.get("overfit_gap")
    trials = e.get("trials", [])
    scores = [t["practice"] for t in trials]
    spark = None
    if scores:
        w, h = 280.0, 64.0
        step = w / max(1, len(scores) - 1)
        pts = [(round(i * step if len(scores) > 1 else w / 2, 1), round(h - 6 - s / 100 * (h - 12), 1)) for i, s in enumerate(scores)]
        best = e.get("best_index", 0)
        spark = {
            "points": " ".join(f"{px},{py}" for px, py in pts),
            "dots": [{"x": px, "y": py, "best": i == best} for i, (px, py) in enumerate(pts)],
            "aria": "Practice score per trial: " + ", ".join(f"{s:.0f}" for s in scores),
        }
    return {
        "kind": row.kind,
        "date": _date(row.created_at),
        "baseline_exam": e.get("baseline_exam"),
        "best_exam": e.get("best_exam"),
        "best_practice": e.get("best_practice"),
        "gap": gap,
        "warn": bool(e.get("overfit_warning")),
        "threshold": e.get("overfit_threshold"),
        "changes": e.get("changes", []),
        "n_trials": len(trials),
        "path": e.get("best_config_path") or "best_config.yaml",
        "spark": spark,
        "n_practice": e.get("n_practice"),
        "n_exam": e.get("n_exam"),
        "stopped": (e.get("stopped_reason") or "").replace("_", " "),
    }


# --- commands --------------------------------------------------------------------


def commands(docs: str | None, adapter: str | None) -> dict:
    docs_arg = _q(docs or "./docs")
    adapter = adapter or str(DEFAULT_ADAPTER_PATH)
    adapter_arg = _q(adapter)
    doctor = "ragaudit doctor" if adapter == str(DEFAULT_ADAPTER_PATH) else f"ragaudit doctor --adapter {adapter_arg}"
    return {
        "easy": [
            {"n": 1, "code": "pip install ragaudit-cli", "what": "Install ragaudit (once)"},
            {"n": 2, "code": "ragaudit ui", "what": "Open the setup page in your browser, then click Run"},
        ],
        "more": [
            {"code": doctor, "what": "Checks Python, your AI settings and your adapter, and says how to fix anything missing"},
            {"code": "ragaudit init --framework demo", "what": "Writes a ready-to-run adapter; use plain, langchain or llamaindex for your own app"},
            {"code": f"ragaudit check --docs {docs_arg} --adapter {adapter_arg}", "what": "One quick test (about 20 questions), then opens this report"},
            {"code": f"ragaudit improve --docs {docs_arg} --adapter {adapter_arg}", "what": "The self-improving loop: tunes settings on practice, scores on the locked exam"},
            {"code": "ragaudit report --open", "what": "Re-opens the latest report"},
        ],
        "again": f'ragaudit check --docs {docs_arg} --adapter {adapter_arg} --note "what you changed"',
        "check": f"ragaudit check --docs {docs_arg} --adapter {adapter_arg}",
        "improve": f"ragaudit improve --docs {docs_arg} --adapter {adapter_arg}",
    }


def all_commands(view: dict) -> list[str]:
    """Every command the report offers to copy (for tests)."""
    c = view["commands"]
    return [x["code"] for x in c["easy"]] + [x["code"] for x in c["more"]] + [c["again"], c["check"], c["improve"]]


# --- assembly ------------------------------------------------------------------


def build_view(
    *,
    cases: list[dict],
    run_id: str,
    created_at: datetime,
    adapter_path: str,
    kind: str = "check",
    history_rows: list[HistoryRow] | None = None,
    docs: str | None = None,
    groundedness_pct: float | None = None,
    grounded_note: str | None = None,
    llm_calls: int | None = None,
    judge_failures: int = 0,
    match_by: str | None = None,
    match_threshold: float | None = None,
    ablation_note: str | None = None,
    scope_note: str | None = None,
    generated_at: datetime | None = None,
) -> dict:
    rows = history_rows or []
    counts = _Counts(cases)
    head = headline(counts)
    failures = [c for c in cases if c["verdict"] != "true_pass"]
    match_text = None
    if match_by == "text":
        match_text = f"matched by text · threshold {match_threshold:g}" if match_threshold is not None else "matched by text"
    elif match_by == "id":
        match_text = "matched by chunk id"
    if ablation_note is None:
        lucky = counts.verdicts["lucky_pass"]
        confirmed = sum(1 for c in cases if c["ablation"] and c["ablation"]["confirms"])
        ablated = sum(1 for c in cases if c["ablation"])
        ablation_note = (
            "Ablation: skipped, no lucky passes" if lucky == 0
            else f"Ablation: {confirmed} of {ablated} lucky pass(es) still right with no passages" if ablated
            else "Ablation: not run"
        )
    view = {
        "run_id": run_id,
        "run_short": run_id[:8],
        "kind": kind,
        "date": _date(created_at),
        "adapter": Path(adapter_path).name if adapter_path else "",
        "badge": badge_for(counts.true_pass_pct),
        "scope_note": scope_note,
        "total": counts.total,
        "counts": counts.verdicts,
        "ring": ring(counts),
        "headline": head,
        "stats": stat_cards(counts, groundedness_pct, grounded_note),
        "bullets": reading_bullets(counts, judge_failures),
        "cases": cases,
        "failures": failures,
        "chart": chart_view(rows, run_id),
        "loop": loop_view(rows),
        "commands": commands(docs, adapter_path),
        "facts": [
            ablation_note,
            f"Judge parse failures: {judge_failures}",
            f"Model calls: {llm_calls}" if llm_calls is not None else None,
            match_text,
        ],
        "match_text": match_text,
        "font_url": FONT_URL,
        "generated": _date(generated_at or datetime.now(timezone.utc)),
    }
    view["facts"] = [f for f in view["facts"] if f]
    view["filters"] = [{"key": "all", "label": f"All ({counts.total})"}] + [
        {"key": v, "label": f"{VERDICT_LABELS[v]} ({counts.verdicts[v]})"} for v in VERDICT_ORDER if counts.verdicts[v]
    ]
    view["data_json"] = dumps_for_html({"history": view["chart"]["runs"]})
    return view


def _asset(name: str) -> str:
    text = resources.files("ragaudit.report").joinpath("templates", "v3", name).read_text(encoding="utf-8")
    if not text.strip():
        raise RuntimeError(f"The report asset ragaudit/report/templates/v3/{name} is empty.")
    return text


def render_html(view: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=select_autoescape(["html", "jinja"]))
    template = env.get_template("v3/report.html.jinja")
    return template.render(v=view, css=Markup(_asset("report.css")), js=Markup(_asset("report.js")))


def write_report(view: dict, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(view), encoding="utf-8")
    return out


def render_v1_run(
    run,
    test_set,
    out: str | Path,
    *,
    history_rows: list[HistoryRow] | None = None,
    docs: Path | str | None = None,
    llm_calls: int | None = None,
    chunk_texts: dict[str, str] | None = None,
    project_store: Path | None = None,
    kind: str = "check",
) -> Path:
    """Render a v1 Run (from `check`, `run` or `report`) as the v3 report."""
    if run.summary is None:
        raise ValueError("Run.summary must be computed before reporting")
    cases = cases_from_v1(run, test_set, chunk_texts)
    match = next((c.retrieval for c in run.scored_cases), None)
    current = next((row for row in (history_rows or []) if row.run_id == run.config.run_id), None)
    extra = current.extra if current else {}
    if llm_calls is None:
        llm_calls = extra.get("llm_calls")
    view = build_view(
        cases=cases,
        run_id=run.config.run_id,
        created_at=run.config.started_at,
        adapter_path=run.config.adapter_path,
        kind=current.kind if current else kind,
        history_rows=history_rows,
        docs=str(docs) if docs is not None else extra.get("docs"),
        groundedness_pct=round(run.summary.mean_groundedness * 100) if run.summary.mean_groundedness is not None else None,
        llm_calls=llm_calls,
        judge_failures=run.summary.judge_failure_count,
        match_by=match.match_by if match else None,
        match_threshold=extra.get("match_threshold") if match and match.match_by == "text" else None,
    )
    return write_report(view, out)
