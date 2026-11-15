"""The v3 report: snapshots of fixture runs, structure, offline safety and copyable commands.

Snapshots live in tests/snapshots/. After an intended change to the report,
regenerate them with:  RAGAUDIT_UPDATE_SNAPSHOTS=1 pytest tests/test_report_v3.py
"""

import os
import re
import shlex
import time
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path

import pytest
import typer

from ragaudit.cli import app
from ragaudit.history import HistoryRow
from ragaudit.report import v3

SNAPSHOTS = Path(__file__).parent / "snapshots"
T0 = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
ALLOWED_HOSTS = ("fonts.googleapis.com", "fonts.gstatic.com")


@pytest.fixture(autouse=True)
def _utc(monkeypatch):
    monkeypatch.setenv("TZ", "UTC")
    if hasattr(time, "tzset"):
        time.tzset()
    yield
    if hasattr(time, "tzset"):
        monkeypatch.undo()
        time.tzset()


def case(i, verdict, contexts=None, rank=1):
    hit = verdict in ("true_pass", "generation_failure")
    return v3.make_case(
        case_id=f"q{i}",
        question=f"How many questions does check write by default ({i})?",
        key="20",
        answer="20" if verdict in ("true_pass", "lucky_pass") else "5",
        verdict=verdict,
        contexts=contexts if contexts is not None else ["generate (default 20 questions total) e.g. --questions 5", "Run without --questions"],
        rank=rank if hit else None,
        hit=hit,
        gold_text="The default is 20 questions." if not hit else None,
        rationale="expected 20" if verdict != "true_pass" else None,
        ablation={"answer": "20", "confirms": True} if verdict == "lucky_pass" else None,
    )


def row(n, tp, hit, *, kind="check", note=None, changed=False, delta=None, shared=0, prev=None, extra=None):
    return HistoryRow(
        run_id=f"run-{n}", kind=kind, created_at=T0 + timedelta(days=n), n_questions=4, true_pass_pct=tp,
        retrieval_hit_pct=hit, note=note, corpus_changed=changed, delta_true_pass=delta, shared_questions=shared,
        prev_run_id=prev, extra=extra or {},
    )


TUNE_EXTRA = {
    "baseline_exam": 50.0, "best_exam": 68.0, "baseline_practice": 55.0, "best_practice": 72.0, "overfit_gap": 4.0,
    "overfit_threshold": 10.0, "overfit_warning": False, "best_index": 2,
    "trials": [{"index": 0, "practice": 55.0}, {"index": 1, "practice": 61.0}, {"index": 2, "practice": 72.0}],
    "changes": [{"knob": "top_k", "before": 3, "after": 5}], "best_config_path": "best_config.yaml",
    "n_practice": 10, "n_exam": 4, "stopped_reason": "max_trials",
}


def fixture_view(name):
    passing = [case(1, "true_pass"), case(2, "true_pass", rank=2), case(3, "true_pass"), case(4, "true_pass")]
    mixed = [case(1, "true_pass"), case(2, "generation_failure"), case(3, "lucky_pass"), case(4, "retrieval_failure", contexts=[])]
    history, cases, run_id = [], passing, "run-1"
    if name == "single_pass":
        history = [row(1, 100, 100)]
    elif name == "history_failures":
        history = [row(1, 75, 100), row(2, 50, 75, note="smaller chunks", changed=True, delta=-25.0, shared=3, prev="run-1")]
        cases, run_id = mixed, "run-2"
    elif name == "tune":
        history = [row(1, 25, 50), row(2, 68, 75, kind="improve", extra=TUNE_EXTRA)]
        cases, run_id = mixed, "run-2"
    elif name == "no_history":
        cases = mixed
    return v3.build_view(
        cases=cases, run_id=run_id, created_at=T0, adapter_path="ragaudit_adapter.py", history_rows=history,
        docs="./docs", groundedness_pct=100, llm_calls=15, match_by="text", match_threshold=0.5, generated_at=T0,
    )


FIXTURES = ("single_pass", "history_failures", "tune", "no_history")


@pytest.mark.parametrize("name", FIXTURES)
def test_snapshot(name):
    html = v3.render_html(fixture_view(name))
    path = SNAPSHOTS / f"report_v3_{name}.html"
    if os.environ.get("RAGAUDIT_UPDATE_SNAPSHOTS") or not path.exists():
        SNAPSHOTS.mkdir(exist_ok=True)
        path.write_text(html, encoding="utf-8")
    assert html == path.read_text(encoding="utf-8"), f"report changed; rerun with RAGAUDIT_UPDATE_SNAPSHOTS=1 if intended ({path})"


class _Collector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls, self.tabs, self.panels, self.tags = [], [], [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.tags.append((tag, a))
        for key in ("src", "href", "action", "poster", "data"):
            if a.get(key):
                self.urls.append(a[key])
        if a.get("role") == "tab":
            self.tabs.append(a)
        if a.get("role") == "tabpanel":
            self.panels.append(a)


def parse(html):
    p = _Collector()
    p.feed(html)
    return p


@pytest.mark.parametrize("name", FIXTURES)
def test_no_external_requests_except_the_font(name):
    html = v3.render_html(fixture_view(name))
    p = parse(html)
    for url in p.urls + re.findall(r"url\(([^)]+)\)", html) + re.findall(r"@import\s+['\"]?([^'\";]+)", html):
        if url.startswith(("#", "data:")):
            continue
        assert url.startswith("https://") and url.split("/")[2] in ALLOWED_HOSTS, url
    assert "<script src" not in html and "fetch(" not in html and "XMLHttpRequest" not in html


def test_structure_tabs_popup_and_accessibility():
    html = v3.render_html(fixture_view("history_failures"))
    p = parse(html)
    assert len(p.tabs) == 8 and len(p.panels) == 8
    assert [t["aria-selected"] for t in p.tabs] == ["true"] + ["false"] * 7
    assert all(t["aria-controls"] == panel["id"] for t, panel in zip(p.tabs, p.panels))
    assert 'role="tablist"' in html and 'role="dialog"' in html and 'aria-modal="true"' in html
    assert 'aria-label="Close"' in html and "Show my results" in html and "How to run it" in html
    assert "Don't show again" not in html
    assert "localStorage" in html and "requestFullscreen" in html and "execCommand('copy')" in html.replace('"', "'")
    assert "prefers-reduced-motion" in html
    # The Overview panel is the only one not hidden, so it shows without JavaScript.
    hidden = ["hidden" in a for tag, a in p.tags if a.get("role") == "tabpanel"]
    assert hidden == [False] + [True] * 7
    assert 'class="no-js"' in html and ".no-js .panel" in html


def test_history_failures_and_tune_content():
    html = v3.render_html(fixture_view("history_failures"))
    assert "smaller chunks" in html and "docs changed" in html and "-25 pts on 3 shared" in html
    assert "a small sample" in html
    assert "THE PAGE IT SHOULD HAVE FOUND" in html and "<b>20</b>" in html
    assert html.count("<article class=\"fail-detail\"") == 3 and "NEEDS WORK" in html
    tune = v3.render_html(fixture_view("tune"))
    assert "Self-improvement loop" in tune and "Practice − exam = 4 pts: no memorising" in tune
    assert "top_k" in tune and "3 → <b>5</b>" in tune and "3 trials" in tune
    single = v3.render_html(fixture_view("single_pass"))
    assert "Run the same check again after a change to see a trend" in single
    assert "Nothing failed." in single and "No improvement run for this project yet" in single
    assert "STRONG" in single


def test_badge_thresholds():
    assert v3.badge_for(80)["label"] == "STRONG" and v3.badge_for(79.9)["label"] == "GOOD"
    assert v3.badge_for(60)["label"] == "GOOD" and v3.badge_for(59)["label"] == "NEEDS WORK"


def test_chart_caps_at_30_runs():
    rows = [row(i, 50 + i % 10, 80) for i in range(35)]
    chart = v3.chart_view(rows, "run-34")
    assert len(chart["runs"]) == 30 and chart["older"] == 5 and chart["runs"][-1]["name"] == "Run 35"


PENDING_COMMANDS = {"improve", "ui"}
"""Added in later v3 phases; until then their copy commands are not parsed. Removed once both exist."""


def _registered(group) -> set[str]:
    return set(group.commands)


def validate_command(command: str) -> None:
    """Parse a copyable command with the real Typer/Click app, exactly as the shell would split it."""
    argv = shlex.split(command)
    if argv[:2] == ["pip", "install"]:
        assert argv[2:] == ["ragaudit-cli"]
        return
    assert argv[0] == "ragaudit", command
    group = typer.main.get_command(app)
    if argv[1] in PENDING_COMMANDS and argv[1] not in _registered(group):
        return
    assert argv[1] in _registered(group), f"{argv[1]} is not a ragaudit command"
    sub = group.commands[argv[1]]
    ctx = sub.make_context(argv[1], argv[2:], resilient_parsing=False)
    assert ctx.params is not None


@pytest.mark.parametrize("docs, adapter", [("./docs", "ragaudit_adapter.py"), ("my docs", "apps/my adapter.py")])
def test_every_copy_command_parses_with_the_real_cli(docs, adapter):
    view = v3.build_view(cases=[case(1, "true_pass")], run_id="r", created_at=T0, adapter_path=adapter, docs=docs)
    commands = v3.all_commands(view)
    assert len(commands) >= 8
    for command in commands:
        validate_command(command)
    html = v3.render_html(view)
    for command in re.findall(r'data-copy="([^"]+)"', html):
        validate_command(command.replace("&#34;", '"').replace("&quot;", '"').replace("&#39;", "'").replace("&amp;", "&"))
