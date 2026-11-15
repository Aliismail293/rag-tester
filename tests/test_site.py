"""The website (docs/): synced with the report's design system, links and anchors intact, honest numbers, valid commands."""

import re
import subprocess
import sys
import tomllib
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

import pytest

from test_report_v3 import validate_command

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
INDEX = (DOCS / "index.html").read_text(encoding="utf-8")
OLD_ANCHORS = ("problem", "how", "reports", "install", "v2", "quickstart")
NAV_ANCHORS = ("top", "what", "how", "reports", "get-started", "main")


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


PAGE = Page()
PAGE.feed(INDEX)


def attrs(tag=None, **match):
    return [a for t, a in PAGE.tags if (tag is None or t == tag) and all(a.get(k) == v for k, v in match.items())]


def test_site_is_in_sync_with_the_report_design_system():
    result = subprocess.run([sys.executable, str(REPO / "scripts" / "sync_site.py"), "--check"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_report_inlines_the_same_design_system():
    from ragaudit.report import v3

    design = (REPO / "src/ragaudit/report/templates/v3/design.css").read_text(encoding="utf-8")
    view = v3.build_view(cases=[], run_id="r", created_at=datetime(2026, 11, 15), adapter_path="ragaudit_adapter.py")
    html = v3.render_html(view)
    assert design.strip() in html
    assert (REPO / "src/ragaudit/report/templates/v3/design.js").read_text(encoding="utf-8").strip() in html


def test_old_and_nav_anchors_all_land_somewhere():
    ids = {a.get("id") for _, a in PAGE.tags if a.get("id")}
    for anchor in OLD_ANCHORS + NAV_ANCHORS:
        assert anchor in ids, f"#{anchor} is missing"
    for _, a in PAGE.tags:
        href = a.get("href") or ""
        if href.startswith("#"):
            assert href[1:] in ids, href


def test_local_links_iframes_and_assets_exist():
    for _, a in PAGE.tags:
        for key in ("href", "src"):
            url = a.get(key)
            if not url or url.startswith(("#", "https://", "mailto:")):
                continue
            assert (DOCS / url.split("#")[0].split("?")[0]).is_file(), url


def test_no_external_scripts_and_only_known_hosts():
    for tag, a in PAGE.tags:
        if tag == "script":
            assert a.get("src", "").startswith("assets/"), a
        for key in ("href", "src"):
            url = a.get(key) or ""
            if url.startswith("http"):
                host = url.split("/")[2]
                assert host in ("fonts.googleapis.com", "fonts.gstatic.com", "github.com"), url


def test_tablists_are_wired_for_assistive_tech():
    lists = re.findall(r'role="tablist".*?(?=role="tablist"|$)', INDEX, re.S)
    assert len(lists) == 2
    tabs = attrs("button", role="tab")
    panels = {a["id"]: a for a in attrs(role="tabpanel")}
    assert len(tabs) == len(panels)
    for tab in tabs:
        panel = panels[tab["aria-controls"]]
        assert panel["aria-labelledby"] == tab["id"]
        assert ("hidden" in panel) == (tab["aria-selected"] == "false")


def test_every_sample_tab_is_labelled_real_demo_or_mock_with_a_date():
    for panel in re.findall(r'<div class="viewer-panel".*?</div>\s*</div>', INDEX, re.S):
        kind = re.search(r'class="kind kind-(real|demo|mock)"', panel)
        assert kind, panel[:120]
        source = re.search(r'class="viewer-source">([^<]+)<', panel).group(1)
        assert re.search(r"\d{1,2} \w{3} 20\d\d", source), source
        src = re.search(r'<iframe[^>]+src="([^"]+)"', panel).group(1)
        report = (DOCS / src).read_text(encoding="utf-8")
        if kind.group(1) == "mock":
            assert "mock" in report.lower()
        else:
            assert 'role="note"' in report  # the label banner inside the report itself


def test_stats_strip_uses_real_numbers_from_the_real_run():
    stats = re.findall(r'<div class="stat" data-source="([^"]+)" data-evidence="([^"]+)">', INDEX)
    assert len(stats) >= 4
    for source, evidence in stats:
        report = (DOCS / source).read_text(encoding="utf-8")
        assert evidence in report, (source, evidence)
        assert "mock" not in report.split("</div>", 1)[0].lower()  # no mock run behind the strip


def test_every_copyable_command_parses_with_the_real_cli():
    commands = [a["data-copy"] for _, a in PAGE.tags if "data-copy" in a and a.get("data-copy-kind") != "code"]
    assert {"pip install ragaudit-cli", "ragaudit ui"} <= set(commands) and len(commands) >= 8
    for command in commands:
        validate_command(command)


def test_copy_buttons_are_real_buttons_with_labels():
    for tag, a in PAGE.tags:
        if "data-copy" in a:
            assert tag == "button" and a.get("type") == "button" and a.get("aria-label"), a


def test_footer_shows_the_package_version():
    version = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"]
    assert f"<span data-version>v{version}</span>" in INDEX


@pytest.mark.parametrize("name", ["sample-report.html", "sample-report-lucky.html", "sample-report-healthy.html", "sample-report-degraded.html"])
def test_v3_samples_are_self_contained_and_have_no_local_paths(name):
    html = (DOCS / name).read_text(encoding="utf-8")
    assert html.count('role="tab"') == 8
    for bad in ("/Users/", "/private/", "/app/", "/home/"):
        assert bad not in html, (name, bad)
    assert "<script src" not in html
