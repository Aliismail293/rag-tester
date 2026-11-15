"""`ragaudit ui`: token and origin enforcement, each endpoint on a temp project, the wizard end to end over HTTP (mock LLM)."""

import json
import shutil
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from ragaudit import history
from ragaudit.cli import app
from ragaudit.storage.store import Store
from ragaudit.ui import server as ui

SAMPLE = Path(__file__).resolve().parent.parent / "demo" / "sample_corpus"
runner = CliRunner()


@pytest.fixture
def clean_env(monkeypatch):
    for key in ui.ENV_KEYS + ("RAGAUDIT_LLM", "RAGAUDIT_DEMO_DOCS"):
        monkeypatch.delenv(key, raising=False)
    yield
    for key in ui.ENV_KEYS:  # SetupApp writes the page's settings into os.environ
        monkeypatch.delenv(key, raising=False)


def start(project: Path, llm: str | None = "mock"):
    srv = ui.UIServer(project, port=0, llm=llm)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_port}"
    client = httpx.Client(base_url=base, headers={"X-Ragaudit-Token": srv.token}, timeout=30)
    return srv, client


@pytest.fixture
def served(tmp_path, clean_env, monkeypatch):
    saved = sys.stdout, sys.stderr
    ui.install_output_router()
    srv, client = start(tmp_path)
    yield tmp_path, srv, client
    client.close()
    srv.shutdown()
    srv.server_close()
    sys.stdout, sys.stderr = saved


def post(client, path, body=None):
    return client.post(path, json=body or {})


def wait_job(client, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get("/api/job").json()["job"]
        if job["state"] != "running":
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


# --- security -------------------------------------------------------------------


def test_every_route_needs_the_token(served):
    _, srv, client = served
    bare = httpx.Client(base_url=str(client.base_url))
    for path in ("/", "/api/state", "/api/job", "/report", "/api/ollama"):
        assert bare.get(path).status_code == 403, path
        assert bare.get(path, params={"t": "wrong"}).status_code == 403, path
    assert bare.get("/", params={"t": srv.token}).status_code == 200  # the page load link
    # POSTs need the header: the ?t= query alone isn't enough, so a plain cross-site form can't post.
    assert bare.post("/api/settings", params={"t": srv.token}, json={}).status_code == 403
    assert bare.post("/api/run", json={"mode": "check"}).status_code == 403
    assert client.get("/api/state").status_code == 200


def test_cross_origin_and_wrong_host_are_refused(served):
    _, srv, client = served
    assert client.get("/api/state", headers={"Origin": "http://evil.example"}).status_code == 403
    assert client.get("/api/state", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.get("/api/state", headers={"Host": "evil.example:80"}).status_code == 403  # DNS rebinding
    assert client.get("/api/state", headers={"Origin": str(client.base_url)}).status_code == 200
    nav = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate"}
    assert client.get("/", headers=nav).status_code == 200  # following the printed link from another page
    assert client.get("/api/state", headers=nav).status_code == 403
    assert client.post("/api/docs", content="{}", headers={"Content-Type": "text/plain"}).status_code == 415


def test_binds_to_loopback_only(served):
    _, srv, _ = served
    assert srv.server_address[0] == "127.0.0.1"
    assert srv.url.startswith(f"http://127.0.0.1:{srv.server_port}/?t=")


def test_paths_stay_inside_the_project(served):
    tmp_path, _, client = served
    for raw in ("..", "../..", "/etc", str(tmp_path.parent)):
        r = post(client, "/api/docs", {"path": raw})
        assert r.status_code == 400 and "inside the project folder" in r.json()["error"], raw


# --- endpoints ------------------------------------------------------------------


def test_state_on_an_empty_project(served):
    tmp_path, _, client = served
    st = client.get("/api/state").json()
    assert st["project"] == str(tmp_path.resolve())
    assert st["settings"]["mock"] and st["settings"]["ready"]
    assert st["steps_done"] == [True, False, False] and st["start_step"] == 0 and st["docs"] == "./docs"
    assert client.get("/report").status_code == 404
    page = client.get("/").text
    assert "STEP 1 OF 4" in page and "Download models" in page and "Open in browser tab" in page


def test_docs_listing(served):
    tmp_path, _, client = served
    missing = post(client, "/api/docs", {"path": "./docs"}).json()
    assert not missing["ok"] and "does not exist" in missing["message"]
    (tmp_path / "docs").mkdir()
    empty = post(client, "/api/docs", {"path": "./docs"}).json()
    assert not empty["ok"] and "No .txt or .md files" in empty["message"]
    shutil.copytree(SAMPLE, tmp_path / "docs", dirs_exist_ok=True)
    (tmp_path / "docs" / "scan.pdf").write_bytes(b"%PDF")
    found = post(client, "/api/docs", {"path": "docs", "save": True}).json()
    assert found["ok"] and found["count"] == len(list(SAMPLE.iterdir())) and found["passages"] > 0
    assert found["skipped"] == 1 and found["path"] == "./docs"
    assert client.get("/api/state").json()["docs_ok"]


def test_adapter_preview_write_and_overwrite_confirm(served):
    tmp_path, _, client = served
    for fw in ui.FRAMEWORKS:
        r = post(client, "/api/adapter/preview", {"framework": fw}).json()
        assert "def " in r["code"] and not r["exists"]
    assert post(client, "/api/adapter/preview", {"framework": "rust"}).status_code == 400
    assert post(client, "/api/adapter", {"framework": "plain"}).json()["ok"]
    original = (tmp_path / "ragaudit_adapter.py").read_text()
    assert "# TODO" in original and (tmp_path / ".env.example").is_file()
    again = post(client, "/api/adapter", {"framework": "demo"}).json()
    assert again["needs_confirm"] and (tmp_path / "ragaudit_adapter.py").read_text() == original
    assert post(client, "/api/adapter", {"framework": "demo", "overwrite": True}).json()["ok"]
    assert (tmp_path / "ragaudit_adapter.py").read_text() != original
    assert post(client, "/api/adapter/preview", {"framework": "demo"}).json()["same"]


def test_connection_reports_the_exact_error_and_fix(served):
    tmp_path, _, client = served
    shutil.copytree(SAMPLE, tmp_path / "docs")
    post(client, "/api/docs", {"path": "./docs", "save": True})
    none = post(client, "/api/test-connection").json()
    assert not none["ok"] and "not found" in none["message"] and "ragaudit init" in none["fix"]
    post(client, "/api/adapter", {"framework": "plain"})
    todo = post(client, "/api/test-connection").json()
    assert not todo["ok"] and "not filled in" in todo["message"] and "TODO" in todo["fix"]
    post(client, "/api/adapter", {"framework": "demo", "overwrite": True})
    ok = post(client, "/api/test-connection").json()
    assert ok["ok"] and "Answered a sample question" in ok["message"] and "passages" in ok["message"]


def test_settings_write_env_and_never_echo_the_key(tmp_path, clean_env, monkeypatch):
    srv, client = start(tmp_path, llm=None)
    try:
        secret = "sk-test-0123456789abcdef"
        bad = post(client, "/api/settings", {"provider": "key", "api_key": secret, "api_base": "nope", "model": "m"})
        assert bad.status_code == 400 and secret not in bad.text
        same = post(client, "/api/settings", {"provider": "key", "api_key": secret, "api_base": "https://x.test/v1", "model": "m", "verifier_model": "m"})
        assert same.status_code == 400 and "different" in same.text
        r = post(client, "/api/settings", {"provider": "key", "api_key": secret, "api_base": "https://x.test/v1", "model": "gpt-4o-mini"})
        assert r.status_code == 200 and secret not in r.text and r.json()["settings"]["key_saved"]
        env = (tmp_path / ".env").read_text()
        assert f"RAGAUDIT_API_KEY={secret}" in env and "RAGAUDIT_MODEL=gpt-4o-mini" in env
        assert "RAGAUDIT_VERIFIER_MODEL" not in env
        assert (tmp_path / ".env").stat().st_mode & 0o077 == 0
        state = client.get("/api/state").text
        assert secret not in state and json.loads(state)["steps_done"][0]
        # Saving again without a key keeps the saved one; other .env lines survive.
        (tmp_path / ".env").write_text(env + "OTHER=1\n")
        post(client, "/api/settings", {"provider": "key", "api_base": "https://y.test/v1", "model": "gpt-4o"})
        env = (tmp_path / ".env").read_text()
        assert f"RAGAUDIT_API_KEY={secret}" in env and "OTHER=1" in env and "https://y.test/v1" in env
        # A failing key test returns the fix, never the key.
        monkeypatch.setattr("ragaudit.llm.client.LLMClient.complete", lambda self, s, u: (_ for _ in ()).throw(RuntimeError(f"status 401 for {secret}")))
        failed = post(client, "/api/test-key", {"api_key": secret, "api_base": "https://x.test/v1", "model": "m"})
        assert not failed.json()["ok"] and "refused" in failed.json()["fix"] and secret not in failed.text
    finally:
        client.close()
        srv.shutdown()
        srv.server_close()


def test_ollama_detection_and_pull(served, monkeypatch):
    _, _, client = served

    def down(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", down)
    o = client.get("/api/ollama").json()
    assert not o["found"] and set(o["missing"]) == set(ui.RECOMMENDED_MODELS)

    class Tags:
        def raise_for_status(self):
            pass

        def json(self):
            return {"models": [{"name": "qwen2.5:7b"}]}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: Tags())
    o = client.get("/api/ollama").json()
    assert o["found"] and o["models"] == ["qwen2.5:7b"] and o["missing"] == ["llama3.1:8b"]
    assert post(client, "/api/ollama/pull", {"models": ["evil/model"]}).status_code == 400

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield json.dumps({"status": "pulling", "total": 100, "completed": 40})
            yield json.dumps({"status": "pulling", "total": 100, "completed": 100})
            yield json.dumps({"status": "success"})

    sent = []
    monkeypatch.setattr(httpx, "stream", lambda method, url, json=None, timeout=None: sent.append((url, json)) or Stream())
    assert post(client, "/api/ollama/pull", {"models": ["llama3.1:8b"]}).status_code == 200
    job = wait_job(client)
    assert job["state"] == "done" and job["kind"] == "pull" and job["done"] == 100
    assert sent == [(f"{ui.OLLAMA_URL}/api/pull", {"model": "llama3.1:8b", "stream": True})]


def test_run_needs_each_step_first(served):
    tmp_path, _, client = served
    r = post(client, "/api/run", {"mode": "check"})
    assert r.status_code == 400 and "step 2" in r.json()["error"]
    shutil.copytree(SAMPLE, tmp_path / "docs")
    post(client, "/api/docs", {"path": "./docs", "save": True})
    assert "step 3" in post(client, "/api/run", {"mode": "check"}).json()["error"]
    post(client, "/api/adapter", {"framework": "demo"})
    assert "whole number" in post(client, "/api/run", {"mode": "check", "budget": "lots"}).json()["error"]
    assert "mode" in post(client, "/api/run", {"mode": "nuke"}).json()["error"]


# --- the wizard, end to end -----------------------------------------------------------


def test_wizard_end_to_end_over_http(served):
    tmp_path, _, client = served
    shutil.copytree(SAMPLE, tmp_path / "docs")
    assert client.get("/api/state").json()["start_step"] == 0
    assert post(client, "/api/settings", {"provider": "ollama"}).json()["ok"]  # mock: nothing to save
    assert post(client, "/api/docs", {"path": "./docs", "save": True}).json()["ok"]
    assert post(client, "/api/adapter", {"framework": "demo"}).json()["ok"]
    assert post(client, "/api/test-connection").json()["ok"]
    st = client.get("/api/state").json()
    assert st["steps_done"] == [True, True, True] and st["start_step"] == 3

    job = post(client, "/api/run", {"mode": "check", "note": "first look"}).json()
    assert job["state"] == "running" and job["kind"] == "check"
    assert post(client, "/api/run", {"mode": "check"}).status_code in (200, 409)  # one job at a time
    job = wait_job(client)
    assert job["state"] == "done", job
    assert job["has_report"] and job["calls"] > 0 and job["total"] > 0
    assert any("questions" in line.lower() for line in job["log"])
    report = client.get("/report")
    assert report.status_code == 200 and "first look" in report.text and 'role="tablist"' in report.text

    second = post(client, "/api/run", {"mode": "check", "note": "again"})
    assert second.status_code == 200
    assert wait_job(client)["state"] == "done"
    with Store(tmp_path / ".ragaudit" / "store.db") as store:
        rows = history.load_history(store)
    assert [r.kind for r in rows] == ["check", "check"] and rows[1].prev_run_id == rows[0].run_id
    assert not (tmp_path / ".env").exists()


def test_improve_job_and_cancel(served):
    tmp_path, _, client = served
    shutil.copytree(SAMPLE, tmp_path / "docs")
    post(client, "/api/docs", {"path": "./docs", "save": True})
    post(client, "/api/adapter", {"framework": "demo"})
    post(client, "/api/run", {"mode": "improve", "budget": 400})
    assert post(client, "/api/job/cancel").json()["ok"]
    job = wait_job(client)
    assert job["state"] in ("cancelled", "done") and job["kind"] == "improve"
    if job["state"] == "cancelled":
        assert "saved" in job["message"]
    post(client, "/api/run", {"mode": "improve"})
    job = wait_job(client, timeout=240)
    assert job["state"] == "done", job
    assert "Self-improvement loop" in client.get("/report").text
    assert (tmp_path / "best_config.yaml").is_file()


# --- the CLI ------------------------------------------------------------------


def test_bare_ragaudit_prints_a_welcome():
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    lines = result.output.strip().splitlines()
    assert len([l for l in lines if l.strip()]) == 4 or len(lines) == 5
    assert "ragaudit ui" in result.output and "ragaudit --help" in result.output


def test_ui_command_prints_the_link(tmp_path, monkeypatch, clean_env):
    seen = {}

    def fake_serve(self):
        seen["url"] = self.url
        raise KeyboardInterrupt

    monkeypatch.setattr(ui.UIServer, "serve_forever", fake_serve)
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    saved = sys.stdout, sys.stderr
    try:
        result = runner.invoke(app, ["ui", "--no-browser", "--port", "0", "--project", str(tmp_path), "--llm", "mock"])
    finally:
        sys.stdout, sys.stderr = saved
    assert result.exit_code == 0, result.output
    assert seen["url"] in result.output and "Press Ctrl+C to stop" in result.output and not opened
