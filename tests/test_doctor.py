"""`ragaudit doctor`: one pass/fail line per check, a fix for each failure, never a setting's value."""

import httpx
import pytest
from typer.testing import CliRunner

from ragaudit.cli import app
from ragaudit.cli_easy import check_python
from ragaudit.llm.client import LLMClient

runner = CliRunner()
SECRET = "sk-test-secret-value-123"

GOOD_ADAPTER = (
    "from ragaudit.models.adapter import RAGResponse\n\n"
    "def query(question):\n"
    "    return RAGResponse(answer='It is about trams.', contexts=['Trams run every ten minutes.'])\n"
)


@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    """Run in an empty folder with a known environment, whatever the developer's shell has set."""
    monkeypatch.chdir(tmp_path)
    for name in ("RAGAUDIT_API_BASE", "RAGAUDIT_API_KEY", "RAGAUDIT_MODEL", "RAGAUDIT_LLM"):
        monkeypatch.delenv(name, raising=False)


def set_api_env(monkeypatch, base="https://api.example.com/v1"):
    monkeypatch.setenv("RAGAUDIT_API_BASE", base)
    monkeypatch.setenv("RAGAUDIT_API_KEY", SECRET)
    monkeypatch.setenv("RAGAUDIT_MODEL", "secret-model-name")


def fake_api(monkeypatch, handler):
    """Route doctor's LLMClient through an in-process transport."""

    def factory(**kwargs):
        return LLMClient(client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)

    monkeypatch.setattr("ragaudit.cli_easy.LLMClient", factory)


def ok_handler(request):
    return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})


def write_adapter(tmp_path, source=GOOD_ADAPTER):
    path = tmp_path / "ragaudit_adapter.py"
    path.write_text(source)
    return path


def doctor(*args):
    return runner.invoke(app, ["doctor", *args])


def test_all_checks_pass(tmp_path, monkeypatch):
    set_api_env(monkeypatch)
    fake_api(monkeypatch, ok_handler)
    write_adapter(tmp_path)
    result = doctor()
    assert result.exit_code == 0, result.output
    assert "FAIL" not in result.output
    assert result.output.count("PASS") == 8
    assert "All checks passed" in result.output and "ragaudit check" in result.output


def test_never_prints_setting_values(tmp_path, monkeypatch):
    set_api_env(monkeypatch)
    fake_api(monkeypatch, lambda request: httpx.Response(401, json={"error": "bad key"}))
    write_adapter(tmp_path)
    result = doctor()
    for value in (SECRET, "secret-model-name", "api.example.com"):
        assert value not in result.output
    assert "RAGAUDIT_API_KEY is set" in result.output


def test_missing_env_vars_fail_with_a_fix(tmp_path):
    write_adapter(tmp_path)
    result = doctor()
    assert result.exit_code == 1
    for name in ("RAGAUDIT_API_BASE", "RAGAUDIT_API_KEY", "RAGAUDIT_MODEL"):
        assert f"FAIL  {name} is not set" in result.output
    assert "cp .env.example .env" in result.output and "RAGAUDIT_API_KEY=ollama" in result.output
    assert "FAIL  Test call to the LLM API: not attempted" in result.output
    assert "PASS  Adapter ragaudit_adapter.py imports" in result.output


def test_settings_are_read_from_a_dotenv_file(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(f"RAGAUDIT_API_BASE=https://api.example.com/v1\nRAGAUDIT_API_KEY={SECRET}\nRAGAUDIT_MODEL=m\n")
    fake_api(monkeypatch, ok_handler)
    write_adapter(tmp_path)
    try:
        result = doctor()
        assert result.exit_code == 0, result.output
        assert "read settings from ./.env" in result.output and SECRET not in result.output
    finally:
        for name in ("RAGAUDIT_API_BASE", "RAGAUDIT_API_KEY", "RAGAUDIT_MODEL"):
            monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "status, base, fix",
    [
        (401, "https://api.example.com/v1", "refused RAGAUDIT_API_KEY"),
        (401, "http://localhost:11434/v1", "RAGAUDIT_API_KEY=ollama"),
        (404, "http://localhost:11434/v1", "ollama pull"),
        (404, "https://api.example.com/v1", "ends in /v1"),
        (429, "https://api.example.com/v1", "quota"),
    ],
)
def test_api_errors_get_a_specific_fix(tmp_path, monkeypatch, status, base, fix):
    set_api_env(monkeypatch, base)
    fake_api(monkeypatch, lambda request: httpx.Response(status, text="error"))
    write_adapter(tmp_path)
    result = doctor()
    assert result.exit_code == 1
    assert "FAIL  Test call to the LLM API failed" in result.output and fix in result.output


def test_unreachable_ollama_says_to_start_it(tmp_path, monkeypatch):
    set_api_env(monkeypatch, "http://localhost:11434/v1")

    def refuse(request):
        raise httpx.ConnectError("connection refused")

    fake_api(monkeypatch, refuse)
    write_adapter(tmp_path)
    result = doctor()
    assert result.exit_code == 1 and "ollama serve" in result.output


def test_api_base_that_is_not_a_url(tmp_path, monkeypatch):
    set_api_env(monkeypatch, "localhost:11434")
    write_adapter(tmp_path)
    result = doctor()
    assert "RAGAUDIT_API_BASE is not a URL" in result.output and "http://" in result.output


def test_mock_llm_skips_env_and_api(tmp_path):
    write_adapter(tmp_path)
    result = doctor("--llm", "mock")
    assert result.exit_code == 0, result.output
    assert "not needed with the mock LLM" in result.output and "mock LLM selected" in result.output


def test_missing_adapter(tmp_path):
    result = doctor("--llm", "mock")
    assert result.exit_code == 1
    assert "FAIL  Adapter ragaudit_adapter.py not found" in result.output and "ragaudit init" in result.output
    assert "FAIL  Adapter returns a valid RAGResponse: not attempted" in result.output


def test_unfilled_template_points_at_the_todo(tmp_path):
    assert runner.invoke(app, ["init"]).exit_code == 0
    result = doctor("--llm", "mock")
    assert result.exit_code == 1
    assert "template is not filled in" in result.output and "TODO in ragaudit_adapter.py" in result.output


def test_adapter_with_a_missing_package(tmp_path):
    write_adapter(tmp_path, "import package_that_does_not_exist_xyz\n")
    result = doctor("--llm", "mock")
    assert "does not import" in result.output and "pip install package_that_does_not_exist_xyz" in result.output


def test_adapter_that_raises(tmp_path):
    write_adapter(tmp_path, "def query(question):\n    raise ValueError('index not loaded')\n")
    result = doctor("--llm", "mock")
    assert "query() raised ValueError: index not loaded" in result.output


def test_adapter_returning_the_wrong_type(tmp_path):
    write_adapter(tmp_path, "def query(question):\n    return 'just a string'\n")
    result = doctor("--llm", "mock")
    assert "query() returned str" in result.output and "RAGResponse(answer=" in result.output


def test_adapter_returning_a_dict_is_accepted(tmp_path):
    write_adapter(tmp_path, "def query(question):\n    return {'answer': 'a', 'contexts': ['c']}\n")
    assert doctor("--llm", "mock").exit_code == 0


def test_empty_answer_and_missing_contexts_fail(tmp_path):
    write_adapter(tmp_path, "from ragaudit.models.adapter import RAGResponse\ndef query(q):\n    return RAGResponse(answer='')\n")
    assert "the answer is empty" in doctor("--llm", "mock").output
    write_adapter(tmp_path, "from ragaudit.models.adapter import RAGResponse\ndef query(q):\n    return RAGResponse(answer='a')\n")
    result = doctor("--llm", "mock")
    assert result.exit_code == 1 and "`contexts` is empty" in result.output and "--match-by id" in result.output


def test_python_version_check():
    assert check_python((3, 11, 0)).ok and check_python((3, 13, 1)).ok
    old = check_python((3, 10, 12))
    assert not old.ok and "3.11" in old.fix
