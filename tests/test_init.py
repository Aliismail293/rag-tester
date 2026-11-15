"""`ragaudit init` templates and .env loading."""

import os
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from ragaudit.adapters.base import load_adapter
from ragaudit.cli import app
from ragaudit.env import load_dotenv, parse_dotenv
from ragaudit.models.adapter import RAGResponse
from ragaudit.scaffold import FRAMEWORKS, adapter_template

runner = CliRunner()


def init(tmp_path, *args):
    return runner.invoke(app, ["init", "--out", str(tmp_path / "ragaudit_adapter.py"), *args])


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_init_writes_adapter_and_env_example(tmp_path, framework):
    result = init(tmp_path, "--framework", framework)
    assert result.exit_code == 0, result.output
    adapter = (tmp_path / "ragaudit_adapter.py").read_text()
    env = (tmp_path / ".env.example").read_text()

    assert adapter.count("# TODO") == 1  # one place to edit (error messages point at it)
    compile(adapter, "ragaudit_adapter.py", "exec")
    assert "ragaudit doctor --adapter ragaudit_adapter.py" in adapter
    assert "RAGAUDIT_API_BASE=http://localhost:11434/v1" in env and "RAGAUDIT_API_KEY=ollama" in env
    assert "RAGAUDIT_API_BASE=https://api.openai.com/v1" in env and "RAGAUDIT_MODEL=" in env
    assert "ragaudit doctor" in result.output and "cp " in result.output


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_unfilled_template_loads_and_explains_what_to_fill_in(tmp_path, framework):
    init(tmp_path, "--framework", framework)
    adapter = load_adapter(str(tmp_path / "ragaudit_adapter.py"))  # importable without the framework installed
    with pytest.raises(NotImplementedError, match="not filled in yet.*TODO"):
        adapter.query("anything")


def load_module(path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("filled_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plain_template_once_filled_returns_a_response(tmp_path):
    init(tmp_path)
    module = load_module(tmp_path / "ragaudit_adapter.py")
    module.ask_my_pipeline = lambda q: ("Paris", ["Paris is the capital of France."])
    response = module.query("Capital of France?")
    assert isinstance(response, RAGResponse)
    assert response.answer == "Paris" and response.contexts == ["Paris is the capital of France."]
    assert response.latency_ms is not None


def test_langchain_template_once_filled_returns_a_response(tmp_path):
    init(tmp_path, "--framework", "langchain")
    module = load_module(tmp_path / "ragaudit_adapter.py")
    docs = [
        SimpleNamespace(page_content="Paris is the capital of France.", metadata={"chunk_id": "doc-0001"}),
        SimpleNamespace(page_content="Lyon is in France.", metadata={}),
    ]
    prompts = []
    module.retriever = SimpleNamespace(invoke=lambda q: docs)
    module.llm = SimpleNamespace(invoke=lambda prompt: prompts.append(prompt) or SimpleNamespace(content="Paris"))
    response = module.query("Capital of France?")
    assert response.answer == "Paris"
    assert response.contexts == [d.page_content for d in docs] and response.retrieved_chunk_ids == ["doc-0001"]
    assert "Paris is the capital of France." in prompts[0] and "Capital of France?" in prompts[0]


def test_langchain_template_accepts_a_plain_string_llm(tmp_path):
    init(tmp_path, "--framework", "langchain")
    module = load_module(tmp_path / "ragaudit_adapter.py")
    module.retriever = SimpleNamespace(invoke=lambda q: [])
    module.llm = SimpleNamespace(invoke=lambda prompt: "I don't know")
    assert module.query("?").answer == "I don't know"


def test_llamaindex_template_once_filled_returns_a_response(tmp_path):
    init(tmp_path, "--framework", "llamaindex")
    module = load_module(tmp_path / "ragaudit_adapter.py")

    def node(text, metadata):
        return SimpleNamespace(node=SimpleNamespace(get_content=lambda: text, metadata=metadata))

    class Response:
        source_nodes = [node("Paris is the capital of France.", {"chunk_id": "doc-0001"}), node("Lyon.", {})]

        def __str__(self):
            return "Paris"

    module.query_engine = SimpleNamespace(query=lambda q: Response())
    response = module.query("Capital of France?")
    assert response.answer == "Paris" and response.contexts == ["Paris is the capital of France.", "Lyon."]
    assert response.retrieved_chunk_ids == ["doc-0001"]


def test_init_refuses_to_overwrite_without_force(tmp_path):
    assert init(tmp_path).exit_code == 0
    (tmp_path / "ragaudit_adapter.py").write_text("# my edits")
    again = init(tmp_path)
    assert again.exit_code != 0 and "--force" in again.output
    assert (tmp_path / "ragaudit_adapter.py").read_text() == "# my edits"
    assert init(tmp_path, "--force").exit_code == 0
    assert "TODO" in (tmp_path / "ragaudit_adapter.py").read_text()


def test_init_rejects_unknown_framework(tmp_path):
    result = init(tmp_path, "--framework", "haystack")
    assert result.exit_code != 0 and "plain, langchain, llamaindex" in result.output
    assert not (tmp_path / "ragaudit_adapter.py").exists()


def test_adapter_template_names_the_file():
    assert "ragaudit doctor --adapter my_rag.py" in adapter_template("plain", "my_rag.py")


# --- .env ---


def test_parse_dotenv():
    text = (
        "# comment\n\nRAGAUDIT_API_BASE=http://localhost:11434/v1\nexport RAGAUDIT_API_KEY='ollama'\n"
        'RAGAUDIT_MODEL="llama3.1:8b"\nRAGAUDIT_LLM=openai  # trailing comment\nnot a setting\n'
    )
    assert parse_dotenv(text) == {
        "RAGAUDIT_API_BASE": "http://localhost:11434/v1",
        "RAGAUDIT_API_KEY": "ollama",
        "RAGAUDIT_MODEL": "llama3.1:8b",
        "RAGAUDIT_LLM": "openai",
    }


def test_load_dotenv_never_overrides_exported_values(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("RAGAUDIT_TEST_A=from-file\nRAGAUDIT_TEST_B=from-file\n")
    monkeypatch.setenv("RAGAUDIT_TEST_A", "exported")
    monkeypatch.delenv("RAGAUDIT_TEST_B", raising=False)
    try:
        assert load_dotenv(env) == ["RAGAUDIT_TEST_B"]
        assert os.environ["RAGAUDIT_TEST_A"] == "exported" and os.environ["RAGAUDIT_TEST_B"] == "from-file"
    finally:
        os.environ.pop("RAGAUDIT_TEST_B", None)


def test_load_dotenv_without_a_file_is_a_no_op(tmp_path):
    assert load_dotenv(tmp_path / ".env") == []
