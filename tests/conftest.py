import pytest


@pytest.fixture(autouse=True)
def _isolate_llm_provider_env(monkeypatch):
    """v2 CLI commands export --llm as RAGAUDIT_LLM for the adapter; keep that from leaking between tests."""
    monkeypatch.delenv("RAGAUDIT_LLM", raising=False)
