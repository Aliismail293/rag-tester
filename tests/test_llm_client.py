import httpx
import pytest

from ragaudit.llm.client import LLMClient, LLMClientError


def make_client(handler, max_retries: int = 3) -> LLMClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = LLMClient(
        api_base="https://fake.example/v1",
        api_key="test-key",
        model="test-model",
        max_retries=max_retries,
        client=http_client,
    )
    client._sleep_backoff = lambda attempt: None  # no real sleeping in tests
    return client


def _completion_response(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def test_complete_returns_content_on_success():
    def handler(request: httpx.Request) -> httpx.Response:
        return _completion_response("hello world")

    client = make_client(handler)
    assert client.complete("sys", "user") == "hello world"


def test_retries_on_429_then_succeeds():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 3:
            return httpx.Response(429, text="rate limited")
        return _completion_response("recovered")

    client = make_client(handler)
    result = client.complete("sys", "user")
    assert result == "recovered"
    assert calls["count"] == 3


def test_retries_on_5xx_then_succeeds():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 2:
            return httpx.Response(503, text="server error")
        return _completion_response("ok")

    client = make_client(handler)
    assert client.complete("sys", "user") == "ok"
    assert calls["count"] == 2


def test_raises_after_max_retries_exhausted():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="always fails")

    client = make_client(handler, max_retries=3)
    with pytest.raises(LLMClientError, match="after 3 attempts"):
        client.complete("sys", "user")


def test_non_retryable_error_raises_immediately():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(400, text="bad request")

    client = make_client(handler)
    with pytest.raises(LLMClientError, match="status 400"):
        client.complete("sys", "user")
    assert calls["count"] == 1
