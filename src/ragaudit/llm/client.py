"""Thin OpenAI-compatible chat completions client over httpx."""

import os
import time

import httpx

DEFAULT_MAX_RETRIES = 3
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class LLMClientError(Exception):
    """Raised when the LLM API call fails after exhausting all retries."""


class LLMClient:
    """Minimal OpenAI-compatible chat completions client.

    Configuration defaults to environment variables:
    - RAGAUDIT_API_BASE: base URL, e.g. https://api.openai.com/v1
    - RAGAUDIT_API_KEY: bearer token
    - RAGAUDIT_MODEL: model name
    """

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_base = (api_base or os.environ["RAGAUDIT_API_BASE"]).rstrip("/")
        self.api_key = api_key or os.environ["RAGAUDIT_API_KEY"]
        self.model = model or os.environ["RAGAUDIT_MODEL"]
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)

    def complete(self, system: str, user: str) -> str:
        """Send one chat completion request and return the assistant's text.

        Retries on 429 and 5xx responses (and transport errors) with
        exponential backoff, up to max_retries attempts total.
        """
        url = f"{self.api_base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }

        last_error: Exception | str | None = None
        for attempt in range(self.max_retries):
            is_last_attempt = attempt == self.max_retries - 1
            try:
                response = self._client.post(url, headers=headers, json=payload)
            except httpx.TransportError as exc:
                last_error = exc
                if is_last_attempt:
                    break
                self._sleep_backoff(attempt)
                continue

            if response.status_code in RETRYABLE_STATUS_CODES:
                last_error = f"status {response.status_code}: {response.text}"
                if is_last_attempt:
                    break
                self._sleep_backoff(attempt)
                continue

            if response.status_code >= 400:
                raise LLMClientError(f"LLM request failed with status {response.status_code}: {response.text}")

            data = response.json()
            return data["choices"][0]["message"]["content"]

        raise LLMClientError(f"LLM request failed after {self.max_retries} attempts: {last_error}")

    def _sleep_backoff(self, attempt: int) -> None:
        time.sleep(2**attempt)
