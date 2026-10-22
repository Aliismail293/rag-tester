"""Builds the LLM client for v2 commands, and enforces --max-llm-calls budgets.

Provider is `mock` (the offline MockLLM) or `openai` (the existing
OpenAI-compatible LLMClient). It comes from the `--llm` flag, else the
RAGAUDIT_LLM environment variable, else `openai`, which keeps v1's
"configure with RAGAUDIT_API_*" behaviour as the default.
"""

import os
from typing import Protocol

from ragaudit.llm.client import LLMClient
from ragaudit.llm.mock import MockLLM

PROVIDER_ENV = "RAGAUDIT_LLM"
PROVIDERS = ("openai", "mock")


class Completer(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str: ...


class LLMBudgetExceeded(Exception):
    """Raised when a call would exceed --max-llm-calls. Callers stop cleanly and save progress."""


def resolve_provider(provider: str | None = None) -> str:
    resolved = (provider or os.environ.get(PROVIDER_ENV) or "openai").lower()
    if resolved not in PROVIDERS:
        raise ValueError(f"unknown LLM provider {resolved!r}; expected one of {', '.join(PROVIDERS)}")
    return resolved


def make_client(provider: str | None = None, model: str | None = None, mock_model: str = "mock-llm") -> Completer:
    """An LLM client for the given provider. Real clients read RAGAUDIT_API_BASE/_API_KEY (and _MODEL if no model)."""
    if resolve_provider(provider) == "mock":
        return MockLLM(model=model or mock_model)
    return LLMClient(model=model)


class CallBudget:
    """A shared counter of LLM calls with an optional cap."""

    def __init__(self, max_calls: int | None = None) -> None:
        self.max_calls = max_calls
        self.used = 0

    @property
    def remaining(self) -> int | None:
        return None if self.max_calls is None else max(self.max_calls - self.used, 0)

    def spend(self) -> None:
        if self.max_calls is not None and self.used >= self.max_calls:
            raise LLMBudgetExceeded(f"LLM call budget of {self.max_calls} reached")
        self.used += 1


class BudgetedCompleter:
    """Wraps a Completer so every call is counted against a shared CallBudget."""

    def __init__(self, inner: Completer, budget: CallBudget) -> None:
        self._inner = inner
        self.budget = budget
        self.model = inner.model

    def complete(self, system: str, user: str) -> str:
        self.budget.spend()
        return self._inner.complete(system, user)
