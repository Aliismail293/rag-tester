"""Wraps a plain callable into the Adapter Protocol."""

from typing import Callable

from ragaudit.models.adapter import RAGResponse


class CallableAdapter:
    """Adapts a plain `query(question: str) -> RAGResponse` callable to Adapter."""

    def __init__(self, fn: Callable[[str], RAGResponse]) -> None:
        self._fn = fn

    def query(self, question: str) -> RAGResponse:
        return self._fn(question)
