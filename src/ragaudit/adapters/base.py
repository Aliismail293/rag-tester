"""The Adapter Protocol and a loader for user-supplied adapter files."""

import importlib.util
from pathlib import Path

from ragaudit.models.adapter import Adapter, RAGResponse

from .callable_adapter import CallableAdapter

__all__ = ["Adapter", "RAGResponse", "AdapterLoadError", "load_adapter"]


class AdapterLoadError(Exception):
    """Raised when an adapter file can't be loaded or doesn't expose an adapter."""


def load_adapter(path: str) -> Adapter:
    """Load an Adapter from a Python file, for the `--adapter` CLI flag.

    The file must define either a module-level `adapter` object satisfying
    the Adapter Protocol, or a module-level `query(question: str) -> RAGResponse`
    function, which is wrapped into an Adapter via CallableAdapter.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise AdapterLoadError(f"Adapter file not found: {path}")

    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    if spec is None or spec.loader is None:
        raise AdapterLoadError(f"Could not load adapter module from: {path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AdapterLoadError(f"Adapter file raised an error on import: {path}") from exc

    adapter = getattr(module, "adapter", None)
    if adapter is not None:
        return adapter

    query_fn = getattr(module, "query", None)
    if callable(query_fn):
        return CallableAdapter(query_fn)

    raise AdapterLoadError(
        f"Adapter file '{path}' must define a module-level `adapter` object "
        "or a `query(question: str) -> RAGResponse` function."
    )
