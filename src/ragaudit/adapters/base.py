"""The Adapter Protocol and loaders for user-supplied adapter files."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Callable

from ragaudit.models.adapter import Adapter, ConfigurableAdapter, RAGResponse

from .callable_adapter import CallableAdapter

__all__ = [
    "Adapter",
    "ConfigurableAdapter",
    "RAGResponse",
    "AdapterLoadError",
    "AdapterFactory",
    "load_adapter",
    "load_adapter_factory",
]

AdapterFactory = Callable[[dict], Adapter]
"""config dict -> Adapter configured with it."""


class AdapterLoadError(Exception):
    """Raised when an adapter file can't be loaded or doesn't expose an adapter."""


def load_adapter(path: str) -> Adapter:
    """Load an Adapter from a Python file, for the `--adapter` CLI flag.

    The file must define either a module-level `adapter` object satisfying
    the Adapter Protocol, or a module-level `query(question: str) -> RAGResponse`
    function, which is wrapped into an Adapter via CallableAdapter.
    """
    module = _import_adapter_module(path)

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


def load_adapter_factory(path: str) -> AdapterFactory:
    """Load an adapter file as a factory that builds a pipeline from a config dict (v2 evaluate/tune).

    Checked in order:

    1. a module-level `make_adapter(config: dict) -> Adapter` function;
    2. an `adapter` object (or `query` function) whose adapter has a
       `configure(config: dict)` method, called before returning it;
    3. a plain v1 adapter, which only accepts an empty config.
    """
    module = _import_adapter_module(path)

    make_adapter = getattr(module, "make_adapter", None)
    if callable(make_adapter):
        return lambda config: make_adapter(dict(config))

    query_fn = getattr(module, "query", None)
    if getattr(module, "adapter", None) is not None:
        base = module.adapter
    elif callable(query_fn):
        base = CallableAdapter(query_fn)
    else:
        raise AdapterLoadError(
            f"Adapter file '{path}' must define `make_adapter(config)`, a module-level `adapter` object, "
            "or a `query(question: str) -> RAGResponse` function."
        )

    def factory(config: dict) -> Adapter:
        configure = getattr(base, "configure", None)
        if callable(configure):
            configure(dict(config))
            return base
        if config:
            raise AdapterLoadError(
                f"Adapter file '{path}' doesn't accept a config: define `make_adapter(config)` or give the "
                "adapter a `configure(config)` method to use --config or `ragaudit tune`."
            )
        return base

    return factory


def _import_adapter_module(path: str) -> ModuleType:
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
    return module
