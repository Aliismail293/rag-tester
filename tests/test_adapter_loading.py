from pathlib import Path

import pytest

from ragaudit.adapters.base import AdapterLoadError, load_adapter

FIXTURES = Path(__file__).parent / "fixtures"


def test_loads_module_with_query_function():
    adapter = load_adapter(str(FIXTURES / "adapter_with_query.py"))
    response = adapter.query("hello")
    assert response.answer == "answer to: hello"
    assert response.retrieved_chunk_ids == ["c1"]


def test_loads_module_with_adapter_object():
    adapter = load_adapter(str(FIXTURES / "adapter_with_object.py"))
    response = adapter.query("hello")
    assert response.answer == "object answer to: hello"
    assert response.retrieved_chunk_ids == ["c2"]


def test_module_with_neither_raises_clear_error():
    with pytest.raises(AdapterLoadError, match="must define a module-level `adapter`"):
        load_adapter(str(FIXTURES / "adapter_with_neither.py"))


def test_missing_file_raises_clear_error():
    with pytest.raises(AdapterLoadError, match="not found"):
        load_adapter(str(FIXTURES / "does_not_exist.py"))


def test_module_that_raises_on_import_is_wrapped():
    with pytest.raises(AdapterLoadError, match="raised an error on import") as exc_info:
        load_adapter(str(FIXTURES / "adapter_raises_on_import.py"))
    assert isinstance(exc_info.value.__cause__, RuntimeError)
