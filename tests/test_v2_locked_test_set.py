"""The locked test set is enforced in code: tune must not be able to read test answers."""

import ast
from pathlib import Path

import pytest

import ragaudit.tune
from ragaudit.models.question import Question, Split
from ragaudit.runner.locked import _load_test_questions
from ragaudit.storage.lock import LockedTestSetError
from ragaudit.storage.store import Store
from ragaudit.tune.devset import DevSet

TUNE_DIR = Path(ragaudit.tune.__file__).parent

FORBIDDEN_IMPORTS = {"sqlite3", "ragaudit.storage.lock", "ragaudit.storage.db", "ragaudit.storage.repository"}
FORBIDDEN_NAMES = {"_unlock_test_split", "_unlocked", "_load_test_questions", "_conn", "load_questions"}


def make_questions(n: int) -> list[Question]:
    return [
        Question.create(
            chunk_ids=[f"c{i}"],
            chunk_hashes=[f"h{i}"],
            fact=f"fact {i}",
            question=f"q {i}?",
            gold_answer=f"secret answer {i}",
            generator_model="gen",
        )
        for i in range(n)
    ]


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "store.db") as s:
        s.save_questions(make_questions(60))
        yield s


def _tune_sources() -> list[Path]:
    sources = sorted(TUNE_DIR.rglob("*.py"))
    assert sources, "expected the tune package to have source files"
    return sources


def test_tune_package_never_imports_the_unlock_machinery_or_raw_storage():
    for path in _tune_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in FORBIDDEN_IMPORTS, f"{path.name} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                assert node.module not in FORBIDDEN_IMPORTS, f"{path.name} imports from {node.module}"
                for alias in node.names:
                    assert alias.name not in FORBIDDEN_NAMES, f"{path.name} imports {alias.name}"
            elif isinstance(node, ast.Attribute):
                assert node.attr not in FORBIDDEN_NAMES, f"{path.name} accesses .{node.attr}"
            elif isinstance(node, ast.Name):
                assert node.id not in FORBIDDEN_NAMES, f"{path.name} references {node.id}"


def test_devset_contains_only_dev_questions(store):
    dev = DevSet(store)
    assert len(dev) > 0
    assert all(q.split == Split.DEV for q in dev.questions)


def test_tune_cannot_read_test_answers_through_the_store(store):
    with pytest.raises(LockedTestSetError):
        store.load_questions(split=Split.TEST)


def test_final_evaluation_path_can_read_test_answers(store):
    test_questions = _load_test_questions(store)
    assert test_questions and all(q.split == Split.TEST for q in test_questions)
    # ...and the unlock does not outlive the call.
    with pytest.raises(LockedTestSetError):
        store.load_questions(split=Split.TEST)
