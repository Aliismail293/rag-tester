"""The final evaluation path: the only code allowed to read test-split gold answers.

Everything here unlocks the test split for exactly as long as it needs,
and hands back aggregate scores rather than questions, answers or
per-question results, so a caller (in particular `ragaudit tune`) learns
how well a config did on the test set without ever seeing what the test
set contains. Per-question test results are still cached in the store,
behind the same lock, so the report can be rebuilt.

The pipeline under test does see the test *questions*: it has to answer
them. That's inherent to evaluating it.
"""

from ragaudit.models.adapter import Adapter
from ragaudit.models.evaluation import EvalSummary
from ragaudit.models.question import Question, Split
from ragaudit.runner.evaluate import CorrectnessJudge, run_evaluation
from ragaudit.storage.lock import _unlock_test_split
from ragaudit.storage.store import Store


def _load_test_questions(store: Store) -> list[Question]:
    """Active test-split questions, with gold answers. Private to this module."""
    with _unlock_test_split():
        return store.load_questions(split=Split.TEST)


def evaluate_test_split(store: Store, adapter: Adapter, judge: CorrectnessJudge, **kwargs) -> EvalSummary:
    """Evaluate the active test split once and return only the aggregate summary."""
    with _unlock_test_split():
        summary, _ = run_evaluation(store, _load_test_questions(store), Split.TEST, adapter, judge, **kwargs)
    return summary


def count_test_questions(store: Store) -> int:
    """Number of active test questions (ids only, no unlock needed)."""
    return len(store.question_ids(Split.TEST))
