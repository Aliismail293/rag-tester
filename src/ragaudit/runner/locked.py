"""The final evaluation path: the only code allowed to read test-split gold answers.

Everything here unlocks the test split for exactly as long as it needs,
and hands back aggregate scores rather than questions or answers, so a
caller (in particular `ragaudit tune`) learns how well a config did on
the test set without ever seeing what the test set contains.
"""

from ragaudit.models.question import Question, Split
from ragaudit.storage.lock import _unlock_test_split
from ragaudit.storage.store import Store


def _load_test_questions(store: Store) -> list[Question]:
    """Active test-split questions, with gold answers. Private to this module."""
    with _unlock_test_split():
        return store.load_questions(split=Split.TEST)
