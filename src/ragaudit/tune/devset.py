"""The data-access layer `tune` uses: dev-split questions only."""

from ragaudit.models.question import Question, Split
from ragaudit.storage.store import Store


class DevSet:
    """Active dev-split questions, with their gold answers. Never contains a test question."""

    def __init__(self, store: Store) -> None:
        self.questions: list[Question] = store.dev_questions()
        if any(q.split != Split.DEV for q in self.questions):
            raise AssertionError("DevSet received a non-dev question")

    def __len__(self) -> int:
        return len(self.questions)
