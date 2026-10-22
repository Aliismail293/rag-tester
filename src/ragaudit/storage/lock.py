"""The test-split lock: test gold answers are readable only inside the final evaluation path.

The store consults `is_test_split_unlocked()` before returning any test-split
question or result. The only code that unlocks it is
`ragaudit.runner.locked`, which returns aggregate scores, never answers.
The `ragaudit.tune` package must never import this module; a unit test
enforces that.

Python can't make this tamper-proof, so the lock exists to stop
accidents (a tuning loop that quietly reads test answers), not a
determined author.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_unlocked: ContextVar[bool] = ContextVar("ragaudit_test_split_unlocked", default=False)


class LockedTestSetError(Exception):
    """Raised when code outside the final evaluation path asks for test-split data."""


def is_test_split_unlocked() -> bool:
    return _unlocked.get()


@contextmanager
def _unlock_test_split() -> Iterator[None]:
    """Unlock test-split reads for the duration of the block. Final evaluation path only."""
    token = _unlocked.set(True)
    try:
        yield
    finally:
        _unlocked.reset(token)
