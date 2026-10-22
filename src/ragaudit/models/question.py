"""v2 questions: a verified fact, the question that tests it, and where it came from."""

import hashlib
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

DEV_FRACTION_PERCENT = 70
"""Share of questions assigned to the dev split; the rest go to the locked test split."""

SPLIT_SALT = "ragaudit-split-v1"
"""Never change this: doing so would reshuffle every existing question between splits."""


class QuestionType(StrEnum):
    SINGLE = "single"
    MULTI_HOP = "multi_hop"


class Split(StrEnum):
    DEV = "dev"
    TEST = "test"


class QuestionStatus(StrEnum):
    ACTIVE = "active"
    STALE = "stale"
    """Its source chunk changed or was deleted. Never scored."""
    REJECTED = "rejected"
    """Failed a generation check; reject_reason says which. Never scored."""


class RejectReason(StrEnum):
    TOO_LEXICAL = "too_lexical"
    RELATIVE_TIME = "relative_time"
    UNSUPPORTED = "unsupported"
    UNANSWERABLE = "unanswerable"
    AMBIGUOUS = "ambiguous"
    VERIFIER_UNPARSEABLE = "verifier_unparseable"


def compute_question_id(chunk_hashes: list[str], fact: str) -> str:
    """Stable id: hash of the source chunk hash(es) plus the fact text.

    The same fact extracted from the same chunk content always gets the
    same id, so re-running generation is idempotent and the split (which
    is derived from this id) never changes.
    """
    payload = "|".join(chunk_hashes) + "\n" + fact.strip()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assign_split(question_id: str) -> Split:
    """Deterministically assign dev (70%) or test (30%) from the question_id alone.

    Depends on nothing but the id, so adding new questions later can never
    move an old question between splits and leak test items into tuning.
    """
    digest = hashlib.sha256(f"{SPLIT_SALT}:{question_id}".encode("utf-8")).hexdigest()
    return Split.DEV if int(digest[:8], 16) % 100 < DEV_FRACTION_PERCENT else Split.TEST


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Question(BaseModel):
    """One generated question, linked to the chunk(s) its gold answer came from."""

    question_id: str
    chunk_ids: list[str]
    """One entry normally; two or more for multi-hop questions."""
    chunk_hashes: list[str]
    """Content hash of each source chunk, parallel to chunk_ids."""
    fact: str
    question: str
    gold_answer: str
    question_type: QuestionType = QuestionType.SINGLE
    split: Split
    status: QuestionStatus = QuestionStatus.ACTIVE
    reject_reason: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    """Absolute UTC timestamp."""
    source_timestamp: datetime | None = None
    """When the source material was captured, if the corpus provides it."""
    generator_model: str
    verifier_model: str | None = None
    lexical_overlap: float | None = None
    """Fraction of the question's content tokens that appear in its source chunk(s)."""
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("created_at", "source_timestamp")
    @classmethod
    def _force_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @classmethod
    def create(
        cls,
        chunk_ids: list[str],
        chunk_hashes: list[str],
        fact: str,
        question: str,
        gold_answer: str,
        generator_model: str,
        **kwargs,
    ) -> "Question":
        """Build a Question with its id and split derived the one canonical way."""
        question_id = compute_question_id(chunk_hashes, fact)
        question_type = QuestionType.MULTI_HOP if len(chunk_ids) > 1 else QuestionType.SINGLE
        return cls(
            question_id=question_id,
            chunk_ids=chunk_ids,
            chunk_hashes=chunk_hashes,
            fact=fact,
            question=question,
            gold_answer=gold_answer,
            question_type=kwargs.pop("question_type", question_type),
            split=assign_split(question_id),
            generator_model=generator_model,
            **kwargs,
        )
