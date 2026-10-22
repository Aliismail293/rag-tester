"""v2 questions: a verified fact, the question that tests it, and where it came from."""

import hashlib
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

DEV_FRACTION_PERCENT = 70
"""Share of chunks assigned to the dev split; the rest go to the locked test split."""

SPLIT_SALT = "ragaudit-chunk-split-v1"
"""Never change this: doing so would reshuffle every chunk (and its questions) between splits."""


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
    CROSS_SPLIT = "cross_split"
    """A multi-hop question whose source chunks are not all in the same split."""


def compute_question_id(chunk_hashes: list[str], fact: str) -> str:
    """Stable id: hash of the source chunk hash(es) plus the fact text.

    The same fact extracted from the same chunk content always gets the
    same id, so re-running generation is idempotent.
    """
    payload = "|".join(chunk_hashes) + "\n" + fact.strip()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assign_split(chunk_id: str) -> Split:
    """Deterministically assign a chunk to dev (70%) or test (30%) from its chunk_id alone.

    The split is per chunk, not per question: if questions from one chunk
    could land on both sides, tuning on the dev ones would teach the
    pipeline that chunk's content and inflate the test score. It hashes
    the chunk *id*, not its content, so editing a chunk never moves it
    between splits, and adding chunks never moves old ones.
    """
    digest = hashlib.sha256(f"{SPLIT_SALT}:{chunk_id}".encode("utf-8")).hexdigest()
    return Split.DEV if int(digest[:8], 16) % 100 < DEV_FRACTION_PERCENT else Split.TEST


def split_for_chunks(chunk_ids: list[str]) -> Split | None:
    """The split shared by every source chunk, or None when they disagree (a cross-split multi-hop question)."""
    splits = {assign_split(chunk_id) for chunk_id in chunk_ids}
    return splits.pop() if len(splits) == 1 else None


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
    split: Split | None
    """Its source chunks' split. None only for questions rejected as cross_split, which belong to neither."""
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

    @model_validator(mode="after")
    def _split_required_unless_rejected(self) -> "Question":
        if self.split is None and self.status != QuestionStatus.REJECTED:
            raise ValueError("only rejected questions may have no split")
        return self

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
        """Build a Question with its id and split derived the one canonical way.

        A question whose source chunks span both splits is always rejected
        as cross_split, whatever status the caller asked for.
        """
        question_id = compute_question_id(chunk_hashes, fact)
        split = split_for_chunks(chunk_ids)
        if split is None:
            kwargs["status"] = QuestionStatus.REJECTED
            kwargs["reject_reason"] = RejectReason.CROSS_SPLIT.value
        question_type = QuestionType.MULTI_HOP if len(chunk_ids) > 1 else QuestionType.SINGLE
        return cls(
            question_id=question_id,
            chunk_ids=chunk_ids,
            chunk_hashes=chunk_hashes,
            fact=fact,
            question=question,
            gold_answer=gold_answer,
            question_type=kwargs.pop("question_type", question_type),
            split=split,
            generator_model=generator_model,
            **kwargs,
        )
