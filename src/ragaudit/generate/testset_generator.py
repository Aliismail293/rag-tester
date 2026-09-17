"""Generates a TestSet by prompting an LLM to write questions per chunk."""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Protocol

from ragaudit.models.document import Chunk, Document
from ragaudit.models.testset import TestCase, TestSet

logger = logging.getLogger(__name__)

DEFAULT_QUESTIONS_PER_CHUNK = 1

SYSTEM_PROMPT = (
    "You are generating a test set for a RAG evaluation. Given a single "
    "passage of text, write questions that can be answered ONLY using facts "
    "stated in that passage — not from general knowledge about the topic, "
    "and not answerable by someone who has never seen this passage but "
    "knows the subject well.\n"
    "\n"
    "Phrase every question exactly as a real user would ask it. Never refer "
    "to \"the passage\", \"the document\", \"the text\", \"this section\", "
    "or any other document structure — the reader must not be able to tell "
    "the question was generated from a document. Do not ask meta-questions "
    "about ordering, sections, or headings.\n"
    "\n"
    "If asked for more than one question, each question must target a "
    "different fact from the passage — no two questions may test the same "
    "piece of information.\n"
    "\n"
    "Respond with strict JSON only, no prose, no markdown fences, in the "
    'form: {"questions": [{"question": str, "answer": str}, ...]}'
)


class Completer(Protocol):
    """The subset of LLMClient this module depends on — easy to stub in tests."""

    model: str

    def complete(self, system: str, user: str) -> str: ...


def _build_user_prompt(chunk_text: str, n: int) -> str:
    return f"Write {n} question(s) answerable only from this passage:\n\n{chunk_text}"


def parse_questions_response(raw: str) -> list[dict[str, str]] | None:
    """Parse the LLM's JSON response into a list of {question, answer} dicts.

    Returns None on any malformed input (bad JSON, wrong shape, missing
    fields) instead of raising, so callers can skip and log without
    crashing a large run.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None
    questions = data.get("questions")
    if not isinstance(questions, list):
        return None

    parsed = []
    for item in questions:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        answer = item.get("answer")
        if isinstance(question, str) and isinstance(answer, str) and question and answer:
            parsed.append({"question": question, "answer": answer})
    return parsed if parsed else None


def _test_id(chunk_id: str, index: int) -> str:
    payload = f"{chunk_id}:{index}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_test_cases_for_chunk(
    client: Completer,
    chunk: Chunk,
    n: int = DEFAULT_QUESTIONS_PER_CHUNK,
) -> list[TestCase]:
    """Prompt the LLM for questions on one chunk, returning TestCases.

    Returns an empty list (logging a warning) if the LLM's response can't
    be parsed as the expected JSON shape — never raises.
    """
    raw = client.complete(SYSTEM_PROMPT, _build_user_prompt(chunk.text, n))
    parsed = parse_questions_response(raw)

    if parsed is None:
        logger.warning("Skipping chunk %s: LLM returned unparseable JSON", chunk.chunk_id)
        return []

    now = datetime.now(timezone.utc)
    return [
        TestCase(
            test_id=_test_id(chunk.chunk_id, index),
            question=item["question"],
            expected_answer=item["answer"],
            source_chunk_id=chunk.chunk_id,
            source_doc_id=chunk.doc_id,
            generation_model=client.model,
            created_at=now,
        )
        for index, item in enumerate(parsed)
    ]


def compute_corpus_hash(documents: list[Document]) -> str:
    """Hash over all Document.content_hash values, sorted for order-independence."""
    payload = "|".join(sorted(doc.content_hash for doc in documents))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_test_set(
    client: Completer,
    documents: list[Document],
    chunks: list[Chunk],
    n_per_chunk: int = DEFAULT_QUESTIONS_PER_CHUNK,
    testset_id: str | None = None,
) -> TestSet:
    """Generate a full TestSet by prompting the LLM once per chunk.

    Chunks whose LLM response can't be parsed are skipped and logged; a
    malformed response from one chunk never aborts the rest of the run.
    """
    test_cases: list[TestCase] = []
    for chunk in chunks:
        test_cases.extend(generate_test_cases_for_chunk(client, chunk, n=n_per_chunk))

    corpus_hash = compute_corpus_hash(documents)
    created_at = datetime.now(timezone.utc)
    return TestSet(
        testset_id=testset_id or hashlib.sha256(f"{corpus_hash}:{created_at.isoformat()}".encode()).hexdigest(),
        corpus_hash=corpus_hash,
        created_at=created_at,
        generation_model=client.model,
        test_cases=test_cases,
    )
