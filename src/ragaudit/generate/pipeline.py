"""v2 test-set generation: fact -> question -> de-wording -> checks -> independent verification.

Extends the v1 generate package (testset_generator.py is untouched). For
each chunk:

1. the generator extracts atomic facts, each with its gold answer;
2. the generator writes one question per fact, with relative time banned;
3. the generator rewrites the question so it doesn't copy the chunk's
   wording, and lexical overlap is measured: above the threshold the
   rewrite is retried once, then the question is rejected `too_lexical`;
4. a regex rejects any remaining relative-time expression
   (`relative_time`). It runs before verification only because it's free;
5. a *different* verifier model checks the fact and answer are supported
   by the chunk and the question is answerable and unambiguous.

Rejected questions are stored too (status `rejected` plus a reason), so
the report can show what was thrown away and why.

Generation is incremental (chunks that already have active questions, or
were already processed at the same content hash, are skipped) and
budgeted (--max-chunks, --max-llm-calls): when a cap is hit the current
chunk's partial work is discarded, everything finished so far is already
saved, and the summary says how much is left.
"""

import json
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime

from pydantic import BaseModel, Field

from ragaudit.generate import prompts
from ragaudit.generate.text import content_tokens, has_relative_time, jaccard, lexical_overlap
from ragaudit.ingest.corpus import chunk_hash, chunk_timestamp
from ragaudit.llm.factory import BudgetedCompleter, CallBudget, Completer, LLMBudgetExceeded
from ragaudit.models.document import Chunk
from ragaudit.models.question import Question, QuestionStatus, RejectReason, utc_now
from ragaudit.storage.store import Store

DEFAULT_MAX_FACTS_PER_CHUNK = 3
DEFAULT_LEXICAL_THRESHOLD = 0.5
DEFAULT_MULTI_HOP_PAIRS = 10
MULTI_HOP_MIN_SIMILARITY = 0.15
MAX_CHUNKS_FOR_SIMILARITY_PAIRS = 2000
"""Above this, multi-hop only pairs adjacent chunks: all-pairs similarity is quadratic."""


class GenerateOptions(BaseModel):
    max_facts_per_chunk: int = DEFAULT_MAX_FACTS_PER_CHUNK
    lexical_threshold: float = DEFAULT_LEXICAL_THRESHOLD
    multi_hop: bool = False
    multi_hop_pairs: int = DEFAULT_MULTI_HOP_PAIRS
    max_chunks: int | None = None
    retry_processed: bool = False
    """Re-process chunks already attempted at this content hash that yielded no active question."""


class GenerateSummary(BaseModel):
    run_id: str
    created_at: datetime
    generator_model: str
    verifier_model: str
    corpus_chunks: int
    index_version: str | None = None
    corpus_diff: dict[str, int] = Field(default_factory=dict)
    chunks_processed: int = 0
    chunks_skipped: int = 0
    chunks_remaining: int = 0
    chunks_unparseable: int = 0
    pairs_processed: int = 0
    pairs_remaining: int = 0
    generated: int = 0
    """New active questions from this run."""
    multi_hop_generated: int = 0
    rejected: int = 0
    reject_reasons: dict[str, int] = Field(default_factory=dict)
    split_counts: dict[str, int] = Field(default_factory=lambda: {"dev": 0, "test": 0})
    duplicates: int = 0
    """Questions whose id was already in the store (same chunk content and fact)."""
    lexical_overlaps: list[float] = Field(default_factory=list)
    """Final lexical overlap of every question that reached the overlap check this run."""
    lexical_threshold: float = DEFAULT_LEXICAL_THRESHOLD
    llm_calls: int = 0
    stopped_reason: str | None = None
    """None when everything pending was processed; else "max_chunks" or "max_llm_calls"."""


# --- response parsing ------------------------------------------------------


def _parse_json_object(raw: str) -> dict | None:
    text = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_facts_response(raw: str) -> list[dict[str, str]] | None:
    """Parse {"facts": [{"fact", "answer"}]}. None if malformed; malformed entries are dropped."""
    data = _parse_json_object(raw)
    if data is None or not isinstance(data.get("facts"), list):
        return None
    facts = []
    for item in data["facts"]:
        if isinstance(item, dict):
            fact, answer = item.get("fact"), item.get("answer")
            if isinstance(fact, str) and isinstance(answer, str) and fact.strip() and answer.strip():
                facts.append({"fact": fact.strip(), "answer": answer.strip()})
    return facts


def parse_question_response(raw: str) -> str | None:
    """Parse {"question": str}, tolerating a bare question string."""
    data = _parse_json_object(raw)
    if data is not None:
        question = data.get("question")
        return question.strip() if isinstance(question, str) and question.strip() else None
    text = raw.strip().strip('"').strip()
    if not text or text.startswith("{") or "\n" in text:
        return None
    return text


def parse_verify_response(raw: str) -> dict | None:
    data = _parse_json_object(raw)
    if data is None:
        return None
    if not all(isinstance(data.get(key), bool) for key in ("supported", "answerable", "unambiguous")):
        return None
    return data


# --- single steps ----------------------------------------------------------


def _passages(chunks: list[Chunk]) -> list[str]:
    return [chunk.text for chunk in chunks]


def _captured(chunks: list[Chunk]) -> str | None:
    stamps = [ts for ts in (chunk_timestamp(c) for c in chunks) if ts is not None]
    return max(stamps).date().isoformat() if stamps else None


def extract_facts(client: Completer, chunk: Chunk, max_facts: int) -> list[dict[str, str]] | None:
    raw = client.complete(prompts.FACTS_SYSTEM_PROMPT, prompts.build_facts_prompt(chunk.text, max_facts, _captured([chunk])))
    facts = parse_facts_response(raw)
    return facts[:max_facts] if facts is not None else None


def write_question(client: Completer, fact: str, answer: str, chunks: list[Chunk]) -> str | None:
    raw = client.complete(
        prompts.QUESTION_SYSTEM_PROMPT, prompts.build_question_prompt(fact, answer, _passages(chunks), _captured(chunks))
    )
    return parse_question_response(raw)


def deword_question(client: Completer, question: str, chunks: list[Chunk], previous: str | None = None) -> str | None:
    raw = client.complete(prompts.DEWORD_SYSTEM_PROMPT, prompts.build_deword_prompt(question, _passages(chunks), previous))
    return parse_question_response(raw)


def verify_question(verifier: Completer, fact: str, question: str, answer: str, chunks: list[Chunk]) -> str | None:
    """Return a reject reason, or None if the verifier accepts the question."""
    raw = verifier.complete(
        prompts.VERIFY_SYSTEM_PROMPT, prompts.build_verify_prompt(fact, question, answer, _passages(chunks))
    )
    verdict = parse_verify_response(raw)
    if verdict is None:
        return RejectReason.VERIFIER_UNPARSEABLE.value
    if not verdict["supported"]:
        return RejectReason.UNSUPPORTED.value
    if not verdict["answerable"]:
        return RejectReason.UNANSWERABLE.value
    if not verdict["unambiguous"]:
        return RejectReason.AMBIGUOUS.value
    return None


def build_checked_question(
    generator: Completer,
    verifier: Completer,
    chunks: list[Chunk],
    fact: str,
    answer: str,
    draft: str,
    lexical_threshold: float,
) -> tuple[Question, float]:
    """Run de-wording, the overlap check (with one retry), the relative-time check and verification.

    Returns the Question (active or rejected) and its final lexical overlap.
    """
    source_text = "\n\n".join(_passages(chunks))
    question = deword_question(generator, draft, chunks) or draft
    overlap = lexical_overlap(question, source_text)
    if overlap > lexical_threshold:
        retry = deword_question(generator, draft, chunks, previous=question)
        if retry:
            question, overlap = retry, lexical_overlap(retry, source_text)

    if overlap > lexical_threshold:
        reject_reason = RejectReason.TOO_LEXICAL.value
    elif has_relative_time(question):
        reject_reason = RejectReason.RELATIVE_TIME.value
    else:
        reject_reason = verify_question(verifier, fact, question, answer, chunks)

    stamps = [ts for ts in (chunk_timestamp(c) for c in chunks) if ts is not None]
    built = Question.create(
        chunk_ids=[c.chunk_id for c in chunks],
        chunk_hashes=[chunk_hash(c) for c in chunks],
        fact=fact,
        question=question,
        gold_answer=answer,
        generator_model=generator.model,
        verifier_model=verifier.model,
        status=QuestionStatus.REJECTED if reject_reason else QuestionStatus.ACTIVE,
        reject_reason=reject_reason,
        source_timestamp=max(stamps) if stamps else None,
        lexical_overlap=round(overlap, 4),
        metadata={"draft_question": draft},
    )
    return built, overlap


def _questions_for_chunk(
    generator: Completer, verifier: Completer, chunk: Chunk, options: GenerateOptions
) -> list[tuple[Question, float]] | None:
    facts = extract_facts(generator, chunk, options.max_facts_per_chunk)
    if facts is None:
        return None
    results = []
    for item in facts:
        draft = write_question(generator, item["fact"], item["answer"], [chunk])
        if draft is None:
            continue
        results.append(
            build_checked_question(
                generator, verifier, [chunk], item["fact"], item["answer"], draft, options.lexical_threshold
            )
        )
    return results


def _questions_for_pair(
    generator: Completer, verifier: Completer, pair: list[Chunk], options: GenerateOptions
) -> list[tuple[Question, float]] | None:
    raw = generator.complete(
        prompts.MULTI_HOP_SYSTEM_PROMPT, prompts.build_multi_hop_prompt(_passages(pair), _captured(pair))
    )
    data = _parse_json_object(raw)
    if data is None or not all(isinstance(data.get(k), str) and data[k].strip() for k in ("fact", "question", "answer")):
        return None
    return [
        build_checked_question(
            generator,
            verifier,
            pair,
            data["fact"].strip(),
            data["answer"].strip(),
            data["question"].strip(),
            options.lexical_threshold,
        )
    ]


# --- incremental bookkeeping -----------------------------------------------


def source_key(chunks: list[Chunk]) -> str:
    """Identity of a generation source: chunk ids at specific content hashes."""
    return "+".join(sorted(f"{c.chunk_id}@{chunk_hash(c)}" for c in chunks))


def candidate_pairs(chunks: list[Chunk]) -> list[list[Chunk]]:
    """Chunk pairs for multi-hop questions: adjacent chunks of a document, then similar chunks across documents."""
    scored: dict[tuple[str, str], tuple[float, list[Chunk]]] = {}

    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.doc_id].append(chunk)
    for doc_chunks in by_doc.values():
        doc_chunks.sort(key=lambda c: c.position)
        for a, b in zip(doc_chunks, doc_chunks[1:]):
            scored[(a.chunk_id, b.chunk_id)] = (1.0 + jaccard(content_tokens(a.text), content_tokens(b.text)), [a, b])

    if len(chunks) <= MAX_CHUNKS_FOR_SIMILARITY_PAIRS:
        tokens = [content_tokens(c.text) for c in chunks]
        for i, a in enumerate(chunks):
            for j in range(i + 1, len(chunks)):
                b = chunks[j]
                if a.doc_id == b.doc_id:
                    continue
                similarity = jaccard(tokens[i], tokens[j])
                if similarity >= MULTI_HOP_MIN_SIMILARITY:
                    scored[(a.chunk_id, b.chunk_id)] = (similarity, [a, b])

    ranked = sorted(scored.items(), key=lambda item: (-item[1][0], item[0]))
    return [pair for _, (_, pair) in ranked]


def _record(store: Store, summary: GenerateSummary, results: list[tuple[Question, float]], multi_hop: bool) -> None:
    questions = [q for q, _ in results]
    inserted_ids = {q.question_id for q in questions if store.save_questions([q])}
    reasons: Counter = Counter(summary.reject_reasons)
    for question, overlap in results:
        summary.lexical_overlaps.append(round(overlap, 4))
        if question.question_id not in inserted_ids:
            summary.duplicates += 1
            continue
        if question.status == QuestionStatus.ACTIVE:
            summary.generated += 1
            summary.split_counts[question.split.value] += 1
            if multi_hop:
                summary.multi_hop_generated += 1
        else:
            summary.rejected += 1
            reasons[question.reject_reason] += 1
    summary.reject_reasons = dict(reasons)


def _outcome(results: list[tuple[Question, float]]) -> str:
    return "active" if any(q.status == QuestionStatus.ACTIVE for q, _ in results) else "no_active"


# --- orchestration ---------------------------------------------------------


def generate_questions(
    store: Store,
    chunks: list[Chunk],
    generator: Completer,
    verifier: Completer,
    options: GenerateOptions | None = None,
    budget: CallBudget | None = None,
) -> GenerateSummary:
    """Generate, check and verify questions for every chunk that needs them, saving as it goes."""
    options = options or GenerateOptions()
    budget = budget or CallBudget()
    gen = BudgetedCompleter(generator, budget)
    ver = BudgetedCompleter(verifier, budget)
    summary = GenerateSummary(
        run_id=str(uuid.uuid4()),
        created_at=utc_now(),
        generator_model=generator.model,
        verifier_model=verifier.model,
        corpus_chunks=len(chunks),
        lexical_threshold=options.lexical_threshold,
    )

    done = store.active_source_keys()
    if not options.retry_processed:
        done |= store.attempted_source_keys()

    pending = [chunk for chunk in chunks if source_key([chunk]) not in done]
    summary.chunks_skipped = len(chunks) - len(pending)

    for index, chunk in enumerate(pending):
        if options.max_chunks is not None and summary.chunks_processed >= options.max_chunks:
            summary.stopped_reason = "max_chunks"
            summary.chunks_remaining = len(pending) - index
            break
        try:
            results = _questions_for_chunk(gen, ver, chunk, options)
        except LLMBudgetExceeded:
            summary.stopped_reason = "max_llm_calls"
            summary.chunks_remaining = len(pending) - index
            break
        summary.chunks_processed += 1
        if results is None:
            summary.chunks_unparseable += 1
            continue
        _record(store, summary, results, multi_hop=False)
        store.record_generation_attempt(source_key([chunk]), _outcome(results))

    if options.multi_hop:
        pending_pairs = [pair for pair in candidate_pairs(chunks) if source_key(pair) not in done]
        pending_pairs = pending_pairs[: options.multi_hop_pairs]
        if summary.stopped_reason is not None:
            summary.pairs_remaining = len(pending_pairs)
        else:
            for index, pair in enumerate(pending_pairs):
                try:
                    results = _questions_for_pair(gen, ver, pair, options)
                except LLMBudgetExceeded:
                    summary.stopped_reason = "max_llm_calls"
                    summary.pairs_remaining = len(pending_pairs) - index
                    break
                summary.pairs_processed += 1
                if results is None:
                    store.record_generation_attempt(source_key(pair), "no_question")
                    continue
                _record(store, summary, results, multi_hop=True)
                store.record_generation_attempt(source_key(pair), _outcome(results))

    summary.llm_calls = budget.used
    store.save_generate_run(summary.run_id, summary.created_at, summary.model_dump_json())
    return summary
