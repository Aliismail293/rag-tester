import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragaudit.cli import app
from ragaudit.generate import prompts
from ragaudit.generate.pipeline import (
    GenerateOptions,
    candidate_pairs,
    generate_questions,
    parse_facts_response,
    parse_question_response,
    parse_verify_response,
)
from ragaudit.generate.text import find_relative_time, has_relative_time, lexical_overlap
from ragaudit.ingest.corpus import load_corpus
from ragaudit.llm.factory import CallBudget
from ragaudit.llm.mock import MockLLM
from ragaudit.models.document import Chunk
from ragaudit.models.question import QuestionStatus, QuestionType, Split
from ragaudit.storage.lock import _unlock_test_split
from ragaudit.storage.store import Store

DEMO_CORPUS = Path(__file__).resolve().parent.parent / "demo" / "v2" / "corpus.jsonl"

PASSAGE = "The Zentrix vault in Pellow stores 412 golden spoons and opens every Thursday."


class ScriptedLLM:
    """Routes each system prompt to a scripted handler; records every call."""

    def __init__(self, model: str, handlers: dict):
        self.model = model
        self.handlers = handlers
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        handler = self.handlers[system]
        return handler(user) if callable(handler) else handler

    def count(self, system: str) -> int:
        return sum(1 for s, _ in self.calls if s == system)


def generator_with(dewords: list[str], facts=None, question="How many golden spoons are in the Zentrix vault?"):
    facts = facts if facts is not None else [{"fact": "The Zentrix vault stores 412 golden spoons.", "answer": "412"}]
    remaining = list(dewords)
    return ScriptedLLM(
        "gen-model",
        {
            prompts.FACTS_SYSTEM_PROMPT: json.dumps({"facts": facts}),
            prompts.QUESTION_SYSTEM_PROMPT: json.dumps({"question": question}),
            prompts.DEWORD_SYSTEM_PROMPT: lambda user: json.dumps({"question": remaining.pop(0)}),
        },
    )


def verifier_with(verdict: dict | str):
    raw = verdict if isinstance(verdict, str) else json.dumps(verdict)
    return ScriptedLLM("verifier-model", {prompts.VERIFY_SYSTEM_PROMPT: raw})


OK_VERDICT = {"supported": True, "answerable": True, "unambiguous": True, "reason": "ok"}
LOW_OVERLAP = "Roughly how much cutlery is kept in that Pellow strongroom, the Zentrix one?"
HIGH_OVERLAP = "How many golden spoons does the Zentrix vault in Pellow store?"


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "store.db") as s:
        yield s


def chunk(chunk_id="c1", text=PASSAGE, doc_id="d1", position=0, timestamp=None) -> Chunk:
    metadata = {"timestamp": timestamp} if timestamp else {}
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, position=position, text=text, metadata=metadata)


def all_questions(store: Store):
    with _unlock_test_split():
        return store.load_questions(split=None, statuses=list(QuestionStatus))


# --- text helpers ---


def test_lexical_overlap_counts_content_words_found_in_the_chunk():
    assert lexical_overlap(HIGH_OVERLAP, PASSAGE) == 1.0
    assert lexical_overlap(LOW_OVERLAP, PASSAGE) < 0.5
    assert lexical_overlap("What is it?", PASSAGE) == 0.0


@pytest.mark.parametrize(
    "text",
    [
        "What happened yesterday at the depot?",
        "Who is the current chair?",
        "What did the board approve last week?",
        "How many trams were ordered recently?",
        "What was the budget two years ago?",
        "What is the latest fare?",
        "Which line opened this year?",
    ],
)
def test_relative_time_is_detected(text):
    assert has_relative_time(text)


@pytest.mark.parametrize(
    "text",
    [
        "What happened on 3 March 2025 at the depot?",
        "Who chaired the board in 2021?",
        "How many trams were ordered in May 2027?",
        "Which line is the newest?",
    ],
)
def test_absolute_time_is_not_flagged(text):
    assert find_relative_time(text) is None


def test_question_prompt_forbids_relative_time_and_asks_for_absolute_time():
    lowered = prompts.QUESTION_SYSTEM_PROMPT.lower()
    for word in ["yesterday", "today", "last week", "recently"]:
        assert word in lowered
    assert "absolute dates" in lowered


# --- parsing ---


def test_parsers_tolerate_fences_and_reject_bad_shapes():
    assert parse_facts_response('```json\n{"facts": [{"fact": "f", "answer": "a"}]}\n```') == [
        {"fact": "f", "answer": "a"}
    ]
    assert parse_facts_response("nope") is None
    assert parse_question_response('{"question": "Q?"}') == "Q?"
    assert parse_question_response('"Bare question?"') == "Bare question?"
    assert parse_question_response("{broken") is None
    assert parse_verify_response('{"supported": true, "answerable": true}') is None


# --- the per-question pipeline ---


def test_low_overlap_question_is_verified_and_activated(store):
    gen, ver = generator_with([LOW_OVERLAP]), verifier_with(OK_VERDICT)
    summary = generate_questions(store, [chunk()], gen, ver)

    [question] = all_questions(store)
    assert question.status == QuestionStatus.ACTIVE
    assert question.question == LOW_OVERLAP
    assert question.gold_answer == "412"
    assert question.generator_model == "gen-model" and question.verifier_model == "verifier-model"
    assert question.chunk_ids == ["c1"] and question.question_type == QuestionType.SINGLE
    assert summary.generated == 1 and summary.rejected == 0
    assert ver.count(prompts.VERIFY_SYSTEM_PROMPT) == 1
    assert gen.count(prompts.VERIFY_SYSTEM_PROMPT) == 0


def test_too_lexical_rewrite_is_retried_once_then_accepted(store):
    gen = generator_with([HIGH_OVERLAP, LOW_OVERLAP])
    generate_questions(store, [chunk()], gen, verifier_with(OK_VERDICT))

    [question] = all_questions(store)
    assert question.status == QuestionStatus.ACTIVE
    assert gen.count(prompts.DEWORD_SYSTEM_PROMPT) == 2
    retry_prompt = [u for s, u in gen.calls if s == prompts.DEWORD_SYSTEM_PROMPT][1]
    assert "Previous rewrite" in retry_prompt


def test_still_too_lexical_after_retry_is_rejected_without_verification(store):
    gen, ver = generator_with([HIGH_OVERLAP, HIGH_OVERLAP]), verifier_with(OK_VERDICT)
    summary = generate_questions(store, [chunk()], gen, ver, GenerateOptions(lexical_threshold=0.5))

    [question] = all_questions(store)
    assert question.status == QuestionStatus.REJECTED
    assert question.reject_reason == "too_lexical"
    assert question.lexical_overlap == 1.0
    assert ver.calls == []
    assert summary.reject_reasons == {"too_lexical": 1}


def test_lexical_threshold_is_configurable(store):
    gen = generator_with([HIGH_OVERLAP])
    generate_questions(store, [chunk()], gen, verifier_with(OK_VERDICT), GenerateOptions(lexical_threshold=1.0))
    assert all_questions(store)[0].status == QuestionStatus.ACTIVE


def test_relative_time_question_is_rejected(store):
    gen = generator_with(["Roughly how much cutlery did that Pellow strongroom hold last week?"])
    generate_questions(store, [chunk()], gen, verifier_with(OK_VERDICT))
    assert all_questions(store)[0].reject_reason == "relative_time"


@pytest.mark.parametrize(
    "verdict, reason",
    [
        ({"supported": False, "answerable": True, "unambiguous": True}, "unsupported"),
        ({"supported": True, "answerable": False, "unambiguous": True}, "unanswerable"),
        ({"supported": True, "answerable": True, "unambiguous": False}, "ambiguous"),
        ("not json", "verifier_unparseable"),
    ],
)
def test_verifier_failures_are_rejected_with_a_reason(store, verdict, reason):
    generate_questions(store, [chunk()], generator_with([LOW_OVERLAP]), verifier_with(verdict))
    [question] = all_questions(store)
    assert question.status == QuestionStatus.REJECTED
    assert question.reject_reason == reason


def test_source_timestamp_comes_from_the_chunk(store):
    generate_questions(store, [chunk(timestamp="2026-02-01T00:00:00+00:00")], generator_with([LOW_OVERLAP]), verifier_with(OK_VERDICT))
    assert all_questions(store)[0].source_timestamp.isoformat() == "2026-02-01T00:00:00+00:00"


# --- incremental and budgets ---


def test_chunks_with_questions_are_skipped_on_rerun(store):
    generate_questions(store, [chunk()], generator_with([LOW_OVERLAP]), verifier_with(OK_VERDICT))
    gen = generator_with([])
    summary = generate_questions(store, [chunk()], gen, verifier_with(OK_VERDICT))
    assert summary.chunks_skipped == 1 and summary.chunks_processed == 0
    assert gen.calls == []


def test_changed_chunk_text_is_processed_again(store):
    generate_questions(store, [chunk()], generator_with([LOW_OVERLAP]), verifier_with(OK_VERDICT))
    edited = chunk(text=PASSAGE.replace("412", "415"))
    summary = generate_questions(store, [edited], generator_with([LOW_OVERLAP], facts=[{"fact": "The Zentrix vault stores 415 golden spoons.", "answer": "415"}]), verifier_with(OK_VERDICT))
    assert summary.chunks_processed == 1


def test_max_chunks_stops_cleanly_and_reports_remaining(store):
    chunks = [chunk(chunk_id=f"c{i}", text=f"{PASSAGE} Batch {i}.") for i in range(3)]
    gen = generator_with([LOW_OVERLAP] * 10)
    summary = generate_questions(store, chunks, gen, verifier_with(OK_VERDICT), GenerateOptions(max_chunks=1))
    assert summary.stopped_reason == "max_chunks"
    assert summary.chunks_processed == 1 and summary.chunks_remaining == 2


def test_max_llm_calls_stops_cleanly_discarding_the_partial_chunk(store):
    chunks = [chunk(chunk_id=f"c{i}", text=f"{PASSAGE} Batch {i}.") for i in range(3)]
    gen = generator_with([LOW_OVERLAP] * 10)
    # One chunk costs 4 calls: facts, question, de-word, verify. 6 calls = one chunk plus a partial one.
    summary = generate_questions(store, chunks, gen, verifier_with(OK_VERDICT), budget=CallBudget(6))
    assert summary.stopped_reason == "max_llm_calls"
    assert summary.chunks_processed == 1 and summary.chunks_remaining == 2
    assert summary.llm_calls == 6
    assert len(all_questions(store)) == 1


# --- multi-hop ---


def test_candidate_pairs_include_adjacent_chunks():
    a = chunk("a", "Alpha depot holds trams.", doc_id="d", position=0)
    b = chunk("b", "Beta yard holds buses.", doc_id="d", position=1)
    c = chunk("c", "Unrelated poem about clouds.", doc_id="e", position=0)
    pairs = candidate_pairs([a, b, c])
    assert [p.chunk_id for p in pairs[0]] == ["a", "b"]


def test_multi_hop_question_lists_both_chunks(store):
    a = chunk("a", "The Zentrix vault in Pellow stores 412 golden spoons.", doc_id="d", position=0)
    b = chunk("b", "The Pellow vault is guarded by Captain Orla Venn.", doc_id="d", position=1)
    gen = ScriptedLLM(
        "gen-model",
        {
            prompts.FACTS_SYSTEM_PROMPT: json.dumps({"facts": []}),
            prompts.MULTI_HOP_SYSTEM_PROMPT: json.dumps(
                {"fact": "f", "question": "Who guards the strongroom with all that cutlery?", "answer": "Orla Venn"}
            ),
            prompts.DEWORD_SYSTEM_PROMPT: json.dumps({"question": "Who keeps watch over the cutlery strongroom?"}),
        },
    )
    summary = generate_questions(store, [a, b], gen, verifier_with(OK_VERDICT), GenerateOptions(multi_hop=True))
    [question] = all_questions(store)
    assert question.question_type == QuestionType.MULTI_HOP
    assert question.chunk_ids == ["a", "b"] and len(question.chunk_hashes) == 2
    assert summary.multi_hop_generated == 1


# --- the offline mock on the demo corpus ---


def test_mock_generation_is_deterministic_and_exercises_every_check(tmp_path):
    chunks = load_corpus(DEMO_CORPUS)
    results = []
    for name in ("a", "b"):
        with Store(tmp_path / f"{name}.db") as s:
            generate_questions(s, chunks, MockLLM("mock-generator"), MockLLM("mock-verifier"))
            results.append(sorted((q.question_id, q.status, q.question) for q in all_questions(s)))
    assert results[0] == results[1]

    statuses = [status for _, status, _ in results[0]]
    assert statuses.count(QuestionStatus.ACTIVE) > len(statuses) / 2
    with Store(tmp_path / "a.db") as s:
        reasons = s.question_counts()["reject_reasons"]
        dev = s.dev_questions()
    assert {"too_lexical", "unsupported"} <= set(reasons)
    assert all(q.split == Split.DEV for q in dev)


# --- corpus loading ---


def test_jsonl_corpus_loads_ids_and_timestamps(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(
        '{"chunk_id": "x", "text": "hello", "timestamp": "2026-01-02"}\n\n{"chunk_id": "y", "text": "bye"}\n'
    )
    chunks = load_corpus(path)
    assert [c.chunk_id for c in chunks] == ["x", "y"]
    assert chunks[0].metadata["timestamp"] == "2026-01-02T00:00:00+00:00"


@pytest.mark.parametrize(
    "content, message",
    [
        ('{"text": "no id"}', "chunk_id"),
        ('{"chunk_id": "x"}', "text"),
        ('{"chunk_id": "x", "text": "a"}\n{"chunk_id": "x", "text": "b"}', "duplicate"),
        ('{"chunk_id": "x", "text": "a", "timestamp": "last tuesday"}', "ISO-8601"),
        ("not json", "invalid JSON"),
    ],
)
def test_bad_jsonl_corpus_raises_clear_errors(tmp_path, content, message):
    from ragaudit.ingest.corpus import CorpusError

    path = tmp_path / "c.jsonl"
    path.write_text(content)
    with pytest.raises(CorpusError, match=message):
        load_corpus(path)


def test_directory_corpus_gets_stable_path_ids(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("first version")
    ids_before = [c.chunk_id for c in load_corpus(tmp_path / "docs")]
    (tmp_path / "docs" / "a.md").write_text("second version")
    ids_after = [c.chunk_id for c in load_corpus(tmp_path / "docs")]
    assert ids_before == ids_after == ["a.md#0"]


# --- CLI ---


runner = CliRunner()


def test_cli_generate_with_corpus_uses_v2_and_mock(tmp_path):
    store_path = tmp_path / "s" / "store.db"
    result = runner.invoke(
        app, ["generate", "--corpus", str(DEMO_CORPUS), "--store", str(store_path), "--llm", "mock", "--max-chunks", "3"]
    )
    assert result.exit_code == 0, result.output
    assert "Rejected by reason" in result.output or "0 rejected" in result.output
    assert "Split of new active questions: dev" in result.output
    assert "Stopped at --max-chunks" in result.output
    assert store_path.exists()


def test_cli_generate_refuses_same_generator_and_verifier(tmp_path):
    result = runner.invoke(
        app,
        [
            "generate", "--corpus", str(DEMO_CORPUS), "--store", str(tmp_path / "s.db"), "--llm", "mock",
            "--generator-model", "same", "--verifier-model", "same",
        ],
    )
    assert result.exit_code != 0
    assert "must differ" in result.output
    assert "Traceback" not in result.output


def test_cli_generate_with_real_llm_requires_a_verifier_model(tmp_path, monkeypatch):
    monkeypatch.delenv("RAGAUDIT_VERIFIER_MODEL", raising=False)
    monkeypatch.delenv("RAGAUDIT_LLM", raising=False)
    result = runner.invoke(app, ["generate", "--corpus", str(DEMO_CORPUS), "--store", str(tmp_path / "s.db")])
    assert result.exit_code != 0
    assert "verifier model is required" in result.output
