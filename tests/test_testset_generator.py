from ragaudit.generate.testset_generator import (
    SYSTEM_PROMPT,
    compute_corpus_hash,
    generate_test_cases_for_chunk,
    generate_test_set,
    parse_questions_response,
)
from ragaudit.models.document import Chunk, Document


class StubClient:
    """A fake Completer that returns pre-scripted responses, no network."""

    def __init__(self, responses: list[str], model: str = "stub-model"):
        self._responses = list(responses)
        self.model = model
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._responses.pop(0)


def make_chunk(chunk_id: str = "chunk1", doc_id: str = "doc1", text: str = "some passage") -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, position=0, text=text)


def make_document(doc_id: str, content_hash: str) -> Document:
    return Document(doc_id=doc_id, source_path=f"{doc_id}.txt", content="x", content_hash=content_hash)


# --- prompt content ---


def test_system_prompt_forbids_document_referential_phrasing():
    lowered = SYSTEM_PROMPT.lower()

    ban_sentence_start = lowered.index('never refer to')
    ban_sentence_end = lowered.index("headings.")
    ban_clause = lowered[ban_sentence_start:ban_sentence_end]

    for phrase in ['"the passage"', '"the document"', '"the text"', '"this section"']:
        assert phrase in ban_clause, f"{phrase} should be explicitly forbidden"

    assert "real user would ask" in lowered
    assert "must not be able to tell the question was generated from a document" in lowered
    assert "meta-questions about ordering, sections, or headings" in lowered
    assert "different fact" in lowered
    assert "general knowledge" in lowered


# --- JSON parsing ---


def test_parse_valid_json_response():
    raw = '{"questions": [{"question": "What is X?", "answer": "X is Y"}]}'
    parsed = parse_questions_response(raw)
    assert parsed == [{"question": "What is X?", "answer": "X is Y"}]


def test_parse_multiple_questions():
    raw = '{"questions": [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]}'
    parsed = parse_questions_response(raw)
    assert len(parsed) == 2


def test_parse_malformed_json_returns_none():
    assert parse_questions_response("not json at all {{{") is None


def test_parse_wrong_shape_returns_none():
    assert parse_questions_response('{"foo": "bar"}') is None
    assert parse_questions_response('["a", "list", "not", "a", "dict"]') is None
    assert parse_questions_response('{"questions": "not a list"}') is None


def test_parse_skips_malformed_entries_but_keeps_valid_ones():
    raw = '{"questions": [{"question": "Q1", "answer": "A1"}, {"question": "missing answer"}, "not a dict"]}'
    parsed = parse_questions_response(raw)
    assert parsed == [{"question": "Q1", "answer": "A1"}]


def test_parse_empty_questions_list_returns_none():
    assert parse_questions_response('{"questions": []}') is None


# --- TestCase assembly ---


def test_generate_test_cases_for_chunk_builds_test_cases():
    client = StubClient(['{"questions": [{"question": "What is X?", "answer": "X is Y"}]}'])
    chunk = make_chunk(chunk_id="chunk1", doc_id="doc1")

    test_cases = generate_test_cases_for_chunk(client, chunk)

    assert len(test_cases) == 1
    case = test_cases[0]
    assert case.question == "What is X?"
    assert case.expected_answer == "X is Y"
    assert case.source_chunk_id == "chunk1"
    assert case.source_doc_id == "doc1"
    assert case.generation_model == "stub-model"


def test_generate_test_cases_for_chunk_returns_empty_on_malformed_json():
    client = StubClient(["garbage, not json"])
    chunk = make_chunk()

    test_cases = generate_test_cases_for_chunk(client, chunk)

    assert test_cases == []


# --- corpus hash ---


def test_corpus_hash_is_order_independent():
    docs_a = [make_document("d1", "hashA"), make_document("d2", "hashB")]
    docs_b = [make_document("d2", "hashB"), make_document("d1", "hashA")]
    assert compute_corpus_hash(docs_a) == compute_corpus_hash(docs_b)


def test_corpus_hash_changes_when_content_changes():
    docs_a = [make_document("d1", "hashA")]
    docs_b = [make_document("d1", "hashDifferent")]
    assert compute_corpus_hash(docs_a) != compute_corpus_hash(docs_b)


# --- full test set generation, including skip-on-error ---


def test_generate_test_set_skips_malformed_chunk_without_crashing():
    client = StubClient(
        [
            '{"questions": [{"question": "Q1", "answer": "A1"}]}',
            "totally malformed {{{",
            '{"questions": [{"question": "Q3", "answer": "A3"}]}',
        ]
    )
    documents = [make_document("doc1", "hash1")]
    chunks = [
        make_chunk(chunk_id="c1", doc_id="doc1"),
        make_chunk(chunk_id="c2", doc_id="doc1"),
        make_chunk(chunk_id="c3", doc_id="doc1"),
    ]

    test_set = generate_test_set(client, documents, chunks)

    assert len(test_set.test_cases) == 2
    assert [c.source_chunk_id for c in test_set.test_cases] == ["c1", "c3"]
    assert test_set.corpus_hash == compute_corpus_hash(documents)
    assert test_set.generation_model == "stub-model"
