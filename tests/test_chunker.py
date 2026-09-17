from ragaudit.ingest.chunker import chunk_document
from ragaudit.models.document import Document


def make_document(content: str, doc_id: str = "doc1") -> Document:
    return Document(doc_id=doc_id, source_path="test.txt", content=content, content_hash="irrelevant")


def test_determinism_same_doc_chunked_twice_matches():
    document = make_document("abcdefgh" * 200)
    chunks_a = chunk_document(document, chunk_size=800, overlap=100)
    chunks_b = chunk_document(document, chunk_size=800, overlap=100)
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]
    assert [c.text for c in chunks_a] == [c.text for c in chunks_b]


def test_overlap_correctness():
    document = make_document("x" * 2500)
    chunks = chunk_document(document, chunk_size=800, overlap=100)
    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        current, following = chunks[i], chunks[i + 1]
        if len(current.text) == 800:
            assert current.text[-100:] == following.text[:100]


def test_document_shorter_than_one_chunk_yields_single_chunk():
    document = make_document("short document")
    chunks = chunk_document(document, chunk_size=800, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].text == "short document"
    assert chunks[0].position == 0


def test_empty_document_yields_no_chunks():
    document = make_document("")
    chunks = chunk_document(document, chunk_size=800, overlap=100)
    assert chunks == []


def test_positions_are_zero_indexed_and_contiguous():
    document = make_document("y" * 3000)
    chunks = chunk_document(document, chunk_size=800, overlap=100)
    assert [c.position for c in chunks] == list(range(len(chunks)))
