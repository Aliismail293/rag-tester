"""A deliberately mediocre keyword/BM25 RAG pipeline for the ragaudit demo.

Retrieval is naive term-overlap BM25 over the chunks ragaudit already
ingested — no embeddings, no vector database, and only the single
best-scoring chunk is retrieved. It is meant to miss sometimes; that is
the point of the demo.

Chunks are read straight out of the ragaudit database (rather than
re-walking and re-chunking the corpus here) so the chunk_ids this
adapter returns are guaranteed to be exactly the chunk_ids `ragaudit
ingest` assigned — see the README's note on why that match matters for
retrieval scoring.
"""

import math
import os
import re
from collections import Counter

from ragaudit.llm.client import LLMClient
from ragaudit.models.adapter import RAGResponse
from ragaudit.models.document import Chunk
from ragaudit.storage.db import get_connection
from ragaudit.storage.repository import load_chunks

DB_PATH = os.environ.get("RAGAUDIT_DB_PATH", "ragaudit.db")
TOP_K = 1  # deliberately mediocre: only the single best-scoring chunk is retrieved

ANSWER_SYSTEM_PROMPT = (
    "Answer the user's question using only the provided context. If the "
    "context does not contain the answer, say you don't know rather than "
    "guessing."
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class _BM25Index:
    """A minimal, dependency-free BM25 index over a fixed list of chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self._chunks = chunks
        self._k1 = k1
        self._b = b
        self._doc_tokens = [_tokenize(chunk.text) for chunk in chunks]
        self._doc_len = [len(tokens) for tokens in self._doc_tokens]
        self._avg_doc_len = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0
        self._term_freqs = [Counter(tokens) for tokens in self._doc_tokens]
        self._doc_freq: Counter = Counter()
        for tokens in self._doc_tokens:
            for term in set(tokens):
                self._doc_freq[term] += 1
        self._n_docs = len(chunks)

    def _idf(self, term: str) -> float:
        df = self._doc_freq.get(term, 0)
        return math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1)

    def top_k(self, query: str, k: int) -> list[Chunk]:
        """Return the k highest-scoring chunks for `query`, best first."""
        if not self._chunks:
            return []

        query_terms = _tokenize(query)
        scores = []
        for i in range(self._n_docs):
            freqs = self._term_freqs[i]
            doc_len = self._doc_len[i]
            score = 0.0
            for term in query_terms:
                freq = freqs.get(term, 0)
                if freq == 0:
                    continue
                idf = self._idf(term)
                length_norm = 1 - self._b + self._b * (doc_len / self._avg_doc_len if self._avg_doc_len else 1)
                score += idf * (freq * (self._k1 + 1)) / (freq + self._k1 * length_norm)
            scores.append((score, i))

        scores.sort(key=lambda pair: pair[0], reverse=True)
        return [self._chunks[i] for _, i in scores[:k]]


def _load_index() -> _BM25Index:
    conn = get_connection(DB_PATH)
    try:
        chunks = load_chunks(conn)
    finally:
        conn.close()
    return _BM25Index(chunks)


_index: _BM25Index | None = None
_client: LLMClient | None = None


def _get_index() -> _BM25Index:
    global _index
    if _index is None:
        _index = _load_index()
    return _index


def _get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def query(question: str) -> RAGResponse:
    """Answer `question` using naive top-1 BM25 retrieval over ingested chunks."""
    top_chunks = _get_index().top_k(question, TOP_K)
    contexts = [chunk.text for chunk in top_chunks]
    retrieved_chunk_ids = [chunk.chunk_id for chunk in top_chunks]

    context_block = "\n---\n".join(contexts) if contexts else "(no context retrieved)"
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}"
    answer = _get_client().complete(ANSWER_SYSTEM_PROMPT, user_prompt)

    return RAGResponse(answer=answer, retrieved_chunk_ids=retrieved_chunk_ids, contexts=contexts)
