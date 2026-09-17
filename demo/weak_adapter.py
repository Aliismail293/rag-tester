"""A degraded RAG pipeline for the ragaudit demo.

Same BM25 retrieval as example_adapter.py, but 30% of the time (seeded,
so the same test set always fails the same way) it substitutes a
uniformly random chunk in place of the true top match. This simulates a
retrieval index that has quietly gone stale — a bad reindex, a partial
embedding backfill, a broken filter — without changing anything about
how answers are generated. Compare its report against
example_adapter.py's to see retrieval quality alone move the verdict
breakdown.
"""

import math
import os
import random
import re
from collections import Counter

from ragaudit.llm.client import LLMClient
from ragaudit.models.adapter import RAGResponse
from ragaudit.models.document import Chunk
from ragaudit.storage.db import get_connection
from ragaudit.storage.repository import load_chunks

DB_PATH = os.environ.get("RAGAUDIT_DB_PATH", "ragaudit.db")
TOP_K = 1
RANDOM_CHUNK_PROBABILITY = 0.3
RANDOM_SEED = 20240517  # fixed: the same test set degrades the same way every run

ANSWER_SYSTEM_PROMPT = (
    "Answer the user's question using only the provided context. If the "
    "context does not contain the answer, say you don't know rather than "
    "guessing."
)

_rng = random.Random(RANDOM_SEED)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class _BM25Index:
    """A minimal, dependency-free BM25 index over a fixed list of chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
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
        if not self.chunks:
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
        return [self.chunks[i] for _, i in scores[:k]]


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


def _retrieve(index: _BM25Index, question: str) -> list[Chunk]:
    """BM25 top-1, except 30% of the time (seeded) return a random chunk instead."""
    if index.chunks and _rng.random() < RANDOM_CHUNK_PROBABILITY:
        return [_rng.choice(index.chunks)]
    return index.top_k(question, TOP_K)


def query(question: str) -> RAGResponse:
    """Answer `question` using BM25 retrieval, degraded by a seeded 30% chance
    of substituting a uniformly random chunk for the true top match.
    """
    top_chunks = _retrieve(_get_index(), question)
    contexts = [chunk.text for chunk in top_chunks]
    retrieved_chunk_ids = [chunk.chunk_id for chunk in top_chunks]

    context_block = "\n---\n".join(contexts) if contexts else "(no context retrieved)"
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}"
    answer = _get_client().complete(ANSWER_SYSTEM_PROMPT, user_prompt)

    return RAGResponse(answer=answer, retrieved_chunk_ids=retrieved_chunk_ids, contexts=contexts)
