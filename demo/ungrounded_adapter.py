"""A RAG pipeline that lets the model fall back on its own knowledge.

Same BM25 retrieval as weak_adapter.py, but even more degraded (70% of
the time, seeded, a random chunk is substituted for the true top match)
and with a permissive generation prompt: the retrieved context is
offered as background, not as the only source of truth, and the model
is explicitly told it may answer from its own knowledge if the context
doesn't help.

This is the adapter that actually produces LUCKY_PASS. weak_adapter.py's
"only use the context" instruction suppresses that fallback — a model
told to refuse when context is unhelpful mostly does, which turns bad
retrieval into a visible failure instead of a hidden one. Real
production prompts are rarely that strict, and a model that quietly
falls back on pretraining despite bad retrieval is exactly the failure
mode ragaudit exists to catch. Compare this adapter's report against
weak_adapter.py's on the same corpus: same retrieval degradation, only
the prompt's honesty about "you may already know this" changed.
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
RANDOM_CHUNK_PROBABILITY = 0.7
RANDOM_SEED = 20240517  # fixed: the same test set degrades the same way every run

ANSWER_SYSTEM_PROMPT = (
    "You are answering a user's question. Some background context has been "
    "retrieved and may or may not be relevant — use it if it helps, but if "
    "it doesn't contain the answer, or seems unrelated to the question, "
    "answer from your own knowledge instead. Prioritize giving a correct, "
    "confident answer over refusing."
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
    """BM25 top-1, except 70% of the time (seeded) return a random chunk instead."""
    if index.chunks and _rng.random() < RANDOM_CHUNK_PROBABILITY:
        return [_rng.choice(index.chunks)]
    return index.top_k(question, TOP_K)


def query(question: str) -> RAGResponse:
    """Answer `question` with heavily degraded retrieval and a permissive
    prompt that lets the model answer from its own knowledge when the
    retrieved context doesn't help.
    """
    top_chunks = _retrieve(_get_index(), question)
    contexts = [chunk.text for chunk in top_chunks]
    retrieved_chunk_ids = [chunk.chunk_id for chunk in top_chunks]

    context_block = "\n---\n".join(contexts) if contexts else "(no context retrieved)"
    user_prompt = f"Background context (may or may not be relevant):\n{context_block}\n\nQuestion: {question}"
    answer = _get_client().complete(ANSWER_SYSTEM_PROMPT, user_prompt)

    return RAGResponse(answer=answer, retrieved_chunk_ids=retrieved_chunk_ids, contexts=contexts)
