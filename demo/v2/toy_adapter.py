"""A small, dependency-free RAG pipeline with real knobs, for the ragaudit v2 demo.

`ragaudit evaluate` and `ragaudit tune` build it from a config dict via
`make_adapter(config)`. Knobs:

- chunk_size / chunk_overlap (words): the pipeline re-chunks each corpus
  chunk into windows of this many words. Small windows sharpen keyword
  retrieval but can cut the answer out of the context the generator sees.
- top_k: how many distinct corpus chunks to retrieve.
- hybrid_weight: 0 = pure BM25, 1 = pure character-trigram similarity (a
  stand-in for a dense embedding that tolerates rewording).
- reranker: "none", or "overlap" to re-rank the top candidates by the share
  of query words they contain.
- prompt_variant: "extractive" (quote the sentence that answers) or "lead"
  (reply with the first sentence of the top passage, a lazier prompt).

Retrieval is scored against the *corpus* chunk ids ragaudit generated
questions from, so each window remembers the chunk it came from and the
pipeline returns those ids. Any pipeline that re-chunks must do the same.

The answer step calls an LLM: the offline MockLLM when RAGAUDIT_LLM=mock,
otherwise the OpenAI-compatible client configured by RAGAUDIT_API_*.
"""

import math
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from ragaudit.ingest.corpus import load_corpus
from ragaudit.llm.factory import make_client
from ragaudit.models.adapter import RAGResponse

CORPUS_PATH = Path(os.environ.get("RAGAUDIT_DEMO_CORPUS", Path(__file__).parent / "corpus.jsonl"))

DEFAULT_CONFIG = {
    "chunk_size": 80,
    "chunk_overlap": 0,
    "top_k": 3,
    "hybrid_weight": 0.0,
    "reranker": "none",
    "prompt_variant": "extractive",
}

PROMPTS = {
    "extractive": (
        "Answer the user's question using only the provided context. Reply with the single sentence from "
        "the context that answers it. If the context does not contain the answer, say you don't know."
    ),
    "lead": (
        "Answer the user's question using only the provided context. Reply with the first sentence of the "
        "most relevant passage."
    ),
}

_STOP = set("a an and are as at be by for from how in is it of on or the to was what when which who with".split())


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def _trigrams(text: str) -> Counter:
    padded = f"  {' '.join(re.findall(r'[a-z0-9]+', text.lower()))}  "
    return Counter(padded[i : i + 3] for i in range(len(padded) - 2))


def _cosine(a: Counter, b: Counter) -> float:
    dot = sum(count * b.get(gram, 0) for gram, count in a.items())
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


@lru_cache(maxsize=1)
def _corpus():
    return load_corpus(CORPUS_PATH)


@lru_cache(maxsize=4)
def _client():
    return make_client(mock_model="mock-answerer")


class ToyPipeline:
    def __init__(self, config: dict) -> None:
        unknown = set(config) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"unknown config keys for the toy pipeline: {sorted(unknown)}")
        self.config = {**DEFAULT_CONFIG, **config}
        if self.config["prompt_variant"] not in PROMPTS:
            raise ValueError(f"unknown prompt_variant {self.config['prompt_variant']!r}")
        self._windows = self._build_windows(int(self.config["chunk_size"]), int(self.config["chunk_overlap"]))
        self._tokens = [_tokens(text) for _, text in self._windows]
        self._trigrams = [_trigrams(text) for _, text in self._windows]
        self._doc_freq = Counter(t for tokens in self._tokens for t in set(tokens))
        self._avg_len = sum(len(t) for t in self._tokens) / len(self._tokens)

    @staticmethod
    def _build_windows(size: int, overlap: int) -> list[tuple[str, str]]:
        """(source chunk_id, window text) pairs: each corpus chunk split into windows of `size` words."""
        step = max(1, size - overlap)
        windows = []
        for chunk in _corpus():
            words = chunk.text.split()
            for start in range(0, len(words), step):
                windows.append((chunk.chunk_id, " ".join(words[start : start + size])))
                if start + size >= len(words):
                    break
        return windows

    def _bm25(self, query: list[str], index: int, k1: float = 1.5, b: float = 0.75) -> float:
        tokens = self._tokens[index]
        freqs = Counter(tokens)
        n = len(self._windows)
        score = 0.0
        for term in query:
            if term not in freqs:
                continue
            df = self._doc_freq[term]
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
            norm = 1 - b + b * len(tokens) / self._avg_len
            score += idf * freqs[term] * (k1 + 1) / (freqs[term] + k1 * norm)
        return score

    def retrieve(self, question: str) -> list[tuple[str, str]]:
        """Top-k (chunk_id, best window text) pairs, best first."""
        query = _tokens(question)
        query_grams = _trigrams(question)
        bm25 = [self._bm25(query, i) for i in range(len(self._windows))]
        top_bm25 = max(bm25) or 1.0
        weight = float(self.config["hybrid_weight"])
        scores = [
            (1 - weight) * bm25[i] / top_bm25 + weight * _cosine(query_grams, self._trigrams[i])
            for i in range(len(self._windows))
        ]
        ranked = sorted(range(len(self._windows)), key=lambda i: scores[i], reverse=True)

        top_k = int(self.config["top_k"])
        if self.config["reranker"] == "overlap":
            query_set = set(query)
            candidates = ranked[: top_k * 4]
            ranked = sorted(
                candidates,
                key=lambda i: (len(query_set & set(self._tokens[i])) / (len(query_set) or 1), scores[i]),
                reverse=True,
            )

        picked: dict[str, str] = {}
        for i in ranked:
            chunk_id, text = self._windows[i]
            if chunk_id not in picked:
                picked[chunk_id] = text
            if len(picked) == top_k:
                break
        return list(picked.items())

    def query(self, question: str) -> RAGResponse:
        retrieved = self.retrieve(question)
        contexts = [text for _, text in retrieved]
        context_block = "\n---\n".join(contexts) if contexts else "(no context retrieved)"
        answer = _client().complete(
            PROMPTS[self.config["prompt_variant"]], f"Context:\n{context_block}\n\nQuestion: {question}"
        )
        return RAGResponse(answer=answer, retrieved_chunk_ids=[cid for cid, _ in retrieved], contexts=contexts)


def make_adapter(config: dict) -> ToyPipeline:
    return ToyPipeline(config)
