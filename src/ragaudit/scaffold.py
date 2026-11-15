"""`ragaudit init`: starter files for plugging a pipeline into ragaudit.

Each adapter template has exactly one TODO, at the spot where the user
connects their own pipeline; everything around it (building the
RAGResponse, timing, a clear error while the TODO is unfilled) is done.

The `demo` template is the exception: a complete toy pipeline (keyword
search over the docs folder, answered by the configured LLM) that works
with no edits, for people who want to see a report before wiring up
their own pipeline.
"""

import os
from pathlib import Path

FRAMEWORKS = ("demo", "plain", "langchain", "llamaindex")
DEMO_DOCS_ENV = "RAGAUDIT_DEMO_DOCS"
"""`ragaudit check` sets this so the demo adapter searches the same folder it writes questions from."""
DEFAULT_DEMO_DOCS = Path("docs")
DEFAULT_ADAPTER_PATH = Path("ragaudit_adapter.py")
ENV_EXAMPLE_NAME = ".env.example"

_HEADER = '''"""ragaudit adapter: connects ragaudit to your RAG pipeline ({framework}).

ragaudit calls query(question) once per test question and scores what you return:

- answer:   your pipeline's final answer.
- contexts: the text of each passage your pipeline retrieved, best first.
            `ragaudit check` (and --match-by text) decides whether retrieval
            found the right passage by comparing these texts with the
            passage each question was written from, so your pipeline can
            keep its own chunking and ids.
- retrieved_chunk_ids: only used with --match-by id, when your index was
            built from ragaudit's own chunks. Fine to leave empty.

Check it with:  ragaudit doctor --adapter {filename}
Then run:       ragaudit check --docs ./your-docs --adapter {filename}
"""

import time

from ragaudit.models.adapter import RAGResponse
'''

_PLAIN = '''

def ask_my_pipeline(question: str) -> tuple[str, list[str]]:
    """Return (answer, texts of the retrieved passages) for one question."""
    # TODO: call your pipeline here and return its answer and retrieved passages, e.g.
    #     result = my_rag.ask(question)
    #     return result.answer, [source.text for source in result.sources]
    raise NotImplementedError("{filename} is not filled in yet: connect your pipeline at the TODO in ask_my_pipeline().")


def query(question: str) -> RAGResponse:
    started = time.perf_counter()
    answer, contexts = ask_my_pipeline(question)
    return RAGResponse(
        answer=str(answer),
        contexts=[str(text) for text in contexts],
        latency_ms=(time.perf_counter() - started) * 1000,
    )
'''

_LANGCHAIN = '''
# Your pipeline's retriever and chat model. Any LangChain retriever works
# (vectorstore.as_retriever(), a hybrid or multi-query retriever, ...), and
# any chat model or LLM (ChatOpenAI, ChatOllama, ...).
# TODO: build or import your own retriever and llm here, e.g.
#     from my_app.rag import vectorstore, llm
#     retriever = vectorstore.as_retriever(search_kwargs={{"k": 4}})
retriever = None
llm = None

# The prompt your pipeline uses to answer from the retrieved context. If
# you already have a chain, copy its prompt here so ragaudit tests the
# same thing your users get.
PROMPT = (
    "Answer the question using only the context below. If the context does not "
    "contain the answer, say you don't know.\\n\\nContext:\\n{{context}}\\n\\nQuestion: {{question}}"
)


def query(question: str) -> RAGResponse:
    if retriever is None or llm is None:
        raise NotImplementedError("{filename} is not filled in yet: set `retriever` and `llm` at the TODO.")
    started = time.perf_counter()
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    reply = llm.invoke(PROMPT.format(context="\\n\\n".join(contexts), question=question))
    answer = getattr(reply, "content", reply)  # chat models return a message, plain LLMs a string
    return RAGResponse(
        answer=str(answer),
        contexts=contexts,
        retrieved_chunk_ids=[str(doc.metadata["chunk_id"]) for doc in docs if "chunk_id" in doc.metadata],
        latency_ms=(time.perf_counter() - started) * 1000,
    )
'''

_LLAMAINDEX = '''
# Your pipeline's query engine: index.as_query_engine(), a RetrieverQueryEngine,
# or anything whose .query(question) returns a Response with source_nodes.
# TODO: build or import your own query engine here, e.g.
#     from my_app.rag import index
#     query_engine = index.as_query_engine(similarity_top_k=4)
query_engine = None


def query(question: str) -> RAGResponse:
    if query_engine is None:
        raise NotImplementedError("{filename} is not filled in yet: set `query_engine` at the TODO.")
    started = time.perf_counter()
    response = query_engine.query(question)
    nodes = [scored.node for scored in (response.source_nodes or [])]
    return RAGResponse(
        answer=str(response),
        contexts=[node.get_content() for node in nodes],
        retrieved_chunk_ids=[str(node.metadata["chunk_id"]) for node in nodes if "chunk_id" in node.metadata],
        latency_ms=(time.perf_counter() - started) * 1000,
    )
'''

_DEMO = r'''"""ragaudit demo adapter: a tiny, complete RAG pipeline that works with no edits.

Retrieval: every .txt/.md file under the docs folder is split into
paragraphs, and the 3 paragraphs sharing the most words with the
question are returned. Generation: the configured ragaudit LLM
(RAGAUDIT_API_BASE / RAGAUDIT_API_KEY / RAGAUDIT_MODEL, or --llm mock)
answers from those paragraphs.

It exists to show what a ragaudit report looks like. Keyword search is
deliberately simple, so expect some retrieval failures and lucky passes.
To test your own pipeline: ragaudit init --framework plain --force

Run it:  ragaudit check --docs __DOCS__ --adapter __FILENAME__
"""

import os
import re
import time
from pathlib import Path

from ragaudit.llm.factory import make_client
from ragaudit.models.adapter import RAGResponse

USES_RAGAUDIT_LLM = True  # tells `ragaudit check` that each answer costs one call to its LLM
TOP_K = 3

# `ragaudit check --docs X` points this at X; otherwise the folder below
# (relative to this file) is used.
DOCS = Path(os.environ.get("__DOCS_ENV__") or Path(__file__).parent / "__DOCS__")

SYSTEM_PROMPT = (
    "Answer the question using only the context. If the context does not contain the answer, "
    "say you don't know. Answer in one or two sentences."
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for", "from", "how", "in", "is", "it",
    "of", "on", "or", "that", "the", "this", "to", "was", "what", "when", "where", "which", "who",
    "why", "with",
}

_paragraphs: list[str] | None = None
_llm = None


def words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOPWORDS}


def paragraphs() -> list[str]:
    """Every paragraph of every .txt/.md file under DOCS, read once."""
    global _paragraphs
    if _paragraphs is None:
        if not DOCS.is_dir():
            raise FileNotFoundError(f"docs folder not found: {DOCS}. Put your .txt/.md files there.")
        files = sorted(p for p in DOCS.rglob("*") if p.suffix.lower() in (".txt", ".md") and p.is_file())
        _paragraphs = [
            " ".join(block.split())
            for path in files
            for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8", errors="replace"))
            if block.strip()
        ]
    return _paragraphs


def retrieve(question: str, k: int = TOP_K) -> list[str]:
    asked = words(question)
    scored = [(len(asked & words(text)), -i, text) for i, text in enumerate(paragraphs())]
    return [text for _, _, text in sorted(scored, reverse=True)[:k]]  # like any search: always the top k


def query(question: str) -> RAGResponse:
    global _llm
    started = time.perf_counter()
    contexts = retrieve(question)
    if _llm is None:
        _llm = make_client(mock_model="mock-answerer")
    context_block = "\n---\n".join(contexts) or "(no context retrieved)"
    answer = _llm.complete(SYSTEM_PROMPT, f"Context:\n{context_block}\n\nQuestion: {question}")
    return RAGResponse(answer=answer.strip(), contexts=contexts, latency_ms=(time.perf_counter() - started) * 1000)
'''

_BODIES = {"plain": _PLAIN, "langchain": _LANGCHAIN, "llamaindex": _LLAMAINDEX}

ENV_EXAMPLE = """# ragaudit reads these settings from the environment, or from a .env file in
# the folder you run it from. Copy this file to .env and keep ONE block:
#     cp .env.example .env
# .env holds your API key: keep it out of git.

# Option A: OpenAI, or any OpenAI-compatible API (Azure, Together, Groq, vLLM, ...).
RAGAUDIT_API_BASE=https://api.openai.com/v1
RAGAUDIT_API_KEY=sk-your-key-here
RAGAUDIT_MODEL=gpt-4o-mini

# Option B: local Ollama. Free, and nothing leaves your machine.
# Install Ollama, run `ollama pull llama3.1:8b`, then use these instead of Option A:
# RAGAUDIT_API_BASE=http://localhost:11434/v1
# RAGAUDIT_API_KEY=ollama
# RAGAUDIT_MODEL=llama3.1:8b
"""


def adapter_template(framework: str, filename: str, docs: str = str(DEFAULT_DEMO_DOCS)) -> str:
    """The adapter file's text. `docs` (demo only) is the docs folder, relative to the adapter's folder."""
    if framework not in FRAMEWORKS:
        raise ValueError(f"--framework must be one of {', '.join(FRAMEWORKS)}, got {framework!r}")
    if framework == "demo":
        return (
            _DEMO.replace("__DOCS_ENV__", DEMO_DOCS_ENV)
            .replace("__FILENAME__", filename)
            .replace("__DOCS__", Path(docs).as_posix())
        )
    label = {"plain": "plain Python", "langchain": "LangChain", "llamaindex": "LlamaIndex"}[framework]
    return _HEADER.format(framework=label, filename=filename) + _BODIES[framework].format(filename=filename)


def write_starter_files(
    framework: str, adapter_path: Path, force: bool = False, docs: Path = DEFAULT_DEMO_DOCS
) -> list[Path]:
    """Write the adapter template and .env.example (next to it). Refuses to overwrite unless force."""
    env_path = adapter_path.parent / ENV_EXAMPLE_NAME
    docs_from_adapter = os.path.relpath(docs.resolve(), adapter_path.parent.resolve())
    content = adapter_template(framework, adapter_path.name, docs_from_adapter)
    existing = [path for path in (adapter_path, env_path) if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"{names} already exists; pass --force to overwrite, or --out to pick another file.")
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_path.write_text(content)
    env_path.write_text(ENV_EXAMPLE)
    return [adapter_path, env_path]
