"""`ragaudit init`: starter files for plugging a pipeline into ragaudit.

Each adapter template has exactly one TODO, at the spot where the user
connects their own pipeline; everything around it (building the
RAGResponse, timing, a clear error while the TODO is unfilled) is done.
"""

from pathlib import Path

FRAMEWORKS = ("plain", "langchain", "llamaindex")
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


def adapter_template(framework: str, filename: str) -> str:
    if framework not in FRAMEWORKS:
        raise ValueError(f"--framework must be one of {', '.join(FRAMEWORKS)}, got {framework!r}")
    label = {"plain": "plain Python", "langchain": "LangChain", "llamaindex": "LlamaIndex"}[framework]
    return _HEADER.format(framework=label, filename=filename) + _BODIES[framework].format(filename=filename)


def write_starter_files(framework: str, adapter_path: Path, force: bool = False) -> list[Path]:
    """Write the adapter template and .env.example (next to it). Refuses to overwrite unless force."""
    env_path = adapter_path.parent / ENV_EXAMPLE_NAME
    content = adapter_template(framework, adapter_path.name)
    existing = [path for path in (adapter_path, env_path) if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"{names} already exists; pass --force to overwrite, or --out to pick another file.")
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_path.write_text(content)
    env_path.write_text(ENV_EXAMPLE)
    return [adapter_path, env_path]
