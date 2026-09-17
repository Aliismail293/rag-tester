from ragaudit.models.adapter import RAGResponse


def query(question: str) -> RAGResponse:
    return RAGResponse(answer=f"answer to: {question}", retrieved_chunk_ids=["c1"], contexts=["ctx"])
