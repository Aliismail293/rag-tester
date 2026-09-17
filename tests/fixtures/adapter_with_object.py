from ragaudit.models.adapter import RAGResponse


class MyAdapter:
    def query(self, question: str) -> RAGResponse:
        return RAGResponse(answer=f"object answer to: {question}", retrieved_chunk_ids=["c2"], contexts=["ctx2"])


adapter = MyAdapter()
