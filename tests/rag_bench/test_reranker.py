from rag_bench.reranker import rerank
from rag_bench.store import RetrievedChunk


def test_rerank_reorders_by_injected_scoring_fn():
    chunks = [
        RetrievedChunk(id="a", document="Mount Everest is a mountain.", distance=0.1),
        RetrievedChunk(id="b", document="The Eiffel Tower is in Paris.", distance=0.2),
    ]

    def fake_score_fn(query: str, documents: list[str]) -> list[float]:
        # pretend the cross-encoder strongly prefers whichever doc mentions "Paris"
        return [1.0 if "Paris" in doc else 0.0 for doc in documents]

    reranked = rerank("Where is the Eiffel Tower?", chunks, score_fn=fake_score_fn, top_k=1)

    assert len(reranked) == 1
    assert reranked[0].id == "b"


def test_rerank_preserves_all_chunks_when_top_k_exceeds_count():
    chunks = [RetrievedChunk(id="a", document="x", distance=0.1)]
    reranked = rerank("q", chunks, score_fn=lambda q, docs: [0.5], top_k=5)
    assert len(reranked) == 1
