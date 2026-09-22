from __future__ import annotations

from rank_bm25 import BM25Okapi


class BM25Index:
    """Sparse (keyword) retrieval, complementary to the dense VectorStore. BM25 wins
    on exact terms (names, ids, numbers) that embeddings often blur together."""

    def __init__(self, ids: list[str], documents: list[str]):
        self._ids = ids
        self._documents = documents
        tokenized = [doc.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized)

    def query(self, query_text: str, top_k: int) -> list[str]:
        scores = self._bm25.get_scores(query_text.lower().split())
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._ids[i] for i in ranked_indices[:top_k]]


def reciprocal_rank_fusion(rankings: list[list[str]], top_k: int, k: int = 60) -> list[str]:
    """Standard RRF: score(id) = sum over rankings of 1/(k + rank). Combines multiple
    ranked lists (e.g. dense + sparse) into one, rewarding ids that rank highly and
    consistently across lists rather than ids that win only one method.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    ranked_ids = sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)
    return ranked_ids[:top_k]
