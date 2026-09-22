from __future__ import annotations

from typing import Callable

from rag_bench.store import RetrievedChunk

ScoreFn = Callable[[str, list[str]], list[float]]

_cross_encoder = None


def default_score_fn(query: str, documents: list[str]) -> list[float]:
    """Lazily loads a cross-encoder so importing this module doesn't force a
    ~100MB model download for callers who only use `rerank` with a fake `score_fn`
    (as every test in this repo does)."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, doc) for doc in documents]
    return list(_cross_encoder.predict(pairs))


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int, score_fn: ScoreFn = default_score_fn) -> list[RetrievedChunk]:
    """Re-scores the initially-retrieved chunks with a (typically more accurate,
    more expensive) cross-encoder and returns the top_k by that score.

    This is the standard two-stage retrieval pattern: a fast dense/sparse retriever
    gets a wide candidate set, a slower cross-encoder reranks it down to what's
    actually shown to the generation model.
    """
    if not chunks:
        return []
    scores = score_fn(query, [c.document for c in chunks])
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_k]]
