from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class QaCase(BaseModel):
    id: str
    question: str
    relevant_doc_id: str
    expected_answer: str


class RagConfig(BaseModel):
    corpus_paths: list[Path]
    chunk_size: int = 200
    overlap: int = 40
    top_k: int = 3
    generation_model: str = "openai/gpt-4o-mini"
    judge_model: str = "openai/gpt-4o-mini"
    use_reranker: bool = False
    rerank_candidates: int = 10  # how many candidates the initial retrieval fetches before reranking down to top_k
    retrieval_mode: str = "dense"  # "dense", "hybrid"
    cases: list[QaCase]
