from __future__ import annotations

import uuid

from pydantic import BaseModel

from common.model_client import ModelClient
from rag_bench.chunking import chunk_text
from rag_bench.config import RagConfig
from rag_bench.faithfulness import score_faithfulness
from rag_bench.hybrid import BM25Index, reciprocal_rank_fusion
from rag_bench.reranker import rerank
from rag_bench.scoring import score_answer
from rag_bench.store import RetrievedChunk, VectorStore

GENERATION_SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context. If the context doesn't contain "
    "the answer, say so. Be concise."
)


class PipelineResult(BaseModel):
    case_id: str
    answer: str
    retrieved_chunk_ids: list[str]
    faithfulness_score: float
    answer_score: float


def _build_store_and_corpus(config: RagConfig) -> tuple[VectorStore, list[str], list[str]]:
    store = VectorStore(collection_name=f"rag-bench-{uuid.uuid4().hex[:8]}")
    ids, documents = [], []
    for path in config.corpus_paths:
        text = path.read_text()
        for i, chunk in enumerate(chunk_text(text, config.chunk_size, config.overlap)):
            ids.append(f"{path.stem}-{i}")
            documents.append(chunk)
    store.add(ids=ids, documents=documents)
    return store, ids, documents


def run_pipeline(client: ModelClient, config: RagConfig) -> list[PipelineResult]:
    store, corpus_ids, corpus_documents = _build_store_and_corpus(config)
    bm25_index = BM25Index(corpus_ids, corpus_documents) if config.retrieval_mode == "hybrid" else None
    doc_by_id = dict(zip(corpus_ids, corpus_documents))
    results = []

    for case in config.cases:
        fetch_k = config.rerank_candidates if config.use_reranker else config.top_k

        if config.retrieval_mode == "hybrid":
            dense_hits = store.query(case.question, top_k=fetch_k)
            dense_ids = [h.id for h in dense_hits]
            sparse_ids = bm25_index.query(case.question, top_k=fetch_k)
            fused_ids = reciprocal_rank_fusion([dense_ids, sparse_ids], top_k=fetch_k)
            retrieved = [RetrievedChunk(id=doc_id, document=doc_by_id[doc_id], distance=0.0) for doc_id in fused_ids]
        else:
            retrieved = store.query(case.question, top_k=fetch_k)

        if config.use_reranker:
            retrieved = rerank(case.question, retrieved, top_k=config.top_k)
        context = [r.document for r in retrieved]
        context_block = "\n---\n".join(context)

        prompt = f"Context:\n{context_block}\n\nQuestion: {case.question}"
        response = client.complete(model=config.generation_model, prompt=prompt, system=GENERATION_SYSTEM_PROMPT, temperature=0.0)

        faithfulness = score_faithfulness(client, answer=response.text, context=context, judge_model=config.judge_model)
        answer_score = score_answer(client, answer=response.text, expected=case.expected_answer, judge_model=config.judge_model)

        results.append(
            PipelineResult(
                case_id=case.id,
                answer=response.text,
                retrieved_chunk_ids=[r.id for r in retrieved],
                faithfulness_score=faithfulness,
                answer_score=answer_score,
            )
        )

    return results
