from __future__ import annotations

from pydantic import BaseModel

from common.model_client import ModelClient
from rag_bench.config import RagConfig
from rag_bench.pipeline import PipelineResult, run_pipeline


class SweepResult(BaseModel):
    chunk_size: int
    top_k: int
    results: list[PipelineResult]

    @property
    def avg_faithfulness(self) -> float:
        return sum(r.faithfulness_score for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def avg_answer_score(self) -> float:
        return sum(r.answer_score for r in self.results) / len(self.results) if self.results else 0.0


def run_sweep(
    client: ModelClient,
    base_config: RagConfig,
    chunk_sizes: list[int],
    top_ks: list[int],
    use_rerankers: list[bool] | None = None,
) -> list[SweepResult]:
    use_rerankers = use_rerankers or [False]
    sweep_results = []
    for chunk_size in chunk_sizes:
        for top_k in top_ks:
            for use_reranker in use_rerankers:
                config = base_config.model_copy(update={"chunk_size": chunk_size, "top_k": top_k, "use_reranker": use_reranker})
                results = run_pipeline(client, config)
                sweep_results.append(SweepResult(chunk_size=chunk_size, top_k=top_k, results=results))
    return sweep_results
