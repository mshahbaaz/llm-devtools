from pathlib import Path
from unittest.mock import MagicMock, patch

from common.model_client import ModelResponse
from rag_bench.config import RagConfig, QaCase
from rag_bench.pipeline import run_pipeline


def test_run_pipeline_returns_result_per_case(tmp_path: Path):
    corpus_file = tmp_path / "doc.txt"
    corpus_file.write_text("The Eiffel Tower is located in Paris, France. It was completed in 1889.")

    config = RagConfig(
        corpus_paths=[corpus_file],
        chunk_size=10,
        overlap=2,
        top_k=2,
        generation_model="openai/gpt-4o-mini",
        judge_model="openai/gpt-4o-mini",
        cases=[QaCase(id="q1", question="Where is the Eiffel Tower?", relevant_doc_id="", expected_answer="Paris")],
    )

    mock_client = MagicMock()
    mock_client.complete.side_effect = [
        ModelResponse(text="The Eiffel Tower is in Paris.", model="openai/gpt-4o-mini", prompt_tokens=50, completion_tokens=10, latency_seconds=0.4),
        ModelResponse(text="VERDICT: grounded\nREASON: matches context.", model="openai/gpt-4o-mini", prompt_tokens=50, completion_tokens=10, latency_seconds=0.3),
        ModelResponse(text="SCORE: 1.0\nREASON: correct.", model="openai/gpt-4o-mini", prompt_tokens=50, completion_tokens=10, latency_seconds=0.3),
    ]

    results = run_pipeline(mock_client, config)

    assert len(results) == 1
    assert results[0].case_id == "q1"
    assert results[0].faithfulness_score == 1.0
    assert results[0].answer_score == 1.0


def test_run_pipeline_applies_reranker_when_enabled(tmp_path):
    corpus_file = tmp_path / "doc.txt"
    corpus_file.write_text("The Eiffel Tower is located in Paris, France. It was completed in 1889. Mount Everest is tall.")

    config = RagConfig(
        corpus_paths=[corpus_file],
        chunk_size=10,
        overlap=2,
        top_k=1,
        use_reranker=True,
        rerank_candidates=5,
        generation_model="openai/gpt-4o-mini",
        judge_model="openai/gpt-4o-mini",
        cases=[QaCase(id="q1", question="Where is the Eiffel Tower?", relevant_doc_id="", expected_answer="Paris")],
    )
    mock_client = MagicMock()
    mock_client.complete.side_effect = [
        ModelResponse(text="Paris.", model="m", prompt_tokens=1, completion_tokens=1, latency_seconds=0.1),
        ModelResponse(text="VERDICT: grounded\nREASON: ok.", model="m", prompt_tokens=1, completion_tokens=1, latency_seconds=0.1),
        ModelResponse(text="SCORE: 1.0\nREASON: ok.", model="m", prompt_tokens=1, completion_tokens=1, latency_seconds=0.1),
    ]

    # Avoid downloading a real cross-encoder model in tests: patch rerank's
    # default scoring behavior with a stand-in that just keeps the input order.
    with patch("rag_bench.pipeline.rerank") as mock_rerank:
        mock_rerank.side_effect = lambda query, chunks, top_k: chunks[:top_k]
        results = run_pipeline(mock_client, config)

    assert len(results) == 1
    assert len(results[0].retrieved_chunk_ids) == 1  # reranked down to top_k despite fetching rerank_candidates


def test_run_pipeline_hybrid_mode_still_returns_one_result_per_case(tmp_path):
    corpus_file = tmp_path / "doc.txt"
    corpus_file.write_text("The Eiffel Tower is located in Paris, France. It was completed in 1889.")

    config = RagConfig(
        corpus_paths=[corpus_file],
        chunk_size=10,
        overlap=2,
        top_k=2,
        retrieval_mode="hybrid",
        generation_model="openai/gpt-4o-mini",
        judge_model="openai/gpt-4o-mini",
        cases=[QaCase(id="q1", question="Where is the Eiffel Tower?", relevant_doc_id="", expected_answer="Paris")],
    )
    mock_client = MagicMock()
    mock_client.complete.side_effect = [
        ModelResponse(text="Paris.", model="m", prompt_tokens=1, completion_tokens=1, latency_seconds=0.1),
        ModelResponse(text="VERDICT: grounded\nREASON: ok.", model="m", prompt_tokens=1, completion_tokens=1, latency_seconds=0.1),
        ModelResponse(text="SCORE: 1.0\nREASON: ok.", model="m", prompt_tokens=1, completion_tokens=1, latency_seconds=0.1),
    ]

    results = run_pipeline(mock_client, config)

    assert len(results) == 1
    assert results[0].case_id == "q1"
