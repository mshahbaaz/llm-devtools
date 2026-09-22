from pathlib import Path
from unittest.mock import MagicMock

from common.model_client import ModelResponse
from rag_bench.config import RagConfig, QaCase
from rag_bench.sweep import run_sweep


def test_run_sweep_runs_pipeline_once_per_config_combo(tmp_path: Path):
    corpus_file = tmp_path / "doc.txt"
    corpus_file.write_text("The Eiffel Tower is in Paris. " * 20)

    base_config = RagConfig(
        corpus_paths=[corpus_file],
        cases=[QaCase(id="q1", question="Where is it?", relevant_doc_id="", expected_answer="Paris")],
    )

    fake_response = ModelResponse(text="Paris", model="m", prompt_tokens=1, completion_tokens=1, latency_seconds=0.1)
    mock_client = MagicMock()
    mock_client.complete.return_value = fake_response

    sweep_results = run_sweep(mock_client, base_config, chunk_sizes=[50, 100], top_ks=[1, 3])

    assert len(sweep_results) == 4  # 2 chunk_sizes x 2 top_ks
    assert {(r.chunk_size, r.top_k) for r in sweep_results} == {(50, 1), (50, 3), (100, 1), (100, 3)}
