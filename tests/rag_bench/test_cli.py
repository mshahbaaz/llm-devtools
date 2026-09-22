from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from common.model_client import ModelResponse
from rag_bench.cli import app

runner = CliRunner()

CONFIG_YAML = """
corpus_paths:
  - {corpus_path}
chunk_size: 20
overlap: 5
top_k: 2
generation_model: openai/gpt-4o-mini
judge_model: openai/gpt-4o-mini
cases:
  - id: q1
    question: "Where is the Eiffel Tower?"
    relevant_doc_id: ""
    expected_answer: "Paris"
"""


def test_run_command(tmp_path: Path, monkeypatch):
    corpus_path = tmp_path / "doc.txt"
    corpus_path.write_text("The Eiffel Tower is in Paris, France.")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_YAML.format(corpus_path=corpus_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    fake_response = ModelResponse(text="Paris", model="openai/gpt-4o-mini", prompt_tokens=5, completion_tokens=1, latency_seconds=0.1)
    with patch("rag_bench.cli.ModelClient") as MockClient:
        MockClient.return_value.complete.return_value = fake_response
        result = runner.invoke(app, ["run", str(config_path)])

    assert result.exit_code == 0
    assert "q1" in result.stdout
