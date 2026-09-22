from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from common.model_client import ModelResponse
from promptbench.cli import app

runner = CliRunner()

CONFIG_YAML = """
models:
  - openai/gpt-4o-mini
prompt_template: "Classify sentiment as positive/negative: {input}"
scorer: exact_match
cases:
  - id: c1
    input: "I love this"
    expected: "positive"
"""


def test_run_command(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_YAML)
    out_path = tmp_path / "results.json"
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    with patch("promptbench.cli.ModelClient") as MockClient:
        MockClient.return_value.complete.return_value = ModelResponse(
            text="positive", model="openai/gpt-4o-mini", prompt_tokens=5, completion_tokens=1, latency_seconds=0.1
        )
        result = runner.invoke(app, ["run", str(config_path), "--out", str(out_path)])

    assert result.exit_code == 0
    assert out_path.exists()
    assert "openai/gpt-4o-mini" in result.stdout
