import subprocess
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from common.model_client import ModelResponse
from git_why.cli import app

runner = CliRunner()


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "a.txt").write_text("hello\n")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add a.txt"], cwd=repo, check=True)
    return repo


def test_cli_explains_head(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    fake_response = ModelResponse(
        text="SUMMARY: Adds a.txt.\nWHY: Initial commit.\nRISK: Low risk.\nCOMMIT_MESSAGE: feat: add a.txt",
        model="openai/gpt-4o-mini",
        prompt_tokens=5,
        completion_tokens=5,
        latency_seconds=0.1,
    )
    with patch("git_why.cli.ModelClient") as MockClient:
        MockClient.return_value.complete.return_value = fake_response
        result = runner.invoke(app, ["explain", "HEAD", "--repo", str(repo)])

    assert result.exit_code == 0
    assert "Adds a.txt." in result.stdout
    assert "feat: add a.txt" in result.stdout
