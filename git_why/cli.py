from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from common.config import get_api_key
from common.model_client import ModelClient
from git_why.context import find_changed_functions, find_test_references
from git_why.diff import get_diff, get_diff_range, get_staged_diff
from git_why.explain import explain_diff
from git_why.hook import install_hook

app = typer.Typer(add_completion=False, help="Explain a git diff or commit in plain English.")
console = Console()

DEFAULT_MODEL = "openai/gpt-4o-mini"


@app.command()
def explain(
    ref: Optional[str] = typer.Argument(None, help="Commit SHA, or 'main..feature' range"),
    staged: bool = typer.Option(False, "--staged", help="Explain currently staged changes"),
    repo: Path = typer.Option(Path("."), "--repo", help="Path to the git repository"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="OpenRouter model id"),
    commit_message_only: bool = typer.Option(False, "--commit-message-only", help="Print only the suggested commit message"),
):
    if staged:
        diff_text = get_staged_diff(repo)
    elif ref and ".." in ref:
        diff_text = get_diff_range(repo, ref)
    elif ref:
        diff_text = get_diff(repo, ref)
    else:
        diff_text = get_diff(repo, "HEAD")

    if not diff_text.strip():
        console.print("[yellow]No changes found to explain.[/yellow]")
        raise typer.Exit(code=0)

    changed_functions = find_changed_functions(diff_text)
    test_references: list[str] = []
    for func_name in changed_functions:
        test_references.extend(find_test_references(repo, func_name))
    test_references = sorted(set(test_references))

    client = ModelClient(api_key=get_api_key())
    explanation = explain_diff(client, diff_text, model=model, test_references=test_references)

    if commit_message_only:
        typer.echo(explanation.commit_message)
        return

    console.print(Panel(explanation.summary, title="Summary", border_style="cyan"))
    console.print(Panel(explanation.why, title="Why", border_style="blue"))
    console.print(Panel(explanation.risk, title="Risk", border_style="yellow"))
    console.print(Panel(explanation.commit_message, title="Suggested commit message", border_style="green"))


@app.command(name="install-hook")
def install_hook_command(repo: Path = typer.Option(Path("."), "--repo", help="Path to the git repository")):
    hook_path = install_hook(repo)
    console.print(f"[green]Installed prepare-commit-msg hook at {hook_path}[/green]")


if __name__ == "__main__":
    app()
