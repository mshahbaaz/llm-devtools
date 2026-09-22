from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from common.config import get_api_key
from common.model_client import ModelClient
from common.report import render_table
from common.results import RunResult
from promptbench.config import load_config
from promptbench.diff import compare_runs
from promptbench.runner import run_eval

app = typer.Typer(add_completion=False, help="Evaluate a prompt across models and test cases.")
console = Console()


@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to an eval YAML config"),
    out: Optional[Path] = typer.Option(None, "--out", help="Write raw JSON results here"),
    judge_model: Optional[str] = typer.Option(None, "--judge-model", help="Model to use when scorer is llm_judge"),
):
    config = load_config(config_path)
    client = ModelClient(api_key=get_api_key())
    result = run_eval(client, config, judge_model=judge_model)

    console.print(render_table(result))

    if out:
        out.write_text(result.model_dump_json(indent=2))
        console.print(f"[green]Wrote results to {out}[/green]")


@app.command()
def compare(old_results: Path, new_results: Path):
    old = RunResult(**json.loads(old_results.read_text()))
    new = RunResult(**json.loads(new_results.read_text()))
    diff = compare_runs(old, new)

    if diff.regressions:
        console.print(f"[red]{len(diff.regressions)} regression(s):[/red]")
        for r in diff.regressions:
            console.print(f"  - {r.case_id} on {r.model}")
    if diff.improvements:
        console.print(f"[green]{len(diff.improvements)} improvement(s):[/green]")
        for r in diff.improvements:
            console.print(f"  - {r.case_id} on {r.model}")
    if not diff.regressions and not diff.improvements:
        console.print("[cyan]No pass/fail changes between runs.[/cyan]")

    if diff.regressions:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
