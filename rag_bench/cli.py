from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from common.config import get_api_key
from common.model_client import ModelClient
from rag_bench.config import RagConfig
from rag_bench.pipeline import run_pipeline
from rag_bench.sweep import run_sweep

app = typer.Typer(add_completion=False, help="Benchmark a RAG pipeline's retrieval and answer quality.")
console = Console()


@app.command()
def run(config_path: Path = typer.Argument(..., help="Path to a rag-bench YAML config")):
    raw = yaml.safe_load(config_path.read_text())
    config = RagConfig(**raw)
    client = ModelClient(api_key=get_api_key())

    results = run_pipeline(client, config)

    table = Table(title="rag-bench results")
    table.add_column("Case")
    table.add_column("Answer")
    table.add_column("Faithfulness")
    table.add_column("Answer score")
    for r in results:
        table.add_row(r.case_id, r.answer[:60], f"{r.faithfulness_score:.1f}", f"{r.answer_score:.1f}")

    console.print(table)


@app.command()
def sweep(
    config_path: Path = typer.Argument(..., help="Path to a rag-bench YAML config (used as the base config)"),
    chunk_sizes: str = typer.Option(..., "--chunk-sizes", help="Comma-separated chunk sizes, e.g. 50,100,200"),
    top_ks: str = typer.Option(..., "--top-ks", help="Comma-separated top_k values, e.g. 1,3,5"),
):
    raw = yaml.safe_load(config_path.read_text())
    base_config = RagConfig(**raw)
    client = ModelClient(api_key=get_api_key())

    chunk_size_list = [int(v.strip()) for v in chunk_sizes.split(",")]
    top_k_list = [int(v.strip()) for v in top_ks.split(",")]

    sweep_results = run_sweep(client, base_config, chunk_sizes=chunk_size_list, top_ks=top_k_list)

    table = Table(title="rag-bench sweep results")
    table.add_column("Chunk size")
    table.add_column("Top k")
    table.add_column("Avg faithfulness")
    table.add_column("Avg answer score")
    for r in sweep_results:
        table.add_row(str(r.chunk_size), str(r.top_k), f"{r.avg_faithfulness:.2f}", f"{r.avg_answer_score:.2f}")

    console.print(table)


if __name__ == "__main__":
    app()
