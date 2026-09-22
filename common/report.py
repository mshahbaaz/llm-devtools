from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.table import Table

from common.results import RunResult
from common.stats import bootstrap_pass_rate_ci


def render_table(run: RunResult) -> str:
    table = Table(title=f"{run.tool} results")
    table.add_column("Model")
    table.add_column("Pass rate")
    table.add_column("Avg latency (s)")
    table.add_column("Total cost ($)")

    for model, results in run.by_model().items():
        pass_rate = sum(1 for r in results if r.passed) / len(results) * 100
        low, high = bootstrap_pass_rate_ci([r.passed for r in results])
        avg_latency = sum(r.latency_seconds for r in results) / len(results)
        total_cost = sum(r.cost_usd for r in results)
        table.add_row(
            model,
            f"{pass_rate:.0f}% (95% CI: {low*100:.0f}-{high*100:.0f}%)",
            f"{avg_latency:.2f}",
            f"{total_cost:.4f}",
        )

    buffer = StringIO()
    console = Console(file=buffer, width=100)
    console.print(table)
    return buffer.getvalue()


def render_html(run: RunResult) -> str:
    rows = ""
    for model, results in run.by_model().items():
        pass_rate = sum(1 for r in results if r.passed) / len(results) * 100
        avg_latency = sum(r.latency_seconds for r in results) / len(results)
        total_cost = sum(r.cost_usd for r in results)
        rows += f"<tr><td>{model}</td><td>{pass_rate:.0f}%</td><td>{avg_latency:.2f}</td><td>{total_cost:.4f}</td></tr>\n"

    return f"""<html><head><title>{run.tool} results</title>
<style>
table {{ border-collapse: collapse; font-family: sans-serif; }}
td, th {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
</style></head>
<body>
<h1>{run.tool} results</h1>
<table>
<tr><th>Model</th><th>Pass rate</th><th>Avg latency (s)</th><th>Total cost ($)</th></tr>
{rows}
</table>
</body></html>"""
