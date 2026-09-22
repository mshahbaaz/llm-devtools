from common.report import render_table, render_html

from common.results import EvalResult, RunResult


def _sample_run() -> RunResult:
    return RunResult(
        tool="promptbench",
        results=[
            EvalResult(case_id="c1", model="openai/gpt-4o-mini", passed=True, score=1.0, latency_seconds=0.5, cost_usd=0.0001),
            EvalResult(case_id="c2", model="openai/gpt-4o-mini", passed=False, score=0.0, latency_seconds=0.4, cost_usd=0.0001),
        ],
    )


def test_render_table_contains_model_and_pass_rate():
    table_str = render_table(_sample_run())
    assert "openai/gpt-4o-mini" in table_str
    assert "50" in table_str  # 50% pass rate shown somewhere


def test_render_html_is_valid_and_contains_scores():
    html = render_html(_sample_run())
    assert "<html" in html.lower()
    assert "openai/gpt-4o-mini" in html
