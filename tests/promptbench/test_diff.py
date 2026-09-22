from common.results import EvalResult, RunResult
from promptbench.diff import compare_runs


def test_compare_runs_flags_regressions():
    old = RunResult(tool="promptbench", results=[
        EvalResult(case_id="c1", model="openai/gpt-4o-mini", passed=True, score=1.0, latency_seconds=0.1, cost_usd=0.0),
    ])
    new = RunResult(tool="promptbench", results=[
        EvalResult(case_id="c1", model="openai/gpt-4o-mini", passed=False, score=0.0, latency_seconds=0.1, cost_usd=0.0),
    ])

    diff = compare_runs(old, new)

    assert len(diff.regressions) == 1
    assert diff.regressions[0].case_id == "c1"
    assert len(diff.improvements) == 0


def test_compare_runs_flags_improvements():
    old = RunResult(tool="promptbench", results=[
        EvalResult(case_id="c1", model="openai/gpt-4o-mini", passed=False, score=0.0, latency_seconds=0.1, cost_usd=0.0),
    ])
    new = RunResult(tool="promptbench", results=[
        EvalResult(case_id="c1", model="openai/gpt-4o-mini", passed=True, score=1.0, latency_seconds=0.1, cost_usd=0.0),
    ])

    diff = compare_runs(old, new)

    assert len(diff.improvements) == 1
    assert len(diff.regressions) == 0
