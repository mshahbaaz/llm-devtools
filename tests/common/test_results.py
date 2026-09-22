from common.results import EvalResult, RunResult


def test_run_result_aggregates_pass_rate_and_cost():
    run = RunResult(
        tool="promptbench",
        results=[
            EvalResult(case_id="c1", model="openai/gpt-4o-mini", passed=True, score=1.0, latency_seconds=0.5, cost_usd=0.0001),
            EvalResult(case_id="c2", model="openai/gpt-4o-mini", passed=False, score=0.0, latency_seconds=0.4, cost_usd=0.0001),
        ],
    )
    assert run.pass_rate == 0.5
    assert abs(run.total_cost_usd - 0.0002) < 1e-9
