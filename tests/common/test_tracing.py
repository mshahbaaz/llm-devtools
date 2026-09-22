import json
from pathlib import Path

from common.tracing import TraceLogger


def test_trace_logger_appends_jsonl_record(tmp_path: Path):
    trace_path = tmp_path / "traces.jsonl"
    logger = TraceLogger(trace_path)

    logger.log(
        tool="promptbench",
        model="openai/gpt-4o-mini",
        prompt="Classify: I love this",
        response_text="positive",
        prompt_tokens=10,
        completion_tokens=1,
        latency_seconds=0.42,
        cost_usd=0.0001,
    )

    lines = trace_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["tool"] == "promptbench"
    assert record["model"] == "openai/gpt-4o-mini"
    assert record["prompt_tokens"] == 10
    assert record["cost_usd"] == 0.0001
    assert "timestamp" in record


def test_trace_logger_appends_multiple_calls(tmp_path: Path):
    trace_path = tmp_path / "traces.jsonl"
    logger = TraceLogger(trace_path)
    for i in range(3):
        logger.log(tool="git-why", model="m", prompt=f"p{i}", response_text="r", prompt_tokens=1, completion_tokens=1, latency_seconds=0.1, cost_usd=0.0)

    assert len(trace_path.read_text().strip().splitlines()) == 3


def test_trace_logger_summary_aggregates_by_model(tmp_path: Path):
    trace_path = tmp_path / "traces.jsonl"
    logger = TraceLogger(trace_path)
    logger.log(tool="t", model="m1", prompt="a", response_text="x", prompt_tokens=10, completion_tokens=5, latency_seconds=0.5, cost_usd=0.001)
    logger.log(tool="t", model="m1", prompt="b", response_text="y", prompt_tokens=20, completion_tokens=5, latency_seconds=0.3, cost_usd=0.002)
    logger.log(tool="t", model="m2", prompt="c", response_text="z", prompt_tokens=5, completion_tokens=5, latency_seconds=0.2, cost_usd=0.0005)

    summary = logger.summary()

    assert summary["m1"]["call_count"] == 2
    assert abs(summary["m1"]["total_cost_usd"] - 0.003) < 1e-9
    assert summary["m2"]["call_count"] == 1
