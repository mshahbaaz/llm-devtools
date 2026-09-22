from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class TraceLogger:
    """Append-only JSONL log of every LLM call: model, tokens, latency, cost, and a
    truncated prompt/response so a run can be replayed or audited after the fact.

    One file per run (pass a fresh path per CLI invocation), not a shared global log,
    so concurrent runs never interleave writes.
    """

    _MAX_TEXT_CHARS = 500

    def __init__(self, path: Path):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        tool: str,
        model: str,
        prompt: str,
        response_text: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_seconds: float,
        cost_usd: float,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "model": model,
            "prompt": prompt[: self._MAX_TEXT_CHARS],
            "response": response_text[: self._MAX_TEXT_CHARS],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_seconds": latency_seconds,
            "cost_usd": cost_usd,
        }
        with self._path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def summary(self) -> dict[str, dict]:
        if not self._path.exists():
            return {}
        by_model: dict[str, dict] = {}
        for line in self._path.read_text().strip().splitlines():
            record = json.loads(line)
            model = record["model"]
            bucket = by_model.setdefault(model, {"call_count": 0, "total_cost_usd": 0.0, "total_latency_seconds": 0.0})
            bucket["call_count"] += 1
            bucket["total_cost_usd"] += record["cost_usd"]
            bucket["total_latency_seconds"] += record["latency_seconds"]
        return by_model
