from __future__ import annotations

from pydantic import BaseModel


class EvalResult(BaseModel):
    case_id: str
    model: str
    passed: bool
    score: float
    latency_seconds: float
    cost_usd: float
    output: str = ""
    error: str | None = None


class RunResult(BaseModel):
    tool: str
    results: list[EvalResult]

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    def by_model(self) -> dict[str, list[EvalResult]]:
        grouped: dict[str, list[EvalResult]] = {}
        for r in self.results:
            grouped.setdefault(r.model, []).append(r)
        return grouped
