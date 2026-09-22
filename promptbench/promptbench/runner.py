from __future__ import annotations

from common.model_client import ModelClient
from common.results import EvalResult, RunResult
from promptbench.config import EvalConfig
from promptbench.scorers import get_scorer, llm_judge

# Rough per-1K-token USD costs for a handful of common OpenRouter models.
# Used only to show an approximate cost estimate in reports, not for billing.
_COST_PER_1K = {
    "openai/gpt-4o-mini": 0.00015,
    "anthropic/claude-3.5-haiku": 0.0008,
    "meta-llama/llama-3.1-8b-instruct:free": 0.0,
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rate = _COST_PER_1K.get(model, 0.001)
    return (prompt_tokens + completion_tokens) / 1000 * rate


def run_eval(client: ModelClient, config: EvalConfig, judge_model: str | None = None) -> RunResult:
    results: list[EvalResult] = []

    for model in config.models:
        for case in config.cases:
            prompt = config.prompt_template.format(input=case.input)
            try:
                response = client.complete(model=model, prompt=prompt)
            except Exception as exc:
                results.append(EvalResult(case_id=case.id, model=model, passed=False, score=0.0, latency_seconds=0.0, cost_usd=0.0, error=str(exc)))
                continue

            if config.scorer == "llm_judge":
                score = llm_judge(client, output=response.text, expected=case.expected, judge_model=judge_model or model)
            else:
                scorer_fn = get_scorer(config.scorer)
                score = scorer_fn(output=response.text, expected=case.expected)

            results.append(
                EvalResult(
                    case_id=case.id,
                    model=model,
                    passed=score >= 0.5,
                    score=score,
                    latency_seconds=response.latency_seconds,
                    cost_usd=_estimate_cost(model, response.prompt_tokens, response.completion_tokens),
                    output=response.text,
                )
            )

    return RunResult(tool="promptbench", results=results)
