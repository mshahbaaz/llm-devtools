from __future__ import annotations

from common.model_client import ModelClient

JUDGE_SYSTEM_PROMPT = (
    "Score how well the given answer matches the expected answer in meaning, from 0.0 to 1.0. "
    "Respond in EXACTLY this format:\nSCORE: <number>\nREASON: <one sentence>"
)


def score_answer(client: ModelClient, answer: str, expected: str, judge_model: str) -> float:
    prompt = f"Expected: {expected}\n\nActual: {answer}"
    response = client.complete(model=judge_model, prompt=prompt, system=JUDGE_SYSTEM_PROMPT, temperature=0.0)
    for line in response.text.splitlines():
        if line.strip().startswith("SCORE:"):
            try:
                return max(0.0, min(1.0, float(line.strip()[len("SCORE:"):].strip())))
            except ValueError:
                return 0.0
    return 0.0
