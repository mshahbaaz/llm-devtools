from __future__ import annotations

import re

from common.model_client import ModelClient

JUDGE_SYSTEM_PROMPT = (
    "You are grading an AI system's output against a reference answer. "
    "Score from 0.0 (completely wrong) to 1.0 (fully correct and equivalent in meaning). "
    "Respond in EXACTLY this format:\n"
    "SCORE: <a number between 0.0 and 1.0>\n"
    "REASON: <one sentence>"
)


def exact_match(output: str, expected: str) -> float:
    return 1.0 if output.strip().lower() == expected.strip().lower() else 0.0


def regex_match(output: str, expected: str) -> float:
    return 1.0 if re.search(expected, output) else 0.0


def llm_judge(client: ModelClient, output: str, expected: str, judge_model: str) -> float:
    prompt = f"Reference answer: {expected}\n\nModel output: {output}"
    response = client.complete(model=judge_model, prompt=prompt, system=JUDGE_SYSTEM_PROMPT, temperature=0.0)
    for line in response.text.splitlines():
        if line.strip().startswith("SCORE:"):
            try:
                return max(0.0, min(1.0, float(line.strip()[len("SCORE:"):].strip())))
            except ValueError:
                return 0.0
    return 0.0


_SCORERS = {"exact_match": exact_match, "regex_match": regex_match}


def get_scorer(name: str):
    if name not in _SCORERS:
        raise ValueError(f"Unknown scorer '{name}'. Available: {list(_SCORERS)} (or 'llm_judge', handled separately).")
    return _SCORERS[name]
