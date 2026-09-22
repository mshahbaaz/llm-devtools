from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class TestCase(BaseModel):
    id: str
    input: str
    expected: str


class EvalConfig(BaseModel):
    models: list[str]
    prompt_template: str
    scorer: str
    cases: list[TestCase]


def load_config(path: Path) -> EvalConfig:
    raw = yaml.safe_load(path.read_text())
    config = EvalConfig(**raw)
    if not config.cases:
        raise ValueError("Config must define at least one test case under 'cases'.")
    if "{input}" not in config.prompt_template:
        raise ValueError("prompt_template must contain an {input} placeholder.")
    return config
