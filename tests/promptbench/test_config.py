from pathlib import Path

from promptbench.config import EvalConfig, load_config


def test_load_config_parses_yaml(tmp_path: Path):
    config_yaml = """
models:
  - openai/gpt-4o-mini
  - anthropic/claude-3.5-haiku
prompt_template: "Classify sentiment as positive/negative: {input}"
scorer: exact_match
cases:
  - id: c1
    input: "I love this product"
    expected: "positive"
  - id: c2
    input: "This is terrible"
    expected: "negative"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)

    config = load_config(config_path)

    assert isinstance(config, EvalConfig)
    assert config.models == ["openai/gpt-4o-mini", "anthropic/claude-3.5-haiku"]
    assert config.scorer == "exact_match"
    assert len(config.cases) == 2
    assert config.cases[0].id == "c1"
    assert config.cases[0].expected == "positive"


def test_load_config_rejects_missing_cases(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("models: [openai/gpt-4o-mini]\nprompt_template: 'x {input}'\nscorer: exact_match\ncases: []\n")
    try:
        load_config(config_path)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "cases" in str(e)
