# promptbench

Evaluate a prompt template across models and test cases.

## Usage

```bash
promptbench run examples/sentiment.yaml --out results/sentiment.json
promptbench compare results/old.json results/new.json   # regression check, exits 1 on regressions
```

## Config format

See `examples/sentiment.yaml`. `scorer` is one of `exact_match`, `regex_match`, or `llm_judge`
(pass `--judge-model` to pick the grading model for `llm_judge`).

## Statistical rigor

Pass rate is reported with a bootstrap 95% confidence interval
(`common/stats.py:bootstrap_pass_rate_ci`), not just a bare percentage. With a
handful of test cases, a point estimate alone hides how little signal it carries.
