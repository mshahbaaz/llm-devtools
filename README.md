# llm-devtools

Three small, focused LLM developer tools, each a real CLI you can install and run.

| Tool | What it does | CLI |
|---|---|---|
| [git-why](git_why/) | Explains a diff/commit in plain English and suggests a commit message, grounded in whether the changed function actually has test coverage | `pip install llm-devtools && git-why explain HEAD` |
| [promptbench](promptbench/) | Evaluates a prompt across models and test cases (exact-match, regex, LLM-judge scoring), with bootstrap confidence intervals on pass rate | `promptbench run config.yaml` |
| [rag-bench](rag_bench/) | Benchmarks a RAG pipeline's retrieval (recall@k, MRR, hybrid BM25+dense search, cross-encoder reranking) and generation (faithfulness) | `rag-bench run config.yaml` |

All three share one `common/` package: an OpenRouter model client, a result schema,
table/HTML report rendering, bootstrap confidence intervals, and JSONL call tracing
(tokens, latency, and cost per model, per LLM call). One model integration, reused everywhere.

## Install

```bash
git clone https://github.com/mshahbaaz/llm-devtools
cd llm-devtools
pip install -e ".[dev,rag]"
export OPENROUTER_API_KEY=sk-or-...   # https://openrouter.ai/keys
```

## Run the tests

```bash
pytest -v
```

## License

MIT
