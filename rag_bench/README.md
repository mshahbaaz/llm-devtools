# rag-bench

Benchmark a RAG pipeline: retrieval quality (recall@k, MRR) and generation quality
(faithfulness/groundedness, answer correctness).

## Usage

```bash
rag-bench run examples/eiffel.yaml
rag-bench sweep examples/eiffel.yaml --chunk-sizes 50,100,200 --top-ks 1,3,5
```

## Config format

See `examples/eiffel.yaml`. Retrieval options in `RagConfig`:

- `retrieval_mode`: `"dense"` (default) or `"hybrid"`. Hybrid fuses dense vector
  search with BM25 keyword search via reciprocal rank fusion (`rag_bench/hybrid.py`),
  since embeddings alone often miss exact-match terms (names, ids, numbers).
- `use_reranker` / `rerank_candidates`: enables a cross-encoder reranking stage
  (`rag_bench/reranker.py`), the standard two-stage retrieve-then-rerank pattern.
