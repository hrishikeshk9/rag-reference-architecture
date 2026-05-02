# rag-reference-architecture

A production-grade RAG system designed for enterprise-scale document Q&A. Built to answer the hard questions: how do you retrieve faithfully, evaluate systematically, and operate sustainably at cost?

This is not a tutorial. It is a reference implementation showing how the pieces fit in production — hybrid retrieval, reranking, citation enforcement, guardrails, an offline eval loop, and Terraform-deployable infrastructure.

---

## Architecture

```mermaid
flowchart TD
    User([User Query]) --> InputGuard[Input Guardrails\npii · jailbreak · topic]
    InputGuard --> Router{Query Router}
    Router -->|structured| BM25[BM25 Retriever]
    Router -->|semantic| VecDB[(Vector DB\nMilvus / pgvector)]
    BM25 --> Fusion[Reciprocal Rank Fusion]
    VecDB --> Fusion
    Fusion --> Reranker[Cross-Encoder Reranker\ncohere/rerank-v3]
    Reranker --> Context[Context Assembly\nchunk selection · dedup]
    Context --> LLM[LLM\nGPT-4o / Claude / self-hosted]
    LLM --> CitationCheck[Citation Enforcer\ngrounding · attribution]
    CitationCheck --> OutputGuard[Output Guardrails\nPII · hallucination signal]
    OutputGuard --> Response([Response + Sources])

    subgraph Eval Loop
        GoldenSet[(Golden Set)] --> OfflineEval[Offline Evaluator]
        OfflineEval --> Metrics[Recall@k · MRR · Faithfulness · Relevance]
        Metrics --> CIGate{CI Gate}
    end

    subgraph Ingest
        Docs[(Raw Docs)] --> Chunker[Chunker\nrecursive + semantic]
        Chunker --> Embedder[Embedder\ntext-embedding-3-large]
        Embedder --> VecDB
        Chunker --> BM25Index[(BM25 Index)]
    end
```

---

## What This Is and Isn't

**This is:**
- A reference for teams building production RAG on top of enterprise documents
- An opinionated implementation of hybrid retrieval with explicit trade-off documentation
- A deployable eval framework with CI integration

**This is not:**
- A library you install with `pip`
- A one-click chatbot demo
- An abstraction layer over LangChain (the retrieval and generation code is explicit and readable)

---

## Performance Characteristics

Benchmarked on an internal enterprise document corpus (financial reports, technical specs, policy docs). Numbers vary by domain and document quality.

| Metric | Baseline (dense-only) | This system |
|---|---|---|
| Recall@5 | 0.61 | **0.79** |
| MRR | 0.58 | **0.74** |
| Faithfulness (LLM-judge) | 0.71 | **0.86** |
| Avg latency p50 (ms) | 420 | 680 |
| Avg latency p99 (ms) | 1100 | 1650 |

The latency cost of reranking is real (+260ms p50). The faithfulness gain justifies it for enterprise use cases where hallucination is a compliance risk. For latency-sensitive cases, see [ADR 0001](adr/0001-hybrid-retrieval.md).

---

## Cost Model (per 1,000 queries)

Based on AWS deployment. Retrieval corpus: 500k chunks. Embedding model: `text-embedding-3-large`. Generation: `gpt-4o`.

| Component | Cost/1k queries |
|---|---|
| Embedding (query-time) | $0.04 |
| Reranker API (Cohere) | $0.20 |
| LLM generation (~2k tokens avg) | $5.00 |
| Vector DB (Milvus on EKS, amortized) | $0.12 |
| **Total** | **~$5.36** |

Switching to a self-hosted reranker and `claude-3-haiku` reduces to ~$1.80/1k queries with <5% quality degradation on most domains. See [docs/cost-model.md](docs/cost-model.md).

---

## Eval Methodology

Evaluation is not bolted on — it is the primary feedback loop.

**Offline eval** runs on a versioned golden set of (query, expected_chunks, expected_answer) triples. The CI gate blocks merges if Recall@5 drops >3% or Faithfulness drops >5%.

**Online eval** samples 2% of production traffic and runs LLM-as-judge asynchronously. Results feed into a weekly drift report.

**Judge model:** `claude-3-5-sonnet-20241022` with structured output. Judge prompts are versioned in `src/eval/prompts/`. See [ADR 0003](adr/0003-eval-strategy.md).

---

## Key Trade-offs

| Decision | Choice | What we gave up |
|---|---|---|
| Retrieval | Hybrid BM25 + vector | Pure vector simplicity |
| Vector DB | Milvus (self-hosted) | Pinecone convenience |
| Reranker | Cohere API | Latency + cost control |
| Guardrails | Input AND output | Slightly higher latency |
| Citations | Strict (block ungrounded) | Some valid responses rejected |

Each decision has a full ADR in [adr/](adr/).

---

## Repository Structure

```
.
├── adr/                        # Architecture Decision Records
├── docs/
│   ├── architecture.md         # Detailed component docs
│   ├── runbook.md              # Ops procedures
│   └── cost-model.md           # Cost breakdown and optimization
├── src/
│   ├── ingest/                 # Chunking and embedding pipelines
│   ├── retrieve/               # Hybrid retrieval: BM25 + vector + rerank
│   ├── generate/               # LLM call with citation enforcement
│   ├── guardrails/             # Input and output validation
│   └── eval/                   # Offline eval suite
├── infra/
│   ├── terraform/              # AWS deployment
│   └── helm/                   # K8s charts
├── examples/
│   ├── enterprise-docs/
│   └── code-search/
├── tests/
└── .github/workflows/          # CI: eval gate on every PR
```

---

## Quickstart

```bash
git clone https://github.com/hrishikeshk9/rag-reference-architecture
cd rag-reference-architecture
pip install -e ".[dev]"

cp .env.example .env  # fill API keys

# Ingest example corpus
python -m src.ingest.pipeline --input examples/enterprise-docs/corpus/ --index local

# Run a query
python -m src.retrieve.hybrid --query "What is the data retention policy?" --top-k 5

# Run offline eval
python -m src.eval.runner --golden-set tests/golden/enterprise_qa.jsonl
```

---

## ADRs

- [0001 — Hybrid vs pure-vector retrieval](adr/0001-hybrid-retrieval.md)
- [0002 — Vector DB choice](adr/0002-vector-db-choice.md)
- [0003 — Eval strategy](adr/0003-eval-strategy.md)
- [0004 — Guardrails placement](adr/0004-guardrails-placement.md)
- [0005 — Citation enforcement](adr/0005-citation-enforcement.md)

---

## Related

- [architecture-decision-records](https://github.com/hrishikeshk9/architecture-decision-records) — broader ADR library for the full LLM platform
- [llm-eval-harness](https://github.com/hrishikeshk9/llm-eval-harness) — standalone eval framework used here
- [transformer-from-scratch](https://github.com/hrishikeshk9/transformer-from-scratch) — the primitives underneath the embedding models

---

## License

Apache-2.0
