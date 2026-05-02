# Architecture — RAG Reference System

## Component Overview

### Ingest Pipeline

**Chunker (`src/ingest/chunker.py`)**

Splits documents into retrieval-sized pieces. Uses a two-pass strategy:
1. Recursive character splitting with target size 512 tokens, 64-token overlap
2. Semantic boundary detection using sentence embeddings — if a split falls in the middle of a logical section, it is moved to the nearest semantic boundary

The semantic boundary pass costs ~50ms per document page but meaningfully improves retrieval quality on long technical documents where paragraph boundaries do not align with logical sections.

**Embedder (`src/ingest/embedder.py`)**

Generates dense vector representations of chunks. Model: `text-embedding-3-large` (1536 dimensions). Batched via OpenAI batch API for ingest jobs over 100 chunks. Synchronous embedding for incremental updates.

**Pipeline (`src/ingest/pipeline.py`)**

Orchestrates ingest: chunker → embedder → parallel writes to vector DB (Milvus) and BM25 index. Transactional: if either write fails, the chunk is queued for retry. Documents are tracked by content hash to avoid re-embedding unchanged content.

---

### Retrieval Layer

**BM25 Retriever (`src/retrieve/bm25.py`)**

Wraps `rank_bm25.BM25Okapi`. Index is loaded from disk at startup and updated incrementally on each ingest run. Returns top-50 candidates with BM25 scores. Query tokenization uses the same tokenizer as the ingest pipeline for consistency.

**Vector Retriever (`src/retrieve/vector.py`)**

Queries Milvus via gRPC. HNSW index with ef=64 (query-time) and M=16 (build-time). Returns top-50 candidates with cosine similarity scores. Metadata filtering by document collection and classification level is applied before ANN search (pre-filter, not post-filter — avoids the empty-result edge case on small filtered sets).

**Hybrid Retriever (`src/retrieve/hybrid.py`)**

Merges BM25 and vector candidate lists using Reciprocal Rank Fusion:
```
RRF_score(doc) = sum(1 / (k + rank_i))  where k=60, rank_i = rank in each list
```
Default k=60 (from the original RRF paper). Returns merged list of up to 100 candidates.

**Reranker (`src/retrieve/reranker.py`)**

Calls Cohere Rerank API (`rerank-v3`). Takes the top-20 RRF candidates and scores each against the query. Returns top-5 by rerank score. The 20→5 reduction happens here; BM25 and vector retrievers return 50 candidates each specifically to give the reranker a wide enough input set.

---

### Generation Layer

**LLM Client (`src/generate/llm.py`)**

Wraps OpenAI, Anthropic, and a vLLM-compatible OpenAI-format endpoint. Provider is selected at runtime via config. All calls use structured output (JSON schema) — see [ADR 0005](../adr/0005-citation-enforcement.md). Async with timeout (30s). Retry with exponential backoff on 429/500.

**Prompt Builder (`src/generate/prompt.py`)**

Assembles the generation prompt from retrieved chunks. Format:
- System: role definition + citation instruction + output schema
- User: query
- Context: retrieved chunks with source metadata inline

Token budget management: if the assembled context exceeds the model's context window budget (default: leave 2k tokens for generation), chunks are truncated starting from the lowest-ranked.

**Citation Enforcer (`src/generate/citation.py`)**

Post-generation verification. For each (claim, source_chunk_id) pair in the structured response:
1. Look up the chunk text by ID
2. Compute cosine similarity between claim embedding and chunk embedding
3. Flag claims below threshold (0.65) as ungrounded

Returns a grounded response object with per-claim grounding scores.

---

### Guardrails

**Input Guard (`src/guardrails/input_guard.py`)**

Three checks in sequence, fast-fail:
1. PII regex scan (SSN, email, phone, credit card patterns)
2. Jailbreak classifier (zero-shot classification using `facebook/bart-large-mnli`, threshold 0.85)
3. Topic scope check (embedding similarity to topic whitelist centroids, threshold 0.3)

Total latency: ~5ms on GPU, ~20ms on CPU.

**Output Guard (`src/guardrails/output_guard.py`)**

Two checks:
1. PII scan of response text (same regex as input)
2. Ungrounded claim ratio check (blocks if >20% claims are ungrounded — see [ADR 0005](../adr/0005-citation-enforcement.md))

Hallucination scoring (LLM-as-judge) runs asynchronously and is stored for monitoring; it does not gate the synchronous response path.

---

### Eval Layer

**Offline Runner (`src/eval/runner.py`)**

Loads a golden set JSONL file. For each example, runs the full retrieval + generation pipeline. Computes retrieval metrics (Recall@k, MRR) by comparing retrieved chunk IDs against expected chunk IDs, and generation metrics (Faithfulness, Relevance) via LLM judge.

Outputs a JSON report and a GitHub PR comment template (for CI integration).

**Judge (`src/eval/judge.py`)**

Calls `claude-3-5-sonnet-20241022` with a structured faithfulness prompt. Returns a score (0-1) and a brief reasoning string. Judge prompt is pinned by version hash; changing the prompt triggers a calibration run requirement.

---

## Data Flow Summary

```
User query
  → Input guardrails (5ms)
  → Parallel: BM25 retrieval + Vector retrieval (80ms)
  → RRF fusion (1ms)
  → Reranker (200ms)
  → Context assembly (5ms)
  → LLM generation (300-700ms)
  → Citation enforcement (10ms)
  → Output guardrails (75ms)
  → Response
```

Total p50: ~680ms. The reranker and LLM generation dominate. Retrieval is under 100ms.

---

## Infrastructure

See `infra/terraform/` for AWS deployment. Key resources:
- EKS cluster with CPU nodegroup (serving stack) and optional GPU nodegroup (self-hosted models)
- Milvus via Helm chart on EKS, storage on S3 via MinIO
- RDS Postgres for metadata and BM25 index persistence
- Application Load Balancer with mTLS for API access
- CloudWatch + Prometheus + Grafana for observability

See `infra/helm/` for K8s manifests.
