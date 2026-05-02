# Cost Model

## Per-Query Cost Breakdown

All figures are USD. AWS us-east-1 pricing as of late 2024.

### Scenario 1: API-first (GPT-4o + Cohere reranker)

| Component | Cost basis | Cost/1k queries |
|---|---|---|
| Query embedding | `text-embedding-3-large`, 512 tokens avg | $0.04 |
| BM25 retrieval | CPU compute (amortized) | ~$0.01 |
| Vector retrieval | Milvus on EKS (amortized, see below) | $0.12 |
| Cohere Rerank v3 | $0.20/1k search units | $0.20 |
| GPT-4o generation | 2k input + 500 output tokens avg | $5.00 |
| Input/output guardrails | CPU inference (amortized) | ~$0.02 |
| **Total** | | **~$5.39** |

### Scenario 2: Cost-optimized (Claude Haiku + self-hosted reranker)

| Component | Cost basis | Cost/1k queries |
|---|---|---|
| Query embedding | `text-embedding-3-large` | $0.04 |
| BM25 retrieval | CPU compute | ~$0.01 |
| Vector retrieval | Milvus on EKS | $0.12 |
| Self-hosted reranker | GPU inference on g4dn.xlarge (amortized) | $0.08 |
| Claude Haiku generation | 2k input + 500 output avg | $0.30 |
| Guardrails | CPU | ~$0.02 |
| **Total** | | **~$0.57** |

Quality comparison on our golden set: Scenario 2 scores 1.5% lower on Faithfulness (0.85 vs 0.86), 3.2% lower on Relevance. Acceptable for cost reduction at scale.

### Scenario 3: Full self-hosted (vLLM + self-hosted embeddings)

| Component | Cost basis | Cost/1k queries |
|---|---|---|
| Query embedding | Self-hosted `bge-large-en-v1.5` on g4dn.xlarge | $0.01 |
| BM25 retrieval | CPU | ~$0.01 |
| Vector retrieval | Milvus | $0.12 |
| Self-hosted reranker | g4dn.xlarge (shared with embedder) | $0.04 |
| vLLM generation (Llama 3.1 70B) | g5.12xlarge (4x A10G) amortized | $0.80 |
| Guardrails | CPU | ~$0.02 |
| **Total** | | **~$1.00** |

Requires 24/7 GPU infrastructure. Economical at >5k queries/day. Below that, APIs are cheaper.

---

## Infrastructure Cost (Monthly)

### EKS Cluster (Scenario 1 — API-first, 100k queries/month)

| Resource | Spec | Monthly cost |
|---|---|---|
| EKS control plane | | $73 |
| CPU nodegroup (serving stack) | 2x c5.2xlarge | $272 |
| Milvus nodegroup | 2x r5.2xlarge (memory-optimized) | $476 |
| RDS Postgres (metadata) | db.t3.medium, Multi-AZ | $95 |
| S3 (MinIO data) | 100GB | $2 |
| ALB | 1x ALB | $25 |
| CloudWatch | Standard usage | $30 |
| **Total infrastructure** | | **~$973/month** |

Per-query infrastructure amortization at 100k queries/month: ~$0.01/query.

### Break-even vs managed alternatives

At 100k queries/month, Pinecone standard tier for 500k vectors costs ~$700/month. Milvus on EKS costs ~$550/month (Milvus nodegroup + storage). Self-hosting saves ~$150/month at this scale — not compelling unless data residency requirements apply.

At 1M queries/month and 5M vectors: Pinecone scales to ~$2,000/month. Milvus on EKS scales to ~$800/month. Self-hosting saves ~$1,200/month. Compelling at this scale.

---

## Optimization Levers

**1. Embedding caching**
If the same query appears frequently (e.g., FAQ-style use cases), cache the query embedding and the top-k retrieval results by query hash. Can reduce embedding and retrieval costs by 20-40% for repetitive query patterns.

**2. Reranker batching**
Batch multiple queries to the Cohere Rerank API. Reduces per-query latency at high throughput and enables bulk pricing tiers.

**3. Tiered generation**
Route simple factual queries (high retrieval confidence, low ambiguity) to Claude Haiku. Route complex synthesis queries to GPT-4o or Claude Sonnet. A routing classifier adds ~5ms and can reduce LLM costs by 40-60% if query distribution supports it.

**4. Corpus-level caching**
For static document corpora (e.g., legal documents that update quarterly), pre-compute and cache the top-10 retrieval results for every query in a representative sample. Serves cached results for exact query matches. High hit rate for FAQ-style deployments.

**5. Context compression**
Rather than passing full chunks to the LLM, extract relevant sentences from each chunk using a smaller model. Reduces input tokens to the LLM by 30-50% with modest quality degradation. See `src/generate/prompt.py` for the `compress_context` option.
