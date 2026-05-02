# ADR 0002 — Vector Database Choice

**Status:** Accepted  
**Date:** 2024-11

---

## Context

The RAG system requires a vector store for embedding-based retrieval. We need to choose a vector database that supports our scale (~500k chunks initially, growing to ~5M), our latency requirements (p99 <200ms for vector search), and our operational model (Kubernetes-native, AWS-hosted).

We evaluated this decision in the context of a team that already operates Kubernetes and Terraform — managed SaaS options trade away operational control for ease. We needed to decide whether that trade is worth making.

---

## Options Considered

**Option 1: Pinecone (managed SaaS)**

Fully managed vector database. No infrastructure to operate. High availability and scaling handled by the vendor.

*Strengths:* Zero operational overhead. Fast time-to-production. Good developer experience and SDK. Metadata filtering is first-class.

*Weaknesses:* Vendor lock-in is significant. Data leaves your infrastructure (non-starter for some enterprise data classification levels). Pricing becomes expensive at scale: at 5M vectors with ~1000 queries/day, cost is approximately $700/month on the standard tier. No visibility into query internals (no explain plan equivalent). Egress costs for large corpus loads.

**Option 2: Weaviate (self-hosted or cloud)**

Open-source vector DB with hybrid search support built in (BM25 + dense in one query). Kubernetes operator available. GraphQL-based query interface.

*Strengths:* Native hybrid search reduces implementation complexity (no separate BM25 index). Strong metadata filtering. Active open-source community.

*Weaknesses:* GraphQL query interface is non-standard and adds a learning curve. Memory-heavy at large scale. Weaviate Cloud (managed) has similar vendor concerns to Pinecone. Our benchmarks showed higher p99 latency than Milvus on our hardware profile (270ms vs 140ms at 500k vectors). Operator maturity is lower than Milvus at the time of decision.

**Option 3: Milvus (self-hosted on Kubernetes)**

CNCF project. Kubernetes-native via Milvus Operator. Separates storage (MinIO/S3) from compute. Supports multiple index types (HNSW, IVF_FLAT, DiskANN). Strong performance benchmarks.

*Strengths:* Open-source with Apache-2.0 license. Data stays in our infrastructure. Performance is excellent — sub-100ms p99 at 500k vectors with HNSW on our instance profile. Rich index configuration. Attu UI for inspection. Active development by Zilliz with strong operational tooling.

*Weaknesses:* Operational complexity: Milvus has multiple components (proxy, datanode, querynode, indexnode, etcd, MinIO). Requires more Kubernetes resources and careful capacity planning. Upgrades need coordination. Steeper learning curve for the team.

**Option 4: pgvector (PostgreSQL extension)**

Vector search as a PostgreSQL extension. Uses existing Postgres infrastructure.

*Strengths:* If you already run Postgres, no new infrastructure. Transactional semantics (vectors update atomically with metadata). SQL-native — familiar to the whole team.

*Weaknesses:* Performance degrades significantly beyond ~1M vectors without careful partitioning and IVF index tuning. Approximate nearest neighbor (ANN) quality is lower than dedicated vector stores. Not suitable as primary vector store at our target scale of 5M+ chunks. Good fit as a dev/staging database; not production primary at scale.

---

## Decision

Use Milvus self-hosted on Kubernetes (Option 3) as the primary vector store, with pgvector for local development and small-scale staging environments.

Key factors:
1. **Data residency** — enterprise document content cannot leave our VPC. Managed SaaS options are disqualifying for the primary deployment.
2. **Performance** — Milvus benchmarked at 140ms p99 for ANN search on 500k vectors (HNSW, m=16, ef=64) on our instance profile. This leaves margin for the full retrieval pipeline to stay under 300ms for the vector component.
3. **Operational fit** — the team already runs Kubernetes. The Milvus Operator is mature. MinIO is already used for model artifact storage.
4. **Cost** — at 5M vectors, Milvus on 2x c5.2xlarge nodes costs approximately $350/month (all-in with storage), versus $700+/month for Pinecone standard.

pgvector is retained for local development because: single binary, no compose orchestration needed, exact SQL queries for debugging retrieval. Dev/prod parity on retrieval logic is maintained through a common interface abstraction.

---

## Consequences

**Positive:**
- No data egress to vendor infrastructure
- Lower cost at scale
- Full control over index configuration and query planning

**Negative:**
- Team owns Milvus operations: upgrades, scaling, backup, incident response
- Milvus component failures are more complex to debug than a managed service
- Local dev requires running pgvector separately from the Milvus cluster

**Mitigations:**
- Milvus is deployed via Operator with all components on dedicated node groups; no shared node contention
- Automated backup to S3 via MinIO lifecycle policies
- Runbook covers common failure modes: querynode OOM, etcd split-brain, index rebuild

---

## Revisit Triggers

- If team size shrinks and Kubernetes operational capacity becomes a constraint, revisit Weaviate Cloud or Pinecone
- If compliance requirements change to allow specific managed vendors (FedRAMP-authorized options exist)
- If pgvector performance catches up to Milvus at our target scale (watch the pgvector roadmap on DiskANN support)
