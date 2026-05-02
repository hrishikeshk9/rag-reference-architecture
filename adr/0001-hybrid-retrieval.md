# ADR 0001 — Hybrid BM25 + Vector Retrieval

**Status:** Accepted  
**Date:** 2024-11

---

## Context

We are building a RAG system for enterprise document Q&A. The retrieval layer must be accurate enough to support high-stakes use cases (policy compliance, financial analysis) where a missed relevant chunk directly causes a wrong or ungrounded answer.

Two dominant approaches exist: dense (vector) retrieval using embedding similarity, and sparse (BM25) retrieval using term overlap. Most recent RAG tutorials default to pure dense retrieval because it is simpler to implement with a single vector store call.

The question is whether this simplicity is worth the accuracy trade-off in an enterprise setting.

---

## Options Considered

**Option 1: Pure dense (vector) retrieval**

Query is embedded, nearest-neighbor search returns top-k chunks by cosine similarity. Implementation is a single API call to the vector DB. Supported natively by all major vector stores (Pinecone, Milvus, Weaviate, pgvector).

*Strengths:* Simple. Fast (single index lookup). Good semantic generalization — finds chunks with similar meaning even when vocabulary differs. Well-supported tooling.

*Weaknesses:* Brittle on exact-match queries. A search for `"SOC 2 Type II compliance requirement"` may rank semantically related but imprecise chunks above an exact match, because the embedding of the full document section captures context that dilutes the specific term. This failure mode appears often in technical and legal document Q&A where precision vocabulary matters.

**Option 2: Pure sparse (BM25) retrieval**

Classic term-frequency scoring. Finds chunks that share vocabulary with the query. Fast and deterministic.

*Strengths:* Exact keyword matching. No embedding inference cost at query time. Interpretable scoring.

*Weaknesses:* Vocabulary mismatch kills recall. Synonyms, paraphrasing, and semantic equivalence all fail. A query about "staff reduction" does not find a chunk about "headcount optimization." This is a hard failure in enterprise Q&A where users rarely know the exact terminology used in the source document.

**Option 3: Hybrid BM25 + vector with Reciprocal Rank Fusion (RRF)**

Run both retrievers independently, combine ranked lists using RRF, feed the merged result into a cross-encoder reranker. The reranker sees a broader candidate set and produces a final ranked list.

*Strengths:* Dense retrieval covers semantic gaps; sparse retrieval anchors on exact terminology. RRF is parameter-free and robust to score scale differences between retrievers. The reranker as a final-pass re-scorer corrects fused ordering errors.

*Weaknesses:* Higher latency (two index lookups + reranker inference). Operationally more complex (two indexes to maintain, sync, and update). Failure modes multiply.

---

## Decision

Use hybrid retrieval (Option 3).

Empirical testing on our internal corpus showed Recall@5 improved from 0.61 (dense-only) to 0.79 (hybrid) on a golden set of 200 query/document pairs drawn from real user questions. The improvement was especially strong on technical specification queries (+24 points) and policy lookups (+19 points), where exact terminology matters most.

The latency cost (+260ms p50) is acceptable for the target use case, which is document Q&A rather than real-time chat. Users in our context tolerate 700ms for a grounded answer.

BM25 index is maintained via a lightweight inverted index (rank_bm25 library) updated on ingest. No additional infrastructure required beyond the vector store.

---

## Consequences

**Positive:**
- Recall improvement justifies the architecture complexity for high-stakes Q&A
- System is more resilient to embedding model changes; BM25 provides a stable fallback

**Negative:**
- Ingestion pipeline must maintain two indexes in sync
- Debugging retrieval failures is harder; need tooling to inspect both retriever outputs independently
- If the vector DB or BM25 index becomes stale at different rates, result quality diverges silently

**Mitigations:**
- Ingest pipeline writes to both indexes transactionally (BM25 update is synchronous, vector embedding is async with retry)
- Retrieval debug endpoint returns both pre-fusion ranked lists for inspection
- Integration tests validate both index contents after every ingest run

---

## Revisit Triggers

- If an embedding model substantially improves keyword-level retrieval precision (making BM25 redundant), revisit Option 1
- If latency requirements tighten to <400ms p50, consider dropping the reranker first, then revisiting pure-vector
- If we move to a domain where vocabulary is highly stable and well-defined (e.g., structured database Q&A), sparse-only may suffice
