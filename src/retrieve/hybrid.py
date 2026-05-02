"""
Hybrid retrieval: BM25 + vector with Reciprocal Rank Fusion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .bm25 import BM25Retriever
from .reranker import Reranker
from .vector import VectorRetriever


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source_doc_id: str
    metadata: dict


class HybridRetriever:
    """
    Merges BM25 and dense vector results via Reciprocal Rank Fusion, then
    applies a cross-encoder reranker as a final scoring pass.

    RRF formula: score(d) = sum_i 1/(k + rank_i(d))
    Default k=60 from the original RRF paper (Cormack et al., 2009).
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        vector: VectorRetriever,
        reranker: Reranker,
        bm25_top_k: int = 50,
        vector_top_k: int = 50,
        rrf_k: int = 60,
        rerank_top_k: int = 20,
        final_top_k: int = 5,
    ):
        self.bm25 = bm25
        self.vector = vector
        self.reranker = reranker
        self.bm25_top_k = bm25_top_k
        self.vector_top_k = vector_top_k
        self.rrf_k = rrf_k
        self.rerank_top_k = rerank_top_k
        self.final_top_k = final_top_k

    async def retrieve(
        self,
        query: str,
        collection: Optional[str] = None,
        metadata_filter: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """
        Full hybrid retrieval pipeline.

        1. BM25 retrieval (top bm25_top_k)
        2. Dense vector retrieval (top vector_top_k)
        3. RRF fusion
        4. Reranker (top rerank_top_k → final_top_k)
        """
        import asyncio

        bm25_results, vector_results = await asyncio.gather(
            self.bm25.retrieve(query, top_k=self.bm25_top_k),
            self.vector.retrieve(query, top_k=self.vector_top_k,
                                 collection=collection, metadata_filter=metadata_filter),
        )

        fused = self._rrf_merge(bm25_results, vector_results)
        top_for_rerank = fused[: self.rerank_top_k]
        reranked = await self.reranker.rerank(query, top_for_rerank, top_k=self.final_top_k)
        return reranked

    def _rrf_merge(
        self,
        list_a: list[RetrievedChunk],
        list_b: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Merge two ranked lists using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(list_a, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (self.rrf_k + rank)
            chunk_map[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(list_b, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (self.rrf_k + rank)
            chunk_map[chunk.chunk_id] = chunk

        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
        result = []
        for cid in sorted_ids:
            chunk = chunk_map[cid]
            result.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=scores[cid],
                source_doc_id=chunk.source_doc_id,
                metadata=chunk.metadata,
            ))
        return result
