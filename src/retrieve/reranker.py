"""
Cross-encoder reranker: the third stage of hybrid retrieval.

Pipeline: BM25 (top-20) + Vector (top-20) → RRF merge (top-20) → Rerank → top-5

Cross-encoder takes (query, chunk) pairs and outputs a single relevance score.
Unlike bi-encoders (which embed query and chunk independently), the cross-encoder
sees both together — dramatically better at judging contextual relevance.
Cost: O(k) forward passes at rerank time, but k is small (≤20 after RRF).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from src.retrieve.bm25 import SearchResult

logger = logging.getLogger(__name__)

# Default model: Cohere Rerank v3 (multilingual, best-in-class for enterprise docs)
# Alternative: local cross-encoder (ms-marco-MiniLM-L-6-v2) for air-gapped environments
DEFAULT_RERANK_MODEL = "rerank-multilingual-v3.0"


@dataclass
class RankedResult:
    chunk_id: str
    text: str
    relevance_score: float  # Cross-encoder output, higher = more relevant
    rrf_score: float        # Upstream RRF score (for debugging/auditing)
    metadata: dict


class CrossEncoderReranker:
    """
    Cohere Rerank API reranker.

    Sends the top-k chunks from RRF to Cohere's rerank endpoint.
    Returns `final_top_k` results sorted by cross-encoder relevance score.

    Cohere's rerank v3 was chosen over a local cross-encoder because:
    - No GPU required for serving (reduces serving infra cost)
    - Better multilingual support (enterprise docs often mix languages)
    - Cohere provides relevance scores on a consistent 0–1 scale
    See ADR 0001 for full evaluation data.
    """

    def __init__(
        self,
        cohere_api_key: str | None = None,
        model: str = DEFAULT_RERANK_MODEL,
        final_top_k: int = 5,
    ):
        self._api_key = cohere_api_key
        self.model = model
        self.final_top_k = final_top_k
        self._client = None

    def _get_client(self):
        if self._client is None:
            import cohere
            self._client = cohere.Client(api_key=self._api_key)
        return self._client

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int | None = None,
    ) -> list[RankedResult]:
        """Rerank candidates using cross-encoder and return top_k."""
        k = top_k or self.final_top_k

        if not candidates:
            return []

        # Build (rrf_score, text) map for later merge
        rrf_score_by_id = {r.chunk_id: r.score for r in candidates}

        documents = [c.text for c in candidates]
        chunk_ids = [c.chunk_id for c in candidates]
        metadata_by_id = {c.chunk_id: c.metadata for c in candidates}

        client = self._get_client()
        loop = asyncio.get_event_loop()

        response = await loop.run_in_executor(
            None,
            lambda: client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_n=k,
            ),
        )

        results = []
        for item in response.results:
            idx = item.index
            chunk_id = chunk_ids[idx]
            results.append(RankedResult(
                chunk_id=chunk_id,
                text=documents[idx],
                relevance_score=item.relevance_score,
                rrf_score=rrf_score_by_id.get(chunk_id, 0.0),
                metadata=metadata_by_id.get(chunk_id, {}),
            ))

        logger.debug(
            f"Reranker: {len(candidates)} → {len(results)} results "
            f"(top score: {results[0].relevance_score:.3f if results else 0})"
        )
        return results
