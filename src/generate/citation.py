"""
Citation verification: ensure every cited chunk actually supports the answer.

Two-stage check:
1. Existence check: cited chunk_id must exist in the retrieved set
2. Grounding check: cosine similarity between answer sentence and cited chunk
   text must exceed threshold (default 0.75)

Uncited chunks are not penalized — only unsupported or hallucinated citations
are flagged. This implements the grounding principle from ADR 0005.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

GROUNDING_THRESHOLD = 0.75  # Cosine similarity minimum for a valid citation


@dataclass
class CitationVerificationResult:
    valid_citations: list[str]       # Citations that pass both checks
    hallucinated_citations: list[str]  # Chunk IDs not in retrieved set
    weak_citations: list[str]        # Exist but low cosine similarity to answer
    grounding_score: float           # Fraction of claims that are grounded


class CitationVerifier:
    """
    Verify that cited chunks support the answer text.

    Uses sentence embeddings (same model as retrieval) to compute similarity
    between the answer and each cited chunk. Low-similarity citations suggest
    the model invented the citation or cited a tangentially related chunk.
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        threshold: float = GROUNDING_THRESHOLD,
    ):
        self.threshold = threshold
        self._openai_api_key = openai_api_key
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            import openai
            self._embedder = openai.OpenAI(api_key=self._openai_api_key)
        return self._embedder

    async def verify(
        self,
        answer: str,
        citations: list[str],
        retrieved_chunks: dict[str, str],  # chunk_id → text
    ) -> CitationVerificationResult:
        """Verify citations against retrieved chunks."""
        if not citations:
            return CitationVerificationResult(
                valid_citations=[],
                hallucinated_citations=[],
                weak_citations=[],
                grounding_score=1.0,  # No citations → no hallucinated citations
            )

        # Stage 1: existence check
        hallucinated = [cid for cid in citations if cid not in retrieved_chunks]
        existing = [cid for cid in citations if cid in retrieved_chunks]

        if not existing:
            return CitationVerificationResult(
                valid_citations=[],
                hallucinated_citations=hallucinated,
                weak_citations=[],
                grounding_score=0.0,
            )

        # Stage 2: grounding check via cosine similarity
        answer_emb = await self._embed(answer)
        valid, weak = [], []

        for chunk_id in existing:
            chunk_text = retrieved_chunks[chunk_id]
            chunk_emb = await self._embed(chunk_text)
            sim = self._cosine(answer_emb, chunk_emb)

            if sim >= self.threshold:
                valid.append(chunk_id)
            else:
                weak.append(chunk_id)
                logger.warning(f"Weak citation {chunk_id}: cosine={sim:.3f} < {self.threshold}")

        total = len(citations)
        grounding_score = len(valid) / total if total > 0 else 1.0

        return CitationVerificationResult(
            valid_citations=valid,
            hallucinated_citations=hallucinated,
            weak_citations=weak,
            grounding_score=grounding_score,
        )

    async def _embed(self, text: str) -> list[float]:
        import asyncio
        embedder = self._get_embedder()
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: embedder.embeddings.create(
                model="text-embedding-3-small", input=text[:8191]
            ),
        )
        return resp.data[0].embedding

    def _cosine(self, a: list[float], b: list[float]) -> float:
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x ** 2 for x in a))
        norm_b = math.sqrt(sum(y ** 2 for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
