"""
Retrieval and generation evaluation metrics.
"""
from __future__ import annotations


def compute_recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> float:
    """Fraction of relevant chunks found in the top-k retrieved chunks."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for rid in relevant_ids if rid in top_k)
    return hits / len(relevant_ids)


def compute_precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> float:
    """Fraction of top-k retrieved chunks that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for rid in top_k if rid in relevant_set)
    return hits / k


def compute_mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant chunk."""
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    import math

    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, rid in enumerate(top_k, start=1)
        if rid in relevant_set
    )

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def faithfulness_score(answer: str, source_chunks: list[str]) -> float:
    """
    Heuristic faithfulness: fraction of answer sentences that have a
    high-similarity match in the source chunks.

    For production use, prefer Judge.score_faithfulness() which uses
    an LLM as judge. This heuristic is fast but less accurate.
    """
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        return 0.0

    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 20]
    if not sentences:
        return 0.0

    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    chunk_embeddings = model.encode(source_chunks, convert_to_tensor=True)

    grounded = 0
    for s_emb in sentence_embeddings:
        similarities = util.cos_sim(s_emb, chunk_embeddings)[0]
        if float(similarities.max()) >= 0.65:
            grounded += 1

    return grounded / len(sentences)
