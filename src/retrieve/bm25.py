"""
BM25 retriever backed by Elasticsearch.

BM25 excels at exact keyword matches — product codes, names, technical terms.
Used as one leg of the hybrid retriever (BM25 + vector → RRF → rerank).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict


class BM25Retriever:
    """
    Elasticsearch BM25 retriever.

    Uses `multi_match` across `text` and `title` fields with field boosting:
    title^2, text^1. `minimum_should_match=1` ensures at least one query term
    is present (prevents false positives from pure fuzzy matching).
    """

    def __init__(
        self,
        es_url: str = "http://localhost:9200",
        index: str = "rag_chunks",
        top_k: int = 20,
    ):
        self.es_url = es_url
        self.index = index
        self.top_k = top_k
        self._client = None

    def _get_client(self):
        if self._client is None:
            from elasticsearch import Elasticsearch
            self._client = Elasticsearch(self.es_url)
        return self._client

    async def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Run BM25 search and return ranked results."""
        k = top_k or self.top_k

        body = {
            "size": k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "text"],
                    "type": "best_fields",
                    "minimum_should_match": "1",
                }
            },
            "_source": ["chunk_id", "text", "title", "source_url", "doc_id"],
        }

        import asyncio
        client = self._get_client()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: client.search(index=self.index, body=body)
        )

        results = []
        for hit in response["hits"]["hits"]:
            src = hit["_source"]
            results.append(SearchResult(
                chunk_id=src["chunk_id"],
                text=src["text"],
                score=hit["_score"],
                metadata={
                    "title": src.get("title", ""),
                    "source_url": src.get("source_url", ""),
                    "doc_id": src.get("doc_id", ""),
                    "retriever": "bm25",
                },
            ))

        logger.debug(f"BM25: {len(results)} results for query '{query[:60]}'")
        return results

    def index_chunks(self, chunks: list[dict]):
        """Bulk index chunks into Elasticsearch. Called during ingestion."""
        from elasticsearch.helpers import bulk

        actions = [
            {
                "_index": self.index,
                "_id": chunk["chunk_id"],
                "_source": {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "title": chunk.get("title", ""),
                    "source_url": chunk.get("source_url", ""),
                    "doc_id": chunk.get("doc_id", ""),
                },
            }
            for chunk in chunks
        ]

        client = self._get_client()
        success, errors = bulk(client, actions)
        if errors:
            logger.error(f"BM25 index errors: {errors[:3]}")
        logger.info(f"BM25: indexed {success} chunks into {self.index}")
