"""
Dense vector retriever backed by Milvus.

Embeds queries with text-embedding-3-small (1536-dim). Uses HNSW index with
cosine similarity. Metadata filtering is pushed down to Milvus (vs post-filter)
to avoid fetching large result sets unnecessarily.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.retrieve.bm25 import SearchResult

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
COLLECTION_NAME = "rag_chunks"


class VectorRetriever:
    """
    Milvus dense retriever with metadata pre-filtering.

    Collection schema: chunk_id (VARCHAR PK), embedding (FLOAT_VECTOR 1536),
    doc_id (VARCHAR), source_url (VARCHAR), title (VARCHAR), text (VARCHAR).

    HNSW index with M=16, ef_construction=256 for ~2ms p99 recall.
    """

    def __init__(
        self,
        milvus_uri: str = "http://localhost:19530",
        collection: str = COLLECTION_NAME,
        top_k: int = 20,
        openai_api_key: str | None = None,
    ):
        self.milvus_uri = milvus_uri
        self.collection_name = collection
        self.top_k = top_k
        self._openai_api_key = openai_api_key
        self._client = None
        self._embedder = None

    def _get_client(self):
        if self._client is None:
            from pymilvus import MilvusClient
            self._client = MilvusClient(uri=self.milvus_uri)
        return self._client

    def _get_embedder(self):
        if self._embedder is None:
            import openai
            self._embedder = openai.OpenAI(api_key=self._openai_api_key)
        return self._embedder

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Embed query and search Milvus with optional metadata pre-filter."""
        k = top_k or self.top_k
        embedding = await self._embed(query)

        filter_expr = self._build_filter(metadata_filter) if metadata_filter else None

        import asyncio
        client = self._get_client()
        loop = asyncio.get_event_loop()

        search_kwargs = {
            "collection_name": self.collection_name,
            "data": [embedding],
            "limit": k,
            "output_fields": ["chunk_id", "text", "title", "source_url", "doc_id"],
            "search_params": {"metric_type": "COSINE", "params": {"ef": 64}},
        }
        if filter_expr:
            search_kwargs["filter"] = filter_expr

        response = await loop.run_in_executor(
            None, lambda: client.search(**search_kwargs)
        )

        results = []
        for hit in response[0]:
            entity = hit["entity"]
            results.append(SearchResult(
                chunk_id=entity["chunk_id"],
                text=entity["text"],
                score=hit["distance"],  # cosine similarity: higher = better
                metadata={
                    "title": entity.get("title", ""),
                    "source_url": entity.get("source_url", ""),
                    "doc_id": entity.get("doc_id", ""),
                    "retriever": "vector",
                },
            ))

        logger.debug(f"Vector: {len(results)} results for query '{query[:60]}'")
        return results

    async def _embed(self, text: str) -> list[float]:
        import asyncio
        embedder = self._get_embedder()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: embedder.embeddings.create(model=EMBEDDING_MODEL, input=text),
        )
        return response.data[0].embedding

    def _build_filter(self, metadata_filter: dict) -> str:
        """Convert metadata dict to Milvus filter expression string."""
        clauses = []
        for key, value in metadata_filter.items():
            if isinstance(value, str):
                clauses.append(f'{key} == "{value}"')
            elif isinstance(value, (int, float)):
                clauses.append(f"{key} == {value}")
            elif isinstance(value, list):
                values_str = ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in value)
                clauses.append(f"{key} in [{values_str}]")
        return " && ".join(clauses) if clauses else ""

    def insert_chunks(self, chunks: list[dict], embeddings: list[list[float]]):
        """Bulk insert chunks and their pre-computed embeddings into Milvus."""
        client = self._get_client()
        data = [
            {
                "chunk_id": chunk["chunk_id"],
                "embedding": emb,
                "text": chunk["text"],
                "title": chunk.get("title", ""),
                "source_url": chunk.get("source_url", ""),
                "doc_id": chunk.get("doc_id", ""),
            }
            for chunk, emb in zip(chunks, embeddings)
        ]
        result = client.insert(collection_name=self.collection_name, data=data)
        logger.info(f"Vector: inserted {result['insert_count']} chunks")
