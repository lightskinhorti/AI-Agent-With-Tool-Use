from __future__ import annotations

import asyncio

import chromadb
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from agent.config import settings
from agent.tools.base import BaseTool


class RAGSearchInput(BaseModel):
    query: str = Field(description="Search query for the knowledge base")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")


class RAGSearchTool(BaseTool):
    name = "rag_search"
    description = (
        "Search the knowledge base using hybrid semantic + keyword search. "
        "Use for domain-specific questions about documents in the corpus."
    )

    def __init__(self) -> None:
        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir)
        )
        self._collection = self._client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )

    def get_schema(self) -> type[BaseModel]:
        return RAGSearchInput

    async def _execute(self, query: str, top_k: int = 5) -> str:
        return await asyncio.to_thread(self._sync_search, query, top_k)

    def _sync_search(self, query: str, top_k: int) -> str:
        doc_count = self._collection.count()
        if doc_count == 0:
            return "Knowledge base is empty. No documents indexed yet."

        n_dense = min(top_k * 2, doc_count)
        dense = self._collection.query(query_texts=[query], n_results=n_dense)

        dense_ids = dense["ids"][0] if dense["ids"] else []
        dense_docs = dense["documents"][0] if dense["documents"] else []
        dense_distances = dense["distances"][0] if dense["distances"] else []
        dense_metadatas = dense["metadatas"][0] if dense["metadatas"] else []

        all_docs = self._collection.get()
        all_ids = all_docs["ids"]
        all_texts = all_docs["documents"] or []

        if not all_texts:
            return "No documents found."

        # BM25 sparse retrieval
        tokenized = [doc.lower().split() for doc in all_texts]
        bm25 = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores(query.lower().split())

        bm25_ranked = sorted(
            zip(all_ids, bm25_scores), key=lambda x: x[1], reverse=True
        )[:n_dense]

        # Reciprocal rank fusion (k=60)
        k = 60
        rrf_scores: dict[str, float] = {}

        for rank, doc_id in enumerate(dense_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        for rank, (doc_id, _) in enumerate(bm25_ranked):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        top_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[
            :top_k
        ]

        # Build result
        id_to_doc = dict(zip(all_ids, all_texts))
        id_to_meta = dict(zip(all_ids, all_docs.get("metadatas") or [{}] * len(all_ids)))

        results = []
        for i, doc_id in enumerate(top_ids, 1):
            text = id_to_doc.get(doc_id, "")
            meta = id_to_meta.get(doc_id, {})
            score = rrf_scores[doc_id]
            source = meta.get("source", "unknown") if meta else "unknown"
            results.append(
                f"[{i}] (score={score:.4f}, source={source})\n{text[:500]}"
            )

        return "\n\n".join(results) if results else "No relevant documents found."
