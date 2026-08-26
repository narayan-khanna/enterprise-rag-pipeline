"""
PgVectorStore
=============
PostgreSQL + pgvector vector store.

Best for:
    - Production: same DB you already run -- no extra infra
    - ACID transactions, row-level security, SQL joins
    - Hybrid search (combine with pg_trgm for BM25-style keyword search)

Prerequisites:
    pip install psycopg2-binary pgvector langchain-community

    -- In PostgreSQL (run once):
    CREATE EXTENSION IF NOT EXISTS vector;

Usage:
    from rag.stores.pgvector import PgVectorStore
    from langchain_openai import OpenAIEmbeddings

    store = PgVectorStore(
        connection_string="postgresql://user:pass@host:5432/mydb",
        embedding_function=OpenAIEmbeddings(),
        collection_name="support_docs",
    )
    store.add_documents(chunks)
    results = store.similarity_search("question", k=5)

    # Filter by metadata:
    results = store.similarity_search(
        "question",
        filter={"source": "handbook.pdf"},
    )

    # Wipe and re-index cleanly:
    store = PgVectorStore(
        connection_string=DB_URL,
        embedding_function=OpenAIEmbeddings(),
        collection_name="my_docs",
        pre_delete_collection=True,   # drops existing data on init
    )
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from langchain.schema import Document
from langchain.embeddings.base import Embeddings

from .base import BaseVectorStore


class PgVectorStore(BaseVectorStore):
    """PostgreSQL + pgvector backed vector store."""

    def __init__(
        self,
        connection_string: str,
        embedding_function: Embeddings,
        collection_name: str = "rag_documents",
        pre_delete_collection: bool = False,
    ):
        try:
            from langchain_community.vectorstores.pgvector import PGVector
        except ImportError:
            raise ImportError(
                "pgvector support requires: pip install psycopg2-binary pgvector langchain-community"
            )

        self._store = PGVector(
            connection_string=connection_string,
            embedding_function=embedding_function,
            collection_name=collection_name,
            pre_delete_collection=pre_delete_collection,
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        return self._store.add_documents(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        return self._store.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        return self._store.similarity_search_with_score(query, k=k, filter=filter)

    def delete(self, ids: List[str]) -> None:
        self._store.delete(ids)
