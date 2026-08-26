"""
ChromaStore
===========
Local, zero-config vector store backed by ChromaDB.
No API key, no cloud account, no Docker — just pip install and go.

Best for:
    - Development and prototyping
    - Datasets under ~100k documents
    - Offline / air-gapped environments

Usage:
    from rag.stores.chroma import ChromaStore
    from langchain_openai import OpenAIEmbeddings

    # In-memory (lost on restart):
    store = ChromaStore(embedding_function=OpenAIEmbeddings())

    # Persistent (survives restarts):
    store = ChromaStore(
        embedding_function=OpenAIEmbeddings(),
        persist_directory="./chroma_db",
    )

    store.add_documents(chunks)
    results = store.similarity_search("my question", k=5)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from langchain.schema import Document
from langchain.embeddings.base import Embeddings

from .base import BaseVectorStore


class ChromaStore(BaseVectorStore):
    """Chroma-backed vector store — zero infra, local-first."""

    def __init__(
        self,
        embedding_function: Embeddings,
        collection_name: str = "rag_collection",
        persist_directory: Optional[str] = None,
    ):
        try:
            from langchain_community.vectorstores import Chroma
        except ImportError:
            raise ImportError(
                "chromadb is required: pip install chromadb langchain-community"
            )

        kwargs: dict = {
            "collection_name": collection_name,
            "embedding_function": embedding_function,
        }
        if persist_directory:
            kwargs["persist_directory"] = persist_directory

        self._store = Chroma(**kwargs)

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

    def clear(self) -> None:
        col = self._store._collection
        existing = col.get()
        if existing["ids"]:
            col.delete(ids=existing["ids"])
