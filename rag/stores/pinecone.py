"""
PineconeStore
=============
Managed vector store backed by Pinecone.

Best for:
    - Production workloads at scale
    - Multi-tenant isolation via namespaces
    - Built-in metadata filtering

Prerequisites:
    pip install pinecone-client langchain-community

    export PINECONE_API_KEY=your-key

    # Create the index first (once):
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key="...")
    pc.create_index(
        name="my-rag-index",
        dimension=1536,           # matches OpenAI text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

Usage:
    from rag.stores.pinecone import PineconeStore
    from langchain_openai import OpenAIEmbeddings

    store = PineconeStore(
        index_name="my-rag-index",
        embedding_function=OpenAIEmbeddings(),
        namespace="prod",          # optional — default ""
    )
    store.add_documents(chunks)
    results = store.similarity_search("question", k=5)

    # Metadata filter (Pinecone filter syntax):
    results = store.similarity_search(
        "question",
        filter={"source": {"$eq": "report.pdf"}},
    )
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from langchain.schema import Document
from langchain.embeddings.base import Embeddings

from .base import BaseVectorStore


class PineconeStore(BaseVectorStore):
    """Pinecone managed vector store."""

    def __init__(
        self,
        index_name: str,
        embedding_function: Embeddings,
        api_key: Optional[str] = None,
        namespace: str = "",
    ):
        try:
            from pinecone import Pinecone
            from langchain_community.vectorstores import Pinecone as LCPinecone
        except ImportError:
            raise ImportError(
                "Pinecone client is required: pip install pinecone-client langchain-community"
            )

        pc = Pinecone(api_key=api_key or os.environ["PINECONE_API_KEY"])
        index = pc.Index(index_name)

        self._store = LCPinecone(
            index=index,
            embedding=embedding_function,
            text_key="text",
            namespace=namespace,
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
