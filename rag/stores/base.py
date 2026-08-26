"""
BaseVectorStore
===============
Abstract interface for vector stores. Implement this to plug in any database:
Pinecone, pgvector, Weaviate, Qdrant, FAISS, Chroma — your choice.

The pipeline only calls:
    add_documents()         — index chunks
    similarity_search()     — retrieve by semantic similarity

Everything else is optional.

Usage (implementing a custom store):
    from rag.stores.base import BaseVectorStore
    from langchain.schema import Document
    from typing import List, Optional

    class MyStore(BaseVectorStore):
        def add_documents(self, documents: List[Document]) -> List[str]:
            # index docs, return their IDs
            ...

        def similarity_search(self, query: str, k: int = 5, ...) -> List[Document]:
            # embed query, retrieve top-k
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from langchain.schema import Document


class BaseVectorStore(ABC):
    """
    Plug-in interface for vector stores.

    Required to implement:
        - add_documents
        - similarity_search

    Optional to override (have sensible defaults):
        - similarity_search_with_score
        - delete
        - clear
        - from_documents  (classmethod)
    """

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Index a list of LangChain Document objects.

        Returns
        -------
        List[str]
            IDs of the indexed documents (for later deletion if needed).
        """
        ...

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        """
        Return the top-k most semantically similar documents.

        Parameters
        ----------
        query : str
            Natural language query.
        k : int
            Number of results to return.
        filter : dict, optional
            Metadata filter — syntax varies by backend.

        Returns
        -------
        List[Document]
        """
        ...

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Return (document, relevance_score) pairs.
        Default implementation calls similarity_search and assigns score=1.0.
        Override for native score support.
        """
        docs = self.similarity_search(query, k=k, filter=filter)
        return [(doc, 1.0) for doc in docs]

    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID. Override if your store supports this."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support delete(). "
            "Override this method to enable deletion."
        )

    def clear(self) -> None:
        """Remove all documents from the store. Override if supported."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support clear(). "
            "Override this method to enable clearing."
        )

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        **kwargs,
    ) -> "BaseVectorStore":
        """
        Convenience: create a store and index documents in one call.

        Example:
            store = ChromaStore.from_documents(
                chunks,
                embedding_function=OpenAIEmbeddings(),
            )
        """
        store = cls(**kwargs)
        store.add_documents(documents)
        return store
