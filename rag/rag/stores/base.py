from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from langchain.schema import Document


class BaseVectorStore(ABC):
    """
    Abstract interface for vector stores.
    Implement add_documents() and similarity_search() to plug in any backend.

    Built-in implementations:
        ChromaStore   -- local, zero-config (chromadb)
        PineconeStore -- managed cloud
        PgVectorStore -- PostgreSQL + pgvector

    Roll your own:
        class MyStore(BaseVectorStore):
            def add_documents(self, documents): ...
            def similarity_search(self, query, k=5, filter=None): ...
    """

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Index documents. Returns list of IDs."""
        ...

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        """Return top-k most semantically similar documents."""
        ...

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Return (document, score) pairs. Override for native score support."""
        docs = self.similarity_search(query, k=k, filter=filter)
        return [(doc, 1.0) for doc in docs]

    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID. Override if your store supports this."""
        raise NotImplementedError(f"{type(self).__name__} does not support delete().")

    def clear(self) -> None:
        """Remove all documents. Override if supported."""
        raise NotImplementedError(f"{type(self).__name__} does not support clear().")

    @classmethod
    def from_documents(cls, documents: List[Document], **kwargs) -> "BaseVectorStore":
        """Create a store and index documents in one call."""
        store = cls(**kwargs)
        store.add_documents(documents)
        return store
