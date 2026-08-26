"""
enterprise-rag-pipeline
=======================
A plug-and-play RAG utility with adaptive chunking and swappable vector stores.

Quick import:
    from rag import RAGPipeline
    from rag.stores.chroma import ChromaStore
    from langchain_openai import OpenAIEmbeddings

    store = ChromaStore(embedding_function=OpenAIEmbeddings())
    rag = RAGPipeline(vector_store=store)
    rag.ingest("my_doc.pdf")
    print(rag.query("What is this about?"))

One-liner:
    from rag import quick_rag
    print(quick_rag("report.pdf", "What are the findings?"))
"""

from .pipeline import RAGPipeline
from .chunkers.adaptive_chunker import AdaptiveChunker
from .loaders.auto_loader import AutoLoader

__version__ = "0.1.0"
__all__ = ["RAGPipeline", "AdaptiveChunker", "AutoLoader", "quick_rag"]


def quick_rag(
    source,
    question: str,
    *,
    persist_dir: str = "./rag_data",
    openai_api_key: str | None = None,
) -> str:
    """
    One-liner RAG. Loads, indexes, and answers in a single call.
    Uses ChromaDB (local, no signup) + OpenAI embeddings.

    Parameters
    ----------
    source : str | list
        File path(s) or URL(s) to ingest.
    question : str
        Your question.
    persist_dir : str
        Where to persist the Chroma index (default: ./rag_data).
    openai_api_key : str, optional
        Override OPENAI_API_KEY env var.

    Example
    -------
    from rag import quick_rag
    answer = quick_rag("report.pdf", "What are the key findings?")
    print(answer)
    """
    import os

    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    from langchain_openai import OpenAIEmbeddings
    from .stores.chroma import ChromaStore

    store = ChromaStore(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=persist_dir,
    )
    pipeline = RAGPipeline(vector_store=store)
    pipeline.ingest(source)
    return pipeline.query(question)
