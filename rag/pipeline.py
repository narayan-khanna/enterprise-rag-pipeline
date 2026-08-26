"""
RAGPipeline
===========
The main entry point. Wires together:

    AutoLoader  →  AdaptiveChunker  →  VectorStore  →  LLM

You bring:
    - Your data (files, URLs, raw text)
    - Your vector store (Chroma, Pinecone, pgvector, or your own)
    - (Optionally) your LLM and embedding model

The pipeline handles the rest.

Quickstart:
    from rag import RAGPipeline
    from rag.stores.chroma import ChromaStore
    from langchain_openai import OpenAIEmbeddings

    store = ChromaStore(embedding_function=OpenAIEmbeddings())
    rag = RAGPipeline(vector_store=store)

    rag.ingest("my_report.pdf")
    answer = rag.query("What are the main findings?")
    print(answer)

    # With sources:
    result = rag.query("Who authored this?", return_sources=True)
    print(result["answer"])
    for doc in result["sources"]:
        print(doc.metadata)

    # Inspect what chunking strategy was chosen:
    profile = rag.profile_text(some_text)
    print(profile)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from .chunkers.adaptive_chunker import AdaptiveChunker
from .loaders.auto_loader import AutoLoader
from .stores.base import BaseVectorStore


_DEFAULT_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the context provided.
If the context doesn't contain enough information to answer confidently, say so.

Context:
{context}

Question: {question}

Answer:"""
)


class RAGPipeline:
    """
    End-to-end RAG pipeline with adaptive chunking and pluggable vector stores.

    Parameters
    ----------
    vector_store : BaseVectorStore
        Any store implementing add_documents() and similarity_search().
        Built-in options: ChromaStore, PineconeStore, PgVectorStore.
        Roll your own by subclassing BaseVectorStore.
    llm : BaseLanguageModel, optional
        Generation model. Defaults to gpt-4o-mini via ChatOpenAI.
    chunk_size : int
        Base chunk size passed to AdaptiveChunker (it may adjust this).
    chunk_overlap : int
        Base overlap passed to AdaptiveChunker.
    top_k : int
        Number of chunks retrieved per query.
    prompt : ChatPromptTemplate, optional
        Override the default RAG prompt.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        llm: Optional[BaseLanguageModel] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
        prompt: Optional[ChatPromptTemplate] = None,
    ):
        self.vector_store = vector_store
        self.chunker = AdaptiveChunker(
            default_chunk_size=chunk_size,
            default_overlap=chunk_overlap,
        )
        self.loader = AutoLoader()
        self.top_k = top_k

        if llm is None:
            try:
                from langchain_openai import ChatOpenAI
                self.llm: BaseLanguageModel = ChatOpenAI(
                    model="gpt-4o-mini", temperature=0
                )
            except ImportError:
                raise ImportError(
                    "langchain-openai is required for the default LLM: "
                    "pip install langchain-openai\n"
                    "Or pass your own llm= parameter."
                )
        else:
            self.llm = llm

        self._prompt = prompt or _DEFAULT_PROMPT
        self._chain = self._build_chain()

    def ingest(
        self,
        source: Union[str, Path, List],
        force_strategy: Optional[str] = None,
        extra_metadata: Optional[dict] = None,
    ) -> int:
        """Load, chunk, and index one or more documents. Returns chunk count."""
        docs = self.loader.load(source)
        if extra_metadata:
            for doc in docs:
                doc.metadata.update(extra_metadata)
        chunks = self.chunker.chunk_documents(docs, force_strategy=force_strategy)
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def ingest_text(
        self,
        text: str,
        source: str = "inline",
        extra_metadata: Optional[dict] = None,
        force_strategy: Optional[str] = None,
    ) -> int:
        """Ingest raw text without creating a file."""
        metadata = {"source": source, **(extra_metadata or {})}
        chunks = self.chunker.chunk(text, metadata=metadata, force_strategy=force_strategy)
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        return_sources: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """Ask a question against indexed documents."""
        top_k = k or self.top_k
        retrieved = self.vector_store.similarity_search(question, k=top_k, filter=filter)
        context = "\n\n---\n\n".join(doc.page_content for doc in retrieved)
        answer = self._chain.invoke({"context": context, "question": question})
        if return_sources:
            return {"answer": answer, "sources": retrieved}
        return answer

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        with_scores: bool = False,
    ) -> List[Document]:
        """Retrieve relevant chunks without generating an answer."""
        top_k = k or self.top_k
        if with_scores:
            return self.vector_store.similarity_search_with_score(query, k=top_k, filter=filter)
        return self.vector_store.similarity_search(query, k=top_k, filter=filter)

    def profile_text(self, text: str) -> dict:
        """Analyze text and show what chunking strategy would be applied."""
        return self.chunker.profile(text).as_dict()

    def _build_chain(self):
        return (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self._prompt
            | self.llm
            | StrOutputParser()
        )
