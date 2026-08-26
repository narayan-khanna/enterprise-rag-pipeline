"""
Quickstart — local RAG in under 5 minutes
==========================================

Prerequisites:
    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...

What happens:
    1. AutoLoader detects file type and loads content
    2. AdaptiveChunker profiles the content, picks chunk size + strategy
    3. Embeddings created with OpenAI, stored in local ChromaDB
    4. Question answered using retrieved context + GPT-4o-mini
"""

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# — Imports ————————————————————————————————————————————————————————————————
from rag import RAGPipeline
from rag.stores.chroma import ChromaStore

# — 1. Choose a vector store ———————————————————————————————————————————————
#    Swap ChromaStore for PineconeStore or PgVectorStore — nothing else changes.
store = ChromaStore(
    embedding_function=OpenAIEmbeddings(),
    collection_name="quickstart",
    persist_directory="./chroma_db",   # remove to use in-memory only
)

# — 2. Build the pipeline ——————————————————————————————————————————————————
rag = RAGPipeline(
    vector_store=store,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    chunk_size=512,    # AdaptiveChunker may adjust this per document type
    top_k=5,
)

# — 3. Ingest your data ————————————————————————————————————————————————————
# Accepts: .pdf, .docx, .txt, .md, .csv, .json, URLs, directories
n_chunks = rag.ingest("your_document.pdf")
print(f"Indexed {n_chunks} chunks")

# — 4. Query ———————————————————————————————————————————————————————————————
answer = rag.query("What are the key points in this document?")
print("\nAnswer:", answer)

# With source attribution:
result = rag.query(
    "What are the main conclusions?",
    return_sources=True,
)
print("\nAnswer:", result["answer"])
print("\nSources:")
for i, doc in enumerate(result["sources"], 1):
    preview = doc.page_content[:120].replace("\n", " ")
    source = doc.metadata.get("source", "unknown")
    print(f"  [{i}] {source} — {preview}...")
