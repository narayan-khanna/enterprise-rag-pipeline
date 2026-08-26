"""
Pinecone example — swap ChromaStore → PineconeStore.
Everything else is identical to quickstart.py.

Prerequisites:
    pip install pinecone-client langchain-community
    export OPENAI_API_KEY=sk-...
    export PINECONE_API_KEY=...

    # Create the index once (run this separately):
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone()
    pc.create_index(
        name="rag-index",
        dimension=1536,        # matches text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from rag import RAGPipeline
from rag.stores.pinecone import PineconeStore   # only change vs quickstart

store = PineconeStore(
    index_name="rag-index",          # your Pinecone index name
    embedding_function=OpenAIEmbeddings(),
    namespace="production",          # optional: isolate by namespace
)

rag = RAGPipeline(
    vector_store=store,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
)

# Ingest multiple sources — all get adaptively chunked and indexed
rag.ingest([
    "docs/handbook.pdf",
    "docs/faq.md",
    "https://yoursite.com/release-notes",
])

# Filter results by source document
answer = rag.query(
    "What changed in the latest release?",
    filter={"source": {"$eq": "https://yoursite.com/release-notes"}},
)
print(answer)
