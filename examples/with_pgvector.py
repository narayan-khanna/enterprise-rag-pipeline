"""
pgvector example — swap ChromaStore → PgVectorStore for PostgreSQL.
Same pipeline, different store. Good for production: no extra infra,
ACID transactions, SQL joins, row-level security.

Prerequisites:
    pip install psycopg2-binary pgvector langchain-community
    export OPENAI_API_KEY=sk-...

    -- In your PostgreSQL database (run once):
    CREATE EXTENSION IF NOT EXISTS vector;
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from rag import RAGPipeline
from rag.stores.pgvector import PgVectorStore   # only change vs quickstart

DB_URL = "postgresql://user:password@localhost:5432/mydb"

store = PgVectorStore(
    connection_string=DB_URL,
    embedding_function=OpenAIEmbeddings(),
    collection_name="support_kb",
    # pre_delete_collection=True,   # wipe and re-index cleanly
)

rag = RAGPipeline(
    vector_store=store,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
)

# Ingest a directory of files — AdaptiveChunker handles each file type
rag.ingest("knowledge_base/", extra_metadata={"department": "support"})

# Filter by your custom metadata
answer = rag.query(
    "How do I reset my password?",
    filter={"department": "support"},
)
print(answer)
