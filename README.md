# 🔗 enterprise-rag-pipeline

> Production-grade RAG system built for enterprise scale — from Naive RAG to Graph RAG to Corrective RAG (CRAG).

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=flat-square&logo=amazonaws&logoColor=white)

---

## The Problem With Most RAG Systems

Most RAG implementations fail in production. They retrieve the wrong chunks, hallucinate on edge cases, and have no mechanism to detect when retrieval confidence is low.

This repo documents the full evolution from a basic RAG pipeline to a production-hardened system.

---

## Architecture Evolution

### Stage 1 — Naive RAG
- Single-vector semantic search
- No query rewriting, no re-ranking
- Fails on multi-hop questions

### Stage 2 — Graph RAG
- Knowledge graph over entities and relationships
- Multi-hop traversal for complex queries
- Handles: "What did the AI team ship in Q3 that impacted the automotive segment?"

### Stage 3 — Corrective RAG (CRAG)
- Self-evaluation layer on retrieved context
- Confidence scoring per chunk
- Falls back to web search when retrieval confidence < threshold
- Dramatically reduces hallucination rate

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain / LangGraph |
| Embeddings | OpenAI text-embedding-3-large |
| Vector Store | Pinecone (prod) / pgvector (dev) |
| Graph DB | Neo4j |
| LLM | AWS Bedrock (Claude 3) |
| Evaluation | RAGAS |
| Deployment | AWS Lambda + API Gateway |

---

## Key Results

- Reduced hallucination rate by ~60% vs Naive RAG baseline
- Multi-hop query accuracy improved 3x with Graph RAG
- CRAG self-correction loop catches low-confidence retrievals before they reach the user

---

## Connect

Built this? Hiring for something similar?

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Narayan_Khanna-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/narayank07051993/)
