from setuptools import setup, find_packages

setup(
    name="enterprise-rag-pipeline",
    version="0.1.0",
    description="Adaptive RAG utility with pluggable vector stores",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Narayan Khanna",
    url="https://github.com/narayan-khanna/enterprise-rag-pipeline",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "langchain>=0.2.0",
        "langchain-community>=0.2.0",
        "langchain-openai>=0.1.0",
        "openai>=1.0.0",
        "chromadb>=0.4.0",
        "pypdf>=3.0.0",
        "tiktoken>=0.5.0",
        "beautifulsoup4>=4.12.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "pinecone": ["pinecone-client>=3.0.0"],
        "pgvector": ["psycopg2-binary>=2.9.0", "pgvector>=0.2.0"],
        "docs": ["unstructured>=0.10.0", "python-docx>=0.8.11", "jq>=1.6.0"],
        "all": [
            "pinecone-client>=3.0.0",
            "psycopg2-binary>=2.9.0",
            "pgvector>=0.2.0",
            "unstructured>=0.10.0",
            "python-docx>=0.8.11",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
