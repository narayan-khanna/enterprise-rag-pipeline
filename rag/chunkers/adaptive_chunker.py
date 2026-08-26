"""
Adaptive Chunker
================
Analyzes document content and selects the optimal chunking strategy
automatically — chunk size, overlap, and splitting method all adapt
to what the data actually looks like.

Content types handled:
    CODE_PYTHON    → PythonCodeTextSplitter (function/class boundaries)
    CODE_JS        → JS-aware RecursiveCharacterTextSplitter
    CODE_GENERIC   → Recursive with code-friendly separators
    MARKDOWN       → Header-first split, then recursive per-section
    JSON           → Large chunks, zero overlap (structure matters)
    CSV            → Row-aware small chunks
    DENSE_TECHNICAL→ Smaller chunks (256-400), more overlap (20%)
    NARRATIVE      → Larger chunks (700-1024), less overlap (10%)
    MIXED          → Defaults

Usage:
    from rag.chunkers.adaptive_chunker import AdaptiveChunker

    chunker = AdaptiveChunker()
    chunks = chunker.chunk(text, metadata={"source": "report.pdf"})

    # See what was detected before chunking:
    profile = chunker.profile(text)
    print(profile.detected_type, profile.recommended_chunk_size)

    # Force a strategy if you know better:
    chunks = chunker.chunk(text, force_strategy="markdown")
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
    Language,
)
from langchain.schema import Document


class ContentType(Enum):
    CODE_PYTHON = auto()
    CODE_JS = auto()
    CODE_GENERIC = auto()
    MARKDOWN = auto()
    JSON = auto()
    CSV = auto()
    DENSE_TECHNICAL = auto()
    NARRATIVE = auto()
    MIXED = auto()


@dataclass
class ContentProfile:
    """Analyzed profile of a document's content."""
    detected_type: ContentType
    avg_sentence_length: float     # words per sentence
    density_score: float           # 0-1, higher = denser/more technical
    structure_score: float         # 0-1, higher = more structured
    code_ratio: float              # fraction of lines that look like code
    recommended_chunk_size: int
    recommended_overlap: int
    recommended_strategy: str      # "recursive" | "markdown" | "python" | "javascript"

    def as_dict(self) -> dict:
        return {
            "detected_type": self.detected_type.name,
            "avg_sentence_length": round(self.avg_sentence_length, 1),
            "density_score": round(self.density_score, 3),
            "structure_score": round(self.structure_score, 3),
            "code_ratio": round(self.code_ratio, 3),
            "recommended_chunk_size": self.recommended_chunk_size,
            "recommended_overlap": self.recommended_overlap,
            "recommended_strategy": self.recommended_strategy,
        }


class AdaptiveChunker:
    """
    Auto-detects content type and selects the best chunking strategy.

    Parameters
    ----------
    default_chunk_size : int
        Base chunk size; the chunker scales from this depending on content.
    default_overlap : int
        Base overlap; scaled with chunk size.
    min_chunk_size : int
        Floor — never go below this.
    max_chunk_size : int
        Ceiling — never exceed this.
    """

    # Detection thresholds
    _CODE_LINE_RATIO = 0.15         # >15% of lines look like code
    _HEADER_STRUCTURED = 5          # >5 markdown headers = structured doc
    _DENSE_TECH_WORD_LEN = 6.5      # avg word length > 6.5 = technical
    _NARRATIVE_SENT_LEN = 25        # avg words/sentence > 25 = narrative

    def __init__(
        self,
        default_chunk_size: int = 512,
        default_overlap: int = 64,
        min_chunk_size: int = 128,
        max_chunk_size: int = 2048,
    ):
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def profile(self, text: str) -> ContentProfile:
        lines = text.split("\n")
        non_empty = [l for l in lines if l.strip()]
        code_line_count = sum(1 for l in non_empty if self._line_is_code(l))
        code_ratio = code_line_count / max(len(non_empty), 1)
        detected_type = self._detect_type(text, code_ratio)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sent_lens = [len(s.split()) for s in sentences if len(s.split()) > 2]
        avg_sentence_length = statistics.mean(sent_lens) if sent_lens else 15.0
        words = text.split()
        cleaned = [w.strip(".,;:!?\"'()[]{}") for w in words[:500]]
        avg_word_len = statistics.mean([len(w) for w in cleaned if w]) if cleaned else 5.0
        density_score = min(avg_word_len / 10.0, 1.0)
        header_count = len(re.findall(r"^#{1,6}\s", text, re.MULTILINE))
        list_count = len(re.findall(r"^[\-\*\+]|\d+\.\s", text, re.MULTILINE))
        structure_score = min((header_count + list_count * 0.5) / 20.0, 1.0)
        chunk_size, overlap, strategy = self._recommend(detected_type, density_score, structure_score, avg_sentence_length, code_ratio)
        return ContentProfile(detected_type=detected_type, avg_sentence_length=avg_sentence_length, density_score=density_score, structure_score=structure_score, code_ratio=code_ratio, recommended_chunk_size=chunk_size, recommended_overlap=overlap, recommended_strategy=strategy)

    def chunk(self, text: str, metadata=None, force_strategy=None):
        metadata = metadata or {}
        profile = self.profile(text)
        strategy = force_strategy or profile.recommended_strategy
        chunk_size = profile.recommended_chunk_size
        overlap = profile.recommended_overlap
        metadata["_rag_content_type"] = profile.detected_type.name
        metadata["_rag_strategy"] = strategy
        if strategy == "python":
            splitter = PythonCodeTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
            return splitter.create_documents([text], metadatas=[metadata])
        if strategy == "javascript":
            splitter = RecursiveCharacterTextSplitter.from_language(language=Language.JS, chunk_size=chunk_size, chunk_overlap=overlap)
            return splitter.create_documents([text], metadatas=[metadata])
        if strategy == "markdown":
            return self._split_markdown(text, metadata, chunk_size, overlap)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""])
        return splitter.create_documents([text], metadatas=[metadata])

    def chunk_documents(self, documents, force_strategy=None):
        result = []
        for doc in documents:
            chunks = self.chunk(doc.page_content, metadata=doc.metadata.copy(), force_strategy=force_strategy)
            result.extend(chunks)
        return result

    def _line_is_code(self, line):
        patterns = [r"^\s*(def |class |import |from |if |for |while |return |async |await )", r"^\s*(const |let |var |function |export |require\()", r"[{};]\s*$", r"^\s*//|^\s*/\*|^\s*\*", r"^\s*(public|private|protected|static|void|int|str|bool)\b", r"^\s*#\s*(include|define|pragma|ifndef|endif)"]
        return any(re.search(p, line) for p in patterns)

    def _detect_type(self, text, code_ratio):
        head = text[:3000]
        if code_ratio > self._CODE_LINE_RATIO:
            if re.search(r"\bdef \w+\(|^import |^from \w+ import", head, re.MULTILINE):
                return ContentType.CODE_PYTHON
            if re.search(r"\bfunction\b|const \w+ =|=>\s*{|require\(", head):
                return ContentType.CODE_JS
            return ContentType.CODE_GENERIC
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                import json; json.loads(stripped[:1000]); return ContentType.JSON
            except: pass
        newlines = text.count("\n") or 1
        if text.count(",") > newlines * 2 and newlines > 3:
            return ContentType.CSV
        header_count = len(re.findall(r"^#{1,6}\s", text, re.MULTILINE))
        if header_count >= self._HEADER_STRUCTURED:
            return ContentType.MARKDOWN
        words = text.split()
        cleaned = [w.strip(".,;:!?\"'()") for w in words[:300] if w]
        avg_word_len = statistics.mean([len(w) for w in cleaned if w]) if cleaned else 5.0
        if avg_word_len >= self._DENSE_TECH_WORD_LEN:
            return ContentType.DENSE_TECHNICAL
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sent_lens = [len(s.split()) for s in sentences if len(s.split()) > 2]
        avg_len = statistics.mean(sent_lens) if sent_lens else 15
        if avg_len >= self._NARRATIVE_SENT_LEN:
            return ContentType.NARRATIVE
        return ContentType.MIXED

    def _recommend(self, content_type, density_score, structure_score, avg_sentence_length, code_ratio):
        if content_type == ContentType.CODE_PYTHON: return 1024, 128, "python"
        if content_type == ContentType.CODE_JS: return 1024, 128, "javascript"
        if content_type == ContentType.CODE_GENERIC: return 800, 100, "recursive"
        if content_type == ContentType.MARKDOWN: return 600, 80, "markdown"
        if content_type == ContentType.JSON: return 1024, 0, "recursive"
        if content_type == ContentType.CSV: return 256, 0, "recursive"
        if content_type == ContentType.DENSE_TECHNICAL:
            size = max(self.min_chunk_size, int(self.default_chunk_size * (1 - density_score * 0.4)))
            size = min(size, self.max_chunk_size)
            return size, int(size * 0.20), "recursive"
        if content_type == ContentType.NARRATIVE:
            size = min(self.max_chunk_size, int(self.default_chunk_size * 1.6))
            return size, int(size * 0.08), "recursive"
        return self.default_chunk_size, self.default_overlap, "recursive"

    def _split_markdown(self, text, metadata, chunk_size, overlap):
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")])
        recursive = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        docs = []
        for section in header_splitter.split_text(text):
            sub = recursive.create_documents([section.page_content], metadatas=[{**metadata, **section.metadata}])
            docs.extend(sub)
        return docs
