"""
AdaptiveChunker demo — see how content type detection works.
Run this to understand what the chunker does to different inputs.
"""

from rag.chunkers.adaptive_chunker import AdaptiveChunker

chunker = AdaptiveChunker(default_chunk_size=512)

samples = {
    "Python code": """
import os
from typing import List

class DataProcessor:
    def __init__(self, config: dict):
        self.config = config

    def process(self, items: List[str]) -> List[str]:
        return [self._clean(item) for item in items]

    def _clean(self, text: str) -> str:
        return text.strip().lower()
""",
    "Markdown docs": """
# Getting Started

## Installation

Install the package using pip:

```bash
pip install my-package
```

## Configuration

### Environment Variables

Set the following environment variables before running:

- `API_KEY` — your API key
- `LOG_LEVEL` — logging level (default: INFO)

## Usage

Import the main class and initialize it:
""",
    "Dense technical": """
The transformer architecture employs multi-head self-attention mechanisms
to compute contextualized representations of input token sequences.
Positional encodings are superimposed onto embedding vectors to preserve
sequential ordering information absent from the attention mechanism itself.
Layer normalization and residual connections stabilize gradient propagation
during backpropagation through deep architectures.
""",
    "CSV data": """id,name,score,category
1,Alice,92.5,A
2,Bob,78.3,B
3,Charlie,88.1,A
4,Diana,95.0,A
5,Eve,71.2,C
""",
}

for name, text in samples.items():
    profile = chunker.profile(text)
    chunks = chunker.chunk(text, metadata={"sample": name})
    print(f"\n{'='*60}")
    print(f"  Sample: {name}")
    print(f"  Detected type:   {profile.detected_type.name}")
    print(f"  Strategy:        {profile.recommended_strategy}")
    print(f"  Chunk size:      {profile.recommended_chunk_size}")
    print(f"  Overlap:         {profile.recommended_overlap}")
    print(f"  Chunks produced: {len(chunks)}")
