from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class ChunkingConfig:
    size: int = 600
    overlap: int = 120
    strategy_name: str = "char_window"


def chunk_text(text: str, config: ChunkingConfig) -> list[str]:
    if not text.strip():
        return []

    chunks: list[str] = []
    step = max(1, config.size - config.overlap)
    start = 0
    while start < len(text):
        end = min(len(text), start + config.size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def chunk_documents(docs: Iterable[dict], config: ChunkingConfig) -> list[dict]:
    out: list[dict] = []
    for doc in docs:
        chunks = chunk_text(doc["text"], config)
        for i, chunk in enumerate(chunks):
            out.append(
                {
                    "chunk_id": f"{doc['doc_id']}::{config.strategy_name}::{i}",
                    "doc_id": doc["doc_id"],
                    "source": doc["source"],
                    "strategy": config.strategy_name,
                    "chunk_index": i,
                    "text": chunk,
                }
            )
    return out
