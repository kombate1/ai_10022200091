from __future__ import annotations

from typing import Sequence


SYSTEM_PROMPT = """You are an Academic City RAG assistant.
Rules:
1) Only answer from provided context.
2) If context is insufficient, clearly say what is missing.
3) Cite the chunk IDs you used.
4) Keep answers concise and factual.
"""


def build_context(chunks: Sequence[dict], max_chars: int = 3500) -> str:
    total = 0
    selected = []
    for c in chunks:
        block = f"[{c['chunk_id']}] {c['text']}\n"
        if total + len(block) > max_chars:
            break
        selected.append(block)
        total += len(block)
    return "\n".join(selected)


def build_prompt(user_query: str, chunks: Sequence[dict]) -> str:
    context = build_context(chunks)
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Context:\n{context}\n"
        f"User Question: {user_query}\n\n"
        "Answer with bullets and include a final line 'Sources: ...' listing chunk IDs."
    )
