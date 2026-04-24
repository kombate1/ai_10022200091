from __future__ import annotations

from typing import Sequence


SYSTEM_PROMPT = """You are an Academic City RAG assistant.
Rules:
1) Only answer from provided context.
2) If context is insufficient, clearly say what is missing.
3) Cite the chunk IDs you used.
4) Keep answers concise and factual.
"""


SYSTEM_PROMPT_QUOTE_FIRST = """You are an Academic City RAG assistant.
Rules:
1) Only answer from provided context.
2) First provide 1-3 short quotes from the context that directly support your answer.
3) If context is insufficient, say 'Not enough context provided' and ask for what is missing.
4) Cite the chunk IDs you used.
"""


SYSTEM_PROMPT_STRICT_REFUSAL = """You are an Academic City RAG assistant.
Rules:
1) Use only the provided context.
2) If the answer is not explicitly in the context, respond with: 'I cannot answer from the provided context.'
3) Do not guess or use external knowledge.
4) Cite the chunk IDs you used.
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


def build_prompt(
    user_query: str,
    chunks: Sequence[dict],
    *,
    max_context_chars: int = 3500,
    prompt_variant: str = "grounded",
) -> str:
    context = build_context(chunks, max_chars=max_context_chars)
    if prompt_variant == "quote_first":
        system = SYSTEM_PROMPT_QUOTE_FIRST
        suffix = "Answer with bullets and include a final line 'Sources: ...' listing chunk IDs."
    elif prompt_variant == "strict_refusal":
        system = SYSTEM_PROMPT_STRICT_REFUSAL
        suffix = "Answer with bullets when possible and include a final line 'Sources: ...' listing chunk IDs."
    else:
        system = SYSTEM_PROMPT
        suffix = "Answer with bullets and include a final line 'Sources: ...' listing chunk IDs."
    return (
        f"{system}\n"
        f"Context:\n{context}\n"
        f"User Question: {user_query}\n\n"
        f"{suffix}"
    )
