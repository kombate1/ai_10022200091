from __future__ import annotations

import os

from openai import OpenAI
import streamlit as st

HF_MODEL = "deepseek-ai/DeepSeek-R1:novita"

def _get_client():
    # Prefer environment variable (matches HuggingFace Router quickstart).
    # This will raise KeyError if not provided, which we convert to a clearer message.
    try:
        hf_token = os.environ["HF_TOKEN"]
    except KeyError as e:
        raise ValueError("HF_TOKEN environment variable is not set.") from e
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token,
    )

def generate_with_llm(prompt: str) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=HF_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""

def generate_without_rag(user_query: str) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=HF_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"Answer this directly without retrieval context: {user_query}",
            }
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""