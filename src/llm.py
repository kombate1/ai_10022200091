from __future__ import annotations

from openai import OpenAI
import streamlit as st

HF_MODEL = "zai-org/GLM-5.1:together"

def _get_client():
    hf_token = st.secrets["HF_TOKEN"]
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