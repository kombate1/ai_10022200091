from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.config import LOG_DIR, TOP_K
from src.llm import generate_with_llm, generate_without_rag
from src.prompting import build_prompt
from src.retriever import HybridRetriever


class RagPipeline:
    def __init__(self, retriever: HybridRetriever, log_file: Path | None = None) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.retriever = retriever
        self.log_file = log_file or LOG_DIR / "pipeline_logs.jsonl"

    def ask(self, query: str) -> dict:
        retrieved = self.retriever.retrieve(query, top_k=TOP_K)
        prompt = build_prompt(query, retrieved)
        answer = generate_with_llm(prompt)
        baseline = generate_without_rag(query)

        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "retrieved": retrieved,
            "prompt": prompt,
            "answer": answer,
            "baseline_answer": baseline,
        }
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        return payload
