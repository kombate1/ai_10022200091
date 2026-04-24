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

    def ask(
        self,
        query: str,
        *,
        top_k: int = TOP_K,
        alpha: float = 0.8,
        use_query_expansion: bool = True,
        use_year_alignment: bool = True,
        prompt_variant: str = "grounded",
        max_context_chars: int = 3500,
        include_baseline: bool = True,
    ) -> dict:
        retrieved = self.retriever.retrieve(
            query,
            top_k=top_k,
            alpha=alpha,
            use_query_expansion=use_query_expansion,
            use_year_alignment=use_year_alignment,
        )
        prompt = build_prompt(query, retrieved, max_context_chars=max_context_chars, prompt_variant=prompt_variant)
        answer = generate_with_llm(prompt)
        baseline = generate_without_rag(query) if include_baseline else ""

        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "retrieved": retrieved,
            "settings": {
                "top_k": top_k,
                "alpha": alpha,
                "use_query_expansion": use_query_expansion,
                "use_year_alignment": use_year_alignment,
                "prompt_variant": prompt_variant,
                "max_context_chars": max_context_chars,
                "include_baseline": include_baseline,
            },
            "prompt": prompt,
            "answer": answer,
            "baseline_answer": baseline,
        }
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        return payload
