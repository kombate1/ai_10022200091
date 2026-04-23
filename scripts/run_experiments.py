from __future__ import annotations

import csv
from pathlib import Path

from src.config import INDEX_METADATA_PATH, INDEX_VECTORS_PATH, VECTORIZER_PATH
from src.embeddings import EmbeddingPipeline
from src.pipeline import RagPipeline
from src.retriever import HybridRetriever
from src.vector_store import NumpyVectorStore


EXPERIMENT_QUERIES = [
    "What are key spending priorities in Ghana's 2025 budget?",
    "Who won most votes in the election dataset and where?",
    "How does the budget discuss inflation and debt?",
    "Tell me the exact 2026 budget numbers.",  # adversarial / unavailable
]


def run() -> None:
    store = NumpyVectorStore(INDEX_VECTORS_PATH, INDEX_METADATA_PATH)
    store.load()
    embedder = EmbeddingPipeline()
    embedder.load(VECTORIZER_PATH)
    retriever = HybridRetriever(embedder, store)
    pipeline = RagPipeline(retriever)

    out_path = Path("logs/experiment_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query", "top_chunk_id", "top_similarity", "top_hybrid", "answer_preview", "baseline_preview"],
        )
        writer.writeheader()
        for query in EXPERIMENT_QUERIES:
            res = pipeline.ask(query)
            top = res["retrieved"][0] if res["retrieved"] else {}
            writer.writerow(
                {
                    "query": query,
                    "top_chunk_id": top.get("chunk_id", ""),
                    "top_similarity": top.get("similarity", ""),
                    "top_hybrid": top.get("hybrid_score", ""),
                    "answer_preview": (res["answer"] or "")[:180],
                    "baseline_preview": (res["baseline_answer"] or "")[:180],
                }
            )

    print(f"Wrote experiment logs to {out_path}")


if __name__ == "__main__":
    run()
