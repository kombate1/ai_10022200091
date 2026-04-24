from __future__ import annotations

import re

from src.embeddings import EmbeddingPipeline
from src.vector_store import NumpyVectorStore


QUERY_EXPANSIONS = {
    "budget": ["fiscal policy", "government spending", "appropriation"],
    "election": ["votes", "constituency", "results"],
    "inflation": ["cpi", "price level", "cost of living"],
}


def expand_query(query: str) -> str:
    q = query.lower()
    additions: list[str] = []
    for key, extra in QUERY_EXPANSIONS.items():
        if key in q:
            additions.extend(extra)
    return query if not additions else f"{query} {' '.join(additions)}"


def keyword_score(query: str, text: str) -> float:
    q_terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
    if not q_terms:
        return 0.0
    txt = text.lower()
    hits = sum(1 for term in q_terms if term in txt)
    return hits / len(q_terms)


def year_alignment_score(query: str, text: str) -> float:
    q_years = set(re.findall(r"\b(19\d{2}|20\d{2})\b", query))
    if not q_years:
        return 0.0
    t_years = set(re.findall(r"\b(19\d{2}|20\d{2})\b", text))
    if not t_years:
        return -0.15
    overlap = len(q_years.intersection(t_years))
    if overlap == 0:
        return -0.1
    return min(0.2, 0.1 * overlap)


class HybridRetriever:
    def __init__(self, embedding: EmbeddingPipeline, store: NumpyVectorStore) -> None:
        self.embedding = embedding
        self.store = store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.8,
        use_query_expansion: bool = True,
        use_year_alignment: bool = True,
    ) -> list[dict]:
        expanded = expand_query(query) if use_query_expansion else query
        q_vec = self.embedding.embed_query(expanded)
        dense_results = self.store.search(q_vec, top_k=max(top_k * 3, 10))

        for row in dense_results:
            ks = keyword_score(query, row["text"])
            ys = year_alignment_score(query, row["text"]) if use_year_alignment else 0.0
            row["keyword_score"] = ks
            row["year_score"] = ys
            row["hybrid_score"] = alpha * row["similarity"] + (1 - alpha) * ks + ys

        ranked = sorted(dense_results, key=lambda x: x["hybrid_score"], reverse=True)[:top_k]
        return ranked
