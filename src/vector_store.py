from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class NumpyVectorStore:
    def __init__(self, vectors_path: Path, metadata_path: Path) -> None:
        self.vectors_path = vectors_path
        self.metadata_path = metadata_path
        self.vectors: np.ndarray | None = None
        self.metadata: list[dict[str, Any]] = []

    def build(self, vectors: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        self.vectors_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.vectors_path, vectors)
        self.metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
        self.vectors = vectors
        self.metadata = metadata

    def load(self) -> None:
        self.vectors = np.load(self.vectors_path)
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        if self.vectors is None:
            self.load()
        assert self.vectors is not None
        scores = self.vectors @ query_vector
        idxs = np.argsort(-scores)[:top_k]
        results = []
        for idx in idxs:
            item = dict(self.metadata[int(idx)])
            item["similarity"] = float(scores[int(idx)])
            results.append(item)
        return results
