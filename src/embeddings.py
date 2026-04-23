from __future__ import annotations

from typing import Sequence
import numpy as np
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import EMBEDDING_MODEL


class EmbeddingPipeline:
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self.vectorizer: TfidfVectorizer | None = None

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        # Manual embedding pipeline based on TF-IDF features.
        self.vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
        x = self.vectorizer.fit_transform(list(texts))
        dense = x.toarray().astype(np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-8
        return dense / norms

    def embed_query(self, query: str) -> np.ndarray:
        if self.vectorizer is None:
            raise RuntimeError("Vectorizer not fitted. Build index before querying.")
        x = self.vectorizer.transform([query]).toarray().astype(np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        return (x / norms)[0]

    def save(self, path: Path) -> None:
        if self.vectorizer is None:
            raise RuntimeError("Vectorizer not fitted.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: Path) -> None:
        with path.open("rb") as f:
            self.vectorizer = pickle.load(f)
