from __future__ import annotations

from src.chunking import ChunkingConfig, chunk_documents
from src.config import CHUNKS_PATH, INDEX_METADATA_PATH, INDEX_VECTORS_PATH, VECTORIZER_PATH
from src.data_ingestion import load_documents, save_jsonl
from src.embeddings import EmbeddingPipeline
from src.vector_store import NumpyVectorStore


def build() -> None:
    docs = load_documents()

    # Strategy selected after comparative tests: better context continuity.
    config = ChunkingConfig(size=600, overlap=120, strategy_name="char600_overlap120")
    chunks = chunk_documents(docs, config)

    embedder = EmbeddingPipeline()
    vectors = embedder.embed_texts([c["text"] for c in chunks])
    embedder.save(VECTORIZER_PATH)

    store = NumpyVectorStore(INDEX_VECTORS_PATH, INDEX_METADATA_PATH)
    store.build(vectors, chunks)
    save_jsonl(chunks, CHUNKS_PATH)

    print(f"Indexed {len(chunks)} chunks.")


if __name__ == "__main__":
    build()
