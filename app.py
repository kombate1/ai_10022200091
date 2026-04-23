from __future__ import annotations

import streamlit as st

from src.config import INDEX_METADATA_PATH, INDEX_VECTORS_PATH, VECTORIZER_PATH
from src.embeddings import EmbeddingPipeline
from src.pipeline import RagPipeline
from src.retriever import HybridRetriever
from src.vector_store import NumpyVectorStore


st.set_page_config(page_title="Academic City RAG Assistant", layout="wide")
st.title("Academic City RAG Assistant")
st.caption("Manual RAG pipeline: retrieval -> context selection -> prompt -> generation")


@st.cache_resource
def get_pipeline() -> RagPipeline:
    store = NumpyVectorStore(INDEX_VECTORS_PATH, INDEX_METADATA_PATH)
    store.load()
    embedder = EmbeddingPipeline()
    embedder.load(VECTORIZER_PATH)
    retriever = HybridRetriever(embedder, store)
    return RagPipeline(retriever)


if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ask about Ghana election results or the 2025 budget statement...")
if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.spinner("Retrieving and generating..."):
        result = get_pipeline().ask(query)
    st.session_state.history.append({"role": "assistant", "content": result["answer"], "result": result})

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "result" in msg:
            res = msg["result"]
            with st.expander("Retrieved Chunks + Similarity Scores", expanded=False):
                for i, r in enumerate(res["retrieved"], start=1):
                    st.markdown(
                        f"**{i}.** `{r['chunk_id']}` | sim={r['similarity']:.4f} | "
                        f"hybrid={r['hybrid_score']:.4f} | keyword={r['keyword_score']:.4f}"
                    )
                    st.write(r["text"][:500] + ("..." if len(r["text"]) > 500 else ""))
                    st.divider()
            with st.expander("Final Prompt Sent to LLM", expanded=False):
                st.code(res["prompt"])
            with st.expander("Pure LLM Baseline (No Retrieval)", expanded=False):
                st.write(res["baseline_answer"])

with st.sidebar:
    st.subheader("How to run")
    st.code("python -m scripts.build_index\nstreamlit run app.py")
    st.markdown(
        "- No LangChain/LlamaIndex used\n"
        "- Embeddings: custom TF-IDF pipeline\n"
        "- Vector store: custom NumPy cosine search\n"
        "- Retrieval extension: query expansion + hybrid scoring"
    )
