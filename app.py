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


def _new_chat_id() -> str:
    n = int(st.session_state.get("chat_counter", 0)) + 1
    st.session_state.chat_counter = n
    return f"chat_{n}"  # stable string key


if "chats" not in st.session_state:
    first_id = "chat_1"
    st.session_state.chat_counter = 1
    st.session_state.chats = {
        first_id: {
            "title": "Chat 1",
            "messages": [],
        }
    }
    st.session_state.current_chat_id = first_id

if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chats:
    st.session_state.current_chat_id = next(iter(st.session_state.chats.keys()))


def _current_messages() -> list[dict]:
    return st.session_state.chats[st.session_state.current_chat_id]["messages"]

with st.sidebar:
    st.subheader("Chat history")

    chat_ids = list(st.session_state.chats.keys())
    chat_labels = [st.session_state.chats[cid]["title"] for cid in chat_ids]

    selected_label = st.radio(
        "Select chat",
        options=chat_labels,
        index=chat_ids.index(st.session_state.current_chat_id) if st.session_state.current_chat_id in chat_ids else 0,
        label_visibility="collapsed",
    )
    selected_id = chat_ids[chat_labels.index(selected_label)]
    if selected_id != st.session_state.current_chat_id:
        st.session_state.current_chat_id = selected_id

    if st.button("New chat", use_container_width=True):
        new_id = _new_chat_id()
        st.session_state.chats[new_id] = {"title": f"Chat {st.session_state.chat_counter}", "messages": []}
        st.session_state.current_chat_id = new_id

    st.divider()
    st.subheader("How to run")
    st.code("python -m scripts.build_index\nstreamlit run app.py")
    st.markdown(
        "- No LangChain/LlamaIndex used\n"
        "- Embeddings: custom TF-IDF pipeline\n"
        "- Vector store: custom NumPy cosine search\n"
        "- Retrieval extension: query expansion + hybrid scoring"
    )


query = st.chat_input("Ask about Ghana election results or the 2025 budget statement...")
if query:
    msgs = _current_messages()
    msgs.append({"role": "user", "content": query})
    with st.spinner("Retrieving and generating..."):
        result = get_pipeline().ask(query)
    # Store full retrieval context inside the chat turn for later review.
    msgs.append({"role": "assistant", "content": result["answer"], "result": result})

for msg in _current_messages():
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
