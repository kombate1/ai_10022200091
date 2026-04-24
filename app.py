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
    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False

    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.subheader("Chats")
    with header_right:
        if st.button("Settings", use_container_width=True):
            st.session_state.show_settings = not st.session_state.show_settings

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
    show_details = st.checkbox("Show answer details (chunks, scores, prompts)", value=True)

    # Settings are hidden unless the user explicitly opens them.
    if st.session_state.show_settings:
        st.subheader("RAG Settings")
        top_k = st.slider("Top-k retrieved chunks", min_value=1, max_value=12, value=5, step=1)
        alpha = st.slider(
            "Hybrid weight (alpha for dense similarity)",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05,
        )
        use_query_expansion = st.checkbox("Use query expansion", value=True)
        use_year_alignment = st.checkbox("Use year-alignment fix", value=True)
        max_context_chars = st.slider("Max context characters", min_value=500, max_value=6000, value=3500, step=250)
        prompt_variant = st.selectbox(
            "Prompt variant",
            options=["grounded", "quote_first", "strict_refusal"],
            index=0,
        )
        include_baseline = st.checkbox("Show pure LLM baseline", value=True)
    else:
        # Defaults used when settings are hidden.
        top_k = 5
        alpha = 0.8
        use_query_expansion = True
        use_year_alignment = True
        max_context_chars = 3500
        prompt_variant = "grounded"
        include_baseline = True

    st.divider()
    st.subheader("How to run")
    st.code("python -m scripts.build_index\nstreamlit run app.py")
    


query = st.chat_input("Ask about Ghana election results or the 2025 budget statement...")
if query:
    msgs = _current_messages()
    msgs.append({"role": "user", "content": query})
    with st.spinner("Retrieving and generating..."):
        result = get_pipeline().ask(
            query,
            top_k=top_k,
            alpha=alpha,
            use_query_expansion=use_query_expansion,
            use_year_alignment=use_year_alignment,
            prompt_variant=prompt_variant,
            max_context_chars=max_context_chars,
            include_baseline=include_baseline,
        )
    # Store full retrieval context inside the chat turn for later review.
    msgs.append({"role": "assistant", "content": result["answer"], "result": result})

for msg in _current_messages():
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "result" in msg:
            res = msg["result"]
            if show_details:
                if "settings" in res:
                    with st.expander("Run Settings", expanded=False):
                        st.json(res["settings"])
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
                if res.get("baseline_answer"):
                    with st.expander("Pure LLM Baseline (No Retrieval)", expanded=False):
                        st.write(res["baseline_answer"])
