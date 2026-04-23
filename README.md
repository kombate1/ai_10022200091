# Academic City Manual RAG Chatbot

This project implements a full Retrieval-Augmented Generation (RAG) chatbot for Academic City using:
- `Ghana_Election_Result.csv`
- Ghana 2025 Budget Statement PDF

It intentionally **does not** use LangChain, LlamaIndex, or any end-to-end RAG framework.
Core pieces are hand-built: ingestion, cleaning, chunking, embeddings, vector search, hybrid ranking, prompt construction, context management, and stage-by-stage logging.

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional for generation:

```bash
set OPENAI_API_KEY=your_key_here
```

## 2) Build Index

```bash
python -m scripts.build_index
```

This downloads the datasets, cleans CSV data, extracts PDF text, chunks documents, computes embeddings, and writes:
- `data/processed/chunks.jsonl`
- `data/index/vectors.npy`
- `data/index/metadata.json`

## 3) Run App

```bash
streamlit run app.py
```

UI features:
- Query input
- Retrieved chunks display
- Similarity and hybrid scores
- Final response
- Final prompt sent to LLM
- Baseline pure LLM output (no retrieval)

## 4) Run Experiments

```bash
python -m scripts.run_experiments
```

Generated logs:
- `logs/pipeline_logs.jsonl`
- `logs/experiment_results.csv`

## 5) Project Structure

- `app.py` - Streamlit UI
- `src/data_ingestion.py` - dataset download, cleaning, PDF extraction
- `src/chunking.py` - configurable chunking
- `src/embeddings.py` - SentenceTransformer embeddings
- `src/vector_store.py` - custom NumPy vector index + cosine similarity
- `src/retriever.py` - top-k retrieval + hybrid search + query expansion
- `src/prompting.py` - prompt templates and context window management
- `src/llm.py` - LLM and pure-LLM baseline
- `src/pipeline.py` - complete RAG pipeline with logging
- `docs/` - architecture, analysis, logs, video walkthrough script

## 6) Constraint Compliance

- No end-to-end RAG frameworks used.
- Manual retrieval logic implemented.
- Manual prompt construction and context filtering implemented.
- Logging for retrieval, scores, and final prompt implemented.
- Embedding pipeline is custom TF-IDF vectorization (implemented manually in code).
