from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
LOG_DIR = ROOT_DIR / "logs"

CSV_URL = "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/Ghana_Election_Result.csv"
PDF_URL = "https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf"

CSV_PATH = RAW_DIR / "Ghana_Election_Result.csv"
PDF_PATH = RAW_DIR / "2025_Budget_Statement.pdf"
PDF_TEXT_PATH = PROCESSED_DIR / "budget_statement_text.txt"
CLEAN_CSV_PATH = PROCESSED_DIR / "ghana_election_clean.csv"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
INDEX_VECTORS_PATH = INDEX_DIR / "vectors.npy"
INDEX_METADATA_PATH = INDEX_DIR / "metadata.json"
VECTORIZER_PATH = INDEX_DIR / "vectorizer.pkl"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_MODEL = os.getenv("HF_MODEL", "zai-org/GLM-5.1:together")
TOP_K = int(os.getenv("TOP_K", "5"))
