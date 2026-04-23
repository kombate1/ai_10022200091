from __future__ import annotations

import json
from pathlib import Path
import re
from typing import List

import pandas as pd
import requests
from pypdf import PdfReader

from src.config import (
    CLEAN_CSV_PATH,
    CSV_PATH,
    CSV_URL,
    PDF_PATH,
    PDF_TEXT_PATH,
    PDF_URL,
    PROCESSED_DIR,
    RAW_DIR,
)


def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, path: Path) -> None:
    if path.exists():
        return
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        cleaned = re.sub(r"[ \t]+", " ", raw).strip()
        pages.append(f"[Page {i + 1}]\n{cleaned}")
    return "\n\n".join(pages)


def clean_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.drop_duplicates()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df


def load_documents() -> list[dict]:
    _ensure_dirs()
    download_file(CSV_URL, CSV_PATH)
    download_file(PDF_URL, PDF_PATH)

    election_df = clean_csv(CSV_PATH)
    election_df.to_csv(CLEAN_CSV_PATH, index=False)

    pdf_text = extract_pdf_text(PDF_PATH)
    PDF_TEXT_PATH.write_text(pdf_text, encoding="utf-8")

    csv_as_text = election_df.to_csv(index=False)
    docs = [
        {"doc_id": "ghana_election_csv", "source": str(CLEAN_CSV_PATH.name), "text": csv_as_text},
        {"doc_id": "ghana_budget_pdf", "source": str(PDF_TEXT_PATH.name), "text": pdf_text},
    ]
    return docs


def save_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
