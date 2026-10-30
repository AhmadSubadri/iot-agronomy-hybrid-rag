"""Konfigurasi terpusat sistem RAG KawalTani (Versi Tesis S2 Terintegrasi)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Database MySQL (Default ke DB eksperimen agar DB asli tetap murni) ---
SOURCE_DB_NAME = "kawaltani_simpul_telkom"
DB = dict(
    host=os.getenv("KT_DB_HOST", "localhost"),
    port=int(os.getenv("KT_DB_PORT", "3306")),
    user=os.getenv("KT_DB_USER", "root"),
    password=os.getenv("KT_DB_PASSWORD", ""),
    database=os.getenv("KT_DB_NAME", "kawaltani_rag_tesis"),
)

# --- Vector Store & Embedding ---
VECTOR_DOCS_DIR = os.getenv("KT_DOCS_DIR", str(BASE_DIR / "vector_docs"))
INDEX_DIR = os.getenv("KT_INDEX_DIR", str(BASE_DIR / "index"))
EMBED_MODEL = os.getenv(
    "KT_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
CHUNK_WORDS = int(os.getenv("KT_CHUNK_WORDS", "180"))
CHUNK_OVERLAP_WORDS = int(os.getenv("KT_CHUNK_OVERLAP", "40"))
TOP_K = int(os.getenv("KT_TOP_K", "4"))
TEMPORAL_TOP_K = int(os.getenv("KT_TEMPORAL_TOP_K", "2"))

# --- LLM Generation (Tabel 6 Proposal: Temperature = 0.1) ---
LLM_PROVIDER = os.getenv("KT_LLM_PROVIDER", "openai")  # openai | gemini | ollama
LLM_MODEL = os.getenv("KT_LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("KT_LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("KT_LLM_MAX_TOKENS", "900"))

# --- Retrieval Window ---
DEFAULT_WINDOW_DAYS = int(os.getenv("KT_DEFAULT_WINDOW_DAYS", "7"))
MAX_WINDOW_DAYS = int(os.getenv("KT_MAX_WINDOW_DAYS", "120"))
MAX_DAILY_POINTS = int(os.getenv("KT_MAX_DAILY_POINTS", "14"))

# --- Pemetaan Area <-> Device & Tanggal Tanam Acuan ---
DEVICE_AREAS = {"TELU0300": [1, 2], "TELU0100": [3, 4, 5, 6]}
AREA_DEVICE = {a: d for d, areas in DEVICE_AREAS.items() for a in areas}

