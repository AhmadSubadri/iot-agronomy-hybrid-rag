import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from kawaltani_rag.pipeline import RAGPipeline
import json

pipe = RAGPipeline()
q = "Berapa rata-rata nitrogen di area 3 dalam tujuh hari terakhir dan apakah perlu pemupukan tambahan?"

print("Mengirim kueri ke Hybrid RAG + Google Gemini...")
res = pipe.answer(q, provider="gemini", model="gemini-3.5-flash-lite")

print("=" * 72)
print("HASIL HYBRID RAG + GOOGLE GEMINI:")
print("Rute Eksekusi:", res["route"])
print("Latensi Retrieval:", res["retrieval_s"], "detik")
print("Latensi Gemini LLM:", res["llm"]["latency_s"], "detik")
print("Total Latensi:", res["total_s"], "detik")
print("Sensors Diakses:", res["sensors"])
print("Dokumen Rujukan:", res["doc_sources"])
print("\n=== REKOMENDASI AGRONOMI DARI GEMINI ===")
print(res["answer"])
print("=" * 72)
