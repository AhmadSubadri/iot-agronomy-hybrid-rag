import urllib.request
import json
import time

time.sleep(3) # Tunggu server binding

url = "http://127.0.0.1:8000/ask"
payload = {
    "question": "Berapa rata-rata nitrogen di area 3 dalam tujuh hari terakhir dan apakah perlu pemupukan tambahan?",
    "mode": "rag",
    "provider": "gemini",
    "model": "gemini-3.5-flash-lite"
}
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

res = json.loads(urllib.request.urlopen(req, timeout=30).read())
print("=== HASIL HYBRID RAG DENGAN GOOGLE GEMINI ===")
print("Model:", res["llm"]["model"])
print("Rute Eksekusi:", res["route"])
print("Latensi Retrieval:", res["latency"]["retrieval_s"], "detik")
print("Latensi Gemini LLM:", res["latency"]["llm_s"], "detik")
print("Total Latensi:", res["latency"]["total_s"], "detik")
print("\n--- JAWABAN REKOMENDASI AGRONOMI OLEH GEMINI ---")
print(res["answer"])

