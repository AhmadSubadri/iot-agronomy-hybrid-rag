import urllib.request
import json

print("=== 1. TEST /health ===")
h = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/health").read())
print("Database:", h["database"])
print("Vector Meta:", h["vector_meta"])

print("\n=== 2. TEST /sensors/overview ===")
s = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/sensors/overview").read())
print(f"Total Area Aktif: {len(s['areas'])}")
for a in s["areas"]:
    print(f"  - Area {a['area']} ({a['dev_id']}): {a.get('plant_name')}, {a.get('hst')} HST, {len(a['metrics'])} sensor")

print("\n=== 3. TEST /ask (Mode Local / Tanpa API Key) ===")
payload = {
    "question": "Berapa rata-rata nitrogen di area 3 dalam tujuh hari terakhir dan apakah perlu pemupukan tambahan?",
    "mode": "local"
}
req = urllib.request.Request(
    "http://127.0.0.1:8000/ask",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
res = json.loads(urllib.request.urlopen(req).read())
print(f"Rute: {res['route']} | Latensi: {res['latency']['retrieval_s']*1000:.1f} ms")
print(f"Sensors: {res['sensors']}")
print(f"Dokumen Rujukan: {res['doc_sources']}")
print(f"Panjang Jawaban: {len(res['answer'])} karakter")
print("\n[SEMUA ENDPOINT WEB UI MERESPONS 100% OK]")

