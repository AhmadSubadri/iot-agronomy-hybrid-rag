import os
import urllib.request
import json
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GEMINI_API_KEY", "")
if not key:
    print("ERROR: GEMINI_API_KEY belum diset di file .env")
    exit(1)
candidates = ['gemini-flash-latest', 'gemini-2.5-flash-lite', 'gemini-pro-latest']

for m in candidates:
    print(f"Mencoba model: {m}...")
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={key}'
    payload = {
        'contents': [{
            'parts': [{'text': 'Tes koneksi RAG KawalTani. Balas dengan singkat: KONEKSI GEMINI BERHASIL.'}]
        }]
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            res = json.loads(r.read().decode('utf-8'))
            txt = res['candidates'][0]['content']['parts'][0]['text']
            print(f'==> Model {m} [BERHASIL]:', txt.strip())
            break
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='ignore')
        print(f'==> Model {m} [FAIL {e.code}]:', body[:200])
    except Exception as e:
        print(f'==> Model {m} [ERROR]:', e)
