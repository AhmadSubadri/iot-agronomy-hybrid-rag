#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_server.py — Peluncur Antarmuka Web Interaktif & Server API Hybrid RAG KawalTani.
Menjalankan server di http://127.0.0.1:8000 dan membuka browser begitu model siap.
"""
import argparse
import threading
import time
import urllib.request
import webbrowser
import uvicorn


def _wait_and_open(url: str):
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                if r.status == 200:
                    webbrowser.open(url)
                    return
        except Exception:
            time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(description="Jalankan Server & UI Interaktif Hybrid RAG KawalTani")
    parser.add_argument("--host", default="127.0.0.1", help="Host server (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port server (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Jangan buka browser otomatis")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print("=" * 72)
    print("  KAWALTANI HYBRID RAG - DASHBOARD & CHATBOT INTERAKTIF")
    print("  Memuat model neural multilingual & menghubungkan database...")
    from kawaltani_rag.api import get_pipeline
    pipe = get_pipeline()
    pipe.store.embedder.encode(["pemanasan awal"])
    print(f"  [OK] Model & Database Terhubung.")
    print(f"  URL Dashboard   : {url}")
    print(f"  Swagger API Docs: {url}/docs")
    print("  Tekan CTRL+C untuk menghentikan server.")
    print("=" * 72)

    if not args.no_browser:
        threading.Thread(target=_wait_and_open, args=(url,), daemon=True).start()

    uvicorn.run("kawaltani_rag.api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
