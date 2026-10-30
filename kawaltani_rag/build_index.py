#!/usr/bin/env python3
"""
CLI: Bangun index Vector Store dari dokumen agronomi + opsional sinkronisasi
incremental ringkasan temporal sensor MySQL.

Contoh:
    python -m kawaltani_rag.build_index --sync-temporal
"""
import argparse
from . import vector_store, incremental_indexer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-st", action="store_true", help="Pakai TF-IDF fallback")
    ap.add_argument("--sync-temporal", action="store_true",
                    help="Sekaligus sinkronkan ringkasan mingguan sensor MySQL ke Vector Store")
    a = ap.parse_args()

    meta = vector_store.build_index(a.docs, a.out, prefer_st=not a.no_st)
    print("[OK] Index dokumen agronomi selesai:", meta)

    if a.sync_temporal:
        # Reset watermark bila baru build ulang index dari nol
        incremental_indexer.save_watermark({"last_indexed_read_date": None, "synced_chunks": 0}, a.out)
        sync_res = incremental_indexer.sync_incremental(index_dir=a.out)
        print("[OK] Sinkronisasi incremental temporal sensor selesai:", sync_res)


if __name__ == "__main__":
    main()

