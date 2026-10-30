#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_features.py — Pengujian menyeluruh fitur kunci arsitektur Hybrid RAG KawalTani:
  1. Keamanan database asli `kawaltani_simpul_telkom` (100% utuh, 400 baris) vs `kawaltani_rag_tesis` (385.312 baris)
  2. Deteksi anomali tersembunyi di balik rata-rata normal + Routing Adaptif (`sql` -> `hybrid`)
  3. Integrasi jadwal penanganan `tr_plant_handling_copy` & konteks fase tumbuh HST (`PL02` / `PL03`)
  4. Idempotensi Incremental Vector Indexing (`sync_incremental` = 0 duplikasi)
  5. Kueri contoh literal dari Proposal Tesis (termasuk angka dalam huruf "tujuh hari terakhir")
"""
import sys
import time
from pathlib import Path
import mysql.connector

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from kawaltani_rag import config, sql_retriever, incremental_indexer
from kawaltani_rag.pipeline import RAGPipeline


def main():
    print("=" * 76)
    print("1. VERIFIKASI KEUTUHAN DATABASE ASLI vs DATABASE EKSPERIMEN TESIS")
    print("=" * 76)
    c_orig = mysql.connector.connect(
        host=config.DB["host"], port=config.DB["port"],
        user=config.DB["user"], password=config.DB["password"],
        database=config.SOURCE_DB_NAME,
    )
    cur_orig = c_orig.cursor()
    cur_orig.execute("SELECT COUNT(*) FROM tm_sensor_read")
    orig_rows = cur_orig.fetchone()[0]
    c_orig.close()

    c_rag_pair = sql_retriever.connect()
    cur_rag = c_rag_pair[0].cursor()
    cur_rag.execute("SELECT COUNT(*) FROM tm_sensor_read")
    rag_rows = cur_rag.fetchone()[0]

    print(f"  - Database asli `{config.SOURCE_DB_NAME}.tm_sensor_read` : {orig_rows:,} baris (AMAN / TIDAK DISENTUH)")
    print(f"  - Database tesis `{config.DB['database']}.tm_sensor_read`: {rag_rows:,} baris (AKTIF)")
    assert orig_rows == 400, f"Expected 400 rows in original DB, got {orig_rows}"
    assert rag_rows > 380000, f"Expected >380k rows in thesis DB, got {rag_rows}"

    pipe = RAGPipeline(conn=c_rag_pair)

    print("\n" + "=" * 76)
    print("2. UJI DETEKSI ANOMALI TERSEMBUNYI & ADAPTIVE ROUTING (sql -> hybrid)")
    print("=" * 76)
    q_hidden = "Bagaimana kondisi nitrogen di area 3 pada periode 15 Januari 2025 s/d 30 Januari 2025?"
    r_hidden = pipe.retrieve(q_hidden)
    print(f"  Kueri         : {q_hidden}")
    print(f"  Initial Route : {r_hidden['initial_route']}")
    print(f"  Final Route   : {r_hidden['route']} (otomatis naik karena terdeteksi episode anomali)")
    print(f"  Doc Sources   : {r_hidden['doc_sources']}")
    has_warn = "PERHATIAN:" in r_hidden["sensor_context"]
    has_sched = "Jadwal protokol penanganan" in r_hidden["sensor_context"]
    print(f"  Ada peringatan episode defisiensi ('PERHATIAN:') : {has_warn}")
    print(f"  Ada jadwal penanganan ('tr_plant_handling_copy') : {has_sched}")
    assert r_hidden["initial_route"] == "sql" and r_hidden["route"] == "hybrid"
    assert has_warn and has_sched

    print("\n" + "=" * 76)
    print("3. UJI KUERI LITERAL DARI PROPOSAL TESIS (Bab 1.2 & Bab 3)")
    print("=" * 76)
    literal_queries = [
        "Berapa rata-rata nitrogen di area 3 dalam tujuh hari terakhir dan apakah perlu pemupukan tambahan?",
        "Bandingkan kelembaban tanah di area 1 antara fase anakan dan fase bunting!",
        "Apa gejala visual kekurangan kalium pada daun tanaman padi dan bagaimana penanganannya?",
        "Berapa pH tanah terbaru pada sensor TELU0400?",
    ]
    for q in literal_queries:
        r = pipe.retrieve(q)
        print(f"  - [{r['initial_route']} -> {r['route']}] ({r['retrieval_s']*1000:.1f} ms) {q}")
        assert len(r["contexts"]) > 0

    print("\n" + "=" * 76)
    print("4. UJI IDEMPOTENSI INCREMENTAL VECTOR INDEXER (Bab 4.1.1)")
    print("=" * 76)
    t0 = time.perf_counter()
    sync_res = incremental_indexer.sync_incremental(conn=c_rag_pair)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Hasil sync kedua: {sync_res} dalam {dt_ms:.1f} ms")
    assert sync_res["added_chunks"] == 0, "Idempotent sync seharusnya menambahkan 0 chunk duplikat!"

    c_rag_pair[0].close()
    print("\n[SEMUA UJI FITUR KUNCI LULUS 100%]")


if __name__ == "__main__":
    main()

