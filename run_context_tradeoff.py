#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_context_tradeoff.py — Eksperimen Tujuan Penelitian ke-4 (Bab 1.4 poin 4 Proposal).

Menganalisis secara kuantitatif trade-off antara kompleksitas/panjang konteks
(`top_k` dokumen agronomi x `max_daily_points` rincian deret waktu harian) terhadap:
  1. Panjang konteks rata-rata (karakter & estimasi token prompt)
  2. Latensi retrieval rata-rata & persentil ke-95 (ms)
  3. Kelengkapan cakupan informasi acuan ground-truth (Lexical Fact Recall %)

Menghasilkan tabel CSV siap pakai untuk Bab IV Tesis di `results/tradeoff_context_vs_latency.csv`.
"""
import csv
import re
import statistics
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def fact_coverage_score(contexts: list[str], ground_truth: str) -> float:
    """Persentase angka kunci & kata kunci agronomi pada ground_truth yang termuat di dalam konteks."""
    ctx_text = " ".join(contexts).lower()
    # Ekstrak angka penting (rata-rata, ambang) dari jawaban referensi
    nums = re.findall(r"\b\d+(?:\.\d+)?\b", ground_truth)
    key_terms = [w for w in ("urea", "sp-36", "kcl", "dolomit", "kapur", "flushing", "drainase",
                             "defisiensi", "berlebih", "normal") if w in ground_truth.lower()]
    targets = nums[:3] + key_terms
    if not targets:
        return 1.0
    hits = sum(1 for t in targets if t in ctx_text)
    return hits / len(targets)


def main():
    from kawaltani_rag.pipeline import RAGPipeline
    from kawaltani_rag import sql_retriever
    conn_shared = sql_retriever.connect()
    pipe = RAGPipeline(conn=conn_shared)

    qa_path = BASE_DIR / "data" / "qa_dataset_100.csv"
    with open(qa_path, encoding="utf-8") as f:
        qa = list(csv.DictReader(f))

    top_k_grid = [2, 4, 6, 8]
    daily_pts_grid = [3, 7, 14]

    out_dir = BASE_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "tradeoff_context_vs_latency.csv"

    # Pemanasan: satu putaran penuh tanpa pencatatan agar konfigurasi pertama tidak
    # menanggung biaya cold start (pemuatan model, cache InnoDB, kompilasi regex).
    print("Pemanasan cache (1 putaran, tidak dicatat)...")
    for row in qa:
        pipe.retrieve(row["question"], top_k=top_k_grid[0], max_daily_points=daily_pts_grid[0])

    summary_rows = []
    print(f"\n{'Konfigurasi':18s} | {'Avg Chars':>10s} | {'Est Tokens':>10s} | {'Avg Lat(ms)':>11s} | {'P95 Lat(ms)':>11s} | {'Fact Coverage':>13s}")
    print("-" * 88)

    for k in top_k_grid:
        for pts in daily_pts_grid:
            chars_list, tok_list, lat_list, cov_list = [], [], [], []
            for row in qa:
                r = pipe.retrieve(row["question"], top_k=k, max_daily_points=pts)
                chars_list.append(r["context_chars"])
                tok_list.append(r["est_prompt_tokens"])
                lat_list.append(r["retrieval_s"] * 1000.0)
                cov_list.append(fact_coverage_score(r["contexts"], row["reference_answer"]) * 100.0)

            lat_sorted = sorted(lat_list)
            p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))]
            cfg_name = f"K={k}, Days={pts}"
            rec = dict(
                config=cfg_name,
                top_k=k,
                max_daily_points=pts,
                avg_context_chars=round(statistics.mean(chars_list), 1),
                avg_est_tokens=round(statistics.mean(tok_list), 1),
                avg_latency_ms=round(statistics.mean(lat_list), 2),
                p95_latency_ms=round(p95, 2),
                fact_coverage_pct=round(statistics.mean(cov_list), 2),
            )
            summary_rows.append(rec)
            print(
                f"{cfg_name:18s} | {rec['avg_context_chars']:10.1f} | {rec['avg_est_tokens']:10.1f} | "
                f"{rec['avg_latency_ms']:11.2f} | {rec['p95_latency_ms']:11.2f} | {rec['fact_coverage_pct']:12.2f}%"
            )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n[OK] Tabel eksperimen Tujuan ke-4 disimpan ke: {out_csv}")


if __name__ == "__main__":
    main()
