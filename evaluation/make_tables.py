#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/make_tables.py — Generator Tabel Bab VI Laporan Tesis (Hasil & Pembahasan).

Menghasilkan tabel markdown & LaTeX untuk:
1. Ringkasan Kinerja 4 Konfigurasi (A, B, C, D) pada Metrik RAGAS & Akurasi Numerik.
2. Breakdown Skor per Kategori Pertanyaan (Nutrisi, Pemupukan, Lingkungan, Komparatif).
3. Tabel Uji Signifikansi Hipotesis H1 & H2.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent


def df_to_md(df: pd.DataFrame) -> str:
    """Format DataFrame ke tabel markdown tanpa dependensi tabulate."""
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def generate_summary_table(rag_csv: str, base_csv: str) -> str:
    df_rag = pd.read_csv(rag_csv)
    df_base = pd.read_csv(base_csv)

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    rows = []

    for m in metrics:
        r_val = df_rag[m].mean() if m in df_rag.columns else 0.0
        r_std = df_rag[m].std() if m in df_rag.columns else 0.0
        b_val = df_base[m].mean() if m in df_base.columns else 0.0
        b_std = df_base[m].std() if m in df_base.columns else 0.0
        delta = r_val - b_val
        rows.append({
            "Metrik": m.replace("_", " ").title(),
            "Konfigurasi A (Baseline)": f"{b_val:.4f} ± {b_std:.4f}",
            "Konfigurasi D (Hybrid RAG)": f"{r_val:.4f} ± {r_std:.4f}",
            "Selisih (Δ)": f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}",
        })

    df_out = pd.DataFrame(rows)
    return df_to_md(df_out)


def generate_category_table(rag_csv: str) -> str:
    df_rag = pd.read_csv(rag_csv)
    if "category" not in df_rag.columns:
        return "Kolom category tidak ditemukan."

    grouped = df_rag.groupby("category")[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean()
    grouped["sample_count"] = df_rag.groupby("category")["id"].count()
    return df_to_md(grouped.reset_index().round(4))


def main():
    parser = argparse.ArgumentParser(description="Generate Tabel Bab VI Laporan Tesis")
    parser.add_argument("--rag-csv", default=str(BASE_DIR / "results" / "hasil_rag_gemini_100_ragas.csv"))
    parser.add_argument("--base-csv", default=str(BASE_DIR / "results" / "hasil_baseline_gemini_100_ragas.csv"))
    args = parser.parse_args()

    print("\n### TABEL 1: Perbandingan RAGAS Baseline (A) vs Hybrid RAG (D)")
    print(generate_summary_table(args.rag_csv, args.base_csv))

    print("\n### TABEL 2: Skor RAGAS Hybrid RAG per Kategori Pertanyaan")
    print(generate_category_table(args.rag_csv))


if __name__ == "__main__":
    main()
