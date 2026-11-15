#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sus/sus_scorer.py — Kalkulator Skor System Usability Scale (SUS) (FR-15 & Bagian 10 PRD).

Formula Standar SUS (Brooke, 1996):
  - Butir Ganjil (1, 3, 5, 7, 9)  : Skor − 1
  - Butir Genap  (2, 4, 6, 8, 10) : 5 − Skor
  - Total Skor Responden          : Jumlah skor terkoreksi × 2.5 (Rentang 0–100)

Target Tesis: Skor Rata-rata SUS ≥ 75.0 (Kategori: 'Good' / 'B').
"""
import argparse
import csv
from pathlib import Path
from typing import List, Dict

BASE_DIR = Path(__file__).resolve().parent

SUS_QUESTIONS = [
    "1. Saya berpikir akan sering menggunakan sistem ini.",
    "2. Saya merasa sistem ini terlalu rumit, padahal bisa dibuat lebih sederhana.",
    "3. Saya merasa sistem ini mudah digunakan.",
    "4. Saya membutuhkan bantuan tenaga teknis untuk dapat menggunakan sistem ini.",
    "5. Saya merasa berbagai fungsi pada sistem ini terintegrasi dengan sangat baik.",
    "6. Saya merasa ada terlalu banyak inkonsistensi (hal yang tidak beraturan) pada sistem ini.",
    "7. Saya merasa kebanyakan pengguna akan memahami cara menggunakan sistem ini dengan sangat cepat.",
    "8. Saya merasa sistem ini membingungkan ketika pertama kali digunakan.",
    "9. Saya merasa sangat percaya diri menggunakan sistem ini.",
    "10. Saya harus belajar banyak hal terlebih dahulu sebelum dapat menggunakan sistem ini dengan lancar."
]


def score_sus_item(item_idx: int, raw_score: int) -> float:
    """item_idx: 1-based index (1..10)"""
    if item_idx % 2 == 1:  # Ganjil (positif)
        return float(raw_score - 1)
    else:                  # Genap (negatif)
        return float(5 - raw_score)


def calculate_individual_sus(answers: List[int]) -> float:
    """Menghitung skor SUS satu responden (10 butir, rentang 1-5)."""
    if len(answers) != 10:
        raise ValueError(f"Diperlukan tepat 10 butir jawaban, diterima {len(answers)}")
    corrected_sum = sum(score_sus_item(i + 1, score) for i, score in enumerate(answers))
    return round(corrected_sum * 2.5, 2)


def get_sus_adjective(score: float) -> str:
    if score >= 85:
        return "Excellent (A+)"
    elif score >= 75:
        return "Good (B) — Target Tesis Tercapai"
    elif score >= 68:
        return "OK / Average (C)"
    elif score >= 50:
        return "Poor (D)"
    else:
        return "Not Acceptable (F)"


def process_sus_csv(csv_path: str) -> Dict:
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    penyuluh_scores = []
    petani_scores = []
    all_scores = []

    for r in rows:
        ans = [int(r[f"Q{i}"]) for i in range(1, 11)]
        s = calculate_individual_sus(ans)
        all_scores.append(s)
        role = r.get("role", "").lower()
        if "penyuluh" in role:
            penyuluh_scores.append(s)
        elif "petani" in role:
            petani_scores.append(s)

    def stats(arr):
        if not arr:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        return {
            "count": len(arr),
            "mean": round(sum(arr) / len(arr), 2),
            "min": min(arr),
            "max": max(arr),
        }

    total_stats = stats(all_scores)
    return {
        "total": total_stats,
        "penyuluh": stats(penyuluh_scores),
        "petani": stats(petani_scores),
        "adjective": get_sus_adjective(total_stats["mean"]),
        "status": "MEMENUHI TARGET (>=75)" if total_stats["mean"] >= 75 else "BELUM MEMENUHI TARGET (<75)",
    }


def main():
    parser = argparse.ArgumentParser(description="Kalkulator Skor SUS Responden Tesis")
    parser.add_argument("--csv", required=True, help="Path file CSV respon kuesioner")
    args = parser.parse_args()

    res = process_sus_csv(args.csv)
    print("\n================ HASIL ANALISIS SYSTEM USABILITY SCALE (SUS) ================")
    print(f"Total Responden       : {res['total']['count']} orang")
    print(f"Rata-rata Skor Total  : {res['total']['mean']} (Min: {res['total']['min']}, Max: {res['total']['max']})")
    print(f"Kelompok Penyuluh     : {res['penyuluh']['mean']} (N={res['penyuluh']['count']})")
    print(f"Kelompok Petani       : {res['petani']['mean']} (N={res['petani']['count']})")
    print(f"Predikat SUS          : {res['adjective']}")
    print(f"Status Target Tesis   : {res['status']}")
    print("=============================================================================\n")


if __name__ == "__main__":
    main()
