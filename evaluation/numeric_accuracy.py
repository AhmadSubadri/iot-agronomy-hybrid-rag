#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/numeric_accuracy.py — Evaluasi Akurasi Numerik (FR-13 & Bab 3.10 Proposal Tesis).

Menjawab keterbatasan RAGAS yang tidak mengecek ketepatan angka sensor:
1. Mengekstrak angka nilai sensor (mg/kg, mg/L, pH, EC, °C, dsb.) dari ground truth.
2. Memeriksa keberadaan angka tersebut pada jawaban yang dihasilkan model dalam toleransi rel = 3%.
3. Menghitung Akurasi Numerik (% pertanyaan dengan angka sensor akurat terhadap data aktual).
"""
import argparse
import json
import re
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent


def extract_sensor_values(text: str) -> List[float]:
    """
    Ekstrak nilai sensor riil:
    - Menghapus pola tanggal (mis. 21 Januari 2025, 2025-01-21)
    - Menghapus label area (mis. area 1, area 2)
    - Menghapus penanda HST (H60, H+7)
    - Menghapus meta hitungan (mis. 120 pembacaan, 15 orang)
    - Mengekstrak angka desimal atau integer dengan satuan sensor
    """
    if not text:
        return []

    clean = re.sub(r"\b\d{1,2}\s+(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+\d{4}\b", "", text, flags=re.I)
    clean = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", clean)
    clean = re.sub(r"\b\d{2}-\d{2}-\d{4}\b", "", clean)
    clean = re.sub(r"\barea\s+\d+\b", "", clean, flags=re.I)
    clean = re.sub(r"\bH\+?\d+\b", "", clean, flags=re.I)
    clean = re.sub(r"\b\d+\s*(?:pembacaan|sampel|hari|butir|orang|menit|detik)\b", "", clean, flags=re.I)

    # Pola: desimal eksplisit ATAU angka dengan satuan agronomi
    decimals = re.findall(r"(?<![a-zA-Z0-9_])(\d+\.\d+)(?![a-zA-Z0-9_])", clean)
    with_units = re.findall(r"(?<![a-zA-Z0-9_])(\d+)\s*(?:mg/kg|mg/L|ppm|°C|mS/cm|µS/cm|uS/cm|ton/ha)(?![a-zA-Z0-9_])", clean)

    nums = []
    for s in decimals + with_units:
        try:
            v = float(s)
            if not (2020 <= v <= 2030):  # Abaikan tahun tersisa
                nums.append(v)
        except ValueError:
            pass
    return nums


def evaluate_numeric_accuracy(results_json_path: str, tol_pct: float = 0.03) -> dict:
    with open(results_json_path, encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("results", [])
    meta = data.get("meta", {})
    cfg = meta.get("configuration") or meta.get("mode") or "Unknown"

    total_with_numbers = 0
    correct_matches = 0
    details = []

    for r in rows:
        gt_text = r.get("ground_truth", "")
        ans_text = r.get("answer", "")
        cat = r.get("category", "")

        gt_nums = extract_sensor_values(gt_text)
        ans_nums = extract_sensor_values(ans_text)

        if gt_nums:
            total_with_numbers += 1
            primary_gt = gt_nums[0]
            matched = any(abs(c - primary_gt) <= max(0.1, abs(primary_gt) * tol_pct) for c in ans_nums)
            if matched:
                correct_matches += 1

            details.append({
                "id": r.get("id"),
                "category": cat,
                "primary_gt": primary_gt,
                "matched": matched,
                "gt_nums": gt_nums[:3],
                "ans_nums": ans_nums[:4],
            })

    accuracy = (correct_matches / total_with_numbers * 100) if total_with_numbers > 0 else 0.0

    summary = {
        "file": str(results_json_path),
        "configuration": cfg,
        "model": meta.get("model", "-"),
        "total_evaluated": total_with_numbers,
        "numeric_correct": correct_matches,
        "numeric_accuracy_pct": round(accuracy, 2),
        "tolerance_pct": tol_pct * 100,
    }
    return summary, details


def main():
    parser = argparse.ArgumentParser(description="Evaluasi Akurasi Numerik Jawaban Sensor IoT (FR-13)")
    parser.add_argument("--results", required=True, help="Path ke file JSON hasil collect")
    parser.add_argument("--tol", type=float, default=0.03, help="Toleransi kecocokan (default 0.03 = 3%)")
    args = parser.parse_args()

    summary, details = evaluate_numeric_accuracy(args.results, tol_pct=args.tol)
    print("\n================ HASIL EVALUASI AKURASI NUMERIK (FR-13) ================")
    print(f"File Hasil        : {summary['file']}")
    print(f"Konfigurasi       : {summary['configuration']}")
    print(f"Model             : {summary['model']}")
    print(f"Total Sampel GT   : {summary['total_evaluated']}")
    print(f"Angka Akurat      : {summary['numeric_correct']} / {summary['total_evaluated']}")
    print(f"Akurasi Numerik   : {summary['numeric_accuracy_pct']}% (Toleransi ±{summary['tolerance_pct']}%)")
    print("========================================================================\n")


if __name__ == "__main__":
    main()
