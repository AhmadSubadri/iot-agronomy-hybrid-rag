#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_qa_dataset.py
=======================
Membangun 100 pasangan Pertanyaan-Jawaban (QA Dataset) tervalidasi SQL langsung dari
database `kawaltani_rag_tesis` sesuai distribusi Bab 3.7.1 Proposal Tesis:
  - 40 soal (40%) : Kondisi Nutrisi Tanah (N, P, K)
  - 30 soal (30%) : Rekomendasi Tindakan Pemupukan & Penanganan Lahan
  - 20 soal (20%) : Kondisi Lingkungan & Dampaknya (pH, Suhu, Kelembapan, EC, Salinitas, TDS)
  - 10 soal (10%) : Komparatif Antar-Periode Waktu

Ditambah 10 soal konseptual agronomi umum (`data/qa_vector_general_10.csv`) untuk
pengujian khusus Jalur 2 (Semantic Vector Search murni).
"""
import csv
import datetime as dt
from pathlib import Path
import mysql.connector

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_NAME = "kawaltani_rag_tesis"

BULAN_ID = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni",
    7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober", 11: "November", 12: "Desember",
}

NUTRIENT_LABEL = {"nitro": "nitrogen (N)", "phos": "fosfor (P)", "pot": "kalium (K)"}
ENV_LABEL = {
    "ph": "pH tanah",
    "temp": "suhu tanah",
    "hum": "kelembapan tanah",
    "con": "konduktivitas listrik (EC)",
    "salin": "salinitas",
    "tds": "TDS (total zat terlarut)",
}

AREA_PLANT_START = {
    1: dt.datetime(2026, 1, 10, 9, 27, 49),
    2: dt.datetime(2026, 1, 10, 9, 27, 49),
    3: dt.datetime(2025, 1, 6, 0, 0, 0),
    4: dt.datetime(2025, 1, 6, 0, 0, 0),
    5: dt.datetime(2025, 1, 6, 0, 0, 0),
    6: dt.datetime(2025, 1, 6, 0, 0, 0),
}

ANOMALIES = [
    dict(area=3, param="nitro", start=15, end=19, category="nutrisi_tanah"),
    dict(area=1, param="phos", start=30, end=34, category="rekomendasi_pemupukan"),
    dict(area=2, param="pot", start=100, end=104, category="rekomendasi_pemupukan"),
    dict(area=4, param="ph", start=45, end=48, category="rekomendasi_pemupukan"),
    dict(area=1, param="nitro", start=50, end=53, category="nutrisi_tanah"),
    dict(area=6, param="pot", start=110, end=113, category="nutrisi_tanah"),
    dict(area=5, param="ph", start=60, end=63, category="kondisi_lingkungan"),
    dict(area=3, param="salin", start=90, end=93, category="kondisi_lingkungan"),
    dict(area=2, param="ph", start=20, end=23, category="kondisi_lingkungan"),
    dict(area=6, param="nitro", start=70, end=76, category="komparatif_periode"),
]


def fmt_id(d: dt.datetime, with_year=True) -> str:
    if with_year:
        return f"{d.day:02d} {BULAN_ID[d.month]} {d.year}"
    return f"{d.day:02d} {BULAN_ID[d.month]}"


def clean_spacing(text: str) -> str:
    import re
    text = re.sub(r"\s+([.,)])", r"\1", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()


def day_bounds(area: int, day_offset: int):
    base = AREA_PLANT_START[area] + dt.timedelta(days=day_offset)
    return (
        base.replace(hour=0, minute=0, second=0, microsecond=0),
        base.replace(hour=23, minute=59, second=59, microsecond=0),
    )


def cond_label(val: float, mn: float, mx: float) -> str:
    if val < mn:
        return "di bawah ambang normal (defisiensi)"
    if val > mx:
        return "di atas ambang normal (berlebih)"
    return "dalam rentang normal"


def rec_text(val: float, mn: float, mx: float, minact: str, maxact: str) -> str:
    if val < mn:
        return minact or "lakukan koreksi pemupukan/penanganan untuk menaikkan nilai ke batas normal"
    if val > mx:
        return maxact or "kurangi input hara terkait dan atur pengairan/drainase"
    return "kondisi optimal/normal sehingga tidak diperlukan tindakan korektif darurat, cukup ikuti jadwal pemeliharaan rutin"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = mysql.connector.connect(host="localhost", user="root", password="", database=DB_NAME)
    cur = conn.cursor(dictionary=True)

    def get_meta(ds_id: str):
        cur.execute("""
            SELECT s.ds_min_norm_value AS mn, s.ds_max_norm_value AS mx,
                   s.min_danger_action AS minact, s.max_danger_action AS maxact,
                   u.unit_symbol
            FROM td_device_sensors s
            LEFT JOIN tr_unit u ON s.unit_id = u.unit_id
            WHERE s.ds_id = %s
        """, (ds_id,))
        r = cur.fetchone()
        unit = (r["unit_symbol"] or "").strip()
        return float(r["mn"]), float(r["mx"]), r["minact"], r["maxact"], unit

    def get_agg(ds_id: str, start_dt: dt.datetime, end_dt: dt.datetime):
        cur.execute("""
            SELECT ROUND(AVG(read_value), 2) AS avg_v,
                   ROUND(MIN(read_value), 2) AS min_v,
                   ROUND(MAX(read_value), 2) AS max_v,
                   COUNT(*) AS n
            FROM tm_sensor_read
            WHERE ds_id = %s AND read_id LIKE 'SIM%%' AND read_date BETWEEN %s AND %s
        """, (ds_id, start_dt, end_dt))
        r = cur.fetchone()
        if not r or not r["n"]:
            return None
        return float(r["avg_v"]), float(r["min_v"]), float(r["max_v"]), int(r["n"])

    rows_out = []

    def add_row(cat, q, ans, ds_id, area, d_info, r_type):
        qid = len(rows_out) + 1
        rows_out.append(dict(
            id=f"QA{qid:03d}",
            category=cat,
            question=clean_spacing(q),
            reference_answer=clean_spacing(ans),
            ds_id=ds_id,
            area=area,
            date_info=d_info,
            retrieval_type=r_type,
        ))

    # 1. 9 Soal Anomali Non-Komparatif -> semuanya butuh penjelasan agronomis ('hybrid')
    for a in ANOMALIES:
        if a["category"] == "komparatif_periode":
            continue
        area, param = a["area"], a["param"]
        ds_id = f"soil_{param}{area}"
        mn, mx, minact, maxact, unit = get_meta(ds_id)
        s0, _ = day_bounds(area, a["start"])
        _, e1 = day_bounds(area, a["end"])
        avg, amin, amax, _ = get_agg(ds_id, s0, e1)
        label = NUTRIENT_LABEL.get(param) or ENV_LABEL.get(param)
        cond = cond_label(avg, mn, mx)
        rec = rec_text(avg, mn, mx, minact, maxact)
        u = f" {unit}" if unit else ""
        t_str = f"{fmt_id(s0)} s/d {fmt_id(e1)}"

        if a["category"] == "nutrisi_tanah":
            q = f"Bagaimana kondisi {label} di area {area} pada periode {t_str}?"
            ans = (f"Pada periode {t_str}, rata-rata {label} di area {area} adalah {avg}{u} "
                   f"(kisaran {amin}–{amax}{u}), yaitu {cond} (ambang normal: {mn}–{mx}{u}). "
                   f"Tindakan yang disarankan: {rec}.")
        elif a["category"] == "rekomendasi_pemupukan":
            q = f"Apa rekomendasi tindakan untuk area {area} berdasarkan kondisi {label} pada {t_str}?"
            ans = (f"Rata-rata {label} di area {area} pada {t_str} tercatat {avg}{u} "
                   f"({cond}; ambang normal {mn}–{mx}{u}). Rekomendasi tindakan: {rec}.")
        else:
            q = f"Bagaimana kondisi {label} di area {area} selama {t_str}, dan apa dampaknya bagi tanaman padi?"
            ans = (f"Rata-rata {label} di area {area} selama {t_str} adalah {avg}{u} "
                   f"({cond}; ambang normal {mn}–{mx}{u}). Tindakan korektif: {rec}.")
        add_row(a["category"], q, ans, ds_id, area, t_str, "hybrid")

    # 2. 1 Soal Anomali Komparatif (H60-H69 vs H70-H76 pada area 6)
    a = ANOMALIES[-1]
    area, param = a["area"], a["param"]
    ds_id = f"soil_{param}{area}"
    mn, mx, _, _, unit = get_meta(ds_id)
    b0, _ = day_bounds(area, a["start"] - 10)
    _, b1 = day_bounds(area, a["start"] - 1)
    d0, _ = day_bounds(area, a["start"])
    _, d1 = day_bounds(area, a["end"])
    before = get_agg(ds_id, b0, b1)
    during = get_agg(ds_id, d0, d1)
    q = (f"Bandingkan kondisi nitrogen di area {area} sebelum (H{a['start']-10} s/d H{a['start']-1}) "
         f"dan selama (H{a['start']} s/d H{a['end']}) periode lonjakan pemupukan.")
    ans = (f"Sebelum lonjakan (H60–H69), rata-rata nitrogen di area {area} adalah {before[0]} {unit} "
           f"(kisaran {before[1]}–{before[2]} {unit}, normal). Selama periode H70–H76, rata-rata melonjak "
           f"menjadi {during[0]} {unit} (kisaran {during[1]}–{during[2]} {unit}), melampaui ambang normal "
           f"{mn}–{mx} {unit} sehingga pemberian Urea perlu dihentikan sementara.")
    add_row("komparatif_periode", q, ans, ds_id, area, "H60-H76", "hybrid")

    # 3. Nutrisi Tanah Normal (genapkan menjadi 40 soal total)
    days_sample = [6, 12, 25, 38, 56, 66, 80, 94, 106, 114]
    week_starts = [3, 24, 40, 58, 82, 96]

    combo_i = 0
    while len([r for r in rows_out if r["category"] == "nutrisi_tanah"]) < 22:
        area = [1, 2, 3, 4, 5, 6][combo_i % 6]
        param = ["nitro", "phos", "pot"][(combo_i // 6) % 3]
        day = days_sample[combo_i % len(days_sample)]
        combo_i += 1
        # Hindari tabrakan dengan window anomali
        if any(an["area"] == area and an["param"] == param and an["start"] <= day <= an["end"] for an in ANOMALIES):
            continue
        ds_id = f"soil_{param}{area}"
        mn, mx, _, _, unit = get_meta(ds_id)
        s0, e1 = day_bounds(area, day)
        agg = get_agg(ds_id, s0, e1)
        if not agg:
            continue
        avg, amin, amax, _ = agg
        cond = cond_label(avg, mn, mx)
        label = NUTRIENT_LABEL[param]
        q = f"Berapa kadar {label} di area {area} pada tanggal {fmt_id(s0)}?"
        ans = (f"Kadar {label} di area {area} pada {fmt_id(s0)} rata-rata {avg} {unit} "
               f"(kisaran harian {amin}–{amax} {unit}), berada {cond} (ambang normal: {mn}–{mx} {unit}).")
        add_row("nutrisi_tanah", q, ans, ds_id, area, s0.strftime("%Y-%m-%d"), "sql")

    combo_i = 0
    while len([r for r in rows_out if r["category"] == "nutrisi_tanah"]) < 40:
        area = [1, 2, 3, 4, 5, 6][combo_i % 6]
        param = ["nitro", "phos", "pot"][(combo_i // 6) % 3]
        wstart = week_starts[combo_i % len(week_starts)]
        combo_i += 1
        if any(an["area"] == area and an["param"] == param and not (wstart + 6 < an["start"] or wstart > an["end"]) for an in ANOMALIES):
            continue
        ds_id = f"soil_{param}{area}"
        mn, mx, _, _, unit = get_meta(ds_id)
        s0, _ = day_bounds(area, wstart)
        _, e1 = day_bounds(area, wstart + 6)
        agg = get_agg(ds_id, s0, e1)
        if not agg:
            continue
        avg, amin, amax, _ = agg
        cond = cond_label(avg, mn, mx)
        label = NUTRIENT_LABEL[param]
        q = f"Bagaimana rata-rata {label} di area {area} selama seminggu ({fmt_id(s0, with_year=False)} - {fmt_id(e1)})?"
        ans = (f"Rata-rata {label} di area {area} selama periode {fmt_id(s0)} s/d {fmt_id(e1)} adalah "
               f"{avg} {unit} (kisaran {amin}–{amax} {unit}), {cond} (ambang normal: {mn}–{mx} {unit}).")
        add_row("nutrisi_tanah", q, ans, ds_id, area, f"{s0.date()} - {e1.date()}", "sql")

    # 4. Rekomendasi Pemupukan (genapkan menjadi 30 soal total)
    combo_i = 0
    while len([r for r in rows_out if r["category"] == "rekomendasi_pemupukan"]) < 30:
        area = [1, 2, 3, 4, 5, 6][combo_i % 6]
        param = ["nitro", "phos", "pot", "ph"][(combo_i // 6) % 4]
        day = days_sample[combo_i % len(days_sample)]
        combo_i += 1
        if any(an["area"] == area and an["param"] == param and an["start"] <= day <= an["end"] for an in ANOMALIES):
            continue
        ds_id = f"soil_{param}{area}"
        mn, mx, minact, maxact, unit = get_meta(ds_id)
        s0, e1 = day_bounds(area, day)
        agg = get_agg(ds_id, s0, e1)
        if not agg:
            continue
        avg, _, _, _ = agg
        cond = cond_label(avg, mn, mx)
        rec = rec_text(avg, mn, mx, minact, maxact)
        label = NUTRIENT_LABEL.get(param) or ENV_LABEL.get(param)
        u = f" {unit}" if unit else ""
        q = f"Apakah area {area} memerlukan tindakan pemupukan atau koreksi terkait {label} pada tanggal {fmt_id(s0)}?"
        ans = (f"Nilai rata-rata {label} di area {area} pada {fmt_id(s0)} adalah {avg}{u} "
               f"({cond}; ambang normal {mn}–{mx}{u}). Rekomendasi: {rec}.")
        add_row("rekomendasi_pemupukan", q, ans, ds_id, area, s0.strftime("%Y-%m-%d"), "hybrid")

    # 5. Kondisi Lingkungan (genapkan menjadi 20 soal total)
    combo_i = 0
    while len([r for r in rows_out if r["category"] == "kondisi_lingkungan"]) < 20:
        area = [1, 2, 3, 4, 5, 6][combo_i % 6]
        param = ["temp", "hum", "con", "tds", "salin"][(combo_i // 6) % 5]
        day = days_sample[combo_i % len(days_sample)]
        combo_i += 1
        if any(an["area"] == area and an["param"] == param and an["start"] <= day <= an["end"] for an in ANOMALIES):
            continue
        ds_id = f"soil_{param}{area}"
        mn, mx, _, _, unit = get_meta(ds_id)
        s0, e1 = day_bounds(area, day)
        agg = get_agg(ds_id, s0, e1)
        if not agg:
            continue
        avg, amin, amax, _ = agg
        cond = cond_label(avg, mn, mx)
        label = ENV_LABEL[param]
        u = f" {unit}" if unit else ""
        q = f"Bagaimana kondisi {label} di area {area} pada tanggal {fmt_id(s0)}?"
        ans = (f"{label.capitalize()} di area {area} pada {fmt_id(s0)} rata-rata {avg}{u} "
               f"(kisaran harian {amin}–{amax}{u}), berada {cond} (ambang normal: {mn}–{mx}{u}).")
        add_row("kondisi_lingkungan", q, ans, ds_id, area, s0.strftime("%Y-%m-%d"), "sql")

    # 6. Komparatif Antar-Periode (genapkan menjadi 10 soal total)
    comparisons = [
        (1, "nitro", 10, 95),
        (2, "ph", 14, 90),
        (3, "pot", 20, 105),
        (4, "phos", 10, 95),
        (5, "temp", 8, 108),
        (6, "hum", 12, 112),
        (1, "pot", 12, 108),
        (3, "ph", 18, 92),
        (5, "nitro", 22, 98),
    ]
    for area, param, day_a, day_b in comparisons:
        ds_id = f"soil_{param}{area}"
        mn, mx, _, _, unit = get_meta(ds_id)
        sA, eA = day_bounds(area, day_a)
        sB, eB = day_bounds(area, day_b)
        aggA = get_agg(ds_id, sA, eA)
        aggB = get_agg(ds_id, sB, eB)
        avgA, avgB = aggA[0], aggB[0]
        arah = "meningkat" if avgB > avgA else ("menurun" if avgB < avgA else "stabil")
        label = NUTRIENT_LABEL.get(param) or ENV_LABEL.get(param)
        u = f" {unit}" if unit else ""
        q = f"Bandingkan {label} di area {area} antara tanggal {fmt_id(sA)} dan {fmt_id(sB)}."
        ans = (f"Pada {fmt_id(sA)}, rata-rata {label} di area {area} tercatat {avgA}{u}. "
               f"Pada {fmt_id(sB)}, nilainya menjadi {avgB}{u} ({arah} dibanding periode pertama; "
               f"ambang normal: {mn}–{mx}{u}).")
        add_row("komparatif_periode", q, ans, ds_id, area, f"{sA.date()} vs {sB.date()}", "sql")

    out_csv = DATA_DIR / "qa_dataset_100.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "category", "question", "reference_answer",
            "ds_id", "area", "date_info", "retrieval_type",
        ])
        writer.writeheader()
        writer.writerows(rows_out)

    # Buat 10 soal konseptual murni untuk verifikasi eksplisit Jalur 2 (Vector Search)
    vec_questions = [
        ("VQ01", "Apa gejala kekurangan unsur nitrogen pada daun tanaman padi?",
         "Daun tua bagian bawah menguning merata (klorosis) dari ujung ke pangkal, tanaman kerdil, dan jumlah anakan sedikit."),
        ("VQ02", "Bagaimana cara mengatasi tanah sawah yang terlalu masam (pH di bawah 5,5)?",
         "Aplikasikan kapur pertanian atau dolomit (1–3 ton/ha sesuai tingkat kemasaman), lakukan drainase berkala untuk mencuci keracunan besi, dan hindari pupuk masam berlebih."),
        ("VQ03", "Mengapa pemberian pupuk Urea dilarang saat padi sudah memasuki fase pemasakan bulir di atas 90 HST?",
         "Karena penyerapan N tanah sudah rendah, memicu anakan terlambat yang tidak produktif, mengundang hama wereng/walangsangit, serta menunda pematangan bulir."),
        ("VQ04", "Apa fungsi utama unsur kalium (K) saat tanaman padi memasuki fase bunting dan pengisian malai?",
         "Kalium memperkuat batang agar tahan rebah, meningkatkan ketahanan penyakit, dan mentranslokasikan karbohidrat ke bulir gabah agar bernas."),
        ("VQ05", "Apa dampak kelebihan nitrogen di atas 120 mg/kg terhadap risiko penyakit tanaman padi?",
         "Tajuk terlalu rimbun dan sukulen sehingga rentan penyakit blas, hawar daun bakteri (kresek), serangan wereng cokelat, dan mudah rebah."),
        ("VQ06", "Bagaimana prinsip pengairan berselang (Alternate Wetting and Drying) pada fase vegetatif hingga menjelang panen?",
         "Macak-macak saat awal tanam dan pemupukan, basah-kering berselang saat vegetatif, genangan stabil 3–5 cm saat bunting/pembungaan, dan kering total 10–14 hari sebelum panen."),
        ("VQ07", "Bagaimana cara menangani sawah padi yang mengalami cekaman salinitas atau nilai EC tinggi?",
         "Lakukan pencucian garam (flushing) dengan air tawar dan perbaikan drainase, tunda pupuk KCl dosis tinggi, beri gypsum dan kompos organik, atau gunakan varietas toleran seperti Inpari 34 Salin."),
        ("VQ08", "Kapan waktu paling tepat memberikan pupuk fosfor (SP-36) pada budidaya padi sawah dan mengapa?",
         "Diberikan 100% pada pemupukan dasar (0–10 HST) karena fosfor bersifat lambat larut dan dibutuhkan sejak dini untuk perkembangan akar serta pembentukan anakan."),
        ("VQ09", "Mengapa pupuk Urea tidak boleh dicampur langsung dengan kapur pertanian atau dolomit pada hari yang sama?",
         "Reaksi basa dari kapur/dolomit mempercepat penguapan nitrogen menjadi gas amonia (volatilisasi) sehingga pupuk Urea hilang ke udara."),
        ("VQ10", "Apa gejala defisiensi fosfor pada tanaman padi fase vegetatif?",
         "Tanaman tumbuh kerdil tegak kaku, anakan sangat sedikit, daun hijau tua kusam atau keunguan, perakaran pendek kecokelatan, dan waktu berbunga mundur."),
    ]
    vec_csv = DATA_DIR / "qa_vector_general_10.csv"
    with open(vec_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "reference_answer", "retrieval_type"])
        writer.writeheader()
        for vid, vq, va in vec_questions:
            writer.writerow({"id": vid, "question": vq, "reference_answer": va, "retrieval_type": "vector"})

    by_cat = {}
    for r in rows_out:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1

    cur.close()
    conn.close()
    print(f"[OK] {len(rows_out)} QA pair utama ditulis ke {out_csv} -> Distribusi: {by_cat}")
    print(f"[OK] {len(vec_questions)} QA konseptual Jalur Vector ditulis ke {vec_csv}")


if __name__ == "__main__":
    main()

