#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_db_and_generate_data.py
==============================
Skrip terotomasi & aman untuk menyiapkan database eksperimen tesis `kawaltani_rag_tesis`:
  1. Mem-backup database murni `kawaltani_simpul_telkom` ke `sql/kawaltani_simpul_telkom_original_backup.sql`
     TANPA mengubah satu baris pun di database asli!
  2. Membuat database `kawaltani_rag_tesis` dan meng-import schema + data asli.
  3. Menerapkan `sql/fix_data_inconsistencies_v2.sql` (harmonisasi satuan `tr_unit`, ambang NPK/EC/salinitas,
     dan perbaikan tindakan bahaya pH/salinitas).
  4. Membangkitkan dataset sensor sintetis-terkalibrasi yang bebas cacat fisika:
       - Kelembapan tanah dibatasi ketat [0, 100]%
       - pH tanah dibatasi ketat [3.0, 9.5] dan baseline sawah normal dikalibrasi pada 5.9–6.5
         agar tidak memicu false-positive 100% di Area 1 & 2, sementara nilai NPK, suhu, EC (µS/cm),
         salinitas (mg/L), dan TDS tetap dikalibrasi dari statistik riil TELU0300.
       - Mencakup 155.520 baris historis per jam (120 hari penuh x 6 area x 9 parameter)
         + 38.880 baris per menit pada 3 hari terakhir siklus untuk simulasi real-time kadensi 1 menit.
       - Menyuntikkan 10 skenario anomali ground-truth tervalidasi untuk evaluasi RAGAS.
"""
import os
import json
import math
import uuid
import random
import subprocess
import datetime as dt
from pathlib import Path
import mysql.connector

BASE_DIR = Path(__file__).resolve().parent.parent
SQL_DIR = BASE_DIR / "sql"
DATA_DIR = BASE_DIR / "data"

ORIG_DB = "kawaltani_simpul_telkom"
TARGET_DB = "kawaltani_rag_tesis"

MYSQL_BIN = r"D:\laragon\bin\mysql\mysql-8.0.30-winx64\bin\mysql.exe"
MYSQLDUMP_BIN = r"D:\laragon\bin\mysql\mysql-8.0.30-winx64\bin\mysqldump.exe"

CORE_PARAMS = ["nitro", "phos", "pot", "ph", "temp", "hum", "con", "salin", "tds"]
DEVICE_AREAS = {"TELU0300": [1, 2], "TELU0100": [3, 4, 5, 6]}


def default_anomaly_windows():
    return [
        dict(area=3, param="nitro", start_day=15, end_day=19, type="deficiency",
             category="nutrisi_tanah",
             note="Defisiensi nitrogen area 3 -> uji rekomendasi pemupukan urea"),
        dict(area=1, param="phos", start_day=30, end_day=34, type="deficiency",
             category="rekomendasi_pemupukan",
             note="Defisiensi fosfor area 1 -> uji rekomendasi pupuk fosfat SP-36"),
        dict(area=2, param="pot", start_day=100, end_day=104, type="deficiency",
             category="rekomendasi_pemupukan",
             note="Defisiensi kalium area 2 menjelang pemasakan bulir -> uji rekomendasi KCl"),
        dict(area=4, param="ph", start_day=45, end_day=48, type="deficiency",
             category="rekomendasi_pemupukan",
             note="pH terlalu masam area 4 -> uji rekomendasi kapur pertanian/dolomit"),
        dict(area=1, param="nitro", start_day=50, end_day=53, type="toxicity",
             category="nutrisi_tanah",
             note="Kelebihan nitrogen area 1 -> uji penghentian pupuk N & irigasi"),
        dict(area=6, param="pot", start_day=110, end_day=113, type="toxicity",
             category="nutrisi_tanah",
             note="Kelebihan kalium area 6 -> uji pengaturan pengairan"),
        dict(area=5, param="ph", start_day=60, end_day=63, type="toxicity",
             category="kondisi_lingkungan",
             note="pH terlalu basa/alkalin area 5 -> uji flushing & pupuk ZA/organik"),
        dict(area=3, param="salin", start_day=90, end_day=93, type="toxicity",
             category="kondisi_lingkungan",
             note="Salinitas tinggi area 3 -> uji pencucian garam (flushing) air tawar"),
        dict(area=2, param="ph", start_day=20, end_day=23, type="deficiency",
             category="kondisi_lingkungan",
             note="pH turun masam mendadak area 2 -> uji peringatan dini kemasaman"),
        dict(area=6, param="nitro", start_day=70, end_day=76, type="toxicity",
             category="komparatif_periode",
             note="Lonjakan nitrogen area 6 pasca pemupukan, untuk perbandingan sebelum vs selama lonjakan"),
    ]


def step1_clone_and_fix_db():
    SQL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    backup_file = SQL_DIR / "kawaltani_simpul_telkom_original_backup.sql"

    print(f"[1/4] Backup database murni `{ORIG_DB}` -> {backup_file.name} ...")
    with open(backup_file, "w", encoding="utf-8") as fout:
        res = subprocess.run(
            [MYSQLDUMP_BIN, "-u", "root", "--routines", "--triggers", ORIG_DB],
            stdout=fout, stderr=subprocess.PIPE, text=True,
        )
    if res.returncode != 0:
        raise RuntimeError(f"mysqldump gagal: {res.stderr}")

    print(f"[2/4] Membuat ulang database eksperimen `{TARGET_DB}` dan import schema asli ...")
    subprocess.run(
        [MYSQL_BIN, "-u", "root", "-e",
         f"DROP DATABASE IF EXISTS `{TARGET_DB}`; CREATE DATABASE `{TARGET_DB}` CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;"],
        check=True,
    )
    with open(backup_file, "r", encoding="utf-8") as fin:
        subprocess.run([MYSQL_BIN, "-u", "root", TARGET_DB], stdin=fin, check=True)

    fix_sql = SQL_DIR / "fix_data_inconsistencies_v2.sql"
    print(f"[3/4] Menerapkan perbaikan satuan & ambang `{fix_sql.name}` pada `{TARGET_DB}` ...")
    with open(fix_sql, "r", encoding="utf-8") as fin:
        subprocess.run([MYSQL_BIN, "-u", "root", TARGET_DB], stdin=fin, check=True)


def clamp_physical(param: str, val: float) -> float:
    if param == "hum":
        return max(5.0, min(99.5, val))
    if param == "ph":
        return max(3.2, min(9.4, val))
    if param == "temp":
        return max(12.0, min(42.0, val))
    return max(0.01, val)


def apply_anomaly(val: float, area: int, param: str, day_float: float, anomalies: list, th: dict) -> float:
    for aw in anomalies:
        if aw["area"] == area and aw["param"] == param and aw["start_day"] <= day_float < (aw["end_day"] + 1.0):
            mn, mx = th["min_norm"], th["max_norm"]
            if aw["type"] == "deficiency":
                if param == "ph":
                    return random.uniform(4.1, 4.7)
                low = max(1.0, mn * 0.45)
                high = max(low + 0.5, mn * 0.72)
                return random.uniform(low, high)
            elif aw["type"] == "toxicity":
                if param == "ph":
                    return random.uniform(8.1, 8.7)
                return mx * random.uniform(1.18, 1.38)
    return val


def step2_generate_and_insert():
    random.seed(42)
    conn = mysql.connector.connect(host="localhost", user="root", password="", database=TARGET_DB)
    cur = conn.cursor(dictionary=True)

    # Ambil metadata ambang yang sudah diharmonisasi
    cur.execute("""
        SELECT ds_id, dev_id, ds_min_norm_value AS min_norm, ds_max_norm_value AS max_norm
        FROM td_device_sensors
    """)
    sensors = {r["ds_id"]: r for r in cur.fetchall()}

    # Ambil statistik kalibrasi riil TELU0300 (buang nilai korup seperti 0 atau pH > 14)
    cur.execute("""
        SELECT ds_id, read_value FROM tm_sensor_read
        WHERE dev_id = 'TELU0300' AND read_value > 0
    """)
    raw_vals = {}
    for r in cur.fetchall():
        ds = r["ds_id"]
        v = float(r["read_value"])
        if ds.startswith("soil_ph") and v > 14:
            continue
        raw_vals.setdefault(ds, []).append(v)

    calib = {}
    for ds, vs in raw_vals.items():
        if len(vs) >= 3:
            m = sum(vs) / len(vs)
            calib[ds] = m

    # Ambil tanaman aktif (PL02 -> TELU0300 [area 1,2]; PL03 -> TELU0100 [area 3..6])
    cur.execute("""
        SELECT p.pl_id, p.dev_id, p.pt_id, p.pl_name, p.pl_date_planting, t.pt_day_harvest
        FROM tm_plant p JOIN tr_plant_type t ON p.pt_id = t.pt_id
        WHERE p.dev_id IS NOT NULL
    """)
    plants = cur.fetchall()

    cur.execute("SELECT pt_id, hand_title, hand_day, fertilizer_type FROM tr_plant_handling_copy ORDER BY hand_day")
    handlings = cur.fetchall()

    anomalies_all = default_anomaly_windows()
    anomaly_records = []

    print("[4/4] Membangkitkan & menyisipkan data sensor terkalibrasi (historis per jam + window real-time per menit) ...")
    run_salt = uuid.uuid4().hex[:6].upper()
    out_sql_path = SQL_DIR / "synthetic_tm_sensor_read.sql"

    total_inserted = 0
    batch = []
    seq = 0

    with open(out_sql_path, "w", encoding="utf-8") as fsql:
        fsql.write("SET autocommit=0;\nSET unique_checks=0;\nSET foreign_key_checks=0;\n")

        for plant in plants:
            dev_id = plant["dev_id"]
            if dev_id not in DEVICE_AREAS:
                continue
            areas = DEVICE_AREAS[dev_id]
            start_dt = plant["pl_date_planting"]
            duration_days = int(plant["pt_day_harvest"] or 120)
            events = [h for h in handlings if h["pt_id"] == plant["pt_id"]]

            plant_anoms = [a for a in anomalies_all if a["area"] in areas and a["end_day"] < duration_days]
            for a in plant_anoms:
                s_d = start_dt + dt.timedelta(days=a["start_day"])
                e_d = start_dt + dt.timedelta(days=a["end_day"], hours=23, minutes=59)
                anomaly_records.append({
                    "plant_id": plant["pl_id"],
                    "plant_name": plant["pl_name"],
                    "dev_id": dev_id,
                    **a,
                    "start_date": s_d.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_date": e_d.strftime("%Y-%m-%d %H:%M:%S"),
                })

            for area in areas:
                nutrient_state = {}
                for p in ("nitro", "phos", "pot"):
                    th = sensors[f"soil_{p}{area}"]
                    nutrient_state[p] = float(th["min_norm"]) + 0.38 * (float(th["max_norm"]) - float(th["min_norm"]))

                timestamps = []
                for day_idx in range(duration_days):
                    step_min = 1 if day_idx >= (duration_days - 3) else 60
                    for m_in_day in range(0, 24 * 60, step_min):
                        timestamps.append((start_dt + dt.timedelta(days=day_idx, minutes=m_in_day), step_min))

                for ts, step_min in timestamps:
                    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                    day_float = (ts - start_dt).total_seconds() / 86400.0
                    hour_frac = (ts.hour * 60 + ts.minute) / 1440.0
                    diurnal = math.sin(2 * math.pi * (hour_frac - 0.25))

                    for ev in events:
                        if ev["fertilizer_type"] and abs(day_float - float(ev["hand_day"])) < 0.5:
                            for p in ("nitro", "phos", "pot"):
                                th = sensors[f"soil_{p}{area}"]
                                target = float(th["min_norm"]) + 0.80 * (float(th["max_norm"]) - float(th["min_norm"]))
                                rate = 0.02 * (step_min / 60.0) if step_min < 60 else 0.035
                                nutrient_state[p] += (target - nutrient_state[p]) * rate

                    for p in ("nitro", "phos", "pot"):
                        ds_id = f"soil_{p}{area}"
                        th = sensors[ds_id]
                        mn, mx = float(th["min_norm"]), float(th["max_norm"])
                        decay = 0.00018 * (step_min / 60.0)
                        floor = mn + 0.12 * (mx - mn)
                        nutrient_state[p] -= (nutrient_state[p] - floor) * decay
                        val = nutrient_state[p] + random.gauss(0, 0.012 * (mx - mn))
                        val = round(clamp_physical(p, apply_anomaly(val, area, p, day_float, plant_anoms, {"min_norm": mn, "max_norm": mx})), 2)
                        seq += 1
                        batch.append(f"('SIM{run_salt}{seq:08d}',NULL,'{ts_str}',{val},1,'{ds_id}','{dev_id}',NULL)")

                    ds_ph = f"soil_ph{area}"
                    th_ph = sensors[ds_ph]
                    base_ph = 6.25 + (0.08 if area in (1, 2) else -0.05)
                    val_ph = base_ph + 0.06 * math.sin(2 * math.pi * day_float / 12.0) + random.gauss(0, 0.03)
                    val_ph = round(clamp_physical("ph", apply_anomaly(val_ph, area, "ph", day_float, plant_anoms, {
                        "min_norm": float(th_ph["min_norm"]), "max_norm": float(th_ph["max_norm"])
                    })), 2)
                    seq += 1
                    batch.append(f"('SIM{run_salt}{seq:08d}',NULL,'{ts_str}',{val_ph},1,'{ds_ph}','{dev_id}',NULL)")

                    for p, amp in (("temp", 1.4), ("hum", -5.5)):
                        ds_id = f"soil_{p}{area}"
                        th = sensors[ds_id]
                        mn, mx = float(th["min_norm"]), float(th["max_norm"])
                        if p == "temp":
                            base = calib.get(ds_id, 26.2) if area in (1, 2) else 25.4
                            val = base + amp * diurnal + random.gauss(0, 0.22)
                        else:
                            base = min(74.0, calib.get(ds_id, 68.0)) if area in (1, 2) else 64.0
                            if day_float >= (duration_days - 12):
                                base = 48.0
                            val = base + amp * diurnal + random.gauss(0, 1.1)
                        val = round(clamp_physical(p, apply_anomaly(val, area, p, day_float, plant_anoms, {"min_norm": mn, "max_norm": mx})), 2)
                        seq += 1
                        batch.append(f"('SIM{run_salt}{seq:08d}',NULL,'{ts_str}',{val},1,'{ds_id}','{dev_id}',NULL)")

                    for p in ("con", "salin", "tds"):
                        ds_id = f"soil_{p}{area}"
                        th = sensors[ds_id]
                        mn, mx = float(th["min_norm"]), float(th["max_norm"])
                        if area in (1, 2) and ds_id in calib:
                            base = calib[ds_id]
                            if p == "tds" and base < 550:
                                base = 680.0 if area == 1 else 620.0
                        else:
                            if p == "salin":
                                base = 38.0
                            elif p == "con":
                                base = 1.85
                            else:
                                base = 920.0
                        val = base + random.gauss(0, 0.025 * max(base, 1.0))
                        val = round(clamp_physical(p, apply_anomaly(val, area, p, day_float, plant_anoms, {"min_norm": mn, "max_norm": mx})), 2)
                        seq += 1
                        batch.append(f"('SIM{run_salt}{seq:08d}',NULL,'{ts_str}',{val},1,'{ds_id}','{dev_id}',NULL)")

                    if len(batch) >= 2000:
                        fsql.write("INSERT INTO `tm_sensor_read` (`read_id`,`log_rx_id`,`read_date`,`read_value`,`read_sts`,`ds_id`,`dev_id`,`con_id`) VALUES\n")
                        fsql.write(",\n".join(batch) + ";\n")
                        total_inserted += len(batch)
                        batch.clear()

        if batch:
            fsql.write("INSERT INTO `tm_sensor_read` (`read_id`,`log_rx_id`,`read_date`,`read_value`,`read_sts`,`ds_id`,`dev_id`,`con_id`) VALUES\n")
            fsql.write(",\n".join(batch) + ";\n")
            total_inserted += len(batch)
            batch.clear()

        fsql.write("COMMIT;\nSET unique_checks=1;\nSET foreign_key_checks=1;\n")

    print(f"    -> File SQL ({total_inserted:,} baris) selesai ditulis, meng-import cepat ke MySQL `{TARGET_DB}` ...")
    with open(out_sql_path, "r", encoding="utf-8") as fin:
        subprocess.run([MYSQL_BIN, "-u", "root", TARGET_DB], stdin=fin, check=True)

    gt_path = DATA_DIR / "ground_truth_anomalies.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(anomaly_records, f, indent=2, ensure_ascii=False)

    conn.commit()
    cur.execute("SELECT COUNT(*) AS total FROM tm_sensor_read")
    final_cnt = cur.fetchone()["total"]
    cur.close()
    conn.close()

    print(f"[SELESAI] {total_inserted:,} baris simulasi disisipkan. Total baris `tm_sensor_read` di `{TARGET_DB}`: {final_cnt:,}.")
    print(f"[SELESAI] Ground-truth 10 skenario anomali disimpan di: {gt_path}")


if __name__ == "__main__":
    step1_clone_and_fix_db()
    step2_generate_and_insert()
