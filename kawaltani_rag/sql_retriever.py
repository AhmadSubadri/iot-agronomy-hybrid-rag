"""
Jalur 1 — Structured Retrieval (SQL) + Data-to-Text Verbalization + Context Enrichment.

Penyempurnaan v2 sesuai Proposal Tesis:
  1. Membaca satuan sensor langsung dari tabel `tr_unit` (`JOIN tr_unit`) agar tidak salah satuan.
  2. Membaca jadwal & dosis pemupukan dari tabel `tr_plant_handling_copy` sesuai usia HST tanaman.
  3. Mendukung resolusi jendela waktu berbasis HST (`H60 s/d H69`) berdasarkan `tm_plant.pl_date_planting`.
  4. Sliding window aggregation + daily temporal chunking + deteksi episode di luar ambang (`out_of_range_stats`).
  5. Mendukung driver `mysql-connector-python` maupun `PyMySQL` secara otomatis.
"""
import datetime as dt
from . import config

PARAM_LABEL = {
    "nitro": "Nitrogen (N)",
    "phos": "Fosfor (P)",
    "pot": "Kalium (K)",
    "ph": "pH tanah",
    "temp": "Suhu tanah",
    "hum": "Kelembapan tanah",
    "con": "Konduktivitas listrik (EC)",
    "salin": "Salinitas",
    "tds": "TDS (Total Zat Terlarut)",
}

FALLBACK_UNIT = {
    "nitro": "mg/kg",
    "phos": "mg/kg",
    "pot": "mg/kg",
    "ph": "",
    "temp": "°C",
    "hum": "%",
    "con": "mS/cm",
    "salin": "mg/L",
    "tds": "ppm",
}


def connect():
    """Koneksi MySQL fleksibel (mendukung mysql-connector-python maupun PyMySQL)."""
    try:
        import mysql.connector
        return mysql.connector.connect(**config.DB), "mysql.connector"
    except ImportError:
        import pymysql
        conn = pymysql.connect(
            host=config.DB["host"],
            port=config.DB.get("port", 3306),
            user=config.DB["user"],
            password=config.DB["password"],
            database=config.DB["database"],
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
        )
        return conn, "pymysql"


def _rows(conn_pair, sql, args=()):
    conn, drv = conn_pair if isinstance(conn_pair, tuple) else (conn_pair, "mysql.connector")
    cur = conn.cursor(dictionary=True) if drv == "mysql.connector" else conn.cursor()
    cur.execute(sql, args)
    out = cur.fetchall()
    cur.close()
    return out


def get_sensor_meta(conn, ds_id: str):
    r = _rows(conn, """
        SELECT s.ds_id, s.ds_name, s.dev_id, s.unit_id,
               s.ds_min_norm_value AS min_norm, s.ds_max_norm_value AS max_norm,
               s.min_danger_action, s.max_danger_action,
               u.unit_symbol
        FROM td_device_sensors s
        LEFT JOIN tr_unit u ON s.unit_id = u.unit_id
        WHERE s.ds_id = %s
    """, (ds_id,))
    return r[0] if r else None


def get_plant_context(conn, dev_id: str, when: dt.datetime = None):
    """Ambil data tanaman aktif di perangkat + usia HST + jadwal pemupukan (`tr_plant_handling_copy`)."""
    r = _rows(conn, """
        SELECT p.pl_id, p.pl_name, p.pt_id, p.pl_date_planting, p.pl_area,
               t.pt_name, t.pt_day_harvest
        FROM tm_plant p
        LEFT JOIN tr_plant_type t ON p.pt_id = t.pt_id
        WHERE p.dev_id = %s
        ORDER BY p.pl_date_planting DESC LIMIT 1
    """, (dev_id,))
    if not r:
        return None
    p = dict(r[0])
    ref = when or dt.datetime.now()
    p["hst"] = (ref - p["pl_date_planting"]).days

    # Ambil protokol penanganan tanaman dari `tr_plant_handling_copy` (Bab 3.3.1 Proposal)
    sched = _rows(conn, """
        SELECT hand_id, hand_title, hand_day, hand_day_toleran, fertilizer_type, dose
        FROM tr_plant_handling_copy
        WHERE pt_id = %s
        ORDER BY hand_day ASC
    """, (p["pt_id"],))
    p["schedule"] = sched or []
    return p


def format_schedule_context(plant: dict) -> str:
    if not plant or not plant.get("schedule"):
        return ""
    hst = plant["hst"]
    items = []
    active_note = None
    for s in plant["schedule"]:
        day = int(s["hand_day"])
        tol = int(s["hand_day_toleran"] or 0)
        fert = (s.get("fertilizer_type") or "").strip()
        title = (s.get("hand_title") or "").replace("Pemupukan Pemupukan", "Pemupukan Dasar").strip()
        fert_info = f" ({fert})" if fert else ""
        items.append(f"H+{day} (±{tol} hari): {title}{fert_info}")
        if day <= hst <= day + tol:
            active_note = f"SAAT INI MASUK JENDELA JADWAL {title.upper()} (H+{day} s/d H+{day+tol}){fert_info}"
    text = "Jadwal protokol penanganan (`tr_plant_handling_copy`): " + "; ".join(items) + "."
    if active_note:
        text += f" [{active_note}]"
    return text


def get_latest_timestamp(conn, ds_ids=None) -> dt.datetime:
    """Ambil timestamp terbaru di database bila kueri 'saat ini/terbaru' dijalankan di luar periode tanam."""
    if ds_ids:
        placeholders = ",".join(["%s"] * len(ds_ids))
        r = _rows(conn, f"SELECT MAX(read_date) AS mx FROM tm_sensor_read WHERE ds_id IN ({placeholders})", tuple(ds_ids))
    else:
        r = _rows(conn, "SELECT MAX(read_date) AS mx FROM tm_sensor_read")
    return r[0]["mx"] if r and r[0]["mx"] else dt.datetime.now()


def aggregate(conn, ds_id: str, start: dt.datetime, end: dt.datetime):
    r = _rows(conn, """
        SELECT COUNT(*) AS n,
               ROUND(AVG(read_value), 2) AS avg_val,
               ROUND(MIN(read_value), 2) AS min_val,
               ROUND(MAX(read_value), 2) AS max_val,
               ROUND(STDDEV_POP(read_value), 2) AS std_val,
               MIN(read_date) AS first_at,
               MAX(read_date) AS last_at
        FROM tm_sensor_read
        WHERE ds_id = %s AND read_date BETWEEN %s AND %s
    """, (ds_id, start, end))
    a = r[0] if r else None
    return a if a and a["n"] else None


def daily_series(conn, ds_id: str, start: dt.datetime, end: dt.datetime, max_points: int = None):
    max_points = max_points if max_points is not None else config.MAX_DAILY_POINTS
    rows = _rows(conn, """
        SELECT DATE(read_date) AS tanggal,
               ROUND(AVG(read_value), 2) AS avg_val,
               ROUND(MIN(read_value), 2) AS min_val,
               ROUND(MAX(read_value), 2) AS max_val
        FROM tm_sensor_read
        WHERE ds_id = %s AND read_date BETWEEN %s AND %s
        GROUP BY DATE(read_date)
        ORDER BY tanggal
    """, (ds_id, start, end))
    if max_points > 0 and len(rows) > max_points:
        step = len(rows) / max_points
        rows = [rows[int(i * step)] for i in range(max_points)]
    return rows


def out_of_range_stats(conn, ds_id: str, start: dt.datetime, end: dt.datetime, mn: float, mx: float):
    r = _rows(conn, """
        SELECT
          SUM(CASE WHEN read_value < %s THEN 1 ELSE 0 END) AS n_below,
          SUM(CASE WHEN read_value > %s THEN 1 ELSE 0 END) AS n_above,
          COUNT(*) AS n_total,
          MIN(CASE WHEN read_value < %s THEN read_date END) AS below_first,
          MAX(CASE WHEN read_value < %s THEN read_date END) AS below_last,
          MIN(CASE WHEN read_value > %s THEN read_date END) AS above_first,
          MAX(CASE WHEN read_value > %s THEN read_date END) AS above_last
        FROM tm_sensor_read
        WHERE ds_id = %s AND read_date BETWEEN %s AND %s
    """, (mn, mx, mn, mn, mx, mx, ds_id, start, end))
    return r[0] if r else None


def condition_label(val, mn: float, mx: float) -> str:
    if val is None:
        return "tidak diketahui"
    if val < mn:
        return "DI BAWAH ambang normal (defisiensi)"
    if val > mx:
        return "DI ATAS ambang normal (berlebih)"
    return "dalam rentang NORMAL"


def trend_label(series) -> str:
    if not series or len(series) < 3:
        return None
    vals = [float(r["avg_val"]) for r in series]
    half = len(vals) // 2
    a = sum(vals[:half]) / half
    b = sum(vals[half:]) / (len(vals) - half)
    if b == a:
        return "stabil"
    delta = (b - a) / abs(a) * 100 if a else 0
    if abs(delta) < 3:
        return "relatif stabil"
    return f"cenderung {'naik' if delta > 0 else 'turun'} (~{abs(delta):.1f}%)"


def resolve_unit(meta: dict, param_key: str) -> str:
    sym = (meta.get("unit_symbol") or "").strip()
    if sym and sym != "-":
        if sym.lower() in ("us/c", "us/cm"):
            return "µS/cm"
        if sym.lower() == "ppm" and param_key in ("nitro", "phos", "pot"):
            return "mg/kg"
        return sym
    return FALLBACK_UNIT.get(param_key, "")


def verbalize(meta: dict, agg: dict, series: list, plant: dict,
              start: dt.datetime, end: dt.datetime, oor: dict = None) -> str:
    param = next((k for k in PARAM_LABEL if meta["ds_id"].startswith(f"soil_{k}")), None)
    unit = resolve_unit(meta, param)
    u = f" {unit}" if unit else ""
    mn, mx = float(meta["min_norm"]), float(meta["max_norm"])
    avg = float(agg["avg_val"])
    cond = condition_label(avg, mn, mx)

    lines = [
        f"[DATA SENSOR — {meta['ds_name']} | kode={meta['ds_id']} | perangkat={meta['dev_id']}]",
        f"Periode pengukuran: {start:%d-%m-%Y %H:%M} s/d {end:%d-%m-%Y %H:%M} ({agg['n']} pembacaan).",
        f"Statistik: rata-rata {avg}{u} | minimum {agg['min_val']}{u} | maksimum {agg['max_val']}{u} | standar deviasi {agg['std_val']}.",
        f"Ambang normal agronomis padi: {mn}–{mx}{u}.",
        f"STATUS (berdasarkan rata-rata periode): {cond}.",
    ]

    if oor and oor["n_total"]:
        n_below = int(oor["n_below"] or 0)
        n_above = int(oor["n_above"] or 0)
        total = int(oor["n_total"])
        if n_below:
            pct = n_below / total * 100
            lines.append(
                f"PERHATIAN: {n_below} dari {total} pembacaan ({pct:.0f}%) berada DI BAWAH ambang normal "
                f"antara {oor['below_first']:%d-%m-%Y %H:%M} dan {oor['below_last']:%d-%m-%Y %H:%M}. "
                f"Episode defisiensi ini wajib diperhatikan walau rata-rata tampak normal."
            )
        if n_above:
            pct = n_above / total * 100
            lines.append(
                f"PERHATIAN: {n_above} dari {total} pembacaan ({pct:.0f}%) berada DI ATAS ambang normal "
                f"antara {oor['above_first']:%d-%m-%Y %H:%M} dan {oor['above_last']:%d-%m-%Y %H:%M}. "
                f"Episode kelebihan ini wajib diperhatikan walau rata-rata tampak normal."
            )
        if not n_below and not n_above:
            lines.append(f"Seluruh {total} pembacaan berada stabil dalam rentang normal.")

    tr = trend_label(series)
    if tr:
        lines.append(f"Tren temporal: {tr}.")

    if plant:
        hst = plant["hst"]
        fase = (
            "vegetatif" if hst <= 55 else
            "reproduktif/generatif" if hst <= 90 else
            "pemasakan bulir"
        )
        lines.append(
            f"Konteks tanaman: {plant['pl_name']} ({plant.get('pt_name') or 'Padi'}), "
            f"lokasi {plant.get('pl_area') or '-'}, tanam {plant['pl_date_planting']:%d-%m-%Y}, "
            f"usia saat periode ini {hst} HST (fase {fase})."
        )
        sched_txt = format_schedule_context(plant)
        if sched_txt:
            lines.append(sched_txt)

    had_below = avg < mn or (oor and int(oor.get("n_below") or 0) > 0)
    had_above = avg > mx or (oor and int(oor.get("n_above") or 0) > 0)
    if had_below and meta.get("min_danger_action"):
        lines.append(f"Tindakan terdaftar (kondisi rendah): {meta['min_danger_action']}.")
    if had_above and meta.get("max_danger_action"):
        lines.append(f"Tindakan terdaftar (kondisi tinggi): {meta['max_danger_action']}.")

    if series and len(series) > 1:
        pts = "; ".join(f"{r['tanggal']:%d-%m}: {r['avg_val']}" for r in series)
        lines.append(f"Rincian harian (rata-rata): {pts}.")

    return "\n".join(lines)


def retrieve(intent, conn=None, max_daily_points: int = None):
    """Ambil & verbalisasi seluruh konteks sensor yang relevan dengan intent."""
    own = conn is None
    conn_pair = connect() if own else conn
    try:
        areas = intent.areas
        if not areas and intent.devices:
            areas = []
            for d in intent.devices:
                areas.extend([a for a in config.DEVICE_AREAS.get(d, []) if isinstance(a, int)])
        areas = areas or [1, 2, 3, 4, 5, 6]
        params = intent.params or ["nitro", "phos", "pot", "ph"]

        blocks, used = [], []
        first_window = None

        for param in params:
            for area in areas:
                ds_id = f"soil_{param}{area}"
                meta = get_sensor_meta(conn_pair, ds_id)
                if not meta:
                    continue

                # Resolusi window waktu: HST -> compare_windows -> start/end -> latest available window
                windows = []
                if intent.hst_windows:
                    plant_ref = get_plant_context(conn_pair, meta["dev_id"])
                    if plant_ref and plant_ref.get("pl_date_planting"):
                        p0 = plant_ref["pl_date_planting"]
                        for h_start, h_end in intent.hst_windows:
                            ws = (p0 + dt.timedelta(days=h_start)).replace(hour=0, minute=0, second=0)
                            we = (p0 + dt.timedelta(days=h_end)).replace(hour=23, minute=59, second=59)
                            windows.append((ws, we))

                if not windows and intent.compare_windows:
                    windows = list(intent.compare_windows)

                if not windows:
                    if intent.start is not None and intent.end is not None:
                        ws, we = intent.start, intent.end
                    else:
                        latest = get_latest_timestamp(conn_pair, [ds_id])
                        we = latest
                        ws = latest - dt.timedelta(days=config.DEFAULT_WINDOW_DAYS)
                    if (we - ws).days > config.MAX_WINDOW_DAYS:
                        ws = we - dt.timedelta(days=config.MAX_WINDOW_DAYS)
                    windows = [(ws, we)]

                parts = []
                for (ws, we) in windows:
                    agg = aggregate(conn_pair, ds_id, ws, we)
                    # Bila user menyebut 'terakhir/terbaru/saat ini' tapi timestamp relatif hari ini di luar periode rekam sensor,
                    # fallback otomatis ke jendela terakhir data sensor yang tercatat pada ds_id tersebut
                    q_low = intent.raw_query.lower()
                    if not agg and any(w in q_low for w in ("terakhir", "terbaru", "saat ini", "sekarang", "hari ini")):
                        latest = get_latest_timestamp(conn_pair, [ds_id])
                        we = latest
                        ws = latest - dt.timedelta(days=config.DEFAULT_WINDOW_DAYS)
                        agg = aggregate(conn_pair, ds_id, ws, we)
                    if not agg:
                        continue
                    if first_window is None:
                        first_window = (ws, we)
                    series = daily_series(conn_pair, ds_id, ws, we, max_points=max_daily_points)
                    plant = get_plant_context(conn_pair, meta["dev_id"], we)
                    oor = out_of_range_stats(
                        conn_pair, ds_id, ws, we, float(meta["min_norm"]), float(meta["max_norm"])
                    )
                    parts.append(verbalize(meta, agg, series, plant, ws, we, oor))

                if parts:
                    blocks.append("\n\n".join(parts))
                    used.append(ds_id)

        return dict(
            context="\n\n".join(blocks),
            sensors=used,
            window=first_window,
            found=bool(blocks),
        )
    finally:
        if own:
            raw_conn = conn_pair[0] if isinstance(conn_pair, tuple) else conn_pair
            raw_conn.close()

