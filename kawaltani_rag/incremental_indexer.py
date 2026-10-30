"""
Incremental Temporal Chunking & Vector Synchronization (Novelty #1 & Bab 4.1.1 Proposal).

Menyinkronkan pembacaan baru dari tabel MySQL `tm_sensor_read` ke dalam Vector Store
tanpa melakukan re-indexing penuh dokumen lama. Setiap jendela waktu mingguan per area
diringkas menjadi narasi multi-parameter (NPK, pH, suhu, kelembapan, EC, salinitas, TDS)
dan ditandai dengan watermark `last_indexed_read_date`.
"""
import os
import json
import datetime as dt
from . import config, sql_retriever, vector_store


def _watermark_path(index_dir: str) -> str:
    return os.path.join(index_dir, "watermark.json")


def load_watermark(index_dir: str = None) -> dict:
    p = _watermark_path(index_dir or config.INDEX_DIR)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {"last_indexed_read_date": None, "synced_chunks": 0}


def save_watermark(wm: dict, index_dir: str = None):
    p = _watermark_path(index_dir or config.INDEX_DIR)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(wm, f, indent=2)


def build_weekly_temporal_chunks(conn=None, since: dt.datetime = None) -> tuple[list, dt.datetime]:
    """
    Kelompokkan pembacaan `tm_sensor_read` per Area (1..6) dan per minggu kalender (ISO year-week)
    menjadi ringkasan narasi multi-parameter yang kaya konteks semantik untuk embedding.
    """
    own = conn is None
    conn_pair = sql_retriever.connect() if own else conn
    try:
        where_since = "WHERE r.read_date > %s" if since else ""
        args = (since,) if since else ()
        rows = sql_retriever._rows(conn_pair, f"""
            SELECT r.ds_id, r.dev_id,
                   YEARWEEK(r.read_date, 1) AS yw,
                   MIN(r.read_date) AS w_start,
                   MAX(r.read_date) AS w_end,
                   COUNT(*) AS n_reads,
                   ROUND(AVG(r.read_value), 2) AS avg_val,
                   ROUND(MIN(r.read_value), 2) AS min_val,
                   ROUND(MAX(r.read_value), 2) AS max_val,
                   s.ds_min_norm_value AS min_norm,
                   s.ds_max_norm_value AS max_norm,
                   u.unit_symbol
            FROM tm_sensor_read r
            JOIN td_device_sensors s ON r.ds_id = s.ds_id
            LEFT JOIN tr_unit u ON s.unit_id = u.unit_id
            {where_since}
            GROUP BY r.ds_id, r.dev_id, YEARWEEK(r.read_date, 1),
                     s.ds_min_norm_value, s.ds_max_norm_value, u.unit_symbol
            HAVING r.ds_id REGEXP '^soil_[a-z]+[1-6]$'
            ORDER BY yw ASC, r.ds_id ASC
        """, args)

        if not rows:
            return [], since

        grouped = {}
        max_dt = since
        for r in rows:
            ds_id = r["ds_id"]
            area = int(ds_id[-1])
            yw = r["yw"]
            key = (area, yw)
            grouped.setdefault(key, []).append(r)
            if max_dt is None or r["w_end"] > max_dt:
                max_dt = r["w_end"]

        chunks = []
        for (area, yw), items in sorted(grouped.items()):
            w_start = min(x["w_start"] for x in items)
            w_end = max(x["w_end"] for x in items)
            dev_id = items[0]["dev_id"]
            plant = sql_retriever.get_plant_context(conn_pair, dev_id, w_end)
            plant_str = (
                f"{plant['pl_name']} ({plant['hst']} HST)"
                if plant else f"Perangkat {dev_id}"
            )

            bullet_lines = []
            anomalies = []
            for x in items:
                param_prefix = x["ds_id"][5:-1]
                label = sql_retriever.PARAM_LABEL.get(param_prefix, x["ds_id"])
                unit = sql_retriever.resolve_unit(x, param_prefix)
                u = f" {unit}" if unit else ""
                mn, mx = float(x["min_norm"]), float(x["max_norm"])
                avg = float(x["avg_val"])
                status = sql_retriever.condition_label(avg, mn, mx)
                bullet_lines.append(
                    f"- {label}: rata-rata {avg}{u} (rentang {x['min_val']}–{x['max_val']}{u}; "
                    f"normal {mn}–{mx}{u}) -> {status}"
                )
                if avg < mn or avg > mx:
                    anomalies.append(f"{label} {status}")

            heading = f"Ringkasan Temporal Area {area} ({w_start:%d-%m-%Y} s/d {w_end:%d-%m-%Y})"
            anom_summary = (
                "Perhatian anomali minggu ini: " + ", ".join(anomalies) + "."
                if anomalies else "Seluruh parameter rata-rata mingguan dalam batas normal."
            )
            body = (
                f"Area {area} ({plant_str}) periode {w_start:%d-%m-%Y} s/d {w_end:%d-%m-%Y}:\n"
                + "\n".join(bullet_lines)
                + f"\nKesimpulan mingguan: {anom_summary}"
            )
            chunks.append(dict(
                id=f"temporal_area{area}_{yw}",
                chunk_type="temporal_sensor",
                source=f"mysql:tm_sensor_read:area{area}",
                doc_title=f"Memori Sensor Area {area}",
                heading=heading,
                text=body,
                embed_text=f"{heading}\n{body}",
                kategori="temporal_sensor",
                parameter_terkait="nitro,phos,pot,ph,temp,hum,con,salin,tds",
            ))

        return chunks, max_dt
    finally:
        if own:
            raw_conn = conn_pair[0] if isinstance(conn_pair, tuple) else conn_pair
            raw_conn.close()


def sync_incremental(index_dir: str = None, conn=None) -> dict:
    """Jalankan sinkronisasi incremental dari MySQL ke Vector Store berdasarkan watermark."""
    idx_dir = index_dir or config.INDEX_DIR
    wm = load_watermark(idx_dir)
    since = (
        dt.datetime.fromisoformat(wm["last_indexed_read_date"])
        if wm.get("last_indexed_read_date") else None
    )
    new_chunks, max_dt = build_weekly_temporal_chunks(conn=conn, since=since)
    store = vector_store.VectorStore(idx_dir)
    added = store.append_chunks(new_chunks) if new_chunks else 0

    if max_dt:
        wm["last_indexed_read_date"] = max_dt.isoformat()
    wm["synced_chunks"] = int(wm.get("synced_chunks", 0)) + added
    wm["total_temporal_chunks"] = store.meta.get("n_temporal", 0)
    save_watermark(wm, idx_dir)
    return dict(
        added_chunks=added,
        total_temporal_chunks=store.meta.get("n_temporal", 0),
        watermark=wm.get("last_indexed_read_date"),
    )

