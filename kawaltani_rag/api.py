"""
API & Interface Layer (Lapisan 5, Gambar 1 Proposal Tesis).

Menyediakan:
  - Antarmuka web (`GET /`): Konsultasi, Data Lahan, dan Evaluasi
  - REST API: /health, /sensors/overview, /sensors/series, /results/summary,
              /retrieve, /ask, /sync-incremental
  - WebSocket: /ws/chat

Cara menjalankan:
    python run_server.py
    atau: uvicorn kawaltani_rag.api:app --port 8000
"""
import csv
import datetime as dt
import json
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import config, incremental_indexer, sql_retriever
from . import intent as intent_mod
from .pipeline import RAGPipeline
from .llm import LLMError

STATIC_DIR = Path(__file__).resolve().parent / "static"
RESULTS_DIR = config.BASE_DIR / "results"
LOGS_DIR = config.BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "queries.jsonl"


def _log_query(entry: dict):
    """Mencatat query ke logs/queries.jsonl sesuai kebutuhan FR-11 PRD."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

app = FastAPI(
    title="KawalTani Hybrid RAG API",
    description="Sistem Rekomendasi Tindakan Agronomi Tanaman Padi Berbasis Data Sensor IoT Real-Time",
    version="2.2.0",
)

_pipeline: Optional[RAGPipeline] = None
_pipe_lock = threading.Lock()

PARAMS_UI = ("nitro", "phos", "pot", "ph", "hum", "temp", "con")


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        with _pipe_lock:
            if _pipeline is None:
                _pipeline = RAGPipeline()
    return _pipeline


# ---------------------------------------------------------------- models
class AskRequest(BaseModel):
    question: str
    mode: Optional[str] = "rag"          # rag | local | baseline | compare | A | B | C | D | sql_only | vector_only
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    top_k: Optional[int] = None
    max_daily_points: Optional[int] = None


class ChatRequest(BaseModel):
    """Sesuai Kontrak API Bagian 11 PRD Sistem RAG KawalTani."""
    question: str
    mode: Optional[str] = "D"            # A (LLM saja) | B (SQL-only) | C (Vector-only) | D (Hybrid)
    llm: Optional[str] = None            # model name e.g. gpt-4o-mini, gemini-3.5-flash-lite, llama3:8b
    provider: Optional[str] = None       # openai | gemini | ollama
    api_key: Optional[str] = None


class RetrieveRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    max_daily_points: Optional[int] = None
    include_temporal_vec: Optional[bool] = False


# ---------------------------------------------------------------- helpers
def _fmt(v):
    if isinstance(v, dt.datetime):
        return v.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(v, dt.date):
        return v.strftime("%Y-%m-%d")
    return v


def _intent_dict(question: str) -> dict:
    """Hasil ekstraksi intent (regex, murah) untuk ditampilkan pada jejak pipeline."""
    it = intent_mod.classify(question)
    return {
        "route": it.route,
        "params": it.params,
        "areas": it.areas,
        "devices": it.devices,
        "start": _fmt(it.start),
        "end": _fmt(it.end),
        "is_comparison": it.is_comparison,
        "needs_advice": it.needs_advice,
        "compare_windows": [[_fmt(a), _fmt(b)] for a, b in (it.compare_windows or [])],
        "hst_windows": it.hst_windows,
    }


def _payload(question: str, mode: str, r: dict, answer: str, llm: dict,
             llm_s: float = 0.0, baseline: dict = None, notice: str = None) -> dict:
    """Satu bentuk respons untuk semua mode agar antarmuka konsisten."""
    out = {
        "mode": mode,
        "question": question,
        "answer": answer,
        "is_local_fallback": notice is not None,
        "llm_notice": notice,
        "initial_route": r.get("initial_route"),
        "route": r.get("route"),
        "intent": _intent_dict(question),
        "sensors": r.get("sensors", []),
        "doc_sources": r.get("doc_sources", []),
        "sensor_context": r.get("sensor_context", ""),
        "agro_context": r.get("agro_context", ""),
        "context_chars": r.get("context_chars", 0),
        "est_prompt_tokens": r.get("est_prompt_tokens", 0),
        "stage_s": r.get("stage_s", {}),
        "latency": {
            "retrieval_s": r.get("retrieval_s", 0.0),
            "llm_s": llm_s,
            "total_s": round(r.get("retrieval_s", 0.0) + llm_s, 3),
        },
        "llm": {
            "provider": llm.get("provider"),
            "model": llm.get("model"),
            "prompt_tokens": llm.get("prompt_tokens"),
            "completion_tokens": llm.get("completion_tokens"),
        },
    }
    if baseline is not None:
        out["baseline"] = baseline
    return out


# ---------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_file = STATIC_DIR / "index.html"
    if not html_file.exists():
        return HTMLResponse("<h2>index.html tidak ditemukan.</h2>", status_code=404)
    return HTMLResponse(html_file.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    pipe = get_pipeline()
    return {
        "status": "ok",
        "database": config.DB["database"],
        "source_database_untouched": config.SOURCE_DB_NAME,
        "llm_provider": config.LLM_PROVIDER,
        "llm_model": config.LLM_MODEL,
        "vector_meta": pipe.store.meta,
    }


# ---------------------------------------------------------------- sensors
@app.get("/sensors/overview")
def sensors_overview():
    """Pembacaan terakhir tiap sensor Area 1..6 (memanfaatkan indeks idx_ds_readdate)."""
    conn_pair = sql_retriever.connect()
    try:
        meta_rows = sql_retriever._rows(conn_pair, """
            SELECT s.ds_id, s.dev_id, s.ds_name,
                   s.ds_min_norm_value AS min_norm, s.ds_max_norm_value AS max_norm,
                   u.unit_symbol
            FROM td_device_sensors s
            LEFT JOIN tr_unit u ON s.unit_id = u.unit_id
            WHERE s.ds_id REGEXP '^soil_(nitro|phos|pot|ph|hum|temp|con)[1-6]$'
            ORDER BY s.ds_id ASC
        """)
        areas = {}
        for m in meta_rows:
            ds_id = m["ds_id"]
            latest = sql_retriever._rows(
                conn_pair,
                "SELECT read_value, read_date, dev_id FROM tm_sensor_read "
                "WHERE ds_id = %s ORDER BY read_date DESC LIMIT 1",
                (ds_id,),
            )
            if not latest:
                continue
            r = latest[0]
            area_num = int(ds_id[-1])
            param = ds_id[5:-1]
            val = float(r["read_value"])
            mn, mx = float(m["min_norm"]), float(m["max_norm"])
            a_obj = areas.setdefault(area_num, {
                "area": area_num,
                "dev_id": r["dev_id"],
                "last_read": _fmt(r["read_date"]),
                "_dt": r["read_date"],
                "metrics": [],
            })
            if r["read_date"] > a_obj["_dt"]:
                a_obj["_dt"], a_obj["last_read"] = r["read_date"], _fmt(r["read_date"])
            a_obj["metrics"].append({
                "param": param,
                "label": sql_retriever.PARAM_LABEL.get(param, param),
                "value": round(val, 2),
                "unit": sql_retriever.resolve_unit(m, param),
                "min_norm": mn,
                "max_norm": mx,
                "state": "low" if val < mn else ("high" if val > mx else "normal"),
            })

        plant_cache = {}
        for a_obj in areas.values():
            dev_id = a_obj["dev_id"]
            ref = a_obj.pop("_dt")
            if dev_id not in plant_cache:
                plant_cache[dev_id] = sql_retriever.get_plant_context(conn_pair, dev_id, ref)
            plant = plant_cache[dev_id]
            if plant:
                a_obj["plant_name"] = plant["pl_name"]
                a_obj["hst"] = plant["hst"]
                a_obj["planted"] = _fmt(plant["pl_date_planting"])
            order = {p: i for i, p in enumerate(PARAMS_UI)}
            a_obj["metrics"].sort(key=lambda x: order.get(x["param"], 99))
        return {"database": config.DB["database"], "areas": [areas[k] for k in sorted(areas)]}
    finally:
        conn_pair[0].close()


@app.get("/sensors/series")
def sensors_series(area: int, param: str, days: int = 120):
    """Deret rata-rata harian satu sensor + ambang normal, untuk grafik pada tab Data Lahan."""
    if param not in PARAMS_UI or not 1 <= area <= 6:
        raise HTTPException(status_code=400, detail="Parameter area/param tidak valid.")
    ds_id = f"soil_{param}{area}"
    conn_pair = sql_retriever.connect()
    try:
        meta = sql_retriever.get_sensor_meta(conn_pair, ds_id)
        if not meta:
            raise HTTPException(status_code=404, detail=f"Sensor {ds_id} tidak ditemukan.")
        end = sql_retriever.get_latest_timestamp(conn_pair, [ds_id])
        start = end - dt.timedelta(days=max(7, min(days, 400)))
        rows = sql_retriever._rows(conn_pair, """
            SELECT DATE(read_date) AS d,
                   ROUND(AVG(read_value), 2) AS avg_val,
                   ROUND(MIN(read_value), 2) AS min_val,
                   ROUND(MAX(read_value), 2) AS max_val,
                   COUNT(*) AS n
            FROM tm_sensor_read
            WHERE ds_id = %s AND read_date BETWEEN %s AND %s
            GROUP BY DATE(read_date)
            ORDER BY d
        """, (ds_id, start, end))
        plant = sql_retriever.get_plant_context(conn_pair, meta["dev_id"], end)
        planted = plant["pl_date_planting"].date() if plant else None
        return {
            "ds_id": ds_id,
            "area": area,
            "param": param,
            "label": sql_retriever.PARAM_LABEL.get(param, param),
            "unit": sql_retriever.resolve_unit(meta, param),
            "min_norm": float(meta["min_norm"]),
            "max_norm": float(meta["max_norm"]),
            "dev_id": meta["dev_id"],
            "planted": _fmt(planted),
            "points": [
                {
                    "d": _fmt(r["d"]),
                    "avg": float(r["avg_val"]),
                    "min": float(r["min_val"]),
                    "max": float(r["max_val"]),
                    "n": int(r["n"]),
                    "hst": (r["d"] - planted).days if planted else None,
                }
                for r in rows
            ],
        }
    finally:
        conn_pair[0].close()


# ---------------------------------------------------------------- evaluation
@app.get("/results/summary")
def results_summary():
    """Ringkasan hasil eksperimen yang sudah dijalankan (audit retrieval & trade-off konteks)."""
    out = {"audit": None, "tradeoff": []}
    audit_file = RESULTS_DIR / "audit_retrieval.json"
    if audit_file.exists():
        out["audit"] = json.loads(audit_file.read_text(encoding="utf-8"))
    trade_file = RESULTS_DIR / "tradeoff_context_vs_latency.csv"
    if trade_file.exists():
        with open(trade_file, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out["tradeoff"].append({k: (float(v) if k != "config" else v) for k, v in row.items()})
    return out


# ---------------------------------------------------------------- RAG
@app.post("/sync-incremental")
def sync_incremental():
    """Sinkronisasi incremental pembacaan baru `tm_sensor_read` ke Vector Store (Bab 4.1.1)."""
    global _pipeline
    t0 = time.time()
    res = incremental_indexer.sync_incremental()
    _pipeline = RAGPipeline()
    return {"status": "synced", "elapsed_s": round(time.time() - t0, 2), **res}


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    r = get_pipeline().retrieve(
        req.question,
        top_k=req.top_k,
        max_daily_points=req.max_daily_points,
        include_temporal_vec=bool(req.include_temporal_vec),
    )
    return _payload(req.question, "retrieve", r, answer="", llm={})


@app.post("/ask")
def ask(req: AskRequest):
    pipe = get_pipeline()
    mode = (req.mode or "rag").lower()
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Pertanyaan kosong.")

    def retrieve_only():
        return pipe.retrieve(q, top_k=req.top_k, max_daily_points=req.max_daily_points)

    if mode == "local":
        r = retrieve_only()
        res = _payload(q, "local", r, pipe.synthesize_local_answer(q, r),
                       llm={"provider": "-", "model": "tanpa LLM"})
        _log_query({"timestamp": dt.datetime.now().isoformat(), "question": q, "mode": "local", "route": r.get("route"), "latency_s": res["latency"]})
        return res

    try:
        if mode in ("baseline", "a", "llm_only"):
            b = pipe.answer_baseline(q, provider=req.provider, model=req.model, api_key=req.api_key)
            empty = dict(initial_route="none", route="baseline_no_rag", retrieval_s=0.0)
            res = _payload(q, "baseline", empty, b["answer"], b["llm"], llm_s=b["llm"]["latency_s"])
            _log_query({"timestamp": dt.datetime.now().isoformat(), "question": q, "mode": "A", "route": "baseline_no_rag", "latency_s": res["latency"], "tokens": res["llm"]})
            return res

        elif mode in ("b", "sql_only", "sql"):
            r = pipe.answer_sql_only(q, provider=req.provider, model=req.model,
                                    max_daily_points=req.max_daily_points, api_key=req.api_key)
            res = _payload(q, "sql_only", r, r["answer"], r["llm"], llm_s=r["llm"]["latency_s"])
            _log_query({"timestamp": dt.datetime.now().isoformat(), "question": q, "mode": "B", "route": r.get("route"), "latency_s": res["latency"], "tokens": res["llm"]})
            return res

        elif mode in ("c", "vector_only", "vector"):
            r = pipe.answer_vector_only(q, provider=req.provider, model=req.model,
                                       top_k=req.top_k, api_key=req.api_key)
            res = _payload(q, "vector_only", r, r["answer"], r["llm"], llm_s=r["llm"]["latency_s"])
            _log_query({"timestamp": dt.datetime.now().isoformat(), "question": q, "mode": "C", "route": r.get("route"), "latency_s": res["latency"], "tokens": res["llm"]})
            return res

        # Mode D (Hybrid / RAG)
        r = pipe.answer(q, provider=req.provider, model=req.model, top_k=req.top_k,
                        max_daily_points=req.max_daily_points, api_key=req.api_key)
        baseline = None
        if mode == "compare":
            b = pipe.answer_baseline(q, provider=req.provider, model=req.model, api_key=req.api_key)
            baseline = {"answer": b["answer"], "latency_s": b["llm"]["latency_s"],
                        "model": b["llm"]["model"]}
        res = _payload(q, mode, r, r["answer"], r["llm"], llm_s=r["llm"]["latency_s"],
                       baseline=baseline)
        _log_query({"timestamp": dt.datetime.now().isoformat(), "question": q, "mode": "D" if mode in ("rag", "d", "hybrid") else mode, "route": r.get("route"), "latency_s": res["latency"], "tokens": res["llm"]})
        return res
    except LLMError as e:
        r = retrieve_only()
        baseline = None
        if mode == "compare":
            baseline = {"answer": None, "latency_s": 0.0, "model": None}
        return _payload(q, mode, r, pipe.synthesize_local_answer(q, r),
                        llm={"provider": "-", "model": "tanpa LLM"}, baseline=baseline,
                        notice=str(e))


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Kontrak API Resmi Bagian 11 PRD Sistem RAG KawalTani:
    Menerima { question, mode: 'A'|'B'|'C'|'D', llm: model_name, provider: provider_name }
    Mengembalikan format terstandarisasi untuk antarmuka chat dan pengujian tesis.
    """
    pipe = get_pipeline()
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Pertanyaan kosong.")

    cfg = (req.mode or "D").strip().upper()
    provider = req.provider or config.LLM_PROVIDER
    model = req.llm or req.provider or config.LLM_MODEL

    try:
        ans_obj = pipe.answer_by_mode(cfg, q, provider=provider, model=model, api_key=req.api_key)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"LLM Generation Error: {e}")

    llm_info = ans_obj.get("llm", {})
    it = ans_obj.get("intent")

    # Format sumber data terstruktur
    sql_source = {}
    if it:
        sql_source = {
            "areas": it.areas,
            "params": it.params,
            "from": _fmt(it.start) if it.start else None,
            "to": _fmt(it.end) if it.end else None,
            "sensors": ans_obj.get("sensors", []),
        }

    out = {
        "answer": ans_obj.get("answer", ""),
        "route": ans_obj.get("route", ""),
        "configuration": cfg,
        "sources": {
            "sql": sql_source if cfg in ("B", "D") else None,
            "docs": ans_obj.get("doc_sources", []) if cfg in ("C", "D") else [],
        },
        "latency_ms": {
            "retrieval": round(ans_obj.get("retrieval_s", 0.0) * 1000, 1),
            "generation": round(llm_info.get("latency_s", 0.0) * 1000, 1),
            "total": round((ans_obj.get("retrieval_s", 0.0) + llm_info.get("latency_s", 0.0)) * 1000, 1),
        },
        "tokens": {
            "in": llm_info.get("prompt_tokens") or 0,
            "out": llm_info.get("completion_tokens") or 0,
        },
    }

    _log_query({
        "timestamp": dt.datetime.now().isoformat(),
        "question": q,
        "mode": cfg,
        "provider": provider,
        "model": model,
        "latency_ms": out["latency_ms"],
        "tokens": out["tokens"],
        "route": out["route"],
    })

    return out


@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    """Endpoint WebSocket untuk integrasi chatbot KawalTani (Bab 3.4 Proposal)."""
    await ws.accept()
    pipe = get_pipeline()
    try:
        while True:
            payload = await ws.receive_json()
            q = (payload.get("question") or "").strip()
            if not q:
                await ws.send_json({"error": "Pertanyaan kosong."})
                continue
            try:
                r = pipe.answer(q, provider=payload.get("provider"), model=payload.get("model"),
                                api_key=payload.get("api_key"))
                await ws.send_json(_payload(q, "rag", r, r["answer"], r["llm"],
                                            llm_s=r["llm"]["latency_s"]))
            except LLMError as e:
                r = pipe.retrieve(q)
                await ws.send_json(_payload(q, "rag", r, pipe.synthesize_local_answer(q, r),
                                            llm={"provider": "-", "model": "tanpa LLM"},
                                            notice=str(e)))
    except WebSocketDisconnect:
        pass
