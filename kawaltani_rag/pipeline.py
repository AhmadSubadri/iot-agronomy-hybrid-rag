"""
RAG Engine — Orkestrasi Hybrid Retrieval + Context Fusion + 4 Konfigurasi Eksperimen (A/B/C/D).

Mendukung 4 konfigurasi eksperimen sesuai PRD & Proposal Tesis (Bab 3.9 & 3.10):
  1. Konfigurasi A (LLM Saja / Baseline):
     Chatbot tanpa retrieval sensor IoT maupun dokumen agronomi.
  2. Konfigurasi B (SQL-Only):
     Retrieval hanya dari data sensor IoT dan jadwal tanam via SQL.
  3. Konfigurasi C (Vector-Only):
     Retrieval hanya dari dokumen agronomi padi via FAISS vector search.
  4. Konfigurasi D (Hybrid RAG - Usulan Tesis):
     Kombinasi SQL Structured Retrieval + FAISS Vector Search + Eskalasi Adaptif.
"""
import time
from . import intent as intent_mod, sql_retriever, vector_store, llm

SYSTEM_PROMPT_RAG = """Anda adalah asisten cerdas agronomi padi pada platform KawalTani yang membantu petani dan penyuluh pertanian di Indonesia menafsirkan data sensor IoT lahan sawah secara akurat.

ATURAN WAJIB:
1. Jawab HANYA berdasarkan KONTEKS yang diberikan. Dilarang mengarang angka sensor, tanggal, ambang batas, atau dosis yang tidak didukung konteks.
2. Bila konteks tidak memuat informasi yang ditanyakan, sampaikan secara jujur bahwa data tersebut tidak tersedia.
3. Saat menyebut nilai sensor, sertakan satuannya dan bandingkan secara eksplisit dengan ambang batas normal agronomis yang tertera di konteks.
4. Bila konteks menandai adanya episode PERHATIAN (pembacaan DI BAWAH atau DI ATAS ambang normal), episode tersebut WAJIB dilaporkan meskipun nilai rata-rata periode tampak normal.
5. Padukan "Tindakan terdaftar", jadwal protokol pemupukan (`tr_plant_handling_copy`), usia tanaman (HST/fase tumbuh), serta panduan agronomi untuk memberikan rekomendasi yang spesifik dan dapat langsung ditindaklanjuti.
6. Gunakan Bahasa Indonesia yang jelas, terstruktur, dan mudah dipahami oleh petani maupun penyuluh lapangan."""

SYSTEM_PROMPT_BASELINE = """Anda adalah chatbot pertanian umum pada platform KawalTani (sistem baseline konvensional tanpa koneksi database sensor IoT real-time).
Jawablah pertanyaan pengguna dalam Bahasa Indonesia berdasarkan pengetahuan umum budidaya tanaman padi. Anda tidak memiliki akses ke pembacaan sensor spesifik lokasi atau tanggal tertentu."""


def _has_anomaly(sensor_ctx: str) -> bool:
    if not sensor_ctx:
        return False
    return (
        "PERHATIAN:" in sensor_ctx
        or "DI BAWAH ambang" in sensor_ctx
        or "DI ATAS ambang" in sensor_ctx
    )


def build_prompt(query: str, sensor_ctx: str, agro_ctx: str) -> str:
    parts = []
    if sensor_ctx:
        parts.append("=== DATA SENSOR IOT & JADWAL TANAM KAWALTANI ===\n" + sensor_ctx)
    if agro_ctx:
        parts.append("=== PANDUAN AGRONOMI TANAMAN PADI ===\n" + agro_ctx)
    if not parts:
        parts.append("(Tidak ada konteks eksternal yang ditemukan.)")
    return (
        "\n\n".join(parts)
        + f"\n\n=== PERTANYAAN PENGGUNA ===\n{query}\n\n"
        + "Berikan jawaban dan rekomendasi sesuai seluruh aturan sistem."
    )


class RAGPipeline:
    def __init__(self, store=None, index_dir=None, conn=None):
        self.store = store or vector_store.VectorStore(index_dir)
        self.conn = conn

    def retrieve(self, query: str, now=None, top_k: int = None,
                 max_daily_points: int = None, include_temporal_vec: bool = False) -> dict:
        t0 = time.time()
        it = intent_mod.classify(query, now=now)
        t_intent = time.time() - t0
        t_sql, t_vec = 0.0, 0.0
        initial_route = it.route
        sensor_ctx, agro_ctx, sensors, sources = "", "", [], []

        if it.route in (intent_mod.SQL, intent_mod.HYBRID):
            ts = time.time()
            r_sql = sql_retriever.retrieve(it, conn=self.conn, max_daily_points=max_daily_points)
            t_sql = time.time() - ts
            sensor_ctx, sensors = r_sql["context"], r_sql["sensors"]
            if not r_sql["found"] and it.route == intent_mod.SQL:
                it.route = intent_mod.HYBRID
            elif it.route == intent_mod.SQL and _has_anomaly(sensor_ctx):
                # Routing adaptif: bila data faktual ternyata anomali, tambahkan panduan agronomi
                it.route = intent_mod.HYBRID

        if it.route in (intent_mod.VECTOR, intent_mod.HYBRID):
            tv = time.time()
            r_vec = vector_store.retrieve(
                query, self.store, k=top_k, include_temporal=include_temporal_vec
            )
            t_vec = time.time() - tv
            agro_ctx, sources = r_vec["context"], r_vec.get("sources", [])

        contexts = [c for c in (sensor_ctx, agro_ctx) if c]
        prompt_preview = build_prompt(query, sensor_ctx, agro_ctx)

        return dict(
            intent=it,
            initial_route=initial_route,
            route=it.route,
            sensor_context=sensor_ctx,
            agro_context=agro_ctx,
            contexts=contexts,
            context_chars=sum(len(c) for c in contexts),
            est_prompt_tokens=max(1, len(prompt_preview) // 4),
            sensors=sensors,
            doc_sources=sources,
            retrieval_s=round(time.time() - t0, 4),
            stage_s=dict(
                intent=round(t_intent, 4),
                sql=round(t_sql, 4),
                vector=round(t_vec, 4),
            ),
        )

    def retrieve_sql_only(self, query: str, now=None, max_daily_points: int = None) -> dict:
        """
        Retrieval Konfigurasi B (SQL-Only):
        Hanya mengambil data sensor IoT dan konteks tanaman dari MySQL, tanpa mencari dokumen agronomi.
        """
        t0 = time.time()
        it = intent_mod.classify(query, now=now)
        t_intent = time.time() - t0
        sensor_ctx, sensors = "", []
        t_sql = 0.0

        # Paksa eksekusi SQL retriever jika ada parameter/area atau intent SQL/HYBRID
        if it.route in (intent_mod.SQL, intent_mod.HYBRID) or it.params or it.areas:
            ts = time.time()
            r_sql = sql_retriever.retrieve(it, conn=self.conn, max_daily_points=max_daily_points)
            t_sql = time.time() - ts
            sensor_ctx, sensors = r_sql["context"], r_sql["sensors"]

        contexts = [sensor_ctx] if sensor_ctx else []
        prompt_preview = build_prompt(query, sensor_ctx, "")

        return dict(
            intent=it,
            initial_route="sql",
            route="sql_only",
            sensor_context=sensor_ctx,
            agro_context="",
            contexts=contexts,
            context_chars=len(sensor_ctx),
            est_prompt_tokens=max(1, len(prompt_preview) // 4),
            sensors=sensors,
            doc_sources=[],
            retrieval_s=round(time.time() - t0, 4),
            stage_s=dict(
                intent=round(t_intent, 4),
                sql=round(t_sql, 4),
                vector=0.0,
            ),
        )

    def retrieve_vector_only(self, query: str, top_k: int = None,
                             include_temporal_vec: bool = False) -> dict:
        """
        Retrieval Konfigurasi C (Vector-Only):
        Hanya mencari dokumen agronomi padi di FAISS, tanpa query database sensor IoT.
        """
        t0 = time.time()
        tv = time.time()
        r_vec = vector_store.retrieve(
            query, self.store, k=top_k, include_temporal=include_temporal_vec
        )
        t_vec = time.time() - tv
        agro_ctx, sources = r_vec["context"], r_vec.get("sources", [])

        contexts = [agro_ctx] if agro_ctx else []
        prompt_preview = build_prompt(query, "", agro_ctx)

        return dict(
            intent=None,
            initial_route="vector",
            route="vector_only",
            sensor_context="",
            agro_context=agro_ctx,
            contexts=contexts,
            context_chars=len(agro_ctx),
            est_prompt_tokens=max(1, len(prompt_preview) // 4),
            sensors=[],
            doc_sources=sources,
            retrieval_s=round(time.time() - t0, 4),
            stage_s=dict(
                intent=0.0,
                sql=0.0,
                vector=round(t_vec, 4),
            ),
        )

    def synthesize_local_answer(self, query: str, r: dict) -> str:
        """
        Sintesis jawaban terstruktur lokal berbasis konteks Hybrid Retrieval tanpa API Key LLM.
        Memungkinkan pengujian antarmuka interaktif secara instan (offline/gratis) sebelum API Key diisi.
        """
        parts = []
        if r["sensor_context"]:
            parts.append("### Data sensor lahan\n" + r["sensor_context"])
        if r["agro_context"]:
            parts.append("### Rujukan agronomi\n" + r["agro_context"])
        if not parts:
            parts.append("Konteks data sensor maupun panduan agronomi tidak ditemukan untuk pertanyaan ini.")
        return "\n\n".join(parts)

    def answer(self, query: str, now=None, provider: str = None, model: str = None,
               top_k: int = None, max_daily_points: int = None, api_key: str = None) -> dict:
        """Konfigurasi D: Hybrid RAG (SQL + Vector + Eskalasi Adaptif)."""
        r = self.retrieve(query, now=now, top_k=top_k, max_daily_points=max_daily_points)
        prompt = build_prompt(query, r["sensor_context"], r["agro_context"])
        gen = llm.generate(SYSTEM_PROMPT_RAG, prompt, provider=provider, model=model, api_key=api_key)
        return dict(
            mode="rag",
            configuration="D",
            question=query,
            answer=gen["text"],
            prompt=prompt,
            **{k: v for k, v in r.items() if k != "intent"},
            llm=gen,
            total_s=round(r["retrieval_s"] + gen["latency_s"], 3),
        )

    def answer_sql_only(self, query: str, now=None, provider: str = None, model: str = None,
                        max_daily_points: int = None, api_key: str = None) -> dict:
        """Konfigurasi B: SQL-Only (hanya data sensor IoT)."""
        r = self.retrieve_sql_only(query, now=now, max_daily_points=max_daily_points)
        prompt = build_prompt(query, r["sensor_context"], "")
        gen = llm.generate(SYSTEM_PROMPT_RAG, prompt, provider=provider, model=model, api_key=api_key)
        return dict(
            mode="sql_only",
            configuration="B",
            question=query,
            answer=gen["text"],
            prompt=prompt,
            **{k: v for k, v in r.items() if k != "intent"},
            llm=gen,
            total_s=round(r["retrieval_s"] + gen["latency_s"], 3),
        )

    def answer_vector_only(self, query: str, provider: str = None, model: str = None,
                           top_k: int = None, api_key: str = None) -> dict:
        """Konfigurasi C: Vector-Only (hanya dokumen agronomi)."""
        r = self.retrieve_vector_only(query, top_k=top_k)
        prompt = build_prompt(query, "", r["agro_context"])
        gen = llm.generate(SYSTEM_PROMPT_RAG, prompt, provider=provider, model=model, api_key=api_key)
        return dict(
            mode="vector_only",
            configuration="C",
            question=query,
            answer=gen["text"],
            prompt=prompt,
            **{k: v for k, v in r.items() if k != "intent"},
            llm=gen,
            total_s=round(r["retrieval_s"] + gen["latency_s"], 3),
        )

    def answer_baseline(self, query: str, provider: str = None, model: str = None,
                        api_key: str = None) -> dict:
        """
        Konfigurasi A: Chatbot konvensional baseline (tanpa retrieval sensor IoT maupun dokumen RAG)
        untuk uji komparatif Hipotesis H1 & H2 terhadap metrik RAGAS.
        """
        r_ref = self.retrieve(query)
        gen = llm.generate(SYSTEM_PROMPT_BASELINE, query, provider=provider, model=model, api_key=api_key)
        return dict(
            mode="baseline",
            configuration="A",
            question=query,
            answer=gen["text"],
            prompt=query,
            initial_route="none",
            route="baseline_no_rag",
            sensor_context="",
            agro_context="",
            contexts=r_ref["contexts"] if r_ref["contexts"] else ["Tidak ada akses data sensor pada sistem baseline."],
            context_chars=0,
            est_prompt_tokens=max(1, len(query) // 4),
            sensors=[],
            doc_sources=[],
            retrieval_s=0.0,
            llm=gen,
            total_s=round(gen["latency_s"], 3),
        )

    def answer_by_mode(self, mode: str, query: str, now=None, provider: str = None,
                       model: str = None, top_k: int = None, max_daily_points: int = None,
                       api_key: str = None) -> dict:
        """
        Dispatcher terpadu untuk 4 konfigurasi eksperimen PRD:
          - A / BASELINE / LLM_ONLY : Tanpa retrieval
          - B / SQL_ONLY / SQL      : Retrieval hanya database sensor IoT
          - C / VECTOR_ONLY / VECTOR: Retrieval hanya dokumen agronomi
          - D / HYBRID / RAG        : Hybrid RAG penuh
        """
        m = (mode or "D").strip().upper()
        if m in ("A", "BASELINE", "LLM_ONLY", "LLM"):
            return self.answer_baseline(query, provider=provider, model=model, api_key=api_key)
        elif m in ("B", "SQL", "SQL_ONLY"):
            return self.answer_sql_only(query, now=now, provider=provider, model=model,
                                        max_daily_points=max_daily_points, api_key=api_key)
        elif m in ("C", "VECTOR", "VECTOR_ONLY"):
            return self.answer_vector_only(query, provider=provider, model=model,
                                          top_k=top_k, api_key=api_key)
        else:  # "D", "HYBRID", "RAG"
            return self.answer(query, now=now, provider=provider, model=model,
                               top_k=top_k, max_daily_points=max_daily_points, api_key=api_key)

