"""
RAG Engine — Orkestrasi Hybrid Retrieval + Context Fusion + Baseline Comparator.

Mendukung:
  1. Mode Hybrid RAG penuh (`answer`) — mengombinasikan SQL Structured Retrieval,
     Vector Search dokumen agronomi, dan Adaptive Routing saat terdeteksi anomali.
  2. Mode Chatbot Baseline (`answer_baseline`) — merepresentasikan sistem chatbot
     konvensional KawalTani tanpa akses RAG ke data sensor IoT real-time, khusus
     untuk pembuktian kuantitatif Hipotesis H1 & H2 (Bab 3.10 Proposal).
  3. Konfigurasi dinamis `top_k` & `max_daily_points` untuk eksperimen Tujuan ke-4
     (analisis trade-off panjang konteks vs latensi respons).
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
        r = self.retrieve(query, now=now, top_k=top_k, max_daily_points=max_daily_points)
        prompt = build_prompt(query, r["sensor_context"], r["agro_context"])
        gen = llm.generate(SYSTEM_PROMPT_RAG, prompt, provider=provider, model=model, api_key=api_key)
        return dict(
            mode="rag",
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
        Chatbot konvensional baseline (tanpa retrieval sensor IoT maupun dokumen RAG)
        untuk uji komparatif Hipotesis H1 & H2 terhadap metrik RAGAS.
        Catatan evaluasi RAGAS: untuk menilai Faithfulness jawaban baseline terhadap fakta
        lapangan aktual secara adil, evaluator tetap dapat membandingkannya terhadap
        ground-truth konteks aktual pertanyaan tersebut.
        """
        r_ref = self.retrieve(query)
        gen = llm.generate(SYSTEM_PROMPT_BASELINE, query, provider=provider, model=model, api_key=api_key)
        return dict(
            mode="baseline",
            question=query,
            answer=gen["text"],
            prompt=query,
            initial_route="none",
            route="baseline_no_rag",
            sensor_context="",
            agro_context="",
            # Simpan referensi konteks faktual agar RAGAS bisa mengukur seberapa jauh baseline
            # gagal/halusinasi terhadap fakta sensor aktual (Faithfulness & Context Recall)
            contexts=r_ref["contexts"] if r_ref["contexts"] else ["Tidak ada akses data sensor pada sistem baseline."],
            context_chars=0,
            est_prompt_tokens=max(1, len(query) // 4),
            sensors=[],
            doc_sources=[],
            retrieval_s=0.0,
            llm=gen,
            total_s=round(gen["latency_s"], 3),
        )

