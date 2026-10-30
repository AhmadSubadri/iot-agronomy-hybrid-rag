#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_evaluation.py — Evaluasi Komprehensif Tesis S2 KawalTani RAG (Bab 3.9 & 3.10 Proposal).

Mendukung 4 sub-perintah lengkap:
  1. `audit-retrieval` : Uji tanpa biaya API LLM (akurasi routing Intent Classifier,
                         kelengkapan konteks 100 QA + 10 Vector QA, dan latensi retrieval).
  2. `collect`         : Kumpulkan jawaban LLM (`--mode rag` atau `--mode baseline`) untuk
                         GPT-4o-mini, Gemini Flash, atau Llama 3 8B (Ollama).
  3. `score`           : Hitung 4 metrik RAGAS (Faithfulness, Answer Relevancy,
                         Context Precision, Context Recall) terhadap target Tabel 5 Proposal.
  4. `hypothesis`      : Uji statistik Hipotesis H1 & H2 (Paired t-test & Wilcoxon Signed-Rank Test)
                         antara hasil RAGAS Sistem RAG vs Sistem Chatbot Baseline, serta
                         perbandingan A/B Testing antar-model LLM + estimasi biaya per 1.000 kueri!
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Tarif referensi API Cloud (USD per 1 juta token) untuk analisis biaya Bab 3.9.3
PRICING_PER_1M = {
    "gpt-4o-mini": (0.15, 0.60),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
    "llama3:8b": (0.0, 0.0),  # On-premise lokal via Ollama (biaya API $0)
}


def cmd_audit_retrieval(args):
    from kawaltani_rag.pipeline import RAGPipeline
    pipe = RAGPipeline(index_dir=args.index_dir)

    with open(args.qa, encoding="utf-8") as f:
        qa = list(csv.DictReader(f))

    latencies = []
    exact_match = 0
    adaptive_escalated = 0
    empty_ctx = 0
    empty_sensor_when_needed = 0

    for row in qa:
        r = pipe.retrieve(row["question"])
        latencies.append(r["retrieval_s"])
        if not r["contexts"]:
            empty_ctx += 1
        if row["retrieval_type"] in ("sql", "hybrid") and not r["sensor_context"]:
            empty_sensor_when_needed += 1
            print(f"  [WARN] Sensor kosong pada {row['id']}: {row['question']}")

        exp = row["retrieval_type"]
        if r["route"] == exp:
            exact_match += 1
        elif exp == "sql" and r["initial_route"] == "sql" and r["route"] == "hybrid":
            adaptive_escalated += 1
        else:
            print(f"  [MISMATCH] {row['id']}: expected={exp}, initial={r['initial_route']}, final={r['route']} | {row['question']}")

    print("\n================ HASIL AUDIT HYBRID RETRIEVAL (100 QA UTAMA) ================")
    print(f"Total Pertanyaan Uji           : {len(qa)}")
    print(f"Konteks Terisi Penuh           : {len(qa) - empty_ctx}/{len(qa)} ({(len(qa)-empty_ctx)/len(qa)*100:.1f}%)")
    print(f"Konteks Sensor SQL Valid       : {len(qa) - empty_sensor_when_needed}/{len(qa)} ({(len(qa)-empty_sensor_when_needed)/len(qa)*100:.1f}%)")
    print(f"Kecocokan Rute Langsung        : {exact_match}/{len(qa)} ({exact_match/len(qa)*100:.1f}%)")
    print(f"Eskalasi Adaptif (SQL->Hybrid) : {adaptive_escalated} kueri")
    print(f"Akurasi Routing Efektif Total  : {exact_match + adaptive_escalated}/{len(qa)} ({(exact_match + adaptive_escalated)/len(qa)*100:.1f}%)")
    print(f"Latensi Retrieval Rata-rata    : {statistics.mean(latencies)*1000:.2f} ms (median {statistics.median(latencies)*1000:.2f} ms, maks {max(latencies)*1000:.2f} ms)")

    vec_file = BASE_DIR / "data" / "qa_vector_general_10.csv"
    if vec_file.exists():
        with open(vec_file, encoding="utf-8") as f:
            vqa = list(csv.DictReader(f))
        v_ok = sum(1 for row in vqa if pipe.retrieve(row["question"])["route"] == "vector")
        print(f"Uji Khusus Jalur 2 (Vector QA) : {v_ok}/{len(vqa)} ({v_ok/len(vqa)*100:.1f}%) tepat masuk jalur `vector`")
    else:
        vqa, v_ok = [], 0

    # Simpan ringkasan audit agar dapat ditampilkan pada tab Evaluasi antarmuka web
    summary = dict(
        n=len(qa),
        context_full=len(qa) - empty_ctx,
        sensor_valid=len(qa) - empty_sensor_when_needed,
        route_exact=exact_match,
        route_escalated=adaptive_escalated,
        route_effective=exact_match + adaptive_escalated,
        latency_mean_ms=round(statistics.mean(latencies) * 1000, 2),
        latency_median_ms=round(statistics.median(latencies) * 1000, 2),
        latency_max_ms=round(max(latencies) * 1000, 2),
        vector_n=len(vqa),
        vector_ok=v_ok,
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    out_dir = BASE_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "audit_retrieval.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"[OK] Ringkasan audit disimpan ke: {out_dir / 'audit_retrieval.json'}")


def cmd_collect(args):
    from kawaltani_rag.pipeline import RAGPipeline
    pipe = RAGPipeline(index_dir=args.index_dir)

    with open(args.qa, encoding="utf-8") as f:
        qa = list(csv.DictReader(f))
    if args.limit:
        qa = qa[:args.limit]

    results, errors = [], 0
    for i, row in enumerate(qa, 1):
        q = row["question"]
        try:
            if args.mode == "baseline":
                r = pipe.answer_baseline(q, provider=args.provider, model=args.model)
            else:
                r = pipe.answer(q, provider=args.provider, model=args.model)
            results.append(dict(
                id=row["id"],
                category=row.get("category", "general"),
                question=q,
                answer=r["answer"],
                ground_truth=row["reference_answer"],
                contexts=r["contexts"],
                mode=args.mode,
                route=r["route"],
                retrieval_s=r["retrieval_s"],
                llm_s=r["llm"]["latency_s"],
                total_s=r["total_s"],
                prompt_tokens=r["llm"].get("prompt_tokens"),
                completion_tokens=r["llm"].get("completion_tokens"),
                expected_route=row.get("retrieval_type"),
            ))
        except Exception as e:
            errors += 1
            print(f"  [{i}/{len(qa)}] GAGAL {row['id']}: {type(e).__name__}: {e}", file=sys.stderr)
        if i % 10 == 0:
            print(f"  ...{i}/{len(qa)} selesai")
        if args.sleep:
            time.sleep(args.sleep)

    meta = dict(mode=args.mode, provider=args.provider, model=args.model, n=len(results), errors=errors)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dict(meta=meta, results=results), f, ensure_ascii=False, indent=1)
    print(f"\n[OK] Tersimpan {len(results)} hasil ({errors} gagal) -> {out_path}")


def cmd_score(args):
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    except ImportError:
        sys.exit("Paket evaluasi belum lengkap. Jalankan: pip install ragas datasets langchain-openai")

    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)
    rows = data["results"]

    ds = Dataset.from_dict({
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "contexts": [r["contexts"] for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    })
    scores = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    df = scores.to_pandas()

    target = dict(faithfulness=0.80, answer_relevancy=0.80, context_precision=0.75, context_recall=0.75)
    print(f"\n=== HASIL RAGAS ({data['meta'].get('mode','rag').upper()} — {data['meta']['model']}) ===")
    for metric, minimum in target.items():
        if metric in df.columns:
            val = df[metric].mean()
            status = "TERCAPAI" if val > minimum else "BELUM TERCAPAI"
            print(f"  {metric:18s}: {val:.4f} (Target > {minimum:.2f}) -> {status}")

    out_csv = args.out or str(args.results).replace(".json", "_ragas.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] Skor per butir disimpan ke: {out_csv}")


def cmd_hypothesis(args):
    """
    Uji Statistik Hipotesis H1 & H2 (Bab 3.10 Proposal):
      - H1: Skor Faithfulness Sistem RAG > Sistem Baseline secara signifikan (p < 0.05)
      - H2: Skor Answer Relevancy Sistem RAG > Sistem Baseline secara signifikan (p < 0.05)
    Menerima 2 file CSV keluaran `score` (`--rag-csv` dan `--baseline-csv`).
    """
    import pandas as pd
    from scipy import stats

    df_rag = pd.read_csv(args.rag_csv)
    df_base = pd.read_csv(args.baseline_csv)

    print("\n================ UJI HIPOTESIS STATISTIK H1 & H2 (BAB 3.10) ================")
    for hyp, col in [("H1 (Faithfulness)", "faithfulness"), ("H2 (Answer Relevancy)", "answer_relevancy")]:
        if col not in df_rag.columns or col not in df_base.columns:
            print(f"[SKIP] Kolom `{col}` tidak ditemukan.")
            continue
        paired = pd.DataFrame({"rag": df_rag[col], "base": df_base[col]}).dropna()
        t_stat, p_two = stats.ttest_rel(paired["rag"], paired["base"])
        p_one_t = p_two / 2 if t_stat > 0 else 1.0 - (p_two / 2)
        try:
            w_stat, p_wilcox = stats.wilcoxon(paired["rag"], paired["base"], alternative="greater")
        except ValueError:
            w_stat, p_wilcox = 0.0, 1.0

        keputusan = "DITERIMA (Signifikan pada alpha=0.05)" if p_wilcox < 0.05 else "DITOLAK"
        print(f"\n{hyp}:")
        print(f"  Rata-rata Sistem RAG      : {paired['rag'].mean():.4f} (±{paired['rag'].std():.4f})")
        print(f"  Rata-rata Sistem Baseline : {paired['base'].mean():.4f} (±{paired['base'].std():.4f})")
        print(f"  Selisih Peningkatan       : +{paired['rag'].mean() - paired['base'].mean():.4f}")
        print(f"  Paired t-test (1-tailed)  : t = {t_stat:.4f}, p = {p_one_t:.6g}")
        print(f"  Wilcoxon Signed-Rank      : W = {w_stat:.4f}, p = {p_wilcox:.6g}")
        print(f"  Keputusan Hipotesis       : {keputusan}")


def cmd_compare(args):
    print(f"\n{'Mode':10s} {'Model':24s} {'N':>4s} {'Latensi(s)':>11s} {'Est.Biaya/1K($)':>16s}")
    print("-" * 70)
    for path in args.results:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        rows, meta = d["results"], d["meta"]
        if not rows:
            continue
        lat = statistics.mean(r["total_s"] for r in rows)
        model_name = meta.get("model", "")
        pt = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens")]
        ct = [r["completion_tokens"] for r in rows if r.get("completion_tokens")]
        cost_str = "N/A"
        if pt and ct:
            avg_p = statistics.mean(pt)
            avg_c = statistics.mean(ct)
            in_rate, out_rate = PRICING_PER_1M.get(model_name, (0.15, 0.60))
            cost_1k = (avg_p * 1000 / 1_000_000 * in_rate) + (avg_c * 1000 / 1_000_000 * out_rate)
            cost_str = f"${cost_1k:.4f}"
        print(f"{meta.get('mode','rag'):10s} {model_name[:24]:24s} {len(rows):4d} {lat:11.2f} {cost_str:>16s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit-retrieval", help="Audit akurasi routing & kelengkapan konteks 100 QA")
    a.add_argument("--qa", default=str(BASE_DIR / "data" / "qa_dataset_100.csv"))
    a.add_argument("--index-dir", default=None)
    a.set_defaults(func=cmd_audit_retrieval)

    c = sub.add_parser("collect", help="Jalankan pengumpulan jawaban LLM (RAG atau Baseline)")
    c.add_argument("--qa", default=str(BASE_DIR / "data" / "qa_dataset_100.csv"))
    c.add_argument("--mode", choices=["rag", "baseline"], default="rag")
    c.add_argument("--provider", default="openai")
    c.add_argument("--model", default="gpt-4o-mini")
    c.add_argument("--index-dir", default=None)
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--sleep", type=float, default=0.0)
    c.add_argument("--out", default=str(BASE_DIR / "results" / "hasil_rag.json"))
    c.set_defaults(func=cmd_collect)

    s = sub.add_parser("score", help="Hitung 4 metrik evaluasi RAGAS")
    s.add_argument("--results", required=True)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_score)

    h = sub.add_parser("hypothesis", help="Uji signifikansi statistik Hipotesis H1 & H2 (RAG vs Baseline)")
    h.add_argument("--rag-csv", required=True)
    h.add_argument("--baseline-csv", required=True)
    h.set_defaults(func=cmd_hypothesis)

    p = sub.add_parser("compare", help="Bandingkan latensi & biaya A/B Testing antar-model LLM")
    p.add_argument("--results", nargs="+", required=True)
    p.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

