"""
Jalur 2 — Semantic Retrieval (Vector Search) + Dukungan Incremental Indexing.

Mendukung dua jenis pengetahuan dalam ruang vektor (Bab 1.5 & Bab 3.4 Proposal):
  1. `agronomi_doc`    : Dokumen panduan agronomi tanaman padi (`vector_docs/*.md`).
  2. `temporal_sensor` : Ringkasan temporal mingguan sensor IoT hasil Data-to-Text
                         yang disinkronkan secara incremental dari MySQL (`incremental_indexer.py`).

Backend indeks: otomatis memakai `faiss` (`IndexFlatIP`) bila tersedia, atau
`ExactFlatIPIndex` berbasis NumPy (identik 100% secara matematis untuk cosine similarity
pada vektor ternormalisasi L2).
"""
import os
import re
import json
import pickle
import numpy as np
from . import config
from .embeddings import get_embedder

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class ExactFlatIPIndex:
    """Fallback eksak setara `faiss.IndexFlatIP` bila paket biner faiss belum terpasang."""
    def __init__(self, dim: int):
        self.dim = dim
        self.vecs = np.zeros((0, dim), dtype="float32")

    @property
    def ntotal(self) -> int:
        return int(self.vecs.shape[0])

    def add(self, x: np.ndarray):
        x = np.asarray(x, dtype="float32")
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self.vecs = np.vstack([self.vecs, x]) if self.vecs.size else x

    def search(self, q: np.ndarray, k: int):
        q = np.asarray(q, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if self.ntotal == 0:
            return np.zeros((q.shape[0], 0), dtype="float32"), np.full((q.shape[0], 0), -1, dtype="int64")
        k = min(k, self.ntotal)
        sims = q @ self.vecs.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, idx, axis=1)
        return scores.astype("float32"), idx.astype("int64")


def _new_index(dim: int):
    return faiss.IndexFlatIP(dim) if HAS_FAISS else ExactFlatIPIndex(dim)


def _save_index(index, index_dir: str):
    if HAS_FAISS and not isinstance(index, ExactFlatIPIndex):
        faiss.write_index(index, os.path.join(index_dir, "agronomi.faiss"))
    else:
        np.save(os.path.join(index_dir, "agronomi_vecs.npy"), index.vecs)


def _load_index(index_dir: str, dim: int):
    faiss_path = os.path.join(index_dir, "agronomi.faiss")
    npy_path = os.path.join(index_dir, "agronomi_vecs.npy")
    if HAS_FAISS and os.path.exists(faiss_path):
        return faiss.read_index(faiss_path)
    if os.path.exists(npy_path):
        vecs = np.load(npy_path)
        idx = _new_index(vecs.shape[1] if vecs.ndim == 2 else dim)
        if vecs.size:
            idx.add(vecs.astype("float32"))
        return idx
    raise FileNotFoundError(f"Index vektor belum ditemukan di {index_dir}. Jalankan build_index terlebih dahulu.")


def parse_frontmatter(text: str):
    m = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, m.group(2)


def chunk_document(text: str, meta: dict, source: str, max_words=None, overlap=None):
    max_words = max_words or config.CHUNK_WORDS
    overlap = overlap or config.CHUNK_OVERLAP_WORDS
    doc_title = meta.get("judul", source)

    SKIP_HEADINGS = (
        "sumber rujukan", "daftar rujukan", "referensi",
        "daftar pustaka", "rekomendasi untuk sistem rag", "catatan penting",
    )

    sections = re.split(r"\r?\n(?=## )", text)
    chunks = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        heading = sec.split("\n", 1)[0].lstrip("# ").strip()
        if heading.lower() in SKIP_HEADINGS:
            continue
        words = sec.split()
        if len(words) <= max_words:
            chunks.append((heading, sec))
            continue
        step = max(1, max_words - overlap)
        for i in range(0, len(words), step):
            piece = " ".join(words[i:i + max_words])
            if len(piece.split()) < 30 and chunks:
                break
            chunks.append((heading, piece))

    out = []
    for i, (heading, body) in enumerate(chunks):
        enriched = f"{doc_title} — {heading}\n\n{body}"
        out.append(dict(
            id=f"{source}#{i}",
            chunk_type="agronomi_doc",
            source=source,
            doc_title=doc_title,
            heading=heading,
            text=body,
            embed_text=enriched,
            kategori=meta.get("kategori", ""),
            parameter_terkait=meta.get("parameter_terkait", ""),
        ))
    return out


def load_chunks(docs_dir=None):
    docs_dir = docs_dir or config.VECTOR_DOCS_DIR
    chunks = []
    for fn in sorted(os.listdir(docs_dir)):
        if not fn.endswith(".md") or fn == "references.md":
            continue
        with open(os.path.join(docs_dir, fn), encoding="utf-8") as f:
            meta, body = parse_frontmatter(f.read())
        chunks.extend(chunk_document(body, meta, fn))
    return chunks


def build_index(docs_dir=None, index_dir=None, prefer_st=True, extra_chunks=None):
    index_dir = index_dir or config.INDEX_DIR
    os.makedirs(index_dir, exist_ok=True)
    chunks = load_chunks(docs_dir)
    if extra_chunks:
        chunks.extend(extra_chunks)
    if not chunks:
        raise RuntimeError(f"Tidak ada dokumen .md di {docs_dir or config.VECTOR_DOCS_DIR}")

    embedder = get_embedder(prefer_st)
    corpus = [c["embed_text"] for c in chunks]
    embedder.fit(corpus)
    vecs = embedder.encode(corpus)

    index = _new_index(vecs.shape[1])
    index.add(vecs)

    _save_index(index, index_dir)
    # Simpan juga salinan numpy agar incremental append selalu cepat dan portabel
    np.save(os.path.join(index_dir, "agronomi_vecs.npy"), vecs)

    with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)
    with open(os.path.join(index_dir, "embedder_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            dict(name=embedder.name, model_name=getattr(embedder, "model_name", "")),
            f, indent=1,
        )
    if embedder.name != "sentence-transformers":
        with open(os.path.join(index_dir, "embedder.pkl"), "wb") as f:
            pickle.dump(embedder, f)

    meta = dict(
        backend="faiss" if HAS_FAISS else "numpy-flat-ip",
        embedder=embedder.name,
        model=getattr(embedder, "model_name", ""),
        dim=int(vecs.shape[1]),
        n_chunks=len(chunks),
        n_agronomi=sum(1 for c in chunks if c.get("chunk_type") == "agronomi_doc"),
        n_temporal=sum(1 for c in chunks if c.get("chunk_type") == "temporal_sensor"),
    )
    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta


class VectorStore:
    def __init__(self, index_dir=None):
        self.index_dir = index_dir or config.INDEX_DIR
        with open(os.path.join(self.index_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        with open(os.path.join(self.index_dir, "chunks.json"), encoding="utf-8") as f:
            self.chunks = json.load(f)

        em_meta_path = os.path.join(self.index_dir, "embedder_meta.json")
        if os.path.exists(em_meta_path):
            with open(em_meta_path, encoding="utf-8") as f:
                em = json.load(f)
            if em["name"] == "sentence-transformers":
                self.embedder = get_embedder(prefer_st=True, model_name=em.get("model_name"))
            else:
                with open(os.path.join(self.index_dir, "embedder.pkl"), "rb") as f:
                    self.embedder = pickle.load(f)
        else:
            with open(os.path.join(self.index_dir, "embedder.pkl"), "rb") as f:
                self.embedder = pickle.load(f)

        self.index = _load_index(self.index_dir, int(self.meta["dim"]))

    def append_chunks(self, new_chunks: list) -> int:
        """Incremental indexing tanpa re-index dokumen lama dari awal (Bab 4.1.1 Proposal)."""
        existing_ids = {c["id"] for c in self.chunks}
        fresh = [c for c in new_chunks if c["id"] not in existing_ids]
        if not fresh:
            return 0
        new_vecs = self.embedder.encode([c["embed_text"] for c in fresh])
        self.index.add(new_vecs)
        self.chunks.extend(fresh)

        npy_path = os.path.join(self.index_dir, "agronomi_vecs.npy")
        if os.path.exists(npy_path):
            all_vecs = np.vstack([np.load(npy_path), new_vecs])
        else:
            all_vecs = new_vecs
        np.save(npy_path, all_vecs)
        _save_index(self.index, self.index_dir)

        with open(os.path.join(self.index_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=1)
        self.meta["n_chunks"] = len(self.chunks)
        self.meta["n_temporal"] = sum(1 for c in self.chunks if c.get("chunk_type") == "temporal_sensor")
        with open(os.path.join(self.index_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=1)
        return len(fresh)

    def search(self, query: str, k=None, chunk_type: str = None):
        k = k or config.TOP_K
        qv = self.embedder.encode([query])
        fetch_k = min(len(self.chunks), max(k * 4, 20) if chunk_type else k)
        scores, idx = self.index.search(qv, fetch_k)
        out = []
        for score, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            c = dict(self.chunks[int(i)])
            if chunk_type and c.get("chunk_type") != chunk_type:
                continue
            c["score"] = float(score)
            out.append(c)
            if len(out) >= k:
                break
        return out


def retrieve(query: str, store: VectorStore, k=None, include_temporal: bool = False):
    k = k if k is not None else config.TOP_K
    agro_hits = store.search(query, k=k, chunk_type="agronomi_doc")
    blocks = [
        f"[PENGETAHUAN AGRONOMI — {h['doc_title']} / {h['heading']}]\n{h['text']}"
        for h in agro_hits
    ]
    sources = [h["id"] for h in agro_hits]
    scores = [h["score"] for h in agro_hits]

    if include_temporal and store.meta.get("n_temporal", 0) > 0:
        temp_hits = store.search(query, k=config.TEMPORAL_TOP_K, chunk_type="temporal_sensor")
        for h in temp_hits:
            blocks.append(f"[MEMORI TEMPORAL SENSOR — {h['heading']}]\n{h['text']}")
            sources.append(h["id"])
            scores.append(h["score"])

    return dict(
        context="\n\n".join(blocks),
        sources=sources,
        scores=scores,
        found=bool(blocks),
    )

