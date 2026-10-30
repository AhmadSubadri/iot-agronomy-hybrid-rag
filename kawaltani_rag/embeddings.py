"""
Lapisan Embedding Semantik — Pluggable (Sentence-Transformers Multilingual + TF-IDF Fallback).
Mendukung penuh Bab 4.1.5 Proposal (terminologi agronomi Bahasa Indonesia).
"""
import numpy as np
from . import config


class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model_name=None):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name or config.EMBED_MODEL
        self.model = SentenceTransformer(self.model_name)

    def fit(self, corpus):
        return self

    def encode(self, texts):
        v = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(v, dtype="float32")


class TfidfEmbedder:
    name = "tfidf-fallback"

    def __init__(self, model_name=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.model_name = "tfidf"
        self.vec = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), sublinear_tf=True)
        self._fitted = False

    def fit(self, corpus):
        self.vec.fit(list(corpus))
        self._fitted = True
        return self

    def encode(self, texts):
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder harus di-fit() dulu dengan korpus.")
        m = self.vec.transform(list(texts)).toarray().astype("float32")
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms


def get_embedder(prefer_st=True, model_name=None):
    if prefer_st:
        try:
            return SentenceTransformerEmbedder(model_name)
        except Exception as e:
            print(
                f"[embeddings] sentence-transformers gagal dimuat ({type(e).__name__}: {e}); "
                f"memakai TF-IDF fallback."
            )
    return TfidfEmbedder(model_name)

