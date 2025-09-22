import os, pickle
from typing import List, Dict, Any
import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

from retriever.embeddings import embed_texts  # ✅ unify with ingest embedding

class VectorStore:
    """
    FAISS index + metadata loader with top-k search using the same embeddings as ingest.
    Returns empty results safely if faiss or index files are unavailable.
    """
    def __init__(self, index_path: str, meta_path: str):
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.meta: List[Dict[str, Any]] = []

        if faiss and os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.meta = pickle.load(f)

    def is_ready(self) -> bool:
        return self.index is not None and len(self.meta) > 0

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.is_ready():
            return []
        # ✅ use the same embedding dim as the index (BGE-ko via retriever.embeddings)
        qv = embed_texts([query]).astype("float32")
        D, I = self.index.search(qv, k)
        out: List[Dict[str, Any]] = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            item = dict(self.meta[idx])
            item["score_vec"] = float(score)
            out.append(item)
        return out

def vector_search(query: str, index_path: str, meta_path: str, k: int = 5) -> List[Dict[str, Any]]:
        store = VectorStore(index_path, meta_path)
        return store.search(query, k=k)