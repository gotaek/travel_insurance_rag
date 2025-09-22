from typing import List, Dict, Any
import os, pickle
import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # faiss 미설치/미사용 환경에서도 안전하게

class VectorStore:
    """
    FAISS 인덱스 + 메타(문서ID/페이지/본문 스니펫)를 로드하고 top-k 검색을 제공.
    - 인덱스/메타 파일이 없거나 faiss가 없으면 안전하게 빈 결과 반환.
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

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        if not self.is_ready():
            return []
        D, I = self.index.search(query_vec.astype("float32"), k)
        out: List[Dict[str, Any]] = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            # meta[idx] 예: {"doc_id": "...", "page": 3, "text": "...", "insurer": "...", "version": "..."}
            item = dict(self.meta[idx])
            item["score_vec"] = float(score)
            out.append(item)
        return out

def _embed_stub(text: str, dim: int = 384) -> np.ndarray:
    """
    임시 임베딩 스텁: 텍스트 해시를 시드로 고정 난수를 생성해 재현성 보장.
    실제 임베딩 연결 전 테스트용.
    """
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.random((1, dim), dtype=np.float32)

def vector_search(query: str, index_path: str, meta_path: str, k: int = 5) -> List[Dict[str, Any]]:
    store = VectorStore(index_path, meta_path)
    qv = _embed_stub(query)
    return store.search(qv, k=k)