from typing import Dict, Any, List
import os
import pickle

from retriever.vector import vector_search, VectorStore, _embed_stub
from retriever.keyword import keyword_search
from retriever.hybrid import hybrid_search
from app.deps import get_settings

def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    질문을 받아 하이브리드 검색 수행
    - vector_search: FAISS 인덱스 기반
    - keyword_search: 메타 텍스트 기반 BM25
    - hybrid_search: 두 결과 병합
    """
    q = state.get("question", "")
    s = get_settings()

    index_path = os.path.join(s.VECTOR_DIR, "index.faiss")
    meta_path = os.path.join(s.VECTOR_DIR, "index.pkl")

    # 메타 로드 (BM25용)
    corpus_meta: List[Dict[str, Any]] = []
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            corpus_meta = pickle.load(f)

    # vector 검색
    vec_results = vector_search(q, index_path, meta_path, k=5)

    # keyword 검색
    kw_results = keyword_search(q, corpus_meta, k=5)

    # hybrid
    merged = hybrid_search(q, vec_results, kw_results, k=5)

    return {**state, "passages": merged}