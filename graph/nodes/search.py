from typing import Dict, Any
import os

from retriever.hybrid import hybrid_search
from app.deps import get_settings

def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    질문을 받아 하이브리드 검색 수행
    - 벡터 검색: FAISS 인덱스 기반
    - 키워드 검색: BM25 기반
    - 하이브리드: 두 결과 가중치 기반 병합
    """
    q = state.get("question", "")
    s = get_settings()

    index_path = os.path.join(s.VECTOR_DIR, "index.faiss")
    meta_path = os.path.join(s.VECTOR_DIR, "index.pkl")

    # 하이브리드 검색 (벡터 + 키워드 통합)
    merged = hybrid_search(q, index_path, meta_path, k=5)

    return {**state, "passages": merged}