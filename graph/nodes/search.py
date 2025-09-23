from typing import Dict, Any, List
import os

from retriever.vector import vector_search
from retriever.keyword import keyword_search
from retriever.hybrid import hybrid_search
from app.deps import get_settings

def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    질문을 받아 하이브리드 검색 수행
    - 벡터 검색: Chroma DB 기반
    - 키워드 검색: BM25 기반
    - 하이브리드: 두 결과 가중치 기반 병합
    """
    q = state.get("question", "")
    s = get_settings()

    # Chroma DB 경로 설정
    db_path = s.VECTOR_DIR
    collection_name = "insurance_docs"

    # 벡터 검색 (Chroma DB 사용)
    vec_results = vector_search(q, db_path, collection_name, k=5)
    
    # 키워드 검색을 위한 메타데이터는 벡터 검색 결과에서 추출
    corpus_meta = [{"text": result.get("text", "")} for result in vec_results]
    kw_results = keyword_search(q, corpus_meta, k=5)
    
    # 하이브리드 병합
    merged = hybrid_search(q, vec_results, kw_results, k=5)

    return {**state, "passages": merged}