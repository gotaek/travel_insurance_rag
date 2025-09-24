from typing import Dict, Any, List, Optional
import os
import re
from collections import Counter

from retriever.vector import vector_search
from retriever.keyword import keyword_search, keyword_search_full_corpus
from retriever.hybrid import hybrid_search
from retriever.korean_tokenizer import (
    extract_insurance_keywords, 
    calculate_keyword_relevance,
    get_keyword_weights
)
from app.deps import get_settings

def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    웹 검색 결과를 활용한 개선된 하이브리드 검색 수행
    - 벡터 검색: Chroma DB 기반
    - 키워드 검색: 전체 코퍼스 BM25 기반 (다양성 향상)
    - 하이브리드: 두 결과 가중치 기반 병합 + 웹 컨텍스트 가중치
    - 웹 검색 결과를 활용한 쿼리 확장 및 컨텍스트 개선
    """
    q = state.get("question", "")
    web_results = state.get("web_results", [])
    s = get_settings()
    
    # 빈 질문 가드
    if not q or not q.strip():
        return {
            **state, 
            "passages": [],
            "search_meta": {
                "reason": "empty_question",
                "k_value": 0,
                "candidates_count": 0,
                "used_query": "",
                "web_keywords": [],
                "from_cache": False
            }
        }

    # Chroma DB 경로 설정
    db_path = s.VECTOR_DIR
    collection_name = "insurance_docs"

    # 웹 검색 결과를 활용한 쿼리 확장
    enhanced_query = _enhance_query_with_web_results(q, web_results)
    
    # 확장된 쿼리 길이 기반 k 값 조정
    k = _determine_k_value(enhanced_query, web_results)
    
    # 검색 메타데이터 초기화
    search_meta = {
        "k_value": k,
        "candidates_count": 0,
        "used_query": enhanced_query,
        "web_keywords": _extract_keywords_from_web_results(web_results)[:5],  # 상위 5개
        "from_cache": False
    }

    try:
        # 벡터 검색 (Chroma DB 사용) - 더 많은 후보 검색
        vec_k = min(k * 3, 50)  # 벡터 검색은 더 많은 후보 검색
        vec_results = vector_search(enhanced_query, db_path, collection_name, k=vec_k)
        
        # 전체 코퍼스에서 BM25 키워드 검색 (다양성 향상)
        kw_k = min(k * 2, 30)  # 키워드 검색도 더 많은 후보 검색
        kw_results = keyword_search_full_corpus(enhanced_query, k=kw_k)
        
        # 벡터 검색 실패 시 키워드/웹만으로 진행
        if not vec_results and not kw_results:
            return {
                **state,
                "passages": [],
                "search_meta": {
                    **search_meta,
                    "reason": "no_search_results",
                    "candidates_count": 0
                }
            }
        
        # 웹 결과를 직접 패시지 후보로 포함
        web_passages = _convert_web_results_to_passages(web_results)
        
        # 향상된 하이브리드 검색 (웹 컨텍스트 가중치 반영)
        merged = _enhanced_hybrid_search_with_web_weight(
            enhanced_query, 
            vec_results, 
            kw_results, 
            web_passages,
            k=k
        )
        
        search_meta["candidates_count"] = len(merged)
        
        return {**state, "passages": merged, "search_meta": search_meta}
        
    except Exception as e:
        # 예외 발생 시 빈 결과와 에러 메타데이터 반환
        return {
            **state,
            "passages": [],
            "search_meta": {
                **search_meta,
                "reason": f"search_error: {str(e)}",
                "candidates_count": 0
            }
        }

def _enhance_query_with_web_results(original_query: str, web_results: List[Dict[str, Any]]) -> str:
    """
    웹 검색 결과를 활용하여 검색 쿼리를 확장합니다.
    
    Args:
        original_query: 원본 질문
        web_results: 웹 검색 결과 리스트
        
    Returns:
        확장된 검색 쿼리
    """
    if not web_results:
        return original_query
    
    # 웹 결과에서 키워드 추출
    web_keywords = _extract_keywords_from_web_results(web_results)
    
    # 원본 쿼리와 웹 키워드 결합
    if web_keywords:
        # 상위 3개 키워드만 추가하여 노이즈 방지
        top_keywords = web_keywords[:3]
        enhanced_query = f"{original_query} {' '.join(top_keywords)}"
        return enhanced_query
    
    return original_query

def _extract_keywords_from_web_results(web_results: List[Dict[str, Any]]) -> List[str]:
    """
    웹 검색 결과에서 관련 키워드를 추출합니다.
    
    Args:
        web_results: 웹 검색 결과 리스트
        
    Returns:
        추출된 키워드 리스트 (빈도순 정렬)
    """
    if not web_results:
        return []
    
    # 웹 결과에서 텍스트 수집
    all_text = []
    for result in web_results:
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        if title:
            all_text.append(title)
        if snippet:
            all_text.append(snippet)
    
    # 개선된 한국어 토크나이저를 사용한 키워드 추출
    insurance_keywords = extract_insurance_keywords(" ".join(all_text), min_frequency=1)
    
    return insurance_keywords


def _determine_k_value(query: str, web_results: List[Dict[str, Any]]) -> int:
    """
    확장된 쿼리 길이와 웹 검색 결과에 따라 동적으로 k 값을 조정합니다.
    
    Args:
        query: 확장된 검색 쿼리
        web_results: 웹 검색 결과
        
    Returns:
        조정된 k 값
    """
    base_k = 5
    
    # 웹 결과가 있으면 더 많은 로컬 문서 검색
    if web_results:
        base_k += 3
    
    # 확장된 쿼리 길이 기반 조정
    query_tokens = len(query.split())
    if query_tokens > 10:
        base_k += 2
    elif query_tokens > 5:
        base_k += 1
    
    # 쿼리 길이 기반 조정
    if len(query) > 30:
        base_k += 2
    elif len(query) > 15:
        base_k += 1
    
    # 최대 15개로 제한
    return min(base_k, 15)

def _enhanced_hybrid_search(
    query: str,
    vector_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    web_results: List[Dict[str, Any]],
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    웹 검색 결과를 고려한 향상된 하이브리드 검색을 수행합니다.
    
    Args:
        query: 검색 쿼리
        vector_results: 벡터 검색 결과
        keyword_results: 키워드 검색 결과
        web_results: 웹 검색 결과
        k: 반환할 결과 수
        
    Returns:
        통합된 검색 결과
    """
    # 기본 하이브리드 검색 수행
    merged = hybrid_search(query, vector_results, keyword_results, k=k)
    
    # 웹 결과가 있으면 로컬 문서 결과에 웹 컨텍스트 정보 추가
    if web_results:
        merged = _add_web_context_to_results(merged, web_results)
    
    return merged

def _add_web_context_to_results(
    local_results: List[Dict[str, Any]], 
    web_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    로컬 검색 결과에 웹 컨텍스트 정보를 추가합니다.
    
    Args:
        local_results: 로컬 검색 결과
        web_results: 웹 검색 결과
        
    Returns:
        웹 컨텍스트가 추가된 검색 결과
    """
    if not web_results:
        return local_results
    
    # 웹 결과에서 상위 3개만 선택
    top_web_results = web_results[:3]
    
    # 각 로컬 결과에 웹 컨텍스트 정보 추가
    enhanced_results = []
    for result in local_results:
        enhanced_result = dict(result)
        
        # 웹 컨텍스트 정보 추가
        enhanced_result["web_context"] = {
            "has_web_info": True,
            "web_sources_count": len(top_web_results),
            "web_relevance_score": _calculate_web_relevance(result, top_web_results)
        }
        
        enhanced_results.append(enhanced_result)
    
    return enhanced_results

def _calculate_web_relevance(
    local_result: Dict[str, Any], 
    web_results: List[Dict[str, Any]]
) -> float:
    """
    로컬 검색 결과와 웹 검색 결과 간의 관련성을 계산합니다.
    
    Args:
        local_result: 로컬 검색 결과
        web_results: 웹 검색 결과
        
    Returns:
        관련성 점수 (0.0 ~ 1.0)
    """
    if not web_results:
        return 0.0
    
    local_text = local_result.get("text", "")
    relevance_scores = []
    
    for web_result in web_results:
        web_title = web_result.get("title", "")
        web_snippet = web_result.get("snippet", "")
        web_text = f"{web_title} {web_snippet}"
        
        # 개선된 키워드 관련성 계산
        relevance_score = calculate_keyword_relevance(local_text, [web_text])
        relevance_scores.append(relevance_score)
    
    # 평균 관련성 점수 반환
    return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

def _convert_web_results_to_passages(web_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    웹 검색 결과를 패시지 형태로 변환합니다.
    
    Args:
        web_results: 웹 검색 결과 리스트
        
    Returns:
        패시지 형태로 변환된 웹 결과
    """
    if not web_results:
        return []
    
    passages = []
    for i, result in enumerate(web_results[:3]):  # 상위 3개만 사용
        passage = {
            "text": f"{result.get('title', '')} {result.get('snippet', '')}",
            "source": "web",
            "url": result.get("url", ""),
            "title": result.get("title", ""),
            "score_web": result.get("score_web", 0.5),  # 기본 웹 점수
            "web_relevance_score": result.get("relevance_score", 0.5),
            "doc_id": f"web_{i}",
            "page": 0,
            "timestamp": result.get("timestamp", "")
        }
        passages.append(passage)
    
    return passages

def _enhanced_hybrid_search_with_web_weight(
    query: str,
    vector_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    web_passages: List[Dict[str, Any]],
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    웹 컨텍스트 가중치를 반영한 향상된 하이브리드 검색을 수행합니다.
    
    Args:
        query: 검색 쿼리
        vector_results: 벡터 검색 결과
        keyword_results: 키워드 검색 결과
        web_passages: 웹 패시지
        k: 반환할 결과 수
        
    Returns:
        웹 가중치가 반영된 통합 검색 결과
    """
    # 기본 하이브리드 검색 수행
    merged = hybrid_search(query, vector_results, keyword_results, k=k*2)  # 더 많은 후보
    
    # 웹 패시지 추가
    all_results = merged + web_passages
    
    # 웹 컨텍스트 가중치 적용
    weighted_results = []
    for result in all_results:
        weighted_result = dict(result)
        
        # 웹 컨텍스트 가중치 계산
        web_weight = 0.0
        if "web_relevance_score" in result:
            web_weight = result["web_relevance_score"] * 0.2  # λ=0.2 가중치
        
        # 기본 점수에 웹 가중치 적용
        base_score = result.get("score", 0.0)
        if base_score > 0:
            final_score = base_score * (1 + web_weight)
        else:
            # 웹 결과의 경우 기본 점수 사용
            final_score = result.get("score_web", 0.5)
        
        weighted_result["score"] = min(final_score, 1.0)  # 1.0으로 제한
        weighted_results.append(weighted_result)
    
    # 점수 기준 정렬
    weighted_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    
    return weighted_results[:k]