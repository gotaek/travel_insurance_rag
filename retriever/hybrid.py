from typing import List, Dict, Any, Optional
import math
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

def _minmax_norm(values: List[float]) -> List[float]:
    """Min-Max 정규화"""
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if math.isclose(vmin, vmax):
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]

def _z_score_norm(values: List[float]) -> List[float]:
    """Z-Score 정규화 (평균 0, 표준편차 1)"""
    if not values or len(values) < 2:
        return [0.0 for _ in values]
    
    mean_val = sum(values) / len(values)
    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
    std_dev = math.sqrt(variance)
    
    if std_dev == 0:
        return [0.0 for _ in values]
    
    return [(x - mean_val) / std_dev for x in values]

def _robust_norm(values: List[float]) -> List[float]:
    """Robust 정규화 (중앙값과 IQR 사용)"""
    if not values:
        return []
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    # 중앙값 계산
    if n % 2 == 0:
        median = (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
    else:
        median = sorted_values[n//2]
    
    # IQR 계산
    q1_idx = n // 4
    q3_idx = 3 * n // 4
    q1 = sorted_values[q1_idx]
    q3 = sorted_values[q3_idx]
    iqr = q3 - q1
    
    if iqr == 0:
        return [0.0 for _ in values]
    
    return [(x - median) / iqr for x in values]

def _merge_by_docpage(
    vec_hits: List[Dict[str, Any]], 
    kw_hits: List[Dict[str, Any]], 
    alpha: float = 0.6,
    norm_method: str = "minmax"
) -> List[Dict[str, Any]]:
    """
    벡터/키워드 결과를 문서ID+페이지 단위로 머지하고 정규화 점수 가중합.
    
    Args:
        vec_hits: 벡터 검색 결과
        kw_hits: 키워드 검색 결과
        alpha: 벡터 가중치(0~1). 0.6이면 벡터 60%, 키워드 40%
        norm_method: 정규화 방법 ("minmax", "zscore", "robust")
    """
    def key(x): return (x.get("doc_id"), x.get("page"))

    # 정규화 방법 선택
    norm_func = {
        "minmax": _minmax_norm,
        "zscore": _z_score_norm,
        "robust": _robust_norm
    }.get(norm_method, _minmax_norm)

    # 정규화
    vec = vec_hits[:]
    kw = kw_hits[:]
    
    try:
        v_scores = norm_func([h.get("score_vec", 0.0) for h in vec])
        k_scores = norm_func([h.get("score_kw", 0.0) for h in kw])
        
        for h, s in zip(vec, v_scores):
            h["_norm_vec"] = s
        for h, s in zip(kw, k_scores):
            h["_norm_kw"] = s
    except Exception as e:
        logger.warning(f"정규화 실패, 기본값 사용: {e}")
        for h in vec:
            h["_norm_vec"] = h.get("score_vec", 0.0)
        for h in kw:
            h["_norm_kw"] = h.get("score_kw", 0.0)

    merged: Dict[Any, Dict[str, Any]] = {}
    for h in vec:
        merged[key(h)] = dict(h)
    for h in kw:
        k_ = key(h)
        if k_ in merged:
            # 병합
            for f in ["score_kw", "_norm_kw"]:
                merged[k_][f] = h.get(f, merged[k_].get(f))
            # 텍스트가 비어있으면 보강
            if not merged[k_].get("text") and h.get("text"):
                merged[k_]["text"] = h["text"]
        else:
            merged[k_] = dict(h)

    # 최종 스코어 계산
    out = []
    for item in merged.values():
        nv = item.get("_norm_vec", 0.0)
        nk = item.get("_norm_kw", 0.0)
        
        # 가중치 적용
        item["score"] = alpha * nv + (1 - alpha) * nk
        
        # 메타데이터 추가
        item["score_components"] = {
            "vector_score": nv,
            "keyword_score": nk,
            "alpha": alpha,
            "norm_method": norm_method
        }
        
        out.append(item)

    out.sort(key=lambda x: x["score"], reverse=True)
    return out

def hybrid_search(
    query: str,
    vector_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    k: int = 5,
    alpha: float = 0.6,
    norm_method: str = "minmax",
    target_insurers: List[str] = None
) -> List[Dict[str, Any]]:
    """
    벡터 검색과 키워드 검색 결과를 병합하여 하이브리드 검색 수행.
    보험사명이 명확히 추출된 경우 키워드 검색 가중치를 동적으로 조정합니다.
    
    Args:
        query: 검색 쿼리
        vector_results: 벡터 검색 결과
        keyword_results: 키워드 검색 결과
        k: 반환할 결과 수
        alpha: 벡터 가중치 (0~1)
        norm_method: 정규화 방법 ("minmax", "zscore", "robust")
        target_insurers: 타겟 보험사명 리스트 (동적 가중치 조정용)
        
    Returns:
        병합된 검색 결과
    """
    try:
        # 입력 검증
        if not vector_results and not keyword_results:
            logger.warning("벡터 검색과 키워드 검색 결과가 모두 비어있음")
            return []
        
        # 동적 가중치 조정: 보험사명이 명확히 추출된 경우 키워드 검색 가중치 증가
        adjusted_alpha = _calculate_dynamic_alpha(alpha, target_insurers, query)
        
        # 병합 수행
        merged = _merge_by_docpage(
            vector_results, 
            keyword_results, 
            alpha=adjusted_alpha,
            norm_method=norm_method
        )
        
        # 결과 제한
        result = merged[:k]
        
        # 로깅
        logger.debug(f"하이브리드 검색 완료: {len(result)}개 결과 (벡터: {len(vector_results)}, 키워드: {len(keyword_results)}, alpha: {adjusted_alpha:.2f})")
        
        return result
        
    except Exception as e:
        logger.error(f"하이브리드 검색 실패: {e}")
        # 폴백: 단순 병합
        all_results = vector_results + keyword_results
        all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return all_results[:k]

def _calculate_dynamic_alpha(base_alpha: float, target_insurers: List[str], query: str) -> float:
    """
    보험사명 추출 여부에 따라 동적으로 가중치를 조정합니다.
    
    Args:
        base_alpha: 기본 벡터 가중치
        target_insurers: 타겟 보험사명 리스트
        query: 검색 쿼리
        
    Returns:
        조정된 가중치
    """
    if not target_insurers:
        return base_alpha
    
    # 보험사명이 명확히 추출된 경우 키워드 검색 가중치 증가
    # 벡터 가중치를 0.4로 감소, 키워드 가중치를 0.6으로 증가
    adjusted_alpha = 0.4
    
    # 쿼리에 보험사명이 직접 포함된 경우 더 강한 조정
    query_lower = query.lower()
    for insurer in target_insurers:
        if insurer.lower() in query_lower:
            adjusted_alpha = 0.3  # 더 강한 키워드 검색 가중치
            break
    
    logger.debug(f"동적 가중치 조정: base_alpha={base_alpha:.2f} -> adjusted_alpha={adjusted_alpha:.2f} (보험사: {target_insurers})")
    
    return adjusted_alpha