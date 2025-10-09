from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

def _collect_doc_ids(items: List[Dict[str, Any]]) -> List[str]:
    """
    문서 리스트에서 doc_id를 추출하여 문자열 리스트로 반환.
    
    Args:
        items: doc_id를 포함하는 딕셔너리 리스트
        
    Returns:
        doc_id 문자열 리스트 (None이거나 빈 값은 제외)
    """
    if not items:
        return []
    
    # 리스트 컴프리헨션으로 성능 최적화
    return [str(item["doc_id"]) for item in items if item.get("doc_id") is not None]

def recall_at_k(
    state: Dict[str, Any], 
    gold_doc_ids: Optional[List[Union[str, int]]], 
    k: int = 5
) -> float:
    """
    검색/정제된 상위 k개 컨텍스트에 정답 문서 ID가 포함되어 있는지 평가.
    
    Args:
        state: RAG 시스템의 상태 딕셔너리 (refined 또는 passages 키 포함)
        gold_doc_ids: 정답 문서 ID 리스트 (문자열 또는 정수)
        k: 평가할 상위 k개 문서 수 (기본값: 5)
        
    Returns:
        float: 정답 문서가 포함되면 1.0, 아니면 0.0
        
    Note:
        - refined 컨텍스트가 있으면 우선 사용, 없으면 passages 사용
        - k가 0 이하이거나 gold_doc_ids가 비어있으면 0.0 반환
    """
    # 입력 검증
    if not isinstance(state, dict):
        logger.warning("state가 딕셔너리가 아닙니다.")
        return 0.0
    
    if k <= 0:
        logger.warning(f"k 값이 유효하지 않습니다: {k}")
        return 0.0
    
    if not gold_doc_ids:
        return 0.0
    
    # 후보 문서 추출 (refined 우선, 없으면 passages 사용)
    candidates = state.get("refined") or state.get("passages") or []
    
    # 상위 k개만 선택
    candidates = candidates[:k]
    
    # 후보 문서 ID 추출 및 집합 변환
    candidate_ids = set(_collect_doc_ids(candidates))
    
    # 정답 문서 ID를 문자열로 변환하여 집합 생성
    gold_ids = set(str(doc_id) for doc_id in gold_doc_ids)
    
    # 교집합이 존재하는지 확인 (Recall@k는 하나라도 맞으면 성공)
    has_hit = bool(candidate_ids.intersection(gold_ids))
    
    return 1.0 if has_hit else 0.0