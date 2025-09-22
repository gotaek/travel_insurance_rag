from typing import Dict, Any, List

def _dedup(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """텍스트 기준 단순 중복 제거"""
    seen = set()
    out = []
    for p in passages:
        text = p.get("text", "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(p)
    return out

def _sort_by_score(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """score 키 기준 내림차순 정렬"""
    return sorted(passages, key=lambda x: x.get("score", 0.0), reverse=True)

def rank_filter_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    검색 결과를 정제:
    - 중복 제거
    - score 기준 정렬
    - 상위 Top-k 선택
    """
    passages = state.get("passages", [])
    if not passages:
        return {**state, "refined": []}

    deduped = _dedup(passages)
    sorted_passages = _sort_by_score(deduped)
    topk = sorted_passages[:5]

    return {**state, "refined": topk}