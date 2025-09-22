from typing import Dict, Any, List, Set

def _to_set(text: str) -> Set[str]:
    return set(text.lower().split())

def simple_faithfulness(state: Dict[str, Any]) -> float:
    """
    초간단 신뢰성 지표(0~1): evidence와 refined 텍스트 간 단어 교집합 비율의 상한치.
    - 정식 entailment가 아님. 엔지니어링 스모크용입니다.
    """
    answer = state.get("draft_answer", {})
    ev: List[str] = answer.get("evidence") or []
    ref = " ".join([p.get("text","") for p in (state.get("refined") or [])])
    if not ev or not ref:
        return 0.0
    ref_set = _to_set(ref)
    if not ref_set:
        return 0.0
    scores = []
    for e in ev:
        es = _to_set(e or "")
        inter = len(es.intersection(ref_set))
        denom = max(1, len(es))
        scores.append(inter/denom)
    return max(scores) if scores else 0.0