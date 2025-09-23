from typing import List, Dict, Any

def _collect_doc_ids(items: List[Dict[str, Any]]) -> List[str]:
    out = []
    for x in items or []:
        did = x.get("doc_id")
        if did:
            out.append(str(did))
    return out

def recall_at_k(state: Dict[str, Any], gold_doc_ids: List[str], k: int = 5) -> float:
    """
    검색/정제된 상위 k 컨텍스트에 gold_doc_ids가 포함되어 있는지 평가.
    - 우선 refined가 있으면 refined 사용, 없으면 passages 사용
    """
    cand = state.get("refined") or state.get("passages") or []
    cand = cand[:k]
    cand_ids = set(_collect_doc_ids(cand))
    gold = set([str(x) for x in gold_doc_ids or []])
    
    
    if not gold:
        return 0.0
    hit = len(cand_ids.intersection(gold)) > 0
    return 1.0 if hit else 0.0