from typing import List, Dict, Any
import math

def _minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if math.isclose(vmin, vmax):
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]

def _merge_by_docpage(vec_hits: List[Dict[str, Any]], kw_hits: List[Dict[str, Any]], alpha: float = 0.6) -> List[Dict[str, Any]]:
    """
    벡터/키워드 결과를 문서ID+페이지 단위로 머지하고 정규화 점수 가중합.
    - alpha: 벡터 가중치(0~1). 0.6이면 벡터 60%, 키워드 40%.
    """
    def key(x): return (x.get("doc_id"), x.get("page"))

    # 정규화
    vec = vec_hits[:]
    kw = kw_hits[:]
    v_scores = _minmax_norm([h.get("score_vec", 0.0) for h in vec])
    k_scores = _minmax_norm([h.get("score_kw", 0.0) for h in kw])
    for h, s in zip(vec, v_scores):
        h["_norm_vec"] = s
    for h, s in zip(kw, k_scores):
        h["_norm_kw"] = s

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

    # 최종 스코어
    out = []
    for item in merged.values():
        nv = item.get("_norm_vec", 0.0)
        nk = item.get("_norm_kw", 0.0)
        item["score"] = alpha * nv + (1 - alpha) * nk
        out.append(item)

    out.sort(key=lambda x: x["score"], reverse=True)
    return out

def hybrid_search(
    query: str,
    vector_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    k: int = 5,
    alpha: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    이미 구해온 vector_results/keyword_results를 병합만 담당.
    - 이후 search 노드에서 vector_search/keyword_search 호출 후 이 함수로 합친다.
    """
    merged = _merge_by_docpage(vector_results, keyword_results, alpha=alpha)
    return merged[:k]