from typing import Dict, Any
from collections import defaultdict

def compare_node(state: Dict[str, Any]) -> Dict[str, Any]:
    refined = state.get("refined", [])
    citations = state.get("citations", [])
    warnings = state.get("warnings", []) or []
    policy_disclaimer = state.get("policy_disclaimer")

    # 간단 비교 스텁: insurer 별로 묶어서 항목 수만 보여줌
    buckets = defaultdict(list)
    for p in refined:
        key = p.get("insurer") or "UNKNOWN_INSURER"
        buckets[key].append(p)

    lines = []
    for insurer, items in buckets.items():
        lines.append(f"- {insurer}: {len(items)}개 항목")

    if not lines:
        lines = ["비교할 컨텍스트가 부족합니다. 서로 다른 보험사/상품의 문맥이 필요합니다."]

    caveats = ["비교는 현재 스텁입니다. 실제 표/정렬/차이점 하이라이트는 이후 단계에서 구현됩니다."]
    caveats += warnings
    if policy_disclaimer:
        caveats.append(policy_disclaimer)

    answer = {
        "conclusion": "보험사별 주요 항목 수 비교(스텁)",
        "evidence": lines[:5],
        "caveats": caveats,
        "quotes": citations,
    }
    return {**state, "draft_answer": answer}