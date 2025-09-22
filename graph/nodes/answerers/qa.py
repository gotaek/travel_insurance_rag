from typing import Dict, Any

def qa_node(state: Dict[str, Any]) -> Dict[str, Any]:
    q = state["question"]
    refined = state.get("refined", [])
    citations = state.get("citations", [])
    policy_disclaimer = state.get("policy_disclaimer")
    warnings = state.get("warnings", [])

    caveats = ["검색 결과를 단순 정제한 상태. 검증/요약 단계는 아직 미적용."]
    caveats += warnings
    if policy_disclaimer:
        caveats.append(policy_disclaimer)

    answer = {
        "conclusion": f"질문 확인: '{q}'",
        "evidence": [p.get("text") for p in refined[:2]],
        "caveats": caveats,
        "quotes": citations,   # 원문 인용 = 문서ID/페이지/버전 요약
    }
    return {**state, "draft_answer": answer}