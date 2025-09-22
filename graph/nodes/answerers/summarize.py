from typing import Dict, Any

def summarize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    refined = state.get("refined", [])
    citations = state.get("citations", [])
    warnings = state.get("warnings", []) or []
    policy_disclaimer = state.get("policy_disclaimer")

    # 간단 요약 스텁: 상위 2개 청크를 이어 붙임
    summary = " / ".join([p.get("text", "") for p in refined[:2]]) or "요약할 컨텍스트가 충분하지 않습니다."

    caveats = ["요약은 현재 규칙 기반 스텁입니다. 실제 LLM 요약은 다음 커밋에서 연결됩니다."]
    caveats += warnings
    if policy_disclaimer:
        caveats.append(policy_disclaimer)

    answer = {
        "conclusion": f"핵심 요약: {summary[:120]}",
        "evidence": [p.get("text") for p in refined[:2]],
        "caveats": caveats,
        "quotes": citations,
    }
    return {**state, "draft_answer": answer}