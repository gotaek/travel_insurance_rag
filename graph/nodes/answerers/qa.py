from typing import Dict, Any

def qa_node(state: Dict[str, Any]) -> Dict[str, Any]:
    q = state["question"]
    refined = state.get("refined", [])
    answer = {
        "conclusion": f"질문 확인: '{q}'",
        "evidence": [p.get("text") for p in refined[:2]],
        "caveats": ["검색 결과를 단순 정제한 상태. 검증/요약 단계는 아직 미적용."],
        "quotes": refined,
    }
    return {**state, "draft_answer": answer}