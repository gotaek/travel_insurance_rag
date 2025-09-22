from typing import Dict, Any

def qa_node(state: Dict[str, Any]) -> Dict[str, Any]:
    q = state["question"]
    passages = state.get("passages", [])
    answer = {
        "conclusion": f"질문 확인: '{q}'",
        "evidence": [p.get("text") for p in passages[:2]],  # 상위 2개 텍스트만 보여줌
        "caveats": ["실제 요약/검증 단계 전. 검색된 raw chunk만 노출."],
        "quotes": passages,
    }
    return {**state, "draft_answer": answer}