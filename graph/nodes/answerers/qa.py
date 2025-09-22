from typing import Dict, Any

def qa_node(state: Dict[str, Any]) -> Dict[str, Any]:
    q = state["question"]
    answer = {
        "conclusion": f"질문 확인: '{q}' → 아직 스텁 응답입니다.",
        "evidence": [],
        "caveats": ["검색/약관 연결 전이라 근거는 없습니다."],
        "quotes": [],
    }
    return {**state, "draft_answer": answer}