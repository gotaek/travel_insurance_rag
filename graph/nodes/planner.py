from typing import Dict, Any

def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    간단한 intent 추정기 (Stub)
    - 모든 질문을 intent="qa"로 처리
    - plan: ["planner", "answer"]
    """
    q = state.get("question", "")
    intent = "qa"
    plan = ["planner", "answer"]

    return {**state, "intent": intent, "needs_web": False, "plan": plan}