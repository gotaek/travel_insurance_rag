from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
from graph.builder import build_graph
from graph.nodes.planner import planner_node

router = APIRouter()

# --- 요청 모델 ---
class AskRequest(BaseModel):
    question: str

# --- 상태 저장 (간단: 최근 trace만 보존) ---
_last_trace: Dict[str, Any] = {}

@router.post("/rag/ask")
def rag_ask(req: AskRequest):
    global _last_trace
    g = build_graph()
    state = {"question": req.question}
    out = g.invoke(state)
    # 직전 trace 저장
    _last_trace = out.get("trace", [])
    return out

@router.post("/rag/plan")
def rag_plan(req: AskRequest):
    """
    실제 실행하지 않고, intent/needs_web/plan만 미리 확인
    """
    state = {"question": req.question}
    out = planner_node(state)
    return {
        "question": req.question,
        "intent": out.get("intent"),
        "needs_web": out.get("needs_web"),
        "plan": out.get("plan"),
    }

@router.get("/rag/trace")
def rag_trace():
    """
    직전 실행된 RAG trace 로그 반환
    """
    global _last_trace
    return {"trace": _last_trace}