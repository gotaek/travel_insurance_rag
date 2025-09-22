from fastapi import APIRouter
from pydantic import BaseModel
from graph.builder import build_graph

router = APIRouter(prefix="/rag", tags=["rag"])

# LangGraph 빌드 (한 번만 실행)
_graph = build_graph()

class AskReq(BaseModel):
    question: str

@router.post("/ask")
def ask(payload: AskReq):
    """
    /rag/ask 엔드포인트
    - 입력: {"question": "..."}
    - 내부: LangGraph 실행 (state 전달)
    - 출력: 스텁 답변 JSON
    """
    state = {"question": payload.question}
    result = _graph.invoke(state)
    return result