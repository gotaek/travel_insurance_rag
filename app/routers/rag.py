from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
from graph.builder import build_graph
from graph.nodes.planner import planner_node
from graph.context import ConversationContext, ConversationTurn, generate_turn_id, generate_session_id
from graph.context_manager import session_manager
from graph.cache_manager import cache_manager
from retriever.embeddings import get_embedding_model_info
from app.compatibility import compatibility_manager

router = APIRouter()

# --- 요청 모델 ---
class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    include_context: bool = True  # 이전 대화 컨텍스트 포함 여부

class MultiTurnAskRequest(BaseModel):
    question: str
    session_id: str
    user_id: Optional[str] = None
    include_context: bool = True

# --- 상태 저장 (간단: 최근 trace만 보존) ---
_last_trace: Dict[str, Any] = {}

@router.post("/rag/ask")
def rag_ask(req: AskRequest):
    """
    RAG 질문 처리 (멀티턴 대화 지원)
    """
    global _last_trace
    
    try:
        # 세션 ID 생성 또는 사용
        session_id = req.session_id or generate_session_id(req.user_id)
        
        # 대화 컨텍스트 로드
        conversation_context = None
        if req.include_context:
            conversation_context = session_manager.load_context(session_id)
            if not conversation_context:
                conversation_context = session_manager.create_new_context(
                    session_id, req.user_id
                )
                # 새로 생성된 컨텍스트 저장
                session_manager.save_context(conversation_context)
        
        # 그래프 실행
        g = build_graph()
        state = {
            "question": req.question,
            "session_id": session_id,
            "conversation_context": conversation_context
        }
        
        out = g.invoke(state)
        
        # 대화 턴 생성 및 저장
        if conversation_context and out.get("draft_answer"):
            turn = ConversationTurn(
                turn_id=generate_turn_id(req.question, session_id),
                question=req.question,
                answer=out.get("draft_answer", {}),
                intent=out.get("intent", "unknown"),
                passages_used=out.get("passages", []),
                tokens_used=out.get("trace", [{}])[-1].get("tokens", 0) if out.get("trace") else 0
            )
            
            # 컨텍스트 업데이트 및 저장
            updated_context = session_manager.update_context_with_turn(conversation_context, turn)
            session_manager.save_context(updated_context)
            
            # 응답에 컨텍스트 정보 추가
            out["conversation_context"] = updated_context.get_context_summary()
        
        # 직전 trace 저장 (기존 호환성)
        _last_trace = out.get("trace", [])
        
        # 호환성 보장
        out = compatibility_manager.ensure_backward_compatibility(out)
        
        return out
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG 처리 중 오류: {str(e)}")

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

@router.post("/rag/multiturn/ask")
def rag_multiturn_ask(req: MultiTurnAskRequest):
    """
    멀티턴 대화 전용 질문 처리
    """
    try:
        # 대화 컨텍스트 로드
        conversation_context = session_manager.load_context(req.session_id)
        if not conversation_context:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        # 그래프 실행
        g = build_graph()
        state = {
            "question": req.question,
            "session_id": req.session_id,
            "conversation_context": conversation_context
        }
        
        out = g.invoke(state)
        
        # 대화 턴 생성 및 저장
        if out.get("draft_answer"):
            turn = ConversationTurn(
                turn_id=generate_turn_id(req.question, req.session_id),
                question=req.question,
                answer=out.get("draft_answer", {}),
                intent=out.get("intent", "unknown"),
                passages_used=out.get("passages", []),
                tokens_used=out.get("trace", [{}])[-1].get("tokens", 0) if out.get("trace") else 0
            )
            
            # 컨텍스트 업데이트 및 저장
            updated_context = session_manager.update_context_with_turn(conversation_context, turn)
            session_manager.save_context(updated_context)
            
            # 응답에 컨텍스트 정보 추가
            out["conversation_context"] = updated_context.get_context_summary()
        
        return out
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"멀티턴 처리 중 오류: {str(e)}")

@router.post("/rag/session/create")
def create_session(user_id: Optional[str] = None):
    """
    새로운 대화 세션 생성
    """
    try:
        session_id = generate_session_id(user_id)
        context = session_manager.create_new_context(session_id, user_id)
        session_manager.save_context(context)
        
        return {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": context.created_at.isoformat(),
            "status": "created"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"세션 생성 중 오류: {str(e)}")

@router.get("/rag/session/{session_id}")
def get_session_info(session_id: str):
    """
    세션 정보 조회
    """
    try:
        context = session_manager.load_context(session_id)
        if not context:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        return {
            "session_id": session_id,
            "context": context.get_context_summary(),
            "recent_turns": [turn.to_dict() for turn in context.get_recent_turns(5)]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"세션 조회 중 오류: {str(e)}")

@router.delete("/rag/session/{session_id}")
def delete_session(session_id: str):
    """
    세션 삭제
    """
    try:
        # Redis에서 세션 삭제
        if session_manager.redis_client:
            key = session_manager.get_session_key(session_id)
            deleted = session_manager.redis_client.delete(key)
            
            return {
                "session_id": session_id,
                "deleted": bool(deleted),
                "status": "deleted" if deleted else "not_found"
            }
        else:
            raise HTTPException(status_code=503, detail="Redis 연결이 없습니다")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"세션 삭제 중 오류: {str(e)}")

@router.get("/rag/cache/stats")
def get_cache_stats():
    """
    캐시 통계 조회
    """
    try:
        stats = cache_manager.get_cache_stats()
        embedding_info = get_embedding_model_info()
        
        return {
            "cache_stats": stats,
            "embedding_model": embedding_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 조회 중 오류: {str(e)}")

@router.post("/rag/cache/clear")
def clear_cache():
    """
    캐시 초기화
    """
    try:
        cleared = {
            "embeddings": cache_manager.invalidate_cache("embeddings:*"),
            "search": cache_manager.invalidate_cache("search:*"),
            "llm_response": cache_manager.invalidate_cache("llm_response:*")
        }
        
        return {
            "cleared": cleared,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"캐시 초기화 중 오류: {str(e)}")