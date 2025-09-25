from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import traceback
from graph.builder import build_graph
from graph.nodes.planner import planner_node
from graph.context import ConversationContext, ConversationTurn, generate_turn_id, generate_session_id
from graph.context_manager import session_manager
from graph.cache_manager import cache_manager
from retriever.embeddings import get_embedding_model_info
from app.compatibility import compatibility_manager

# 로깅 설정
logger = logging.getLogger(__name__)

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
        logger.info(f"RAG 요청 시작: {req.question[:50]}...")
        
        # 세션 ID 생성 또는 사용
        session_id = req.session_id or generate_session_id(req.user_id)
        logger.info(f"세션 ID: {session_id}")
        
        # 대화 컨텍스트 로드
        conversation_context = None
        if req.include_context:
            try:
                conversation_context = session_manager.load_context(session_id)
                if not conversation_context:
                    conversation_context = session_manager.create_new_context(
                        session_id, req.user_id
                    )
                    # 새로 생성된 컨텍스트 저장
                    session_manager.save_context(conversation_context)
                logger.info("대화 컨텍스트 로드 완료")
            except Exception as e:
                logger.error(f"대화 컨텍스트 로드 실패: {str(e)}")
                # 컨텍스트 로드 실패해도 계속 진행
        
        # 그래프 실행
        logger.info("LangGraph 실행 시작")
        g = build_graph()
        state = {
            "question": req.question,
            "session_id": session_id,
            "conversation_context": conversation_context,
            # 무한루프 방지를 위한 초기값 설정
            "replan_count": 0,
            "max_replan_attempts": 3,
            "needs_replan": False
        }
        
        # 재귀 제한 설정 (무한루프 방지)
        config = {"recursion_limit": 25}
        out = g.invoke(state, config=config)
        logger.info("LangGraph 실행 완료")
        
        # 대화 턴 생성 및 저장
        if conversation_context and out.get("draft_answer"):
            try:
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
                logger.info("대화 턴 저장 완료")
            except Exception as e:
                logger.error(f"대화 턴 저장 실패: {str(e)}")
                # 턴 저장 실패해도 계속 진행
        
        # 직전 trace 저장 (기존 호환성)
        _last_trace = out.get("trace", [])
        
        # 호환성 보장
        out = compatibility_manager.ensure_backward_compatibility(out)
        
        logger.info("RAG 요청 처리 완료")
        return out
        
    except Exception as e:
        error_msg = f"RAG 처리 중 오류: {str(e)}"
        logger.error(error_msg)
        logger.error(f"상세 에러: {traceback.format_exc()}")
        print(f"❌ RAG 에러: {error_msg}")
        print(f"상세: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@router.post("/rag/plan")
def rag_plan(req: AskRequest):
    """
    RAG 계획 생성 (실행 없이 계획만)
    """
    try:
        logger.info(f"RAG 계획 요청: {req.question[:50]}...")
        
        # 플래너 노드만 실행
        state = {
            "question": req.question,
            "session_id": req.session_id or generate_session_id(req.user_id)
        }
        
        result = planner_node(state)
        
        return {
            "intent": result.get("intent"),
            "needs_web": result.get("needs_web"),
            "plan": result.get("plan", ""),
            "reasoning": result.get("reasoning", "")
        }
        
    except Exception as e:
        error_msg = f"RAG 계획 생성 중 오류: {str(e)}"
        logger.error(error_msg)
        logger.error(f"상세 에러: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/rag/trace")
def rag_trace():
    """
    최근 실행 trace 조회
    """
    global _last_trace
    return {"trace": _last_trace}

@router.post("/rag/multiturn/ask")
def rag_multiturn_ask(req: MultiTurnAskRequest):
    """
    멀티턴 대화 전용 질문 처리
    """
    try:
        logger.info(f"멀티턴 요청: {req.question[:50]}...")
        
        # 대화 컨텍스트 로드
        conversation_context = session_manager.load_context(req.session_id)
        if not conversation_context:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        # 그래프 실행
        g = build_graph()
        state = {
            "question": req.question,
            "session_id": req.session_id,
            "conversation_context": conversation_context,
            # 무한루프 방지를 위한 초기값 설정
            "replan_count": 0,
            "max_replan_attempts": 3,
            "needs_replan": False
        }
        
        # 재귀 제한 설정 (무한루프 방지)
        config = {"recursion_limit": 25}
        out = g.invoke(state, config=config)
        
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
        error_msg = f"멀티턴 처리 중 오류: {str(e)}"
        logger.error(error_msg)
        logger.error(f"상세 에러: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

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
        error_msg = f"세션 생성 중 오류: {str(e)}"
        logger.error(error_msg)
        logger.error(f"상세 에러: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

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
        error_msg = f"세션 조회 중 오류: {str(e)}"
        logger.error(error_msg)
        logger.error(f"상세 에러: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@router.delete("/rag/session/{session_id}")
def delete_session(session_id: str):
    """
    세션 삭제
    """
    try:
        success = session_manager.delete_context(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        return {"status": "deleted", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"세션 삭제 중 오류: {str(e)}"
        logger.error(error_msg)
        logger.error(f"상세 에러: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/rag/cache/stats")
def get_cache_stats():
    """
    캐시 통계 조회
    """
    try:
        return cache_manager.get_stats()
    except Exception as e:
        error_msg = f"캐시 통계 조회 중 오류: {str(e)}"
        logger.error(error_msg)
        logger.error(f"상세 에러: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/rag/embedding/info")
def get_embedding_info():
    """
    임베딩 모델 정보 조회
    """
    try:
        return get_embedding_model_info()
    except Exception as e:
        error_msg = f"임베딩 정보 조회 중 오류: {str(e)}"
        logger.error(error_msg)
        logger.error(f"상세 에러: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)