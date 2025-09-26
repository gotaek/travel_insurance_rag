from fastapi import APIRouter
from app.deps import get_settings, get_llm
import os
import logging

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)

@router.get("/health")
@router.get("/healthz")
def healthz():
    s = get_settings()
    vector_exists = os.path.isdir(s.VECTOR_DIR)
    docs_exists = os.path.isdir(s.DOCUMENT_DIR)
    return {
        "ok": True,
        "env": s.ENV,
        "vector_dir": s.VECTOR_DIR,
        "documents_dir": s.DOCUMENT_DIR,
        "exists": {"vector": vector_exists, "documents": docs_exists},
    }

@router.get("/readyz")
def readyz():
    return {"ready": True}

@router.get("/api-status")
def api_status():
    """API 연결 상태 및 설정 확인"""
    try:
        s = get_settings()
        
        # Gemini API 키 확인
        api_key_status = "✅ 설정됨" if s.GEMINI_API_KEY else "❌ 미설정"
        model_status = s.GEMINI_MODEL
        
        # Gemini API 연결 테스트
        try:
            llm = get_llm()
            # 간단한 테스트 요청
            response = llm.generate_content("Hello", request_options={"timeout": 10})
            gemini_status = "✅ 연결됨"
            gemini_error = None
        except Exception as e:
            gemini_status = "❌ 연결 실패"
            gemini_error = str(e)
            logger.error(f"Gemini API 연결 실패: {e}")
        
        # Tavily API 키 확인
        tavily_status = "✅ 설정됨" if s.TAVILY_API_KEY else "❌ 미설정"
        
        return {
            "gemini": {
                "status": gemini_status,
                "model": model_status,
                "api_key": api_key_status,
                "error": gemini_error
            },
            "tavily": {
                "status": tavily_status,
                "api_key": tavily_status
            },
            "overall": "✅ 정상" if gemini_status == "✅ 연결됨" else "❌ 문제 있음"
        }
        
    except Exception as e:
        logger.error(f"API 상태 확인 실패: {e}")
        return {
            "error": str(e),
            "overall": "❌ 확인 실패"
        }