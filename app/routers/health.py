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
        
        # 사용 가능한 모델 목록 확인
        try:
            from app.deps import get_available_models
            available_models = get_available_models()
        except Exception as e:
            available_models = []
            logger.error(f"모델 목록 확인 실패: {e}")
        
        # Gemini API 연결 테스트
        try:
            llm = get_llm()
            # 간단한 테스트 요청
            response = llm.generate_content("Hello", request_options={"timeout": 10})
            gemini_status = "✅ 연결됨"
            gemini_error = None
        except Exception as e:
            gemini_status = "❌ 연결 실패"
            error_str = str(e).lower()
            
            # 오류 유형별 상세 메시지
            if "api_key" in error_str or "authentication" in error_str:
                gemini_error = "API 키 인증 실패 - .env 파일의 GEMINI_API_KEY를 확인해주세요"
            elif "quota" in error_str or "limit" in error_str:
                gemini_error = "API 할당량 초과 - 잠시 후 다시 시도해주세요"
            elif "permission" in error_str or "access" in error_str:
                gemini_error = "모델 접근 권한 없음 - API 키에 해당 모델 접근 권한이 있는지 확인해주세요"
            elif "timeout" in error_str or "connection" in error_str:
                gemini_error = "네트워크 연결 실패 - 인터넷 연결을 확인해주세요"
            else:
                gemini_error = f"연결 실패: {str(e)}"
            
            logger.error(f"Gemini API 연결 실패: {e}")
        
        # Tavily API 키 확인
        tavily_status = "✅ 설정됨" if s.TAVILY_API_KEY else "❌ 미설정"
        
        return {
            "gemini": {
                "status": gemini_status,
                "model": model_status,
                "api_key": api_key_status,
                "error": gemini_error,
                "available_models": available_models
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