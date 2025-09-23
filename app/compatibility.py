"""
기존 API와의 호환성 보장 모듈
- 기존 클라이언트가 새로운 멀티턴 기능 없이도 작동하도록 보장
- 점진적 마이그레이션 지원
"""

from typing import Dict, Any, Optional
from graph.context_manager import session_manager
from graph.cache_manager import cache_manager


class CompatibilityManager:
    """호환성 관리자"""
    
    def __init__(self):
        self.session_manager = session_manager
        self.cache_manager = cache_manager
    
    def ensure_backward_compatibility(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        응답의 하위 호환성 보장
        - 기존 필드들이 누락되지 않도록 보장
        - 새로운 필드는 선택적으로 포함
        """
        # 기본 필드 보장
        required_fields = {
            "question": "",
            "intent": "unknown",
            "needs_web": False,
            "plan": [],
            "passages": [],
            "refined": [],
            "draft_answer": {},
            "citations": [],
            "warnings": [],
            "trace": [],
            "web_results": []
        }
        
        # 누락된 필드 추가
        for field, default_value in required_fields.items():
            if field not in response:
                response[field] = default_value
        
        return response
    
    def create_legacy_response(self, new_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        새로운 응답을 기존 형식으로 변환
        """
        legacy_response = {
            "question": new_response.get("question", ""),
            "intent": new_response.get("intent", "unknown"),
            "needs_web": new_response.get("needs_web", False),
            "plan": new_response.get("plan", []),
            "passages": new_response.get("passages", []),
            "refined": new_response.get("refined", []),
            "draft_answer": new_response.get("draft_answer", {}),
            "citations": new_response.get("citations", []),
            "warnings": new_response.get("warnings", []),
            "trace": new_response.get("trace", []),
            "web_results": new_response.get("web_results", [])
        }
        
        return legacy_response
    
    def handle_redis_failure(self, fallback_func, *args, **kwargs):
        """
        Redis 실패 시 폴백 처리
        """
        try:
            return fallback_func(*args, **kwargs)
        except Exception as e:
            print(f"⚠️ Redis 실패, 폴백 모드: {e}")
            # Redis 없이도 기본 기능 작동
            return self._fallback_mode(*args, **kwargs)
    
    def _fallback_mode(self, *args, **kwargs):
        """
        Redis 없이 작동하는 폴백 모드
        """
        # 기본 응답 반환 (캐싱 없음)
        return {
            "question": kwargs.get("question", ""),
            "intent": "unknown",
            "needs_web": False,
            "plan": [],
            "passages": [],
            "refined": [],
            "draft_answer": {"conclusion": "시스템을 준비 중입니다. 잠시 후 다시 시도해주세요."},
            "citations": [],
            "warnings": ["Redis 연결이 없어 제한된 기능으로 작동합니다."],
            "trace": [],
            "web_results": [],
            "fallback_mode": True
        }


# 전역 인스턴스
compatibility_manager = CompatibilityManager()
