"""
LangSmith와 통합된 LLM 호출 래퍼
Google Gemini LLM 호출을 LangSmith로 추적합니다.
"""

import time
from typing import Dict, Any, Optional, List
import google.generativeai as genai
from app.deps import get_llm
from graph.langsmith_integration import create_langsmith_run, update_langsmith_run, is_langsmith_enabled


class LangSmithLLMWrapper:
    """LangSmith 추적이 포함된 LLM 래퍼 클래스"""
    
    def __init__(self):
        self.base_llm = get_llm()
        self.enabled = is_langsmith_enabled()
    
    def generate_content(self, prompt: str, request_options: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        LangSmith 추적이 포함된 LLM 콘텐츠 생성
        
        Args:
            prompt: 입력 프롬프트
            request_options: 요청 옵션 (타임아웃 등)
            **kwargs: 추가 키워드 인수
            
        Returns:
            LLM 응답 객체
        """
        run_id = None
        start_time = time.time()
        
        try:
            # LangSmith 실행 추적 시작
            if self.enabled:
                try:
                    run_id = create_langsmith_run(
                        name="gemini_generate_content",
                        inputs={
                            "prompt": prompt,
                            "request_options": request_options or {},
                            "model": "gemini-1.5-flash"
                        },
                        extra={
                            "metadata": {
                                "provider": "google",
                                "model": "gemini-1.5-flash",
                                "timestamp": start_time
                            }
                        }
                    )
                except Exception as e:
                    print(f"⚠️ LangSmith 실행 추적 시작 실패: {str(e)}")
                    run_id = None
            
            # 실제 LLM 호출
            response = self.base_llm.generate_content(prompt, request_options=request_options, **kwargs)
            
            # 성공 시 LangSmith 업데이트
            if self.enabled and run_id:
                try:
                    end_time = time.time()
                    update_langsmith_run(
                        run_id=run_id,
                        outputs={
                            "response_text": response.text if hasattr(response, 'text') else str(response),
                            "usage_metadata": getattr(response, 'usage_metadata', {}),
                            "finish_reason": getattr(response, 'finish_reason', 'stop')
                        },
                        extra={
                            "metadata": {
                                "latency_seconds": end_time - start_time,
                                "success": True
                            }
                        }
                    )
                except Exception as e:
                    print(f"⚠️ LangSmith 실행 업데이트 실패: {str(e)}")
            
            return response
            
        except Exception as e:
            # 에러 시 LangSmith 업데이트
            if self.enabled and run_id:
                try:
                    end_time = time.time()
                    update_langsmith_run(
                        run_id=run_id,
                        error=str(e),
                        extra={
                            "metadata": {
                                "latency_seconds": end_time - start_time,
                                "success": False,
                                "error_type": type(e).__name__
                            }
                        }
                    )
                except Exception as update_error:
                    print(f"⚠️ LangSmith 에러 업데이트 실패: {str(update_error)}")
            
            # 원본 에러 재발생
            raise e
    
    def count_tokens(self, text: str) -> int:
        """토큰 수 계산 (LangSmith 추적 포함)"""
        run_id = None
        start_time = time.time()
        
        try:
            # LangSmith 실행 추적 시작
            if self.enabled:
                try:
                    run_id = create_langsmith_run(
                        name="gemini_count_tokens",
                        inputs={"text": text},
                        extra={
                            "metadata": {
                                "provider": "google",
                                "model": "gemini-1.5-flash",
                                "timestamp": start_time
                            }
                        }
                    )
                except Exception as e:
                    print(f"⚠️ LangSmith 토큰 계산 추적 시작 실패: {str(e)}")
                    run_id = None
            
            # 실제 토큰 계산
            if hasattr(self.base_llm, 'count_tokens'):
                token_count = self.base_llm.count_tokens(text)
            else:
                # fallback: 간단한 추정
                token_count = len(text.split()) * 1.3  # 대략적인 추정
            
            # 성공 시 LangSmith 업데이트
            if self.enabled and run_id:
                try:
                    end_time = time.time()
                    update_langsmith_run(
                        run_id=run_id,
                        outputs={"token_count": token_count},
                        extra={
                            "metadata": {
                                "latency_seconds": end_time - start_time,
                                "success": True
                            }
                        }
                    )
                except Exception as e:
                    print(f"⚠️ LangSmith 토큰 계산 업데이트 실패: {str(e)}")
            
            return int(token_count)
            
        except Exception as e:
            # 에러 시 LangSmith 업데이트
            if self.enabled and run_id:
                try:
                    end_time = time.time()
                    update_langsmith_run(
                        run_id=run_id,
                        error=str(e),
                        extra={
                            "metadata": {
                                "latency_seconds": end_time - start_time,
                                "success": False,
                                "error_type": type(e).__name__
                            }
                        }
                    )
                except Exception as update_error:
                    print(f"⚠️ LangSmith 토큰 계산 에러 업데이트 실패: {str(update_error)}")
            
            # 원본 에러 재발생
            raise e


# 전역 LLM 래퍼 인스턴스
_langsmith_llm_wrapper = None


def get_langsmith_llm() -> LangSmithLLMWrapper:
    """LangSmith 추적이 포함된 LLM 인스턴스 반환"""
    global _langsmith_llm_wrapper
    if _langsmith_llm_wrapper is None:
        _langsmith_llm_wrapper = LangSmithLLMWrapper()
    return _langsmith_llm_wrapper


def get_llm_with_tracing():
    """기존 get_llm() 함수의 LangSmith 추적 버전"""
    return get_langsmith_llm()