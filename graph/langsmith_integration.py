"""
LangSmith 통합 모듈
LangGraph와 LangSmith를 연결하여 추적 및 모니터링 기능을 제공합니다.
"""

import os
from typing import Optional, List, Any, Dict
from langsmith import Client

# LangChain 버전 호환성을 위한 import
try:
    from langchain_core.callbacks import LangChainTracer
except ImportError:
    try:
        from langchain_core.tracers import LangChainTracer
    except ImportError:
        # 최신 버전에서는 다른 방식으로 import
        from langsmith import Client as LangSmithClient
        LangChainTracer = None


class LangSmithManager:
    """LangSmith 통합을 관리하는 클래스"""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self.tracer: Optional[LangChainTracer] = None
        self.project_name = os.getenv("LANGCHAIN_PROJECT", "travel-insurance-rag")
        self._initialize_client()
    
    def _initialize_client(self):
        """LangSmith 클라이언트 초기화"""
        try:
            # 환경 변수 확인
            api_key = os.getenv("LANGCHAIN_API_KEY")
            if not api_key or api_key == "your_langsmith_api_key_here":
                print("⚠️ LangSmith API 키가 설정되지 않았습니다. 추적이 비활성화됩니다.")
                return
            
            # LangSmith 클라이언트 생성
            self.client = Client(api_key=api_key)
            
            # 트레이서 생성 (LangChain 버전 호환성 고려)
            if LangChainTracer is not None:
                self.tracer = LangChainTracer(
                    project_name=self.project_name,
                    client=self.client
                )
            else:
                # LangChainTracer가 없는 경우 None으로 설정
                self.tracer = None
                print("⚠️ LangChainTracer를 사용할 수 없습니다. 기본 추적만 활성화됩니다.")
            
            print(f"✅ LangSmith 초기화 완료 - 프로젝트: {self.project_name}")
            
        except Exception as e:
            print(f"⚠️ LangSmith 초기화 실패: {str(e)}")
            self.client = None
            self.tracer = None
    
    def get_callbacks(self) -> List[Any]:
        """LangGraph에서 사용할 콜백 리스트 반환"""
        if self.tracer:
            return [self.tracer]
        return []
    
    def is_enabled(self) -> bool:
        """LangSmith 추적이 활성화되어 있는지 확인"""
        return self.tracer is not None
    
    def create_run(self, name: str, inputs: Dict[str, Any], **kwargs) -> Optional[Any]:
        """수동으로 실행 추적 생성"""
        if not self.client:
            return None
        
        try:
            return self.client.create_run(
                name=name,
                inputs=inputs,
                run_type="llm",  # run_type 파라미터 추가
                project_name=self.project_name,
                **kwargs
            )
        except Exception as e:
            print(f"⚠️ LangSmith 실행 생성 실패: {str(e)}")
            return None
    
    def update_run(self, run_id: str, outputs: Dict[str, Any] = None, error: str = None, **kwargs):
        """실행 추적 업데이트"""
        if not self.client:
            return
        
        try:
            self.client.update_run(
                run_id=run_id,
                outputs=outputs,
                error=error,
                **kwargs
            )
        except Exception as e:
            print(f"⚠️ LangSmith 실행 업데이트 실패: {str(e)}")

    def create_fallback_run(self, name: str, inputs: Dict[str, Any], outputs: Dict[str, Any], **kwargs) -> Optional[Any]:
        """Fallback 실행 추적 생성 (LLM 호출 없이)"""
        if not self.client:
            return None
        
        try:
            run = self.client.create_run(
                name=f"{name}_fallback",
                inputs=inputs,
                run_type="tool",  # fallback은 tool로 분류
                project_name=self.project_name,
                **kwargs
            )
            
            # 즉시 완료로 표시
            self.client.update_run(
                run_id=run.id,
                outputs=outputs,
                extra={
                    "metadata": {
                        "fallback": True,
                        "reason": "LLM 호출 실패로 인한 fallback 사용"
                    }
                }
            )
            
            return run
        except Exception as e:
            print(f"⚠️ LangSmith fallback 실행 생성 실패: {str(e)}")
            return None


# 전역 LangSmith 매니저 인스턴스
langsmith_manager = LangSmithManager()


def get_langsmith_callbacks() -> List[Any]:
    """LangGraph에서 사용할 LangSmith 콜백 반환"""
    return langsmith_manager.get_callbacks()


def is_langsmith_enabled() -> bool:
    """LangSmith 추적 활성화 상태 확인"""
    return langsmith_manager.is_enabled()


def create_langsmith_run(name: str, inputs: Dict[str, Any], **kwargs) -> Optional[Any]:
    """수동 실행 추적 생성"""
    return langsmith_manager.create_run(name, inputs, **kwargs)


def update_langsmith_run(run_id: str, outputs: Dict[str, Any] = None, error: str = None, **kwargs):
    """실행 추적 업데이트"""
    langsmith_manager.update_run(run_id, outputs, error, **kwargs)


def create_fallback_run(name: str, inputs: Dict[str, Any], outputs: Dict[str, Any], **kwargs) -> Optional[Any]:
    """Fallback 실행 추적 생성"""
    return langsmith_manager.create_fallback_run(name, inputs, outputs, **kwargs)