"""
LangSmith와 통합된 LLM 호출 래퍼
Google Gemini LLM 호출을 LangSmith로 추적합니다.
"""

import time
import json
import re
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
                    print(f"✅ LangSmith 에러 업데이트 완료: {type(e).__name__}")
                except Exception as update_error:
                    print(f"⚠️ LangSmith 에러 업데이트 실패: {str(update_error)}")
            else:
                print(f"⚠️ LangSmith 실행 ID가 없어 에러 업데이트를 건너뜁니다: {str(e)}")
            
            # API 할당량 초과 에러인지 확인
            error_str = str(e).lower()
            if "quota" in error_str or "limit" in error_str or "429" in error_str:
                print("🚫 Gemini API 할당량이 초과되었습니다. 잠시 후 다시 시도해주세요.")
                print(f"상세 에러: {str(e)}")
            
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
    
    def with_structured_output(self, response_schema: Any) -> 'LangSmithLLMWrapper':
        """
        Structured output을 위한 래퍼 반환
        Gemini의 경우 기본적으로 structured output을 지원하지 않으므로
        프롬프트에 JSON 스키마를 추가하여 처리
        """
        return StructuredOutputWrapper(self, response_schema)


class StructuredOutputWrapper:
    """Structured output을 위한 래퍼 클래스"""
    
    def __init__(self, llm_wrapper: LangSmithLLMWrapper, response_schema: Any):
        self.llm_wrapper = llm_wrapper
        self.response_schema = response_schema
    
    def generate_content(self, prompt: str, request_options: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        Structured output을 위한 프롬프트 수정 및 응답 파싱
        """
        # 스키마 정보를 프롬프트에 추가
        structured_prompt = self._add_schema_to_prompt(prompt)
        
        # 기본 LLM 호출
        response = self.llm_wrapper.generate_content(
            structured_prompt, 
            request_options=request_options, 
            **kwargs
        )
        
        # 응답을 structured format으로 파싱
        return self._parse_structured_response(response)
    
    def _add_schema_to_prompt(self, prompt: str) -> str:
        """프롬프트에 JSON 스키마 정보 추가"""
        # 스키마 타입에 따라 다른 JSON 형식 제공
        schema_name = getattr(self.response_schema, '__name__', '')
        
        if 'Recommend' in schema_name:
            schema_instruction = """

**중요**: 응답은 반드시 다음 JSON 형식으로 작성해주세요:

```json
{
    "conclusion": "답변의 핵심 내용",
    "evidence": ["증거 1", "증거 2"],
    "caveats": ["주의사항 1", "주의사항 2"],
    "quotes": [
        {
            "text": "인용된 텍스트",
            "source": "출처 정보"
        }
    ],
    "recommendations": ["추천 1", "추천 2"],
    "web_info": {
        "sources": ["웹 소스 1", "웹 소스 2"]
    }
}
```

위 JSON 형식을 정확히 따라 응답해주세요."""
        elif 'Compare' in schema_name:
            schema_instruction = """

**중요**: 응답은 반드시 다음 JSON 형식으로 작성해주세요:

```json
{
    "conclusion": "답변의 핵심 내용",
    "evidence": ["증거 1", "증거 2"],
    "caveats": ["주의사항 1", "주의사항 2"],
    "quotes": [
        {
            "text": "인용된 텍스트",
            "source": "출처 정보"
        }
    ],
    "comparison_table": {
        "headers": ["항목", "보험사 A", "보험사 B"],
        "rows": [
            ["항목1", "내용1", "내용2"],
            ["항목2", "내용3", "내용4"]
        ]
    }
}
```

위 JSON 형식을 정확히 따라 응답해주세요."""
        elif 'Quality' in schema_name:
            schema_instruction = """

**중요**: 응답은 반드시 다음 JSON 형식으로 작성해주세요:

```json
{
    "score": 0.8,
    "feedback": "품질 평가 설명",
    "needs_replan": false,
    "replan_query": ""
}
```

위 JSON 형식을 정확히 따라 응답해주세요."""
        elif 'Planner' in schema_name:
            schema_instruction = """

**중요**: 응답은 반드시 다음 JSON 형식으로 작성해주세요:

```json
{
    "intent": "qa",
    "needs_web": false,
    "reasoning": "분류 근거 설명"
}
```

위 JSON 형식을 정확히 따라 응답해주세요."""
        elif 'Replan' in schema_name:
            schema_instruction = """

**중요**: 응답은 반드시 다음 JSON 형식으로 작성해주세요:

```json
{
    "new_question": "개선된 검색 질문",
    "needs_web": false,
    "reasoning": "재검색 질문 개선 근거"
}
```

위 JSON 형식을 정확히 따라 응답해주세요."""
        else:
            # 기본 QA/Summarize 형식
            schema_instruction = """

**중요**: 응답은 반드시 다음 JSON 형식으로 작성해주세요:

```json
{
    "conclusion": "답변의 핵심 내용",
    "evidence": ["증거 1", "증거 2"],
    "caveats": ["주의사항 1", "주의사항 2"],
    "quotes": [
        {
            "text": "인용된 텍스트",
            "source": "출처 정보"
        }
    ]
}
```

위 JSON 형식을 정확히 따라 응답해주세요."""
        
        return prompt + schema_instruction
    
    def _parse_structured_response(self, response: Any) -> Any:
        """LLM 응답을 structured format으로 파싱"""
        try:
            # 응답 텍스트 추출
            if hasattr(response, 'text'):
                response_text = response.text
            else:
                response_text = str(response)
            
            # JSON 블록 추출
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # JSON 블록이 없으면 전체 텍스트에서 JSON 찾기
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("JSON 형식을 찾을 수 없습니다")
            
            # JSON 파싱
            parsed_data = json.loads(json_str)
            
            # 스키마에 맞는 객체 생성 (간단한 네임스페이스 객체)
            class StructuredResponse:
                def __init__(self, data):
                    for key, value in data.items():
                        setattr(self, key, value)
            
            return StructuredResponse(parsed_data)
            
        except Exception as e:
            # 파싱 실패 시 기본 구조 반환
            schema_name = getattr(self.response_schema, '__name__', '')
            
            if 'Recommend' in schema_name:
                class FallbackResponse:
                    def __init__(self):
                        self.conclusion = "답변을 생성하는 중 오류가 발생했습니다."
                        self.evidence = ["응답 파싱 오류"]
                        self.caveats = ["추가 확인이 필요합니다."]
                        self.quotes = []
                        self.recommendations = []
                        self.web_info = {}
            elif 'Compare' in schema_name:
                class FallbackResponse:
                    def __init__(self):
                        self.conclusion = "답변을 생성하는 중 오류가 발생했습니다."
                        self.evidence = ["응답 파싱 오류"]
                        self.caveats = ["추가 확인이 필요합니다."]
                        self.quotes = []
                        self.comparison_table = {
                            "headers": ["항목", "비교 결과"],
                            "rows": [["오류", "파싱 실패"]]
                        }
            elif 'Quality' in schema_name:
                class FallbackResponse:
                    def __init__(self):
                        self.score = 0.5
                        self.feedback = "품질 평가 중 오류가 발생했습니다."
                        self.needs_replan = True
                        self.replan_query = ""
            elif 'Planner' in schema_name:
                class FallbackResponse:
                    def __init__(self):
                        self.intent = "qa"
                        self.needs_web = False
                        self.reasoning = "분류 중 오류가 발생했습니다."
            elif 'Replan' in schema_name:
                class FallbackResponse:
                    def __init__(self):
                        self.new_question = "재검색 질문 생성 중 오류가 발생했습니다."
                        self.needs_web = False
                        self.reasoning = "재검색 질문 생성 중 오류가 발생했습니다."
            else:
                class FallbackResponse:
                    def __init__(self):
                        self.conclusion = "답변을 생성하는 중 오류가 발생했습니다."
                        self.evidence = ["응답 파싱 오류"]
                        self.caveats = ["추가 확인이 필요합니다."]
                        self.quotes = []
            
            return FallbackResponse()


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