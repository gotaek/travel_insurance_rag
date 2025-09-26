from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Optional
import logging
import google.generativeai as genai
import redis

# 로깅 설정
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    # 기존 필드
    ENV: str = "dev"
    REDIS_URL: Optional[str] = "redis://localhost:6379"
    VECTOR_DIR: str = "data/vector_db"
    DOCUMENT_DIR: str = "data/documents"

    # ✅ Gemini(.env) 필드
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"
    
    # ✅ Tavily API 설정
    TAVILY_API_KEY: str = ""
    
    # ✅ Redis 설정
    REDIS_SESSION_TTL: int = 3600  # 세션 만료 시간 (초)
    REDIS_CACHE_TTL: int = 1800    # 캐시 만료 시간 (초)

    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

class StructuredOutputWrapper:
    """Gemini의 structured output을 위한 래퍼 클래스"""
    
    def __init__(self, llm, response_schema):
        self.llm = llm
        self.response_schema = response_schema
    
    def generate_content(self, prompt: str, **kwargs):
        """structured output을 위한 프롬프트 수정 및 응답 파싱"""
        import json
        import re
        
        # Pydantic 모델에서 JSON 스키마 생성
        schema = self.response_schema.model_json_schema()
        
        # structured output을 위한 프롬프트 수정
        structured_prompt = f"""
{prompt}

**중요**: 응답은 반드시 다음 JSON 형식으로만 제공해주세요:

{json.dumps(schema, ensure_ascii=False, indent=2)}

응답 형식:
```json
{{
  "필드명": "값"
}}
```
"""
        
        try:
            # LLM 호출
            response = self.llm.generate_content(structured_prompt, **kwargs)
            response_text = response.text
            
            # JSON 부분 추출
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            json_match = re.search(json_pattern, response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                # JSON 블록이 없는 경우 일반 JSON 패턴 시도
                json_pattern = r'\{.*\}'
                json_match = re.search(json_pattern, response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                else:
                    raise ValueError("JSON 형식의 응답을 찾을 수 없습니다.")
            
            # JSON 파싱
            parsed_data = json.loads(json_str)
            
            # Pydantic 모델로 검증
            validated_response = self.response_schema(**parsed_data)
            return validated_response
            
        except Exception as e:
            error_str = str(e).lower()
            print(f"❌ Structured output 파싱 실패: {str(e)}")
            
            # 특정 오류 타입에 따른 처리
            if "404" in error_str or "publisher" in error_str or "model" in error_str:
                print("🔧 모델 이름 오류 감지 - 기본 모델로 재시도")
                # 모델 이름 오류인 경우 기본값 반환
                fallback_data = {}
                for field_name, field_info in self.response_schema.model_fields.items():
                    field_type = field_info.annotation
                    if field_type == str:
                        fallback_data[field_name] = "모델 설정 오류로 인한 파싱 실패"
                    elif field_type == list:
                        fallback_data[field_name] = []
                    elif field_type == dict:
                        fallback_data[field_name] = {}
                    else:
                        fallback_data[field_name] = f"모델 오류: {str(e)[:30]}"
                return self.response_schema(**fallback_data)
            
            elif "429" in error_str or "quota" in error_str or "exceeded" in error_str:
                print("🔧 API 할당량 초과 감지")
                # 할당량 초과인 경우
                fallback_data = {}
                for field_name, field_info in self.response_schema.model_fields.items():
                    field_type = field_info.annotation
                    if field_type == str:
                        fallback_data[field_name] = "API 할당량 초과로 인한 서비스 제한"
                    elif field_type == list:
                        fallback_data[field_name] = []
                    elif field_type == dict:
                        fallback_data[field_name] = {}
                    else:
                        fallback_data[field_name] = f"할당량 초과: {str(e)[:30]}"
                return self.response_schema(**fallback_data)
            
            elif "validation" in error_str or "pydantic" in error_str:
                print("🔧 Pydantic validation 오류 감지 - 기본값으로 처리")
                # Pydantic validation 오류인 경우 기본값 사용
                try:
                    return self.response_schema()
                except Exception:
                    # 기본값 생성도 실패하면 수동으로 안전한 값 설정
                    fallback_data = {}
                    for field_name, field_info in self.response_schema.model_fields.items():
                        field_type = field_info.annotation
                        if field_type == str:
                            fallback_data[field_name] = "Validation 오류로 인한 기본값"
                        elif field_type == list:
                            fallback_data[field_name] = []
                        elif field_type == dict:
                            fallback_data[field_name] = {}
                        else:
                            fallback_data[field_name] = f"Validation 오류: {str(e)[:30]}"
                    return self.response_schema(**fallback_data)
            
            # 기타 오류에 대한 일반적인 fallback 처리
            try:
                # 먼저 빈 딕셔너리로 모델 생성 시도 (기본값 사용)
                return self.response_schema()
            except Exception as model_error:
                print(f"❌ 모델 기본값 생성도 실패: {str(model_error)}")
                
                # 수동으로 타입에 맞는 기본값 설정
                fallback_data = {}
                for field_name, field_info in self.response_schema.model_fields.items():
                    field_type = field_info.annotation
                    
                    # 타입에 따른 안전한 기본값 설정
                    if field_type == float:
                        fallback_data[field_name] = 0.0
                    elif field_type == bool:
                        fallback_data[field_name] = False
                    elif field_type == int:
                        fallback_data[field_name] = 0
                    elif field_type == str:
                        fallback_data[field_name] = f"파싱 실패: {str(e)[:50]}"
                    elif field_type == list:
                        fallback_data[field_name] = []
                    elif field_type == dict:
                        fallback_data[field_name] = {}
                    elif hasattr(field_type, '__origin__'):
                        # 제네릭 타입 처리
                        if field_type.__origin__ is list:
                            fallback_data[field_name] = []
                        elif field_type.__origin__ is dict:
                            fallback_data[field_name] = {}
                        else:
                            fallback_data[field_name] = f"파싱 실패: {str(e)[:50]}"
                    else:
                        # 기타 타입
                        fallback_data[field_name] = f"파싱 실패: {str(e)[:50]}"
                
                print(f"🔄 수동 Fallback 데이터 생성: {fallback_data}")
                return self.response_schema(**fallback_data)

class LangSmithLLMWrapper:
    """LangSmith 추적을 위한 LLM 래퍼"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def with_structured_output(self, response_schema):
        """Structured output을 위한 래퍼 반환"""
        return StructuredOutputWrapper(self.llm, response_schema)
    
    def generate_content(self, prompt: str, **kwargs):
        """일반 generate_content 호출"""
        return self.llm.generate_content(prompt, **kwargs)

@lru_cache()
def get_llm():
    """
    Gemini LLM 클라이언트 (환경변수는 .env에서 로드)
    """
    s = get_settings()
    if not s.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set in .env")
    
    # API 키 설정
    genai.configure(api_key=s.GEMINI_API_KEY)
    
    # 모델 이름 검증 및 수정
    model_name = s.GEMINI_MODEL.strip()
    
    # 잘못된 모델 이름 패턴 제거 및 정리
    if model_name.startswith('projects/'):
        print(f"⚠️ 잘못된 모델 이름 감지: {model_name}")
        model_name = "gemini-1.5-flash"  # 기본값으로 리셋
    
    # 추가적인 잘못된 패턴들 정리
    if 'generativelanguage' in model_name.lower():
        print(f"⚠️ 잘못된 모델 이름 감지: {model_name}")
        model_name = "gemini-1.5-flash"
    
    # 빈 문자열이나 None 체크
    if not model_name or model_name.strip() == "":
        model_name = "gemini-1.5-flash"
    
    # models/ 접두사 추가
    if not model_name.startswith('models/'):
        model_name = f"models/{model_name}"
    
    print(f"🔧 사용할 Gemini 모델: {model_name}")
    
    # 여러 모델명으로 시도
    model_candidates = [
        model_name,
        "models/gemini-1.5-flash",
        "gemini-1.5-flash",
    ]
    
    for candidate in model_candidates:
        try:
            print(f"🔄 모델 시도: {candidate}")
            base_llm = genai.GenerativeModel(candidate)
            print(f"✅ 모델 생성 성공: {candidate}")
            return LangSmithLLMWrapper(base_llm)
        except Exception as e:
            print(f"❌ 모델 {candidate} 실패: {str(e)}")
            continue
    
    # 모든 시도 실패
    raise RuntimeError(f"모든 Gemini 모델 시도 실패. 마지막 오류: {str(e)}")

@lru_cache()
def get_redis_client():
    """
    Redis 클라이언트 인스턴스 생성
    """
    s = get_settings()
    if not s.REDIS_URL:
        raise RuntimeError("REDIS_URL is not set in .env")
    
    try:
        client = redis.from_url(
            s.REDIS_URL,
            decode_responses=False,  # 바이너리 데이터 지원
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True
        )
        # 연결 테스트
        client.ping()
        return client
    except Exception as e:
        logger.error(f"Redis 연결 실패: {e}")
        return None