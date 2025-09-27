# deps.py — Gemini 503 & Structured Output hotfixes
# -------------------------------------------------
# 변경 요약:
# 1) New SDK 사용 시 response_schema에 Pydantic 클래스 자체를 전달(딕셔너리 아님)
#    → additionalProperties 관련 오류 및 빈 properties 오류 방지
# 2) Legacy SDK 경로는 JSON Schema를 쓰되, $defs를 허용하고
#    additionalProperties 키를 모든 깊이에서 제거하도록 sanitizer 강화
# 3) list_models 실패 시 [] 대신 None을 반환하여 "모델 목록에 없음" 경고가 잘못 찍히는 문제 방지
# 4) 503 재시도 안정화(메시지에 'unavailable' 매칭 추가)

from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Optional, List, Any
import logging
import redis
import time
import random

logger = logging.getLogger(__name__)

# 스키마 Sanitizer - 지원하지 않는 키들을 제거
_ALLOWED_SCHEMA_KEYS = {
    "type", "properties", "required", "items", "enum",
    "anyOf", "oneOf", "description", "$ref", "$defs"  # ← $defs 보존 추가
}


def _sanitize_schema(obj):
    """
    Pydantic model_json_schema() 결과에서 지원하지 않는 키들을 제거.
    - 모든 깊이에서 additionalProperties 제거
    - $ref와 $defs는 보존 (참조 깨짐 방지)
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # 모든 깊이에서 additionalProperties 제거
            if k == "additionalProperties":
                continue
            if k == "properties" and isinstance(v, dict):
                out[k] = {name: _sanitize_schema(s) for name, s in v.items()}
            elif k in _ALLOWED_SCHEMA_KEYS:
                out[k] = _sanitize_schema(v)
            # 그 외 키는 제거
        return out
    elif isinstance(obj, list):
        return [_sanitize_schema(i) for i in obj]
    else:
        return obj


def _is_retryable_error(error: Exception) -> bool:
    """재시도 가능한 오류인지 판단."""
    error_str = str(error).lower()
    retryable_patterns = [
        "503", "502", "504", "500", "429",
        "timeout", "connection", "network",
        "rate limit", "quota", "exceeded",
        "service unavailable", "internal server error", "unavailable",
        "temporarily", "retry", "backoff"  # 추가 패턴
    ]
    is_retryable = any(p in error_str for p in retryable_patterns)
    logger.info(f"🔍 오류 재시도 가능성 검사: '{error_str}' -> {is_retryable}")
    return is_retryable


def _exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
    return delay


def _retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            error_str = str(e)
            logger.warning(f"🔍 API 호출 실패 (시도 {attempt + 1}/{max_retries + 1}): {error_str}")
            
            if attempt == max_retries or not _is_retryable_error(e):
                logger.error(f"❌ 최대 재시도 횟수 초과 또는 재시도 불가능한 오류: {error_str}")
                raise e
            
            delay = _exponential_backoff(attempt, base_delay)
            logger.info(f"⏳ {delay:.2f}초 후 재시도...")
            time.sleep(delay)


# 1) 새 SDK 우선, 구 SDK는 백업
try:
    from google import genai  # new SDK
    from google.genai import types as genai_types
    _USE_NEW_SDK = True
except Exception:
    import google.generativeai as genai  # legacy SDK
    _USE_NEW_SDK = False


class Settings(BaseSettings):
    ENV: str = "dev"
    REDIS_URL: Optional[str] = "redis://localhost:6379"
    VECTOR_DIR: str = "data/vector_db"
    DOCUMENT_DIR: str = "data/documents"

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    TAVILY_API_KEY: str = ""

    REDIS_SESSION_TTL: int = 3600
    REDIS_CACHE_TTL: int = 1800

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


class LangSmithLLMWrapper:
    def __init__(self, client_or_model, model_name: str):
        self._backend = client_or_model
        self.model_name = model_name

    def with_structured_output(self, response_schema, emergency_fallback: bool = False):
        return StructuredOutputWrapper(self._backend, self.model_name, response_schema, emergency_fallback)

    def generate_content(self, prompt: str, **kwargs):
        if _USE_NEW_SDK:
            resp = self._backend.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=kwargs.get("config")
            )
            return type("Resp", (), {"text": getattr(resp, "text", None) or str(resp)})
        else:
            resp = self._backend.generate_content(prompt, **kwargs)
            return type("Resp", (), {"text": getattr(resp, "text", None) or str(resp)})


class StructuredOutputWrapper:
    def __init__(self, client_or_model, model_name: str, response_schema, emergency_fallback: bool = False):
        self._backend = client_or_model
        self.model_name = model_name
        self.response_schema = response_schema  # Pydantic 모델 클래스 기대
        self.emergency_fallback = emergency_fallback

    def generate_content(self, prompt: str, **kwargs):
        try:
            print(f"🔍 [StructuredOutputWrapper] 시작 - 모델: {self.model_name}")
            print(f"🔍 [StructuredOutputWrapper] 프롬프트 길이: {len(prompt)}자")
            print(f"🔍 [StructuredOutputWrapper] 스키마: {self.response_schema.__name__}")
            print(f"🔍 [StructuredOutputWrapper] 긴급 탈출 모드: {self.emergency_fallback}")

            if self.emergency_fallback:
                def _generate_emergency_content():
                    if _USE_NEW_SDK:
                        return self._backend.models.generate_content(
                            model=self.model_name,
                            contents=prompt,
                            config=kwargs.get("config")
                        )
                    else:
                        return self._backend.generate_content(prompt, **kwargs)
                resp = _retry_with_backoff(_generate_emergency_content, max_retries=2, base_delay=0.5)
                raw_text = getattr(resp, "text", None) or str(resp)
                from json import loads
                try:
                    data = loads(raw_text)
                except Exception:
                    # 비구조 텍스트를 후처리 매핑 (기존 유틸 사용)
                    return _parse_unstructured_response(raw_text, self.response_schema)
                return self.response_schema(**data)

            if _USE_NEW_SDK:
                # ✅ New SDK: Pydantic 클래스 자체를 전달 (dict 아님)
                config = genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=self.response_schema,
                )
                print("🔍 [StructuredOutputWrapper] New SDK 사용 - config 생성 완료")

                def _generate_content():
                    return self._backend.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=config,
                    )
                resp = _retry_with_backoff(_generate_content, max_retries=3, base_delay=1.0)
                raw = getattr(resp, "text", None) or "{}"
                print(f"🔍 [StructuredOutputWrapper] New SDK 응답 타입: {type(resp)}")
            else:
                # Legacy SDK: dict(JSON Schema) 필요 → sanitize
                gm = self._backend
                schema = _sanitize_schema(self.response_schema.model_json_schema())
                # quotes 같은 복잡 필드 다운캐스팅 필요 시 여기서 처리
                generation_config = {
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                }
                print("🔍 [StructuredOutputWrapper] Legacy SDK 사용 - generation_config 생성 완료")
                def _generate_content():
                    return gm.generate_content(prompt, generation_config=generation_config)
                resp = _retry_with_backoff(_generate_content, max_retries=3, base_delay=1.0)
                raw = getattr(resp, "text", None) or "{}"
                print(f"🔍 [StructuredOutputWrapper] Legacy SDK 응답 타입: {type(resp)}")

            print(f"🔍 [StructuredOutputWrapper] 원본 응답 텍스트: '{raw}'")
            print(f"🔍 [StructuredOutputWrapper] 응답 텍스트 길이: {len(raw)}자")
            from json import loads
            data = loads(raw)
            print(f"🔍 [StructuredOutputWrapper] JSON 파싱 결과: {data}")
            result = self.response_schema(**data)
            print(f"🔍 [StructuredOutputWrapper] Pydantic 객체 생성 성공: {result}")
            return result

        except Exception as e:
            print(f"❌ [StructuredOutputWrapper] 예외 발생: {str(e)}")
            import traceback; print(f"❌ [StructuredOutputWrapper] 스택 트레이스: {traceback.format_exc()}")
            try:
                print("🔧 [StructuredOutputWrapper] 기본값 생성 시도")
                return self.response_schema()
            except Exception:
                # 필드 타입 기반 수동 기본값
                fallback = {}
                for name, f in self.response_schema.model_fields.items():
                    t = f.annotation
                    if t is int:
                        fallback[name] = 0
                    elif t is float:
                        fallback[name] = 0.0
                    elif t is bool:
                        fallback[name] = False
                    elif t is str:
                        fallback[name] = ""
                    elif getattr(t, "__origin__", None) is list:
                        fallback[name] = []
                    elif getattr(t, "__origin__", None) is dict:
                        fallback[name] = {}
                    else:
                        fallback[name] = None
                return self.response_schema(**fallback)


# -----------------------------
# 모델 선택/생성
# -----------------------------

def _normalize_candidates(name: str) -> List[str]:
    base = (name or "").strip()
    if base.startswith("projects/") or "generativelanguage" in base.lower():
        base = "gemini-2.5-flash"
    if base.startswith("models/"):
        base = base.split("models/", 1)[1]
    prefer = [
        base,
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-pro",
        "gemini-1.5-flash-002",
        "gemini-1.5-pro-002",
    ]
    out, seen = [], set()
    for c in prefer:
        if c and c not in seen:
            seen.add(c); out.append(c)
    return out


def _supports_generate_content(meta: Any) -> bool:
    methods = getattr(meta, "supported_generation_methods", None) or \
              getattr(meta, "supported_methods", None)
    if isinstance(methods, (list, tuple, set)):
        return any("generateContent" in m or "generate_content" in m for m in methods)
    return True


def _list_available_model_names(api_key: str) -> Optional[List[str]]:
    try:
        if _USE_NEW_SDK:
            client = genai.Client(api_key=api_key)
            models = list(client.models.list())
            # New SDK에서는 models/ 접두사가 포함된 전체 이름을 반환하므로 제거
            names = []
            for m in models:
                if _supports_generate_content(m):
                    n = getattr(m, "name", "")
                    # models/ 접두사 제거하여 일관성 유지
                    clean_name = n.split("models/")[-1] if n.startswith("models/") else n
                    names.append(clean_name)
            return names
        else:
            genai.configure(api_key=api_key)
            models = list(genai.list_models())
            names = []
            for m in models:
                if _supports_generate_content(m):
                    n = getattr(m, "name", "")
                    names.append(n.split("models/")[-1] if n else n)
            return names
    except Exception as e:
        logger.warning(f"list_models 실패: {e}")
        return None  # ← 빈 리스트 대신 None으로 반환해 필터 비활성화


@lru_cache()
def get_llm():
    s = get_settings()
    if not s.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set in .env")

    try:
        if _USE_NEW_SDK:
            client = genai.Client(api_key=s.GEMINI_API_KEY)
        else:
            genai.configure(api_key=s.GEMINI_API_KEY)
            client = None
    except Exception as e:
        logger.error(f"❌ SDK 초기화 실패: {e}")
        raise RuntimeError(f"Gemini SDK 초기화 실패: {e}")

    try:
        available = _list_available_model_names(s.GEMINI_API_KEY)
        if available:
            logger.info(f"🔍 사용 가능한 모델: {set(available)}")
        else:
            logger.warning("⚠️ 모델 목록이 비어있음 - API 연결 문제일 수 있음")
    except Exception as e:
        logger.warning(f"⚠️ 모델 목록 확인 실패: {e}")
        available = None

    candidates = _normalize_candidates(s.GEMINI_MODEL)
    logger.info(f"🔍 시도할 모델 후보: {candidates}")

    last_err = None
    for cand in candidates:
        # ← 모델 목록 조회에 실패(None)면 스킵하지 않음
        if available is not None and (cand not in set(available)):
            logger.warning(f"⚠️ 모델 {cand}이 사용 가능한 모델 목록에 없음")
            continue
        try:
            if _USE_NEW_SDK:
                logger.info(f"🔄 모델 시도: {cand}")
                # 가벼운 헬스체크 (재시도 래핑)
                def _count_tokens():
                    return client.models.count_tokens(model=cand, contents="ping")
                _retry_with_backoff(_count_tokens, max_retries=3, base_delay=0.5)
                logger.info(f"✅ 모델 사용 가능: {cand}")
                return LangSmithLLMWrapper(client, cand)
            else:
                logger.info(f"🔄 모델 시도: {cand}")
                gm = genai.GenerativeModel(cand)
                def _count_tokens():
                    return gm.count_tokens("ping")
                _retry_with_backoff(_count_tokens, max_retries=3, base_delay=0.5)
                logger.info(f"✅ 모델 생성 성공: {cand}")
                return LangSmithLLMWrapper(gm, cand)
        except Exception as e:
            last_err = e
            error_str = str(e).lower()
            if "api_key" in error_str or "authentication" in error_str:
                logger.error(f"❌ API 키 인증 실패: {e}")
                raise RuntimeError(f"Gemini API 키 인증 실패: {e}")
            elif "quota" in error_str or "limit" in error_str:
                logger.error(f"❌ API 할당량 초과: {e}")
                raise RuntimeError(f"Gemini API 할당량 초과: {e}")
            elif "permission" in error_str or "access" in error_str:
                logger.error(f"❌ 모델 접근 권한 없음: {e}")
                raise RuntimeError(f"모델 {cand}에 대한 접근 권한이 없습니다: {e}")
            elif "503" in error_str or "unavailable" in error_str:
                logger.warning(f"⚠️ 모델 {cand} 서비스 일시 중단 (503): {e}")
                logger.info("🔄 다른 모델로 시도하거나 잠시 후 재시도하세요")
                continue
            else:
                logger.warning(f"❌ 모델 {cand} 실패: {e}")
                continue

    error_details = []
    if last_err:
        error_details.append(f"마지막 오류: {last_err}")
    if available is None:
        error_details.append("사용 가능한 모델 목록을 확인하지 못함(list_models 실패)")
    if not s.GEMINI_API_KEY:
        error_details.append("GEMINI_API_KEY가 설정되지 않음")

    error_msg = f"모든 Gemini 모델 시도 실패. {' | '.join(error_details)}"
    logger.error(f"❌ {error_msg}")
    raise RuntimeError(error_msg)


def get_available_models():
    s = get_settings()
    try:
        return _list_available_model_names(s.GEMINI_API_KEY) or []
    except Exception as e:
        logger.error(f"모델 목록 확인 실패: {e}")
        return []


@lru_cache()
def get_redis_client():
    s = get_settings()
    if not s.REDIS_URL:
        raise RuntimeError("REDIS_URL is not set in .env")
    try:
        client = redis.from_url(
            s.REDIS_URL,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
        client.ping()
        return client
    except Exception as e:
        logger.error(f"Redis 연결 실패: {e}")
        return None
