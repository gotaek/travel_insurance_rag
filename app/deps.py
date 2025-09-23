from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Optional
import google.generativeai as genai
import redis

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

@lru_cache()
def get_llm():
    """
    Gemini LLM 클라이언트 (환경변수는 .env에서 로드)
    """
    s = get_settings()
    if not s.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set in .env")
    genai.configure(api_key=s.GEMINI_API_KEY)
    return genai.GenerativeModel(s.GEMINI_MODEL)

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
        print(f"⚠️ Redis 연결 실패: {e}")
        return None