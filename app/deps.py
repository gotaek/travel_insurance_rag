from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Optional
import google.generativeai as genai

class Settings(BaseSettings):
    # 기존 필드
    ENV: str = "dev"
    REDIS_URL: Optional[str] = None
    VECTOR_DIR: str = "data/vector_db"
    DOCUMENT_DIR: str = "data/documents"

    # ✅ Gemini(.env) 필드
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"

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