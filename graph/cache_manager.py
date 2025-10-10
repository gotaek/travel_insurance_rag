"""
Redis 기반 캐싱 관리자
- 임베딩 결과 캐싱
- 검색 결과 캐싱
- 세션 데이터 관리
"""

import json
import pickle
import hashlib
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import numpy as np

from app.deps import get_redis_client, get_settings
from graph.normalize_cache import question_normalizer

# 로깅 설정
logger = logging.getLogger(__name__)


class CacheManager:
    """Redis 기반 캐싱 관리자"""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = get_redis_client()
    
    def _generate_cache_key(self, prefix: str, content: str, **kwargs) -> str:
        """캐시 키 생성 - 일관성 보장"""
        # 키워드 인자들을 정렬하여 일관성 보장
        sorted_kwargs = sorted(kwargs.items()) if kwargs else []
        key_content = f"{content}:{sorted_kwargs}"
        content_hash = hashlib.md5(key_content.encode()).hexdigest()[:16]
        return f"{prefix}:{content_hash}"
    
    def _generate_texts_cache_key(self, texts: List[str]) -> str:
        """텍스트 리스트용 일관된 캐시 키 생성 (정규화 포함)"""
        if not texts:
            return "embeddings:empty"
        
        # 단일 텍스트인 경우 정규화 적용
        if len(texts) == 1:
            normalized_text = question_normalizer.normalize_question(texts[0])
            content_hash = hashlib.md5(normalized_text.encode()).hexdigest()[:16]
            return f"embeddings:{content_hash}"
        
        # 여러 텍스트인 경우 정규화 후 정렬
        normalized_texts = [question_normalizer.normalize_question(text) for text in texts]
        sorted_texts = sorted(normalized_texts)
        key_content = "|".join(sorted_texts)
        content_hash = hashlib.md5(key_content.encode()).hexdigest()[:16]
        return f"embeddings:{content_hash}"
    
    def cache_embeddings(
        self, 
        texts: List[str], 
        embeddings: np.ndarray,
        ttl: Optional[int] = None
    ) -> bool:
        """임베딩 결과 캐싱"""
        if not self.redis_client or not texts:
            return False
        
        try:
            cache_key = self._generate_texts_cache_key(texts)
            
            # numpy 배열을 바이너리로 직렬화
            data = {
                "texts": texts,
                "embeddings": embeddings.tobytes(),
                "shape": embeddings.shape,
                "dtype": str(embeddings.dtype),
                "timestamp": datetime.now().isoformat()
            }
            
            serialized_data = pickle.dumps(data)
            ttl = ttl or self.settings.REDIS_CACHE_TTL
            
            self.redis_client.setex(cache_key, ttl, serialized_data)
            return True
        except Exception as e:
            logger.error(f"임베딩 캐싱 실패: {e}")
            return False
    
    def get_cached_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """캐시된 임베딩 결과 조회"""
        if not self.redis_client or not texts:
            return None
        
        try:
            cache_key = self._generate_texts_cache_key(texts)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                data = pickle.loads(cached_data)
                # numpy 배열 복원
                embeddings = np.frombuffer(
                    data["embeddings"], 
                    dtype=data["dtype"]
                ).reshape(data["shape"])
                return embeddings
            return None
        except Exception as e:
            logger.error(f"임베딩 캐시 조회 실패: {e}")
            return None
    
    def cache_search_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        search_type: str = "vector",
        k: int = 5,
        ttl: Optional[int] = None
    ) -> bool:
        """검색 결과 캐싱 (정규화 포함)"""
        if not self.redis_client or not results:
            return False
        
        try:
            # 정규화된 쿼리로 캐시 키 생성
            normalized_query = question_normalizer.normalize_question(query)
            cache_key = self._generate_cache_key(
                "search", 
                normalized_query, 
                search_type=search_type, 
                k=k
            )
            
            data = {
                "original_query": query,
                "normalized_query": normalized_query,
                "results": results,
                "search_type": search_type,
                "k": k,
                "timestamp": datetime.now().isoformat()
            }
            
            serialized_data = pickle.dumps(data)
            ttl = ttl or self.settings.REDIS_CACHE_TTL
            
            self.redis_client.setex(cache_key, ttl, serialized_data)
            logger.debug(f"검색 결과 캐시 저장 (정규화): '{query}' → '{normalized_query}'")
            return True
        except Exception as e:
            logger.error(f"검색 결과 캐싱 실패: {e}")
            return False
    
    def get_cached_search_results(
        self,
        query: str,
        search_type: str = "vector",
        k: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """캐시된 검색 결과 조회 (정규화 포함)"""
        if not self.redis_client:
            return None
        
        try:
            # 정규화된 쿼리로 캐시 키 생성
            normalized_query = question_normalizer.normalize_question(query)
            cache_key = self._generate_cache_key(
                "search", 
                normalized_query, 
                search_type=search_type, 
                k=k
            )
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                data = pickle.loads(cached_data)
                logger.debug(f"검색 결과 캐시 히트 (정규화): '{query}' → '{normalized_query}'")
                return data["results"]
            return None
        except Exception as e:
            logger.error(f"검색 결과 캐시 조회 실패: {e}")
            return None
    
    def cache_llm_response(
        self,
        prompt_hash: str,
        response: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """LLM 응답 캐싱"""
        if not self.redis_client or not response:
            return False
        
        try:
            cache_key = f"llm_response:{prompt_hash}"
            
            data = {
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
            
            serialized_data = pickle.dumps(data)
            ttl = ttl or self.settings.REDIS_CACHE_TTL
            
            self.redis_client.setex(cache_key, ttl, serialized_data)
            return True
        except Exception as e:
            logger.error(f"LLM 응답 캐싱 실패: {e}")
            return False
    
    def get_cached_llm_response(self, prompt_hash: str) -> Optional[Dict[str, Any]]:
        """캐시된 LLM 응답 조회"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = f"llm_response:{prompt_hash}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                data = pickle.loads(cached_data)
                return data["response"]
            return None
        except Exception as e:
            logger.error(f"LLM 응답 캐시 조회 실패: {e}")
            return None
    
    def generate_prompt_hash(self, prompt: str, **kwargs) -> str:
        """프롬프트 해시 생성 (질문 정규화 포함)"""
        # 프롬프트에서 질문 부분을 찾아서 정규화
        normalized_prompt = self._normalize_prompt_question(prompt)
        content = f"{normalized_prompt}:{sorted(kwargs.items()) if kwargs else ''}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _normalize_prompt_question(self, prompt: str) -> str:
        """프롬프트에서 질문 부분을 정규화"""
        try:
            # "## 질문" 섹션을 찾아서 정규화
            if "## 질문" in prompt:
                parts = prompt.split("## 질문")
                if len(parts) >= 2:
                    # 질문 부분 추출
                    question_part = parts[1].split("\n")[0].strip()
                    # 정규화
                    normalized_question = question_normalizer.normalize_question(question_part)
                    # 프롬프트 재구성
                    return prompt.replace(question_part, normalized_question)
        except Exception as e:
            logger.warning(f"프롬프트 질문 정규화 실패: {e}")
        
        return prompt
    
    def invalidate_cache(self, pattern: str) -> int:
        """패턴에 맞는 캐시 무효화"""
        if not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"캐시 무효화 실패: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보"""
        if not self.redis_client:
            return {"error": "Redis 연결 없음"}
        
        try:
            stats = {}
            
            # 각 타입별 키 개수
            for cache_type in ["embeddings", "search", "llm_response", "session"]:
                pattern = f"{cache_type}:*"
                keys = self.redis_client.keys(pattern)
                stats[f"{cache_type}_count"] = len(keys)
            
            # 메모리 사용량 (Redis INFO 명령)
            info = self.redis_client.info("memory")
            stats["memory_used"] = info.get("used_memory_human", "N/A")
            stats["memory_peak"] = info.get("used_memory_peak_human", "N/A")
            
            return stats
        except Exception as e:
            return {"error": f"통계 조회 실패: {e}"}
    
    def cleanup_expired_cache(self) -> Dict[str, int]:
        """만료된 캐시 정리 (Redis TTL이 자동 처리)"""
        if not self.redis_client:
            return {"error": "Redis 연결 없음"}
        
        try:
            # Redis는 TTL이 자동으로 만료된 키를 정리하므로
            # 여기서는 현재 상태만 반환
            stats = {}
            for cache_type in ["embeddings", "search", "llm_response", "session"]:
                pattern = f"{cache_type}:*"
                keys = self.redis_client.keys(pattern)
                stats[f"{cache_type}_active"] = len(keys)
            
            return stats
        except Exception as e:
            return {"error": f"정리 실패: {e}"}


# 전역 인스턴스
cache_manager = CacheManager()
