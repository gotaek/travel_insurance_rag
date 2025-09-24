import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None

from retriever.embeddings import embed_texts  # ✅ unify with ingest embedding
from graph.cache_manager import cache_manager

# 로깅 설정
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Chroma DB 벡터 스토어 with multilingual-e5-small-ko 임베딩 모델
    Returns empty results safely if chromadb or collection is unavailable.
    """
    def __init__(self, db_path: str, collection_name: str = "insurance_docs"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None

        if chromadb:
            try:
                # Chroma DB 클라이언트 초기화
                self.client = chromadb.PersistentClient(
                    path=db_path,
                    settings=Settings(anonymized_telemetry=False)
                )
                # 컬렉션 가져오기 (존재하지 않으면 빈 결과 반환)
                try:
                    self.collection = self.client.get_collection(collection_name)
                except Exception:
                    # 컬렉션이 존재하지 않는 경우
                    self.collection = None
            except Exception:
                self.client = None
                self.collection = None

    def is_ready(self) -> bool:
        """Chroma DB가 준비되었는지 확인"""
        return self.collection is not None

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """쿼리에 대한 벡터 검색 수행 - multilingual-e5-small-ko 모델 사용 (캐싱 지원)"""
        if not self.is_ready():
            return []
        
        # 캐시에서 먼저 확인
        cached_results = cache_manager.get_cached_search_results(query, "vector", k)
        if cached_results is not None:
            logger.debug(f"벡터 검색 캐시 히트: {query[:50]}...")
            return cached_results
        
        try:
            # multilingual-e5-small-ko 모델을 사용하여 쿼리 임베딩 생성
            query_embedding = embed_texts([query])
            
            # Chroma DB에서 검색 수행 (e5 임베딩 기반)
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=k
            )
            
            out: List[Dict[str, Any]] = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0] if results['metadatas'] else [{}] * len(results['documents'][0]),
                    results['distances'][0] if results['distances'] else [0.0] * len(results['documents'][0])
                )):
                    item = dict(metadata) if metadata else {}
                    item["text"] = doc
                    item["score_vec"] = float(1.0 - distance)  # 거리를 유사도 점수로 변환
                    out.append(item)
            
            # 결과 캐싱
            cache_manager.cache_search_results(query, out, "vector", k)
            logger.debug(f"벡터 검색 완료 및 캐싱: {len(out)}개 결과")
            
            return out
        except Exception as e:
            logger.error(f"벡터 검색 실패: {e}")
            return []

# 전역 VectorStore 인스턴스 (싱글톤)
_vector_store_cache: Dict[str, VectorStore] = {}

def vector_search(query: str, db_path: str, collection_name: str = "insurance_docs", k: int = 5) -> List[Dict[str, Any]]:
    """
    벡터 검색 함수 - multilingual-e5-small-ko 모델 사용
    VectorStore 인스턴스를 캐싱하여 성능 최적화
    """
    # 캐시 키 생성
    cache_key = f"{db_path}:{collection_name}"
    
    # 캐시된 인스턴스가 없으면 새로 생성
    if cache_key not in _vector_store_cache:
        _vector_store_cache[cache_key] = VectorStore(db_path, collection_name)
    
    store = _vector_store_cache[cache_key]
    return store.search(query, k=k)

def get_vector_store_info() -> Dict[str, Any]:
    """VectorStore 캐시 정보 반환"""
    return {
        "cached_stores": len(_vector_store_cache),
        "store_keys": list(_vector_store_cache.keys())
    }

def clear_vector_store_cache():
    """VectorStore 캐시 초기화"""
    global _vector_store_cache
    _vector_store_cache.clear()