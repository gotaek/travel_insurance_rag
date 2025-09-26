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

    def search(self, query: str, k: int = 5, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        쿼리에 대한 벡터 검색 수행 - multilingual-e5-small-ko 모델 사용 (캐싱 지원)
        Chroma DB 네이티브 필터링 지원
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            where: Chroma DB where 절 필터 (선택사항)
        """
        if not self.is_ready():
            return []
        
        # 캐시 키 생성 (where 절 포함)
        cache_key = f"{query}:{k}:{str(where)}" if where else f"{query}:{k}"
        cached_results = cache_manager.get_cached_search_results(cache_key, "vector", k)
        if cached_results is not None:
            logger.debug(f"벡터 검색 캐시 히트: {query[:50]}...")
            return cached_results
        
        try:
            # multilingual-e5-small-ko 모델을 사용하여 쿼리 임베딩 생성
            query_embedding = embed_texts([query])
            
            # Chroma DB에서 검색 수행 (e5 임베딩 기반, 네이티브 필터링 지원)
            query_params = {
                "query_embeddings": query_embedding.tolist(),
                "n_results": k
            }
            
            # where 절이 있으면 추가
            if where:
                query_params["where"] = where
                logger.debug(f"Chroma DB 네이티브 필터링 적용: {where}")
            
            results = self.collection.query(**query_params)
            
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
            cache_manager.cache_search_results(cache_key, out, "vector", k)
            logger.debug(f"벡터 검색 완료 및 캐싱: {len(out)}개 결과")
            
            return out
        except Exception as e:
            logger.error(f"벡터 검색 실패: {e}")
            return []

def _create_insurer_where_clause(insurer_filter: List[str]) -> Dict[str, Any]:
    """
    보험사 필터를 Chroma DB where 절로 변환합니다.
    insurer 필드를 사용하여 정확한 매칭 수행
    
    Args:
        insurer_filter: 보험사 필터 리스트
        
    Returns:
        Chroma DB where 절 딕셔너리
    """
    if not insurer_filter:
        return {}
    
    # 단일 보험사인 경우
    if len(insurer_filter) == 1:
        return {"insurer": insurer_filter[0]}
    
    # 여러 보험사인 경우 $in 연산자 사용
    return {"insurer": {"$in": insurer_filter}}

# 전역 VectorStore 인스턴스 (싱글톤)
_vector_store_cache: Dict[str, VectorStore] = {}

def vector_search(query: str, db_path: str, collection_name: str = "insurance_docs", k: int = 5, insurer_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    벡터 검색 함수 - multilingual-e5-small-ko 모델 사용
    사후 필터링 지원 (Chroma DB 네이티브 필터링 문제로 인해)
    
    Args:
        query: 검색 쿼리
        db_path: 벡터 DB 경로
        collection_name: 컬렉션 이름
        k: 반환할 결과 수
        insurer_filter: 보험사 필터 (선택사항)
    """
    # 캐시 키 생성
    cache_key = f"{db_path}:{collection_name}"
    
    # 캐시된 인스턴스가 없으면 새로 생성
    if cache_key not in _vector_store_cache:
        _vector_store_cache[cache_key] = VectorStore(db_path, collection_name)
    
    store = _vector_store_cache[cache_key]
    
    # 필터링이 있는 경우 더 많은 결과를 검색한 후 필터링
    search_k = k * 3 if insurer_filter else k  # 필터링이 있으면 3배 더 검색
    
    # 벡터 검색 수행 (필터링 없이)
    results = store.search(query, k=search_k)
    
    # 보험사 필터링 적용 (사후 필터링)
    if insurer_filter and results:
        results = _apply_insurer_filter_to_results(results, insurer_filter)
        # 필터링 후 결과 수 제한
        results = results[:k]
    
    return results

# 기존 사후 필터링 함수는 네이티브 필터링으로 대체됨
# 이 함수는 더 이상 사용되지 않음 (하위 호환성을 위해 유지)
def _apply_insurer_filter_to_results(results: List[Dict[str, Any]], insurer_filter: List[str]) -> List[Dict[str, Any]]:
    """
    벡터 검색 결과에 보험사 필터를 적용합니다.
    부분 매칭을 지원하여 '카카오'로 '카카오페이' 검색 가능
    중복 제거 및 효율적인 매칭 지원
    
    Args:
        results: 벡터 검색 결과
        insurer_filter: 보험사 필터 리스트
        
    Returns:
        필터링된 검색 결과
    """
    if not insurer_filter:
        return results
    
    import unicodedata
    
    def normalize_korean(text: str) -> str:
        """한글 정규화 (완성형 -> 조합형) - DB가 NFD 형태로 저장됨"""
        return unicodedata.normalize('NFD', text)
    
    def is_match(doc_insurer: str, filter_insurer: str) -> bool:
        """보험사 이름과 필터가 매칭되는지 확인"""
        normalized_filter = normalize_korean(filter_insurer).lower()
        
        # 정확한 매칭
        if doc_insurer == normalized_filter:
            return True
        
        # 부분 매칭 (필터가 보험사 이름에 포함되거나, 보험사 이름이 필터에 포함)
        if (normalized_filter in doc_insurer or 
            doc_insurer in normalized_filter):
            return True
        
        # 단어 단위 매칭
        filter_words = normalized_filter.split()
        doc_words = doc_insurer.split()
        
        # 필터의 모든 단어가 보험사 이름에 포함되는지 확인
        if all(any(word in doc_word or doc_word in word for doc_word in doc_words) for word in filter_words):
            return True
        
        return False
    
    # 중복 제거를 위한 set 사용
    seen_results = set()
    filtered_results = []
    
    for result in results:
        doc_insurer = normalize_korean(result.get("insurer", "")).lower()
        
        # 보험사 필터와 매칭되는지 확인
        matched = False
        for filter_insurer in insurer_filter:
            if is_match(doc_insurer, filter_insurer):
                # 중복 제거를 위한 고유 키 생성
                result_key = f"{result.get('doc_id', '')}_{result.get('page', '')}_{result.get('chunk_no', '')}"
                
                if result_key not in seen_results:
                    seen_results.add(result_key)
                    filtered_results.append(result)
                matched = True
                break
    
    return filtered_results

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