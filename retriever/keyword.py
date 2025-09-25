from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import os
import json
from pathlib import Path

try:
    import chromadb
except Exception:
    chromadb = None

def _tokenize(s: str) -> List[str]:
    # 간단 토크나이저(공백 분리). 한국어 형태소 분석 대체용 스텁.
    return s.lower().split()

def _load_full_corpus() -> List[Dict[str, Any]]:
    """
    전체 코퍼스를 로드합니다.
    벡터 DB에서 모든 문서를 가져오거나, 별도 인덱스에서 로드.
    """
    try:
        # 벡터 DB에서 전체 문서 로드 시도
        from app.deps import get_settings
        settings = get_settings()
        
        # Chroma DB에서 전체 문서 가져오기
        if chromadb:
            try:
                client = chromadb.PersistentClient(
                    path=settings.VECTOR_DIR,
                    settings=chromadb.config.Settings(anonymized_telemetry=False)
                )
                collection = client.get_collection("insurance_docs")
                
                # 전체 문서 가져오기 (최대 10000개)
                results = collection.get(limit=10000)
                
                if results and results.get('documents'):
                    corpus = []
                    for i, doc in enumerate(results['documents']):
                        metadata = results['metadatas'][i] if results.get('metadatas') else {}
                        corpus.append({
                            "text": doc,
                            "doc_id": metadata.get("doc_id", f"doc_{i}"),
                            "page": metadata.get("page", 0),
                            **metadata
                        })
                    return corpus
            except Exception as e:
                print(f"⚠️ 벡터 DB에서 전체 코퍼스 로드 실패: {e}")
        
        # 폴백: 빈 리스트 반환
        return []
        
    except Exception as e:
        print(f"⚠️ 전체 코퍼스 로드 실패: {e}")
        return []

class KeywordStore:
    """
    Simple BM25 keyword search over pre-tokenized corpus.
    - docs: List[Dict] with at least a "text" field.
    """
    def __init__(self, docs: List[Dict[str, Any]]):
        self.docs = docs or []
        corpus = [ _tokenize(d.get("text", "")) for d in self.docs ]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.docs or self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(
            (
                {**self.docs[i], "score_kw": float(scores[i])}
                for i in range(len(self.docs))
            ),
            key=lambda x: x["score_kw"],
            reverse=True,
        )
        return ranked[:k]

# Backward-compatible functional wrapper
def keyword_search(query: str, corpus_meta: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    """
    corpus_meta: [{"text": "...", "doc_id": "...", "page": 1, ...}, ...]
    - corpus_meta가 비었으면 빈 리스트 반환(안전)
    """
    store = KeywordStore(corpus_meta)
    return store.search(query, k=k)

# 전역 KeywordStore 인스턴스 (싱글톤)
_keyword_store_cache: Optional[KeywordStore] = None

def keyword_search_full_corpus(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    전체 코퍼스에서 BM25 키워드 검색을 수행합니다.
    KeywordStore 인스턴스를 캐싱하여 성능 최적화
    
    Args:
        query: 검색 쿼리
        k: 반환할 결과 수
        
    Returns:
        검색 결과 리스트
    """
    global _keyword_store_cache
    
    # 캐시된 인스턴스가 없으면 새로 생성
    if _keyword_store_cache is None:
        full_corpus = _load_full_corpus()
        if not full_corpus:
            return []
        _keyword_store_cache = KeywordStore(full_corpus)
    
    # BM25 검색 수행
    return _keyword_store_cache.search(query, k=k)

def get_keyword_store_info() -> Dict[str, Any]:
    """KeywordStore 캐시 정보 반환"""
    global _keyword_store_cache
    return {
        "is_cached": _keyword_store_cache is not None,
        "corpus_size": len(_keyword_store_cache.docs) if _keyword_store_cache else 0
    }

def clear_keyword_store_cache():
    """KeywordStore 캐시 초기화"""
    global _keyword_store_cache
    _keyword_store_cache = None

__all__ = ["KeywordStore", "keyword_search", "keyword_search_full_corpus"]