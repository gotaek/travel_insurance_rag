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
                # include 메타데이터를 명시적으로 포함
                results = collection.get(
                    limit=10000,
                    include=['documents', 'metadatas']
                )
                
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
                    
                    # 보험사별 분포 확인
                    insurer_counts = {}
                    for item in corpus:
                        insurer = item.get("insurer", "Unknown")
                        insurer_counts[insurer] = insurer_counts.get(insurer, 0) + 1
                    
                    print(f"📊 키워드 검색용 전체 코퍼스 로드 완료: {len(corpus)}개 문서")
                    print(f"📋 보험사별 분포: {insurer_counts}")
                    
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

def keyword_search_full_corpus(query: str, k: int = 5, insurer_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    전체 코퍼스에서 BM25 키워드 검색을 수행합니다.
    KeywordStore 인스턴스를 캐싱하여 성능 최적화
    
    Args:
        query: 검색 쿼리
        k: 반환할 결과 수
        insurer_filter: 보험사 필터 (선택사항)
        
    Returns:
        검색 결과 리스트
    """
    global _keyword_store_cache
    
    # 캐시된 인스턴스가 없으면 새로 생성
    if _keyword_store_cache is None:
        print("🔄 키워드 검색용 전체 코퍼스 로드 중...")
        full_corpus = _load_full_corpus()
        if not full_corpus:
            print("⚠️ 전체 코퍼스 로드 실패")
            return []
        _keyword_store_cache = KeywordStore(full_corpus)
        print(f"✅ 키워드 검색용 코퍼스 준비 완료: {len(_keyword_store_cache.docs)}개 문서")
    
    # 보험사 필터링이 있는 경우 사전 필터링 적용
    if insurer_filter:
        filtered_docs = _apply_insurer_filter_to_corpus(_keyword_store_cache.docs, insurer_filter)
        if not filtered_docs:
            return []
        
        # 필터링된 문서로 임시 KeywordStore 생성
        temp_store = KeywordStore(filtered_docs)
        
        # BM25 검색 수행
        results = temp_store.search(query, k=k)
        
        return results
    else:
        # BM25 검색 수행
        results = _keyword_store_cache.search(query, k=k)
        
        return results

def _apply_insurer_filter_to_corpus(docs: List[Dict[str, Any]], insurer_filter: List[str]) -> List[Dict[str, Any]]:
    """
    코퍼스에서 보험사 필터를 적용하여 필터링된 문서만 반환합니다.
    insurer 필드를 사용하여 정확한 매칭 수행
    
    Args:
        docs: 전체 문서 코퍼스
        insurer_filter: 보험사 필터 리스트
        
    Returns:
        필터링된 문서 리스트
    """
    if not insurer_filter:
        return docs
    
    import unicodedata
    
    def normalize_korean(text: str) -> str:
        """한글 정규화 (완성형 -> 조합형) - DB가 NFD 형태로 저장됨"""
        return unicodedata.normalize('NFD', text)
    
    filtered_docs = []
    for doc in docs:
        doc_insurer = doc.get("insurer", "")
        doc_insurer_normalized = normalize_korean(doc_insurer).lower()
        
        # 보험사 필터와 매칭되는지 확인
        matched = False
        for filter_insurer in insurer_filter:
            normalized_filter = normalize_korean(filter_insurer).lower()
            
            # 정확한 매칭 우선 시도
            if doc_insurer_normalized == normalized_filter:
                filtered_docs.append(doc)
                matched = True
                break
            
            # 부분 매칭 시도 (카카오 -> 카카오페이)
            if normalized_filter in doc_insurer_normalized or doc_insurer_normalized in normalized_filter:
                filtered_docs.append(doc)
                matched = True
                break
    
    return filtered_docs

def _apply_insurer_filter_to_keyword_results(results: List[Dict[str, Any]], insurer_filter: List[str]) -> List[Dict[str, Any]]:
    """
    키워드 검색 결과에 보험사 필터를 적용합니다.
    한글 정규화를 통해 조합형/완성형 한글을 통일하여 매칭합니다.
    
    Args:
        results: 키워드 검색 결과
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
    
    filtered_results = []
    for result in results:
        doc_insurer = result.get("insurer", "")
        doc_insurer_normalized = normalize_korean(doc_insurer).lower()
        
        # 보험사 필터와 매칭되는지 확인
        matched = False
        for filter_insurer in insurer_filter:
            normalized_filter = normalize_korean(filter_insurer).lower()
            
            # 정확한 매칭 우선 시도
            if doc_insurer_normalized == normalized_filter:
                filtered_results.append(result)
                matched = True
                break
            
            # 부분 매칭 시도
            if normalized_filter in doc_insurer_normalized or doc_insurer_normalized in normalized_filter:
                filtered_results.append(result)
                matched = True
                break
    
    return filtered_results

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