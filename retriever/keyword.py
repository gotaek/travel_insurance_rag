from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

def _tokenize(s: str) -> List[str]:
    # 간단 토크나이저(공백 분리). 한국어 형태소 분석 대체용 스텁.
    return s.lower().split()

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

__all__ = ["KeywordStore", "keyword_search"]