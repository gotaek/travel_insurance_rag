from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

def _tokenize(s: str) -> List[str]:
    # 간단 토크나이저(공백 분리). 한국어 형태소 분석 대체용 스텁.
    return s.lower().split()

def keyword_search(query: str, corpus_meta: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    """
    corpus_meta: [{"text": "...", "doc_id": "...", "page": 1, ...}, ...]
    - corpus_meta가 비었으면 빈 리스트 반환(안전)
    """
    if not corpus_meta:
        return []
    texts = [c.get("text", "") for c in corpus_meta]
    tokenized_corpus = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(
        [
            {**corpus_meta[i], "score_kw": float(scores[i])}
            for i in range(len(corpus_meta))
        ],
        key=lambda x: x["score_kw"],
        reverse=True,
    )
    return ranked[:k]