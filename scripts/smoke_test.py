from retriever.hybrid import hybrid_search

def main():
    vec = [
        {"doc_id": "db_202501", "page": 2, "text": "항공기 지연 보장", "score_vec": 1.2},
        {"doc_id": "db_202501", "page": 3, "text": "수하물 지연", "score_vec": 0.8},
    ]
    kw = [
        {"doc_id": "db_202501", "page": 2, "text": "항공기 지연 보장", "score_kw": 3.0},
        {"doc_id": "kakao_202412", "page": 1, "text": "여행자 보험 기초", "score_kw": 2.0},
    ]
    out = hybrid_search("연착 서류", vec, kw, k=3)
    assert len(out) == 3 or len(out) == 2
    print("OK hybrid_search:", out)

if __name__ == "__main__":
    main()