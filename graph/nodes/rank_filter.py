from typing import Dict, Any, List
import math
import logging
from collections import Counter

# 로깅 설정
logger = logging.getLogger(__name__)

def _dedup(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """텍스트 기준 단순 중복 제거"""
    seen = set()
    out = []
    for p in passages:
        text = p.get("text", "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(p)
    return out


def _rerank_with_advanced_scoring(passages: List[Dict[str, Any]], question: str, insurer_filter: List[str] = None) -> List[Dict[str, Any]]:
    """
    고급 점수 계산을 통한 리랭크
    - 질문-문서 간 의미적 유사도 강화
    - 키워드 매칭 가중치
    - 문서 품질 점수
    - 보험사별 우선순위 적용
    """
    if not passages or not question:
        return passages
    
    question_lower = question.lower()
    question_words = set(question_lower.split())
    
    
    reranked = []
    for passage in passages:
        text = passage.get("text", "").lower()
        title = passage.get("title", "").lower()
        
        # 기본 점수
        base_score = passage.get("score", 0.0)
        
        # 키워드 매칭 점수 (질문 단어가 문서에 얼마나 포함되는지)
        text_words = set(text.split())
        title_words = set(title.split())
        
        # 질문 단어와의 교집합
        text_matches = len(question_words.intersection(text_words))
        title_matches = len(question_words.intersection(title_words))
        
        # 키워드 매칭 점수 (제목 매칭에 더 높은 가중치)
        keyword_score = (text_matches * 0.3 + title_matches * 0.7) / max(len(question_words), 1)
        
        # 문서 품질 점수 (길이, 구조 등)
        quality_score = min(len(text) / 500, 1.0)  # 적절한 길이의 문서 선호
        
        # 보험 관련 키워드 가중치
        insurance_keywords = ["보험", "보장", "보상", "손해", "위험", "보험료", "보험금", "보험사"]
        insurance_bonus = sum(1 for kw in insurance_keywords if kw in text) * 0.1
        
        # 최종 점수 계산 (보험사 가중치 제거)
        final_score = (
            base_score * 0.5 +           # 기본 검색 점수
            keyword_score * 0.25 +       # 키워드 매칭
            quality_score * 0.1 +        # 문서 품질
            insurance_bonus              # 보험 전문성 보너스
        )
        
        # 점수 업데이트
        passage_copy = dict(passage)
        passage_copy["score"] = min(final_score, 1.0)
        passage_copy["rerank_score"] = final_score
        passage_copy["keyword_matches"] = text_matches + title_matches
        passage_copy["score_breakdown"] = {
            "base_score": base_score * 0.5,
            "keyword_score": keyword_score * 0.25,
            "quality_score": quality_score * 0.1,
            "insurance_bonus": insurance_bonus
        }
        reranked.append(passage_copy)
    
    return reranked

def _apply_mmr(passages: List[Dict[str, Any]], question: str, lambda_param: float = 0.7, insurer_filter: List[str] = None) -> List[Dict[str, Any]]:
    """
    MMR (Maximal Marginal Relevance) 적용
    - 관련성과 다양성의 균형
    - 중복 내용 제거
    - 보험사 필터링은 Search 단계에서 이미 적용됨
    """
    if not passages:
        return passages
    
    # 점수 기준 정렬
    passages.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    
    
    selected = []
    remaining = passages.copy()
    
    # 첫 번째 문서는 가장 높은 점수
    if remaining:
        selected.append(remaining.pop(0))
    
    
    # MMR 알고리즘 적용
    while remaining and len(selected) < 5:  # 최대 5개 선택
        best_idx = 0
        best_mmr_score = -1
        
        for i, candidate in enumerate(remaining):
            # 관련성 점수
            relevance_score = candidate.get("score", 0.0)
            
            
            # 다양성 점수 (이미 선택된 문서들과의 유사도)
            max_similarity = 0.0
            for selected_doc in selected:
                similarity = _calculate_similarity(candidate, selected_doc)
                max_similarity = max(max_similarity, similarity)
            
            # MMR 점수 계산
            mmr_score = lambda_param * relevance_score - (1 - lambda_param) * max_similarity
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = i
        
        # 최적 문서 선택
        chosen_doc = remaining.pop(best_idx)
        selected.append(chosen_doc)
        
    
    return selected


def _calculate_similarity(doc1: Dict[str, Any], doc2: Dict[str, Any]) -> float:
    """두 문서 간의 유사도 계산 (간단한 Jaccard 유사도)"""
    text1 = set(doc1.get("text", "").lower().split())
    text2 = set(doc2.get("text", "").lower().split())
    
    if not text1 or not text2:
        return 0.0
    
    intersection = len(text1.intersection(text2))
    union = len(text1.union(text2))
    
    return intersection / union if union > 0 else 0.0

def _quality_filter(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    품질 기준 필터링
    - 최소 점수 임계값
    - 텍스트 길이 검증
    """
    if not passages:
        return passages
    
    filtered = []
    for passage in passages:
        # 최소 점수 임계값
        if passage.get("score", 0.0) < 0.1:
            continue
        
        # 텍스트 길이 검증 (너무 짧거나 긴 문서 제외)
        text_length = len(passage.get("text", ""))
        if text_length < 50 or text_length > 2000:
            continue
        
        filtered.append(passage)
    
    return filtered

def _sort_by_score(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """score 키 기준 내림차순 정렬"""
    return sorted(passages, key=lambda x: x.get("score", 0.0), reverse=True)

def rank_filter_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    전통적 리랭크 및 필터링:
    - 중복 제거
    - 고급 점수 계산 (리랭크)
    - MMR 다양성 확보
    - 품질 필터링
    - 상위 Top-k 선택
    - 보험사 우선순위 샘플링
    """
    passages = state.get("passages", [])
    question = state.get("question", "")
    insurer_filter = state.get("insurer_filter", None)  # Planner에서 전달된 보험사 필터
    
    if not passages:
        return {**state, "refined": []}

    # 1. 중복 제거
    deduped = _dedup(passages)
    
    # 2. 고급 리랭크 (질문-문서 간 정교한 관련성 계산, 보험사 우선순위 포함)
    logger.info(f"전통적 리랭크 사용: {len(deduped)}개 후보, 보험사 필터: {insurer_filter}")
    reranked = _rerank_with_advanced_scoring(deduped, question, insurer_filter)
    
    # 3. MMR 적용 (다양성 확보, 보험사 우선순위 샘플링 포함)
    diverse = _apply_mmr(reranked, question, lambda_param=0.7, insurer_filter=insurer_filter)
    
    # 4. 품질 필터링
    filtered = _quality_filter(diverse)
    
    # 5. 최종 정렬 및 Top-k 선택
    sorted_passages = _sort_by_score(filtered)
    topk = sorted_passages[:5]
    
    
    # 메타데이터 추가
    rank_meta = {
        "original_count": len(passages),
        "deduped_count": len(deduped),
        "reranked_count": len(reranked),
        "diverse_count": len(diverse),
        "filtered_count": len(filtered),
        "final_count": len(topk),
        "rerank_method": "traditional",
        "rerank_applied": True,
        "mmr_applied": True,
        "insurer_filter": insurer_filter,
        "insurer_priority_sampling": insurer_filter is not None
    }
    
    return {**state, "refined": topk, "rank_meta": rank_meta}