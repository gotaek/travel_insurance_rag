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

def _extract_insurers_from_question(question: str) -> List[str]:
    """
    질문에서 보험사명을 추출합니다. 컨텍스트를 고려한 정확한 매핑을 수행합니다.
    """
    if not question:
        return []
    
    question_lower = question.lower()
    insurers = []
    
    # 개선된 보험사명 매핑 테이블 (컨텍스트 기반)
    insurer_mapping = {
        "카카오페이": {
            "exact": ["카카오페이", "카카오페이보험"],
            "partial": ["카카오"],
            "context": ["카카오페이", "카카오"]
        },
        "현대해상": {
            "exact": ["현대해상", "현대해상보험"],
            "partial": ["현대"],
            "context": ["현대해상", "현대"]
        },
        "db손해보험": {
            "exact": ["db손해보험", "db손보", "db손해보험"],
            "partial": ["db"],
            "context": ["db손해보험", "db손보", "db"]
        },
        "kb손해보험": {
            "exact": ["kb손해보험", "kb손보", "kb손해보험"],
            "partial": ["kb"],
            "context": ["kb손해보험", "kb손보", "kb"]
        },
        "삼성화재": {
            "exact": ["삼성화재", "삼성화재보험"],
            "partial": ["삼성"],
            "context": ["삼성화재", "삼성"]
        }
    }
    
    # 1단계: 정확한 매칭 우선 검색
    for standard_name, patterns in insurer_mapping.items():
        for exact_pattern in patterns["exact"]:
            if exact_pattern in question_lower:
                if standard_name not in insurers:
                    insurers.append(standard_name)
                    break
    
    # 2단계: 컨텍스트 기반 부분 매칭 (보험 관련 키워드와 함께 사용된 경우)
    if not insurers:  # 정확한 매칭이 없을 때만 부분 매칭 수행
        insurance_context_keywords = ["보험", "여행자보험", "여행보험", "보장", "약관", "상품"]
        
        for standard_name, patterns in insurer_mapping.items():
            for partial_pattern in patterns["partial"]:
                if partial_pattern in question_lower:
                    # 보험 관련 컨텍스트 확인
                    has_insurance_context = any(
                        context_kw in question_lower for context_kw in insurance_context_keywords
                    )
                    
                    if has_insurance_context and standard_name not in insurers:
                        insurers.append(standard_name)
                        break
    
    # 3단계: 질문 패턴 기반 추론 (예: "DB 여행자 보험" -> "DB손해보험")
    if not insurers:
        question_words = question_lower.split()
        for i, word in enumerate(question_words):
            if word in ["db", "kb"]:
                # 다음 단어가 보험 관련인지 확인
                if i + 1 < len(question_words):
                    next_word = question_words[i + 1]
                    if any(insurance_kw in next_word for insurance_kw in ["보험", "여행자", "여행"]):
                        if word == "db" and "db손해보험" not in insurers:
                            insurers.append("db손해보험")
                        elif word == "kb" and "kb손해보험" not in insurers:
                            insurers.append("kb손해보험")
    
    return insurers

def _rerank_with_advanced_scoring(passages: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
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
    
    # 질문에서 보험사명 추출
    target_insurers = _extract_insurers_from_question(question)
    
    reranked = []
    for passage in passages:
        text = passage.get("text", "").lower()
        title = passage.get("title", "").lower()
        insurer = passage.get("insurer", "").lower()
        
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
        
        # 보험사별 우선순위 가중치 (강화된 로직)
        insurer_boost = 0.0
        insurer_match_quality = 0.0
        
        if target_insurers and insurer:
            # 정확한 보험사명 매칭 확인
            is_target_insurer = any(
                target_insurer.lower() in insurer or insurer in target_insurer.lower()
                for target_insurer in target_insurers
            )
            
            if is_target_insurer:
                # 기본 보험사 부스트
                insurer_boost = 0.4  # 30% -> 40%로 증가
                
                # 보험사명 매칭 품질 점수 (정확한 매칭에 더 높은 점수)
                for target_insurer in target_insurers:
                    if target_insurer.lower() == insurer.lower():
                        insurer_match_quality = 0.2  # 정확한 매칭
                        break
                    elif target_insurer.lower() in insurer.lower() or insurer.lower() in target_insurer.lower():
                        insurer_match_quality = 0.1  # 부분 매칭
                        break
        
        # 질문에서 보험사명이 직접 언급된 경우 추가 가중치
        direct_mention_bonus = 0.0
        if target_insurers:
            for target_insurer in target_insurers:
                if target_insurer.lower() in question_lower:
                    direct_mention_bonus = 0.15  # 직접 언급 보너스
                    break
        
        # 최종 점수 계산 (보험사명 매칭 가중치 강화)
        final_score = (
            base_score * 0.4 +           # 기본 검색 점수 (가중치 감소: 0.5 -> 0.4)
            keyword_score * 0.2 +        # 키워드 매칭 (가중치 감소: 0.25 -> 0.2)
            quality_score * 0.1 +        # 문서 품질
            insurance_bonus +            # 보험 전문성 보너스
            insurer_boost +              # 보험사 우선순위 가중치
            insurer_match_quality +      # 보험사명 매칭 품질 점수
            direct_mention_bonus         # 직접 언급 보너스
        )
        
        # 점수 업데이트
        passage_copy = dict(passage)
        passage_copy["score"] = min(final_score, 1.0)
        passage_copy["rerank_score"] = final_score
        passage_copy["keyword_matches"] = text_matches + title_matches
        passage_copy["insurer_boost"] = insurer_boost > 0
        passage_copy["target_insurer"] = insurer_boost > 0
        passage_copy["insurer_match_quality"] = insurer_match_quality
        passage_copy["direct_mention_bonus"] = direct_mention_bonus
        passage_copy["score_breakdown"] = {
            "base_score": base_score * 0.4,
            "keyword_score": keyword_score * 0.2,
            "quality_score": quality_score * 0.1,
            "insurance_bonus": insurance_bonus,
            "insurer_boost": insurer_boost,
            "insurer_match_quality": insurer_match_quality,
            "direct_mention_bonus": direct_mention_bonus
        }
        reranked.append(passage_copy)
    
    return reranked

def _apply_mmr(passages: List[Dict[str, Any]], question: str, lambda_param: float = 0.7) -> List[Dict[str, Any]]:
    """
    MMR (Maximal Marginal Relevance) 적용 - 보험사별 그룹핑 고려
    - 관련성과 다양성의 균형
    - 보험사별 문서 그룹핑을 고려한 다양성 확보
    - 중복 내용 제거
    """
    if not passages:
        return passages
    
    # 점수 기준 정렬
    passages.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    
    # 보험사별 그룹핑
    insurer_groups = {}
    for passage in passages:
        insurer = passage.get("insurer", "unknown").lower()
        if insurer not in insurer_groups:
            insurer_groups[insurer] = []
        insurer_groups[insurer].append(passage)
    
    selected = []
    remaining = passages.copy()
    
    # 첫 번째 문서는 가장 높은 점수
    if remaining:
        selected.append(remaining.pop(0))
    
    # 보험사별 그룹 통계
    selected_insurers = set()
    if selected:
        selected_insurers.add(selected[0].get("insurer", "unknown").lower())
    
    # MMR 알고리즘 적용 (보험사별 그룹핑 고려)
    while remaining and len(selected) < 5:  # 최대 5개 선택
        best_idx = 0
        best_mmr_score = -1
        
        for i, candidate in enumerate(remaining):
            # 관련성 점수
            relevance_score = candidate.get("score", 0.0)
            
            # 보험사별 다양성 보너스
            candidate_insurer = candidate.get("insurer", "unknown").lower()
            insurer_diversity_bonus = 0.0
            
            # 새로운 보험사인 경우 다양성 보너스
            if candidate_insurer not in selected_insurers:
                insurer_diversity_bonus = 0.1  # 보험사 다양성 보너스
            
            # 다양성 점수 (이미 선택된 문서들과의 유사도)
            max_similarity = 0.0
            for selected_doc in selected:
                similarity = _calculate_similarity(candidate, selected_doc)
                max_similarity = max(max_similarity, similarity)
            
            # 보험사별 그룹 내 유사도 계산
            same_insurer_similarity = 0.0
            if candidate_insurer in insurer_groups:
                for doc in insurer_groups[candidate_insurer]:
                    if doc in selected:
                        similarity = _calculate_similarity(candidate, doc)
                        same_insurer_similarity = max(same_insurer_similarity, similarity)
            
            # MMR 점수 계산 (보험사별 그룹핑 고려)
            # 같은 보험사 내에서는 유사도 패널티를 줄임
            adjusted_similarity = max_similarity * 0.7 + same_insurer_similarity * 0.3
            mmr_score = lambda_param * relevance_score - (1 - lambda_param) * adjusted_similarity + insurer_diversity_bonus
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = i
        
        # 최적 문서 선택
        chosen_doc = remaining.pop(best_idx)
        selected.append(chosen_doc)
        
        # 선택된 보험사 업데이트
        chosen_insurer = chosen_doc.get("insurer", "unknown").lower()
        selected_insurers.add(chosen_insurer)
    
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
    """
    passages = state.get("passages", [])
    question = state.get("question", "")
    
    if not passages:
        return {**state, "refined": []}

    # 1. 중복 제거
    deduped = _dedup(passages)
    
    # 2. 고급 리랭크 (질문-문서 간 정교한 관련성 계산)
    logger.info(f"전통적 리랭크 사용: {len(deduped)}개 후보")
    reranked = _rerank_with_advanced_scoring(deduped, question)
    
    # 3. MMR 적용 (다양성 확보) - 보험사명 추출 여부에 따른 동적 조정
    target_insurers = _extract_insurers_from_question(question)
    lambda_param = 0.8 if target_insurers else 0.7  # 보험사명이 있으면 관련성 중시
    diverse = _apply_mmr(reranked, question, lambda_param=lambda_param)
    
    # 4. 품질 필터링
    filtered = _quality_filter(diverse)
    
    # 5. 최종 정렬 및 Top-k 선택
    sorted_passages = _sort_by_score(filtered)
    topk = sorted_passages[:5]
    
    # 보험사별 통계 계산
    target_insurers = _extract_insurers_from_question(question)
    insurer_stats = {}
    if target_insurers:
        for insurer in target_insurers:
            count = sum(1 for p in topk if p.get("target_insurer", False) and insurer.lower() in p.get("insurer", "").lower())
            insurer_stats[insurer] = count
    
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
        "target_insurers": target_insurers,
        "insurer_boost_applied": len(target_insurers) > 0,
        "insurer_stats": insurer_stats
    }
    
    return {**state, "refined": topk, "rank_meta": rank_meta}