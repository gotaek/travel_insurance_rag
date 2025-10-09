from typing import Dict, Any, List, Set, Optional
import re
import logging

logger = logging.getLogger(__name__)

def _preprocess_text(text: str) -> Set[str]:
    """
    텍스트를 전처리하여 단어 집합으로 변환.
    
    Args:
        text: 전처리할 텍스트
        
    Returns:
        전처리된 단어들의 집합 (소문자, 특수문자 제거)
    """
    if not text or not isinstance(text, str):
        return set()
    
    # 특수문자 제거 및 소문자 변환, 공백으로 분리
    cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
    words = cleaned.split()
    
    # 빈 문자열과 너무 짧은 단어 제거 (1글자 단어 제외)
    return {word for word in words if len(word) > 1}

def _calculate_word_overlap(evidence_words: Set[str], context_words: Set[str]) -> float:
    """
    두 단어 집합 간의 겹침 비율을 계산.
    
    Args:
        evidence_words: 답변의 증거 단어 집합
        context_words: 컨텍스트 단어 집합
        
    Returns:
        겹침 비율 (0.0 ~ 1.0)
    """
    if not evidence_words:
        return 0.0
    
    intersection = evidence_words.intersection(context_words)
    return len(intersection) / len(evidence_words)

def simple_faithfulness(state: Dict[str, Any]) -> float:
    """
    간단한 신뢰성 지표(0~1): 답변의 증거와 검색된 컨텍스트 간 단어 겹침 비율.
    
    이 함수는 답변이 검색된 문서에 기반하는지를 평가하는 휴리스틱 지표입니다.
    정식 entailment 평가가 아닌 엔지니어링용 근사치입니다.
    
    Args:
        state: RAG 시스템의 상태 딕셔너리
            - draft_answer: 답변 정보 (evidence 키 포함)
            - refined: 정제된 컨텍스트 문서 리스트
            
    Returns:
        float: 신뢰성 점수 (0.0 ~ 1.0)
            - 1.0: 모든 증거가 컨텍스트에 포함됨
            - 0.0: 증거가 없거나 컨텍스트와 겹치지 않음
            
    Note:
        - evidence가 여러 개인 경우 가장 높은 점수를 반환
        - 빈 문자열이나 None 값은 안전하게 처리됨
    """
    # 입력 검증
    if not isinstance(state, dict):
        logger.warning("state가 딕셔너리가 아닙니다.")
        return 0.0
    
    # 답변 정보 추출
    answer = state.get("draft_answer", {})
    if not isinstance(answer, dict):
        logger.warning("draft_answer가 딕셔너리가 아닙니다.")
        return 0.0
    
    # 증거 문장들 추출
    evidence_list: List[str] = answer.get("evidence") or []
    if not evidence_list:
        return 0.0
    
    # 컨텍스트 문서들에서 텍스트 추출
    refined_docs = state.get("refined") or []
    if not refined_docs:
        return 0.0
    
    # 모든 컨텍스트 텍스트를 하나로 합치기
    context_texts = []
    for doc in refined_docs:
        if isinstance(doc, dict) and doc.get("text"):
            context_texts.append(doc["text"])
    
    if not context_texts:
        return 0.0
    
    # 컨텍스트 텍스트를 전처리하여 단어 집합 생성
    full_context = " ".join(context_texts)
    context_words = _preprocess_text(full_context)
    
    if not context_words:
        return 0.0
    
    # 각 증거 문장에 대해 겹침 비율 계산
    overlap_scores = []
    for evidence in evidence_list:
        if not evidence or not isinstance(evidence, str):
            continue
            
        evidence_words = _preprocess_text(evidence)
        if not evidence_words:
            continue
            
        overlap_score = _calculate_word_overlap(evidence_words, context_words)
        overlap_scores.append(overlap_score)
    
    # 가장 높은 겹침 비율 반환 (최소 하나의 증거가 컨텍스트에 기반해야 함)
    return max(overlap_scores) if overlap_scores else 0.0