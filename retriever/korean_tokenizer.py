"""
한국어 토크나이저 유틸리티 모듈
여행자보험 도메인에 특화된 한국어 텍스트 처리 기능을 제공합니다.
"""

import re
from typing import List, Set
from collections import Counter

# 여행자보험 도메인 특화 키워드 사전
INSURANCE_DOMAIN_KEYWORDS = {
    # 보험 관련 기본 키워드
    "보험", "보장", "특약", "보험료", "보험금", "보험사", "보험상품",
    
    # 여행 관련 키워드
    "여행", "해외여행", "국내여행", "여행자", "여행지", "여행일정",
    
    # 보험사 키워드
    "db손해보험", "kb손해보험", "삼성화재", "현대해상", "카카오페이",
    "db", "kb", "삼성", "현대", "카카오",
    
    # 보장 내용 키워드
    "상해", "질병", "휴대품", "배상책임", "항공기", "여행지연", "항공기연착",
    "의료비", "치료비", "입원", "통원", "응급처치",
    
    # 특약 관련 키워드
    "특약", "선택특약", "기본특약", "고액특약", "스포츠특약",
    
    # 가입 조건 키워드
    "가입", "가입조건", "가입연령", "보험기간", "보험가입금액",
    
    # 배상 관련 키워드
    "배상", "배상책임", "배상한도", "무한배상", "배상보험"
}

# 불용어 (제거할 키워드)
STOP_WORDS = {
    "이", "가", "을", "를", "은", "는", "에", "에서", "로", "으로",
    "와", "과", "의", "도", "만", "까지", "부터", "부터", "까지",
    "그", "그것", "이것", "저것", "이런", "그런", "저런",
    "있", "없", "되", "안", "못", "할", "하는", "한", "된"
}

def tokenize_korean_text(text: str) -> List[str]:
    """
    한국어 텍스트를 토크나이징합니다.
    
    Args:
        text: 토크나이징할 텍스트
        
    Returns:
        토큰화된 단어 리스트
    """
    if not text:
        return []
    
    # 특수문자 제거 및 정규화
    cleaned_text = _normalize_text(text)
    
    # 공백으로 분리하여 기본 토큰화
    tokens = cleaned_text.split()
    
    # 불용어 제거 및 필터링
    filtered_tokens = _filter_tokens(tokens)
    
    return filtered_tokens

def _normalize_text(text: str) -> str:
    """
    텍스트를 정규화합니다.
    
    Args:
        text: 원본 텍스트
        
    Returns:
        정규화된 텍스트
    """
    # 소문자 변환
    text = text.lower()
    
    # 특수문자 제거 (한글, 영문, 숫자만 유지)
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    
    # 연속된 공백을 단일 공백으로 변환
    text = re.sub(r'\s+', ' ', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text

def _normalize_keyword(keyword: str) -> str:
    """
    키워드를 정규화합니다.
    
    Args:
        keyword: 원본 키워드
        
    Returns:
        정규화된 키워드
    """
    # 조사, 어미 제거 (간단한 정규화)
    normalized = keyword
    
    # 일반적인 조사/어미 패턴 제거
    patterns_to_remove = [
        r'의$', r'에$', r'을$', r'를$', r'이$', r'가$', r'은$', r'는$',
        r'과$', r'와$', r'로$', r'으로$', r'에서$', r'부터$', r'까지$',
        r'에$', r'에서$', r'에게$', r'한테$', r'께$', r'에$', r'에게$'
    ]
    
    for pattern in patterns_to_remove:
        normalized = re.sub(pattern, '', normalized)
    
    return normalized.strip()

def _filter_tokens(tokens: List[str]) -> List[str]:
    """
    토큰을 필터링합니다.
    
    Args:
        tokens: 토큰 리스트
        
    Returns:
        필터링된 토큰 리스트
    """
    filtered_tokens = []
    
    for token in tokens:
        # 길이 필터링 (2글자 이상, 15글자 이하)
        if not (2 <= len(token) <= 15):
            continue
        
        # 불용어 제거
        if token in STOP_WORDS:
            continue
        
        # 숫자만으로 구성된 토큰 제거
        if token.isdigit():
            continue
        
        # 한 글자 토큰 제거 (한글 제외)
        if len(token) == 1 and not _is_korean_char(token):
            continue
        
        filtered_tokens.append(token)
    
    return filtered_tokens

def _is_korean_char(char: str) -> bool:
    """
    문자가 한글인지 확인합니다.
    
    Args:
        char: 확인할 문자
        
    Returns:
        한글 여부
    """
    return '가' <= char <= '힣'

def extract_insurance_keywords(text: str, min_frequency: int = 1) -> List[str]:
    """
    여행자보험 도메인에 특화된 키워드를 추출합니다.
    중복 제거, 불용어 제거, 품사 필터링을 적용합니다.
    
    Args:
        text: 분석할 텍스트
        min_frequency: 최소 빈도수
        
    Returns:
        추출된 키워드 리스트 (빈도순 정렬)
    """
    # 기본 토크나이징
    tokens = tokenize_korean_text(text)
    
    # 도메인 특화 키워드 필터링 및 정규화
    domain_keywords = []
    for token in tokens:
        if _is_insurance_keyword(token):
            # 키워드 정규화
            normalized_token = _normalize_keyword(token)
            if normalized_token and len(normalized_token) >= 2:  # 최소 길이 체크
                domain_keywords.append(normalized_token)
    
    # 중복 제거 및 불용어 제거
    unique_keywords = _remove_duplicates_and_stopwords(domain_keywords)
    
    # 빈도 계산
    keyword_counts = Counter(unique_keywords)
    
    # 최소 빈도수 이상인 키워드만 선택
    filtered_keywords = [
        keyword for keyword, count in keyword_counts.most_common()
        if count >= min_frequency
    ]
    
    # 도메인 키워드 가중치 적용 (화이트리스트 우선)
    weighted_keywords = _apply_domain_weights(filtered_keywords)
    
    return weighted_keywords

def _remove_duplicates_and_stopwords(keywords: List[str]) -> List[str]:
    """
    중복 제거 및 불용어 제거를 수행합니다.
    
    Args:
        keywords: 키워드 리스트
        
    Returns:
        정제된 키워드 리스트
    """
    # 중복 제거
    unique_keywords = list(set(keywords))
    
    # 불용어 제거
    filtered_keywords = []
    for keyword in unique_keywords:
        if not _is_stopword(keyword):
            filtered_keywords.append(keyword)
    
    return filtered_keywords

def _is_stopword(keyword: str) -> bool:
    """
    키워드가 불용어인지 확인합니다.
    
    Args:
        keyword: 확인할 키워드
        
    Returns:
        불용어 여부
    """
    # 기본 불용어
    basic_stopwords = {
        "이", "가", "을", "를", "은", "는", "에", "에서", "로", "으로",
        "와", "과", "의", "도", "만", "까지", "부터", "부터", "까지",
        "그", "그것", "이것", "저것", "이런", "그런", "저런",
        "있", "없", "되", "안", "못", "할", "하는", "한", "된"
    }
    
    if keyword in basic_stopwords:
        return True
    
    # 길이가 너무 짧거나 긴 경우
    if len(keyword) < 2 or len(keyword) > 15:
        return True
    
    # 숫자만으로 구성된 경우
    if keyword.isdigit():
        return True
    
    return False

def _apply_domain_weights(keywords: List[str]) -> List[str]:
    """
    도메인 키워드 가중치를 적용합니다.
    
    Args:
        keywords: 키워드 리스트
        
    Returns:
        가중치가 적용된 키워드 리스트
    """
    # 도메인 화이트리스트 (높은 가중치)
    domain_whitelist = {
        "여행자보험", "해외여행보험", "여행보험", "보험", "보장", "특약",
        "상해보장", "질병보장", "휴대품보장", "배상책임", "의료비",
        "DB손해보험", "KB손해보험", "삼성화재", "현대해상", "카카오페이"
    }
    
    # 가중치 적용
    weighted_keywords = []
    
    # 화이트리스트 키워드 우선 추가
    for keyword in keywords:
        if keyword in domain_whitelist:
            weighted_keywords.append(keyword)
    
    # 나머지 키워드 추가
    for keyword in keywords:
        if keyword not in domain_whitelist:
            weighted_keywords.append(keyword)
    
    return weighted_keywords

def _is_insurance_keyword(token: str) -> bool:
    """
    토큰이 여행자보험 도메인 키워드인지 확인합니다.
    
    Args:
        token: 확인할 토큰
        
    Returns:
        도메인 키워드 여부
    """
    # 정확히 일치하는 키워드 확인
    if token in INSURANCE_DOMAIN_KEYWORDS:
        return True
    
    # 부분 일치하는 키워드 확인
    for domain_keyword in INSURANCE_DOMAIN_KEYWORDS:
        if domain_keyword in token or token in domain_keyword:
            return True
    
    # 패턴 매칭 (더 유연한 매칭)
    insurance_patterns = [
        r'.*보험.*', r'.*여행.*', r'.*해외.*', r'.*보장.*', r'.*특약.*',
        r'.*손해.*', r'.*화재.*', r'.*배상.*', r'.*의료.*', r'.*치료.*',
        r'.*상해.*', r'.*질병.*', r'.*휴대품.*', r'.*항공기.*', r'.*여행지연.*'
    ]
    
    for pattern in insurance_patterns:
        if re.search(pattern, token, re.IGNORECASE):
            return True
    
    return False

def calculate_keyword_relevance(
    text: str, 
    target_texts: List[str]
) -> float:
    """
    텍스트와 대상 텍스트들 간의 관련성을 계산합니다.
    
    Args:
        text: 분석할 텍스트
        target_texts: 대상 텍스트 리스트
        
    Returns:
        관련성 점수 (0.0 ~ 1.0)
    """
    if not text or not target_texts:
        return 0.0
    
    # 텍스트에서 키워드 추출
    text_keywords = set(extract_insurance_keywords(text, min_frequency=1))
    
    if not text_keywords:
        return 0.0
    
    # 모든 대상 텍스트에서 키워드 추출
    all_target_keywords = set()
    for target_text in target_texts:
        target_keywords = extract_insurance_keywords(target_text, min_frequency=1)
        all_target_keywords.update(target_keywords)
    
    if not all_target_keywords:
        return 0.0
    
    # 공통 키워드 개수 계산
    common_keywords = text_keywords.intersection(all_target_keywords)
    
    # 관련성 점수 계산 (교집합 크기 / 합집합 크기)
    union_keywords = text_keywords.union(all_target_keywords)
    relevance_score = len(common_keywords) / len(union_keywords) if union_keywords else 0.0
    
    return min(relevance_score, 1.0)

def get_keyword_weights(keywords: List[str]) -> dict:
    """
    키워드별 가중치를 계산합니다.
    
    Args:
        keywords: 키워드 리스트
        
    Returns:
        키워드별 가중치 딕셔너리
    """
    if not keywords:
        return {}
    
    # 빈도 계산
    keyword_counts = Counter(keywords)
    total_count = sum(keyword_counts.values())
    
    # 가중치 계산 (빈도 기반)
    weights = {}
    for keyword, count in keyword_counts.items():
        weights[keyword] = count / total_count
    
    return weights
