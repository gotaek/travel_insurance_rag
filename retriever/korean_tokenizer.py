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
    "의료비", "치료비", "입원", "통원", "응급처치", "후유장해", "수하물지연",
    
    # 특약 관련 키워드
    "특약", "선택특약", "기본특약", "고액특약", "스포츠특약",
    
    # 가입 조건 키워드
    "가입", "가입조건", "가입연령", "보험기간", "보험가입금액",
    
    # 배상 관련 키워드
    "배상", "배상책임", "배상한도", "무한배상", "배상보험"
}

# 여행자보험 동의어(유사어) 정규화 맵
# - 좌변 형태가 등장하면 우변 표준어로 치환하여 매칭 일관성을 높임
SYNONYM_MAP = {
   # ========== 상품명/일반 용어 ==========
    "여행보험": "여행자보험",
    "해외여행보험": "여행자보험",
    "해외보험": "여행자보험",
    "트래블보험": "여행자보험",
    "여행자보장": "여행자보험",
    
    # ========== 휴대품/소지품 계열 ==========
    "휴대물품": "휴대품",
    "휴대품손해": "휴대품",
    "휴대품손해담보": "휴대품",
    "휴대품분실": "휴대품",
    "휴대품도난": "휴대품",
    "소지품": "휴대품",
    "소지품손해": "휴대품",
    "개인소지품": "휴대품",
    "소지물": "휴대품",
    "도난": "휴대품",
    "도품": "휴대품",
    "분실": "휴대품",
    "분실물": "휴대품",
    "파손": "휴대품",
    "손상": "휴대품",
    
    # ========== 배상책임 계열 ==========
    "개인배상책임": "배상책임",
    "일상배상책임": "배상책임",
    "배상": "배상책임",
    "배상책임담보": "배상책임",
    "타인배상": "배상책임",
    "대인배상": "배상책임",
    "대물배상": "배상책임",
    "법률상배상책임": "배상책임",
    "손해배상": "배상책임",
    
    # ========== 의료비/치료비 계열 ==========
    "상해의료비": "의료비",
    "질병의료비": "의료비",
    "해외의료비": "의료비",
    "치료비": "의료비",
    "진료비": "의료비",
    "의료실비": "의료비",
    "실손의료비": "의료비",
    "통원의료비": "통원",
    "통원치료비": "통원",
    "외래진료": "통원",
    "입원의료비": "입원",
    "입원치료비": "입원",
    "입원비": "입원",
    "수술비": "수술",
    "수술의료비": "수술",
    
    # ========== 사망/장해 계열 ==========
    "상해후유장해": "후유장해",
    "후유장애": "후유장해",
    "영구장해": "후유장해",
    "후유증": "후유장해",
    "장해": "후유장해",
    "장애": "후유장해",
    "상해사망": "사망",
    "질병사망": "사망",
    "사망위로금": "사망",
    "유족위로금": "사망",
    
    # ========== 항공/지연 계열 ==========
    "항공기지연": "항공기연착",
    "항공기지연도착": "항공기연착",
    "비행기지연": "항공기연착",
    "항공편지연": "항공기연착",
    "출발지연": "항공기연착",
    "도착지연": "항공기연착",
    "연착": "항공기연착",
    "지연": "항공기연착",
    "비행기납치":"항공기납치",
    "여행지연보장": "여행지연",
    "여행출발지연": "여행지연",
    "일정지연": "여행지연",
    "수하물딜레이": "수하물지연",
    "수하물지연보장": "수하물지연",
    "수하물연착": "수하물지연",
    "짐지연": "수하물지연",
    "화물지연": "수하물지연",
    "수하물분실": "수하물분실",
    "수하물파손": "수하물파손",
    "짐분실": "수하물분실",
    
    # ========== 감염병/질병 계열 ==========
    "코로나": "코로나19",
    "코비드": "코로나19",
    "covid": "코로나19",
    "covid19": "코로나19",
    "신종감염병": "감염병",
    "전염병": "감염병",
    "법정전염병": "감염병",
    "뎅기열": "감염병",
    "말라리아": "감염병",
    "지카바이러스": "감염병",
    
    # ========== 여행 중단/취소 계열 ==========
    "여행중단": "여행취소",
    "여행포기": "여행취소",
    "예약취소": "여행취소",
    "일정취소": "여행취소",
    "여행변경": "일정변경",
    "스케줄변경": "일정변경",
    
    # ========== 구조/송환 계열 ==========
    "긴급구조": "긴급구조송환",
    "긴급이송": "긴급구조송환",
    "긴급후송": "긴급구조송환",
    "응급이송": "긴급구조송환",
    "의료후송": "�긴급구조송환",
    "송환비용": "긴급구조송환",
    "귀국비용": "긴급구조송환",
    "본국송환": "긴급구조송환",
    
    # ========== 특약/부가 계열 ==========
    "선택특약": "특약",
    "부가특약": "특약",
    "추가특약": "특약",
    "옵션": "특약",
    "선택보장": "특약",
    "부가보장": "특약",
    
    # ========== 보험료/가격 계열 ==========
    "보험료": "가격",
    "보험비": "가격",
    "보험금액": "가격",
    "가입비": "가격",
    "비용": "가격",
    "요금": "가격",
    "프리미엄": "가격",
    
    # ========== 보험금/보상 계열 ==========
    "보험금": "보상",
    "보상금": "보상",
    "보상액": "보상",
    "지급금": "보상",
    "보장금액": "보상",
    "지급액": "보상",
    
    # ========== 가입/청약 계열 ==========
    "가입": "청약",
    "신청": "청약",
    "계약": "청약",
    "인수": "청약",
    
    # ========== 청구/신청 계열 ==========
    "보험금청구": "청구",
    "보상청구": "청구",
    "클레임": "청구",
    "신청": "청구",
    
    # ========== 면책/제외 계열 ==========
    "면책사항": "면책",
    "보상제외": "면책",
    "제외사항": "면책",
    "부담보": "면책",
    "면책조항": "면책",
    
    # ========== 기간/일수 계열 ==========
    "보험기간": "가입기간",
    "보장기간": "가입기간",
    "계약기간": "가입기간",
    "여행기간": "가입기간",
    "여행일수": "가입기간",
    
    # ========== 지역/국가 계열 ==========
    "여행지": "목적지",
    "여행국가": "목적지",
    "방문국": "목적지",
    "도착지": "목적지",
    "여행목적지": "목적지",
    
    # ========== 가족/단체 계열 ==========
    "가족형": "가족",
    "가족보험": "가족",
    "가족플랜": "가족",
    "패밀리": "가족",
    "단체형": "단체",
    "그룹": "단체",
    "개인형": "개인",
    "1인": "개인",
    
    # ========== 상해/사고 계열 ==========
    "사고": "상해",
    "부상": "상해",
    "골절": "상해",
    "화상": "상해",
    "외상": "상해",
    
    # ========== 질병 계열 ==========
    "질환": "질병",
    "병": "질병",
    "발병": "질병",
    
    # ========== 약관/조건 계열 ==========
    "약관": "보험약관",
    "보험조건": "보험약관",
    "보험계약": "보험약관",
    "계약조건": "보험약관",
    
    # ========== 할인/혜택 계열 ==========
    "할인": "혜택",
    "할인율": "혜택",
    "디스카운트": "혜택",
    "특가": "혜택",
    "프로모션": "혜택",
    "이벤트": "혜택",
    "리워드": "혜택",
    "캐시백": "혜택",
    "적립": "혜택",
    "포인트": "혜택",
}

# 문구(멀티 토큰) 동의어 정규화 맵
# 공백/형태 차이를 흡수하기 위해 토큰화 이전에 치환
PHRASE_SYNONYM_MAP = {
    # ========== 보장 항목 ==========
    "휴대품 손해": "휴대품",
    "휴대물품 손해": "휴대품",
    "휴대품 분실": "휴대품",
    "휴대품 도난": "휴대품",
    "소지품 손해": "휴대품",
    "개인 소지품": "휴대품",
    
    "개인 배상 책임": "배상책임",
    "일상 배상 책임": "배상책임",
    "법률상 배상 책임": "배상책임",
    "타인 배상 책임": "배상책임",
    
    "여행 지연": "여행지연",
    "여행 출발 지연": "여행지연",
    "일정 지연": "여행지연",
    
    "항공기 지연": "항공기연착",
    "항공기 연착": "항공기연착",
    "비행기 지연": "항공기연착",
    "항공편 지연": "항공기연착",
    "출발 지연": "항공기연착",
    
    "수하물 지연": "수하물지연",
    "수하물 연착": "수하물지연",
    "수하물 분실": "수하물분실",
    "수하물 파손": "수하물파손",
    
    "상해 후유 장해": "후유장해",
    "후유 장애": "후유장해",
    "영구 장해": "후유장해",
    
    "여행 중단": "여행취소",
    "여행 취소": "여행취소",
    "일정 취소": "여행취소",
    "예약 취소": "여행취소",
    
    # ========== 비용/의료 표현 ==========
    "입원 의료비": "입원",
    "입원 치료비": "입원",
    "통원 의료비": "통원",
    "통원 치료비": "통원",
    "외래 진료비": "통원",
    "상해 의료비": "의료비",
    "질병 의료비": "의료비",
    "해외 의료비": "의료비",
    "의료 실비": "의료비",
    "실손 의료비": "의료비",
    
    "수술 비용": "수술",
    "수술 의료비": "수술",
    
    # ========== 사망/보상 ==========
    "상해 사망": "사망",
    "질병 사망": "사망",
    "사망 위로금": "사망",
    "유족 위로금": "사망",
    
    # ========== 구조/송환 ==========
    "긴급 구조": "긴급구조송환",
    "긴급 이송": "긴급구조송환",
    "긴급 후송": "긴급구조송환",
    "응급 이송": "긴급구조송환",
    "의료 후송": "긴급구조송환",
    "본국 송환": "긴급구조송환",
    "귀국 비용": "긴급구조송환",
    
    # ========== 보험금/청구 ==========
    "보험금 청구": "청구",
    "보상금 청구": "청구",
    "보험금 지급": "보상",
    "보상금 지급": "보상",
    "보장 금액": "보상",
    
    # ========== 가입/계약 ==========
    "보험 가입": "청약",
    "계약 신청": "청약",
    "온라인 가입": "청약",
    
    # ========== 기간/조건 ==========
    "보험 기간": "가입기간",
    "보장 기간": "가입기간",
    "계약 기간": "가입기간",
    "여행 기간": "가입기간",
    "여행 일수": "가입기간",
    
    "면책 사항": "면책",
    "보상 제외": "면책",
    "제외 사항": "면책",
    
    # ========== 특약/옵션 ==========
    "선택 특약": "특약",
    "부가 특약": "특약",
    "추가 특약": "특약",
    "선택 보장": "특약",
    "부가 보장": "특약",
    
    # ========== 가족/단체 ==========
    "가족 보험": "가족",
    "가족 플랜": "가족",
    "단체 보험": "단체",
    "개인 보험": "개인",
    
    # ========== 지역/목적 ==========
    "여행 목적지": "목적지",
    "여행 국가": "목적지",
    "방문 국가": "목적지",
    
    # ========== 감염병 ==========
    "코로나 19": "코로나19",
    "코로나19": "코로나19",
    "신종 감염병": "감염병",
    "법정 전염병": "감염병",
    
    # ========== 할인/혜택 ==========
    "할인 혜택": "혜택",
    "가입 혜택": "혜택",
    "이벤트 혜택": "혜택",
    "캐시 백": "혜택",
    "포인트 적립": "혜택",
    
    # ========== 보험사별 상품명 정규화 ==========
    "삼성화재 여행자보험": "여행자보험",
    "현대해상 여행자보험": "여행자보험",
    "DB손해보험 여행자보험": "여행자보험",
    "KB손해보험 여행자보험": "여행자보험",
    "메리츠화재 여행자보험": "여행자보험",
    
    # ========== 레저/활동 ==========
    "레저 활동": "레저",
    "스포츠 활동": "레저",
    "익스트림 스포츠": "레저",
    "위험 활동": "레저",
    
    # ========== 천재지변 ==========
    "자연 재해": "천재지변",
    "자연 재난": "천재지변",
    "천재 지변": "천재지변",
    
    # ========== 테러/전쟁 ==========
    "테러 위험": "테러",
    "전쟁 위험": "전쟁",
    "내란 위험": "전쟁",
}

def _apply_synonym(word: str) -> str:
    """단일 단어 동의어를 표준어로 치환합니다."""
    if not word:
        return word
    # 정확 일치 기반 간단 매핑
    return SYNONYM_MAP.get(word, word)

def _apply_phrase_synonyms(text: str) -> str:
    """문구 동의어를 표준어로 치환합니다."""
    if not text:
        return text
    # 긴 패턴부터 치환하여 부분 매칭 충돌을 방지
    for phrase in sorted(PHRASE_SYNONYM_MAP.keys(), key=len, reverse=True):
        replacement = PHRASE_SYNONYM_MAP[phrase]
        # 공백/형태 차이가 있어도 정규식으로 정확히 해당 구를 치환
        pattern = re.escape(phrase)
        text = re.sub(pattern, replacement, text)
    return text

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
    - 공백 기반 토큰화 후 불용어/숫자/길이 필터링을 적용합니다.
    - 동의어 정규화를 적용해 표현 차이를 흡수합니다.
    """
    if not text:
        return []
    
    # 특수문자 제거 및 정규화
    cleaned_text = _normalize_text(text)
    
    # 공백으로 분리하여 기본 토큰화
    tokens = cleaned_text.split()
    
    # 불용어 제거 및 필터링
    filtered_tokens = _filter_tokens(tokens)
    
    # 동의어 정규화 적용
    normalized_tokens = [_apply_synonym(tok) for tok in filtered_tokens]
    
    return normalized_tokens

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
    
    # 문구 동의어를 우선 치환 (토큰화 이전)
    text = _apply_phrase_synonyms(text)
    
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
    - 조사/어미 제거 후 동의어 정규화를 적용해 표준형으로 변환합니다.
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
    
    normalized = normalized.strip()
    
    # 동의어 정규화 적용
    normalized = _apply_synonym(normalized)
    return normalized

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
    - 토크나이징 → 도메인 키워드 필터링 → 정규화(조사 제거+동의어 맵) → 중복/불용어 제거 → 가중치 적용.
    """
    # 기본 토크나이징
    tokens = tokenize_korean_text(text)
    
    # 도메인 특화 키워드 필터링 및 정규화
    domain_keywords = []
    for token in tokens:
        if _is_insurance_keyword(token):
            # 키워드 정규화(조사 제거 + 동의어 정규화)
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
