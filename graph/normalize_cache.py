"""
Normalization Cache 구현
- 질문 정규화를 통한 캐시 히트율 향상
- 간단하고 효과적인 캐시 전략
"""

import re
import hashlib
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class QuestionNormalizer:
    """질문 정규화 클래스"""
    
    def __init__(self):
        # 동의어 사전
        self.synonyms = {
            # 요청 표현
            "알려주세요": "알려줘",
            "설명해주세요": "알려줘", 
            "말씀해주세요": "알려줘",
            "가르쳐주세요": "알려줘",
            "안내해주세요": "알려줘",
            
            # 의문사
            "어떤": "무엇",
            "어느": "무엇",
            "얼마": "가격",
            "어디": "어느",
            "언제": "몇시",
            "왜": "이유",
            "어떻게": "방법",
            "몇": "가격",
            
            # 조사 정규화
            "은": "는",
            "이": "가",
            "을": "를",
            "의": "의",
            
            # 보험 관련 용어
            "여행자보험": "여행보험",
            "해외여행보험": "여행보험",
            "여행보험": "여행보험",
            
            # 보험사 정규화
            "kb손해보험": "kb보험",
            "kb손보": "kb보험",
            "삼성화재": "삼성보험",
            "현대해상": "현대보험",
            "db손해보험": "db보험",
            "db손보": "db보험",
        }
        
        # 불용어 제거
        self.stop_words = {
            "그런데", "그리고", "그러면", "그래서", "그런", "그러니까",
            "아", "어", "음", "네", "예", "좋아요", "감사합니다"
        }
    
    def normalize_question(self, question: str) -> str:
        """질문 정규화"""
        if not question:
            return ""
        
        # 1. 소문자 변환
        normalized = question.lower()
        
        # 2. 특수문자 제거 (한글, 영문, 숫자, 공백만 유지)
        normalized = re.sub(r'[^\w\s가-힣]', '', normalized)
        
        # 3. 공백 정규화
        normalized = " ".join(normalized.split())
        
        # 4. 동의어 치환
        for old_word, new_word in self.synonyms.items():
            normalized = normalized.replace(old_word, new_word)
        
        # 5. 불용어 제거
        words = normalized.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        normalized = " ".join(filtered_words)
        
        # 6. 중복 공백 제거
        normalized = " ".join(normalized.split())
        
        return normalized.strip()
    
    def generate_normalized_cache_key(self, question: str, prefix: str = "") -> str:
        """정규화된 질문으로 캐시 키 생성"""
        normalized = self.normalize_question(question)
        content_hash = hashlib.md5(normalized.encode()).hexdigest()[:16]
        return f"{prefix}:{content_hash}" if prefix else content_hash
    
    def get_cache_variations(self, question: str) -> list:
        """질문의 다양한 변형들을 반환 (캐시 검색용)"""
        variations = []
        
        # 원본 질문
        variations.append(question)
        
        # 정규화된 질문
        normalized = self.normalize_question(question)
        if normalized != question:
            variations.append(normalized)
        
        # 공백 제거 버전
        no_space = question.replace(" ", "")
        if no_space != question:
            variations.append(no_space)
        
        # 마침표 제거 버전
        no_dot = question.replace(".", "").replace("?", "").replace("!", "")
        if no_dot != question:
            variations.append(no_dot)
        
        return list(set(variations))  # 중복 제거


# 전역 인스턴스
question_normalizer = QuestionNormalizer()


def normalize_question(question: str) -> str:
    """질문 정규화 함수 (편의용)"""
    return question_normalizer.normalize_question(question)


def generate_normalized_cache_key(question: str, prefix: str = "") -> str:
    """정규화된 캐시 키 생성 함수 (편의용)"""
    return question_normalizer.generate_normalized_cache_key(question, prefix)
