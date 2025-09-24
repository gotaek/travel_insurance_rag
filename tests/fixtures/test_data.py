"""
테스트 데이터 및 픽스처
"""

import pytest


@pytest.fixture
def sample_questions():
    """샘플 질문 데이터"""
    return {
        "qa": [
            "여행자보험 보장 내용이 뭐야?",
            "보험료는 얼마인가요?",
            "가입 조건은 어떻게 되나요?",
            "보험 기간은 얼마나 되나요?"
        ],
        "summary": [
            "여행자보험 약관을 요약해주세요",
            "상품 내용을 간단히 정리해주세요",
            "핵심 내용을 한눈에 보여주세요",
            "전체 약관을 요약해주세요"
        ],
        "compare": [
            "DB손해보험과 KB손해보험의 차이점을 비교해주세요",
            "휴대품 관련 조항은 어떻게 돼?",
            "여러 보험 상품의 보장 내용을 비교해주세요",
            "개인용품 보상 규정은 어떻게 되나요?"
        ],
        "recommend": [
            "일본 여행에 추천하는 보험은?",
            "나에게 적합한 특약을 추천해주세요",
            "어떤 보험이 가장 좋을까요?",
            "2025년 3월 일본 도쿄 여행에 추천하는 보험은?"
        ]
    }


@pytest.fixture
def web_search_questions():
    """웹 검색이 필요한 질문들"""
    return [
        "2025년 3월 일본 도쿄 여행 보험 추천해주세요",
        "현재 도쿄의 안전 상황은 어떤가요?",
        "최신 여행 제한 사항을 알려주세요",
        "현재 일본의 코로나 상황은 어떤가요?"
    ]


@pytest.fixture
def edge_case_questions():
    """엣지 케이스 질문들"""
    return [
        "",
        "안녕하세요",
        "보험",
        "어떻게 어떻게 어떻게",
        "???",
        "123456789",
        "a" * 1000  # 매우 긴 질문
    ]


@pytest.fixture
def expected_classifications():
    """예상 분류 결과"""
    return {
        "여행자보험 보장 내용이 뭐야?": {"intent": "qa", "needs_web": False},
        "휴대품 관련 조항은 어떻게 돼?": {"intent": "compare", "needs_web": False},
        "일본 여행에 추천하는 보험은?": {"intent": "recommend", "needs_web": True},
        "여행자보험 약관을 요약해주세요": {"intent": "summary", "needs_web": False},
        "DB손해보험과 KB손해보험의 차이점을 비교해주세요": {"intent": "compare", "needs_web": False}
    }


@pytest.fixture
def performance_benchmark_questions():
    """성능 벤치마크용 질문들"""
    return [
        "여행자보험 보장 내용이 뭐야?",
        "휴대품 관련 조항은 어떻게 돼?",
        "일본 여행에 추천하는 보험은?",
        "여행자보험 약관을 요약해주세요",
        "DB손해보험과 KB손해보험의 차이점을 비교해주세요",
        "보험료는 얼마인가요?",
        "가입 조건은 어떻게 되나요?",
        "상품 내용을 간단히 정리해주세요",
        "핵심 내용을 한눈에 보여주세요",
        "여러 보험 상품의 보장 내용을 비교해주세요"
    ] * 5  # 50개 질문
