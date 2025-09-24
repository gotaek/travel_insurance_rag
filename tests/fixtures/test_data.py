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


@pytest.fixture
def websearch_test_data():
    """웹 검색 테스트용 데이터"""
    return {
        "mock_tavily_response": {
            "results": [
                {
                    "url": "https://www.dbinsu.co.kr/travel-insurance",
                    "title": "DB손해보험 여행자보험 보장내용",
                    "content": "해외여행보험의 상세한 보장내용을 안내합니다. 의료비, 휴대품, 여행지연 등 다양한 위험을 보장합니다.",
                    "score": 0.85
                },
                {
                    "url": "https://www.kbinsure.co.kr/products/travel",
                    "title": "KB손해보험 여행자보험 상품안내",
                    "content": "KB손해보험의 여행자보험 상품에 대한 상세 정보를 제공합니다. 보험료, 가입조건, 보장내용을 확인하세요.",
                    "score": 0.82
                },
                {
                    "url": "https://www.naver.com/travel-insurance-guide",
                    "title": "여행자보험 가이드 - 네이버",
                    "content": "여행자보험 선택 가이드와 비교 정보를 제공합니다. 여행 목적에 맞는 보험을 선택하는 방법을 안내합니다.",
                    "score": 0.75
                }
            ]
        },
        "domain_test_urls": {
            "insurance_sites": [
                "https://www.dbinsu.co.kr/travel-insurance",
                "https://www.kbinsure.co.kr/products/travel",
                "https://www.samsungfire.com/insurance/travel",
                "https://www.hyundai.com/insurance/travel"
            ],
            "government_sites": [
                "https://www.fss.or.kr/insurance/travel",
                "https://www.kdi.re.kr/research/travel-insurance",
                "https://www.korea.kr/policy/travel",
                "https://www.visitkorea.or.kr/travel-info"
            ],
            "portal_sites": [
                "https://www.naver.com/travel-insurance",
                "https://www.daum.net/insurance/travel",
                "https://www.kakao.com/insurance"
            ],
            "other_sites": [
                "https://www.example.com/travel-insurance",
                "https://www.unknown-site.com/insurance"
            ]
        },
        "relevance_test_cases": [
            {
                "title": "여행자보험 보장내용 및 보험료 안내",
                "content": "해외여행보험의 보장내용과 보험료에 대한 상세 정보를 제공합니다.",
                "question": "여행자보험 보장내용이 뭐야?",
                "expected_score_range": (0.7, 1.0)
            },
            {
                "title": "일반 뉴스 기사",
                "content": "오늘 날씨가 맑습니다. 경제 뉴스입니다.",
                "question": "여행자보험 보장내용이 뭐야?",
                "expected_score_range": (0.0, 0.3)
            },
            {
                "title": "해외여행 가이드",
                "content": "일본 여행을 위한 준비사항과 관광지 정보",
                "question": "일본 여행 보험 추천",
                "expected_score_range": (0.3, 0.7)
            }
        ]
    }


@pytest.fixture
def websearch_edge_cases():
    """웹 검색 엣지 케이스"""
    return {
        "empty_questions": ["", "   ", "\n", "\t"],
        "very_long_questions": ["여행자보험" * 1000],
        "special_characters": ["???", "!!!", "@@@", "###", "$$$"],
        "numbers_only": ["123456789", "000000000"],
        "mixed_languages": ["여행자보험 travel insurance", "보험 insurance 保険"],
        "unicode_questions": ["여행자보험 🏖️ ✈️", "보험료 💰 💳"]
    }
