"""
Planner Fallback 분류기 단위 테스트
fallback 분류기의 정확도와 로직을 테스트합니다.
"""

import sys
import os
import pytest

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from graph.nodes.planner import _fallback_classify, _analyze_question_context, _determine_web_search_need


@pytest.mark.unit
class TestPlannerFallback:
    """Planner Fallback 분류기 단위 테스트 클래스"""
    
    def test_qa_intent_classification(self):
        """QA intent 분류 테스트"""
        test_cases = [
            {
                "question": "여행자보험 보장 내용이 뭐야?",
                "expected_intent": "qa",
                "description": "기본 QA 질문"
            },
            {
                "question": "보험료는 얼마인가요?",
                "expected_intent": "qa",
                "description": "보험료 관련 QA"
            },
            {
                "question": "가입 조건은 어떻게 되나요?",
                "expected_intent": "qa",
                "description": "가입 조건 QA"
            },
            {
                "question": "휴대품 관련 조항은 어떻게 돼?",
                "expected_intent": "qa",
                "description": "휴대품 조항 QA"
            }
        ]
        
        for case in test_cases:
            result = _fallback_classify(case["question"])
            
            assert result["intent"] == case["expected_intent"], \
                f"❌ {case['description']}: 예상 {case['expected_intent']}, 실제 {result['intent']}"
            
            # 웹 검색 필요성도 검증
            assert isinstance(result["needs_web"], bool)
            
            print(f"✅ {case['description']}: {result['intent']} (웹검색: {result['needs_web']})")
    
    def test_summary_intent_classification(self):
        """Summary intent 분류 테스트"""
        test_cases = [
            {
                "question": "여행자보험 약관을 요약해주세요",
                "expected_intent": "summary",
                "description": "약관 요약 요청"
            },
            {
                "question": "상품 내용을 간단히 정리해주세요",
                "expected_intent": "summary",
                "description": "상품 정리 요청"
            },
            {
                "question": "핵심 내용을 한눈에 보여주세요",
                "expected_intent": "summary",
                "description": "핵심 내용 요약"
            }
        ]
        
        for case in test_cases:
            result = _fallback_classify(case["question"])
            
            assert result["intent"] == case["expected_intent"], \
                f"❌ {case['description']}: 예상 {case['expected_intent']}, 실제 {result['intent']}"
            
            print(f"✅ {case['description']}: {result['intent']} (웹검색: {result['needs_web']})")
    
    def test_compare_intent_classification(self):
        """Compare intent 분류 테스트"""
        test_cases = [
            {
                "question": "DB손해보험과 KB손해보험의 차이점을 비교해주세요",
                "expected_intent": "compare",
                "description": "보험사 비교"
            },
            {
                "question": "여러 보험 상품의 보장 내용을 비교해주세요",
                "expected_intent": "compare",
                "description": "보장 내용 비교"
            },
            {
                "question": "개인용품 보상 규정은 어떻게 되나요?",
                "expected_intent": "qa",  # 수정: 이제 qa로 분류됨
                "description": "개인용품 보상 규정 QA"
            }
        ]
        
        for case in test_cases:
            result = _fallback_classify(case["question"])
            
            assert result["intent"] == case["expected_intent"], \
                f"❌ {case['description']}: 예상 {case['expected_intent']}, 실제 {result['intent']}"
            
            print(f"✅ {case['description']}: {result['intent']} (웹검색: {result['needs_web']})")
    
    def test_recommend_intent_classification(self):
        """Recommend intent 분류 테스트"""
        test_cases = [
            {
                "question": "일본 여행에 추천하는 보험은?",
                "expected_intent": "recommend",
                "description": "여행지별 보험 추천"
            },
            {
                "question": "나에게 적합한 특약을 추천해주세요",
                "expected_intent": "recommend",
                "description": "개인 맞춤 특약 추천"
            },
            {
                "question": "어떤 보험이 가장 좋을까요?",
                "expected_intent": "recommend",
                "description": "최적 보험 추천"
            }
        ]
        
        for case in test_cases:
            result = _fallback_classify(case["question"])
            
            assert result["intent"] == case["expected_intent"], \
                f"❌ {case['description']}: 예상 {case['expected_intent']}, 실제 {result['intent']}"
            
            print(f"✅ {case['description']}: {result['intent']} (웹검색: {result['needs_web']})")
    
    def test_web_search_detection(self):
        """웹 검색 필요성 판단 테스트"""
        test_cases = [
            {
                "question": "2025년 3월 일본 도쿄 여행 보험 추천해주세요",
                "expected_web": True,
                "description": "날짜+지역+추천 조합"
            },
            {
                "question": "현재 도쿄의 안전 상황은 어떤가요?",
                "expected_web": True,
                "description": "실시간 안전 정보"
            },
            {
                "question": "여행자보험 보장 내용이 뭐야?",
                "expected_web": False,
                "description": "일반적인 보장 내용 질문"
            },
            {
                "question": "휴대품 관련 조항은 어떻게 돼?",
                "expected_web": False,
                "description": "보험 조항 비교 (웹 검색 불필요)"
            },
            {
                "question": "여행자보험 가격 비교해주세요",
                "expected_web": True,
                "description": "가격 비교 (웹 검색 필요)"
            }
        ]
        
        for case in test_cases:
            result = _fallback_classify(case["question"])
            
            assert result["needs_web"] == case["expected_web"], \
                f"❌ {case['description']}: 예상 웹검색 {case['expected_web']}, 실제 {result['needs_web']}"
            
            print(f"✅ {case['description']}: 웹검색 {result['needs_web']}")
    
    def test_context_analysis(self):
        """문맥 분석 테스트"""
        test_cases = [
            {
                "question": "여러 보험사의 차이점을 알려주세요",
                "expected_boost": "compare",
                "description": "복수 비교 키워드"
            },
            {
                "question": "전체 약관을 요약해주세요",
                "expected_boost": "summary",
                "description": "전체 요약 키워드"
            },
            {
                "question": "나에게 맞는 보험을 추천해주세요",
                "expected_boost": "recommend",
                "description": "개인화 추천 키워드"
            }
        ]
        
        for case in test_cases:
            context_boost = _analyze_question_context(case["question"])
            # 가장 높은 점수를 받은 intent 확인
            max_intent = max(context_boost, key=context_boost.get)
            assert max_intent == case["expected_boost"], \
                f"❌ {case['description']}: 예상 {case['expected_boost']}, 실제 {max_intent}"
            print(f"✅ {case['description']}: {max_intent} (부스트: {context_boost})")
    
    def test_edge_cases(self):
        """엣지 케이스 테스트"""
        test_cases = [
            {
                "question": "",
                "expected_intent": "qa",
                "description": "빈 질문"
            },
            {
                "question": "안녕하세요",
                "expected_intent": "qa",
                "description": "인사말"
            },
            {
                "question": "보험",
                "expected_intent": "qa",
                "description": "단일 키워드"
            },
            {
                "question": "어떻게 어떻게 어떻게",
                "expected_intent": "qa",
                "description": "반복 키워드"
            }
        ]
        
        for case in test_cases:
            result = _fallback_classify(case["question"])
            
            assert result["intent"] == case["expected_intent"], \
                f"❌ {case['description']}: 예상 {case['expected_intent']}, 실제 {result['intent']}"
            
            print(f"✅ {case['description']}: {result['intent']} (웹검색: {result['needs_web']})")


@pytest.mark.benchmark
def test_fallback_accuracy_benchmark():
    """Fallback 분류기 정확도 벤치마크 테스트"""
    print("\n" + "="*60)
    print("🎯 FALLBACK 분류기 정확도 벤치마크")
    print("="*60)
    
    # 전체 테스트 케이스
    all_test_cases = [
        # QA 테스트 케이스
        ("여행자보험 보장 내용이 뭐야?", "qa"),
        ("보험료는 얼마인가요?", "qa"),
        ("가입 조건은 어떻게 되나요?", "qa"),
        ("휴대품 관련 조항은 어떻게 돼?", "qa"),
        
        # Summary 테스트 케이스
        ("여행자보험 약관을 요약해주세요", "summary"),
        ("상품 내용을 간단히 정리해주세요", "summary"),
        ("핵심 내용을 한눈에 보여주세요", "summary"),
        
        # Compare 테스트 케이스
        ("DB손해보험과 KB손해보험의 차이점을 비교해주세요", "compare"),
        ("여러 보험 상품의 보장 내용을 비교해주세요", "compare"),
        ("개인용품 보상 규정은 어떻게 되나요?", "qa"),  # 수정됨
        
        # Recommend 테스트 케이스
        ("일본 여행에 추천하는 보험은?", "recommend"),
        ("나에게 적합한 특약을 추천해주세요", "recommend"),
        ("어떤 보험이 가장 좋을까요?", "recommend"),
    ]
    
    correct = 0
    total = len(all_test_cases)
    
    for question, expected in all_test_cases:
        result = _fallback_classify(question)
        
        if result["intent"] == expected:
            correct += 1
            print(f"✅ {question[:40]}... → {result['intent']}")
        else:
            print(f"❌ {question[:40]}... → 예상: {expected}, 실제: {result['intent']}")
    
    accuracy = (correct / total) * 100
    print(f"\n📊 정확도: {correct}/{total} ({accuracy:.1f}%)")
    
    # 정확도 기준
    if accuracy >= 90:
        print("🎉 우수한 성능!")
    elif accuracy >= 80:
        print("👍 양호한 성능")
    elif accuracy >= 70:
        print("⚠️ 개선 필요")
    else:
        print("🚨 심각한 문제")
    
    return accuracy


if __name__ == "__main__":
    # 직접 실행 시 벤치마크 테스트
    test_fallback_accuracy_benchmark()
