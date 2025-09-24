"""
Planner 노드 통합 테스트 파일
planner_node의 전체적인 동작과 JSON 반환 형태를 테스트합니다.
"""

import sys
import os
import pytest
from unittest.mock import patch

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from graph.nodes.planner import planner_node


@pytest.mark.integration
class TestPlannerIntegration:
    """Planner 노드 통합 테스트 클래스"""
    
    def test_planner_node_json_structure(self):
        """planner_node의 JSON 반환 구조 테스트"""
        state = {
            "question": "여행자보험 보장 내용이 뭐야?",
            "session_id": "test_session_123",
            "user_id": "test_user"
        }
        
        result = planner_node(state)
        
        # 필수 필드 검증
        required_fields = ["intent", "needs_web", "plan", "classification_reasoning"]
        for field in required_fields:
            assert field in result, f"필수 필드 '{field}'가 누락되었습니다"
        
        # 기존 state 필드들이 그대로 전달되는지 검증
        assert result["question"] == state["question"]
        assert result["session_id"] == state["session_id"]
        assert result["user_id"] == state["user_id"]
        
        # intent 유효성 검증
        valid_intents = ["qa", "summary", "compare", "recommend"]
        assert result["intent"] in valid_intents, f"잘못된 intent: {result['intent']}"
        
        # needs_web 타입 검증
        assert isinstance(result["needs_web"], bool), "needs_web은 boolean이어야 합니다"
        
        # plan 구조 검증
        assert isinstance(result["plan"], list), "plan은 리스트여야 합니다"
        assert len(result["plan"]) >= 4, "plan은 최소 4개 요소를 가져야 합니다"
        assert result["plan"][0] == "planner", "plan의 첫 번째 요소는 'planner'여야 합니다"
        
        print(f"✅ JSON 구조 검증 통과: {result['intent']} (웹검색: {result['needs_web']})")
    
    def test_plan_generation(self):
        """실행 계획 생성 테스트"""
        test_cases = [
            {
                "question": "여행자보험 보장 내용이 뭐야?",
                "expected_plan_pattern": ["planner", "search", "rank_filter", "verify_refine", "answer:qa"],
                "description": "QA 질문 (웹검색 없음)"
            },
            {
                "question": "2025년 3월 일본 도쿄 여행 보험 추천해주세요",
                "expected_plan_pattern": ["planner", "websearch", "search", "rank_filter", "verify_refine", "answer:recommend"],
                "description": "추천 질문 (웹검색 있음)"
            }
        ]
        
        for case in test_cases:
            state = {"question": case["question"]}
            result = planner_node(state)
            
            plan = result["plan"]
            
            # 기본 구조 검증
            assert plan[0] == "planner", "plan의 첫 번째는 planner여야 합니다"
            assert plan[-1].startswith("answer:"), "plan의 마지막은 answer:로 시작해야 합니다"
            
            # 웹 검색 필요성에 따른 plan 구조 검증
            if result["needs_web"]:
                assert "websearch" in plan, "웹 검색이 필요하면 plan에 websearch가 포함되어야 합니다"
                assert plan[1] == "websearch", "웹 검색이 필요하면 plan의 두 번째는 websearch여야 합니다"
            else:
                assert "websearch" not in plan, "웹 검색이 불필요하면 plan에 websearch가 포함되지 않아야 합니다"
                assert plan[1] == "search", "웹 검색이 불필요하면 plan의 두 번째는 search여야 합니다"
            
            print(f"✅ {case['description']}: {plan}")
    
    def test_state_preservation(self):
        """State 보존 테스트 - 기존 state 필드들이 그대로 전달되는지 확인"""
        original_state = {
            "question": "테스트 질문",
            "session_id": "test_session_123",
            "user_id": "test_user",
            "conversation_context": {"test": "context"},
            "custom_field": "custom_value"
        }
        
        result = planner_node(original_state)
        
        # 기존 state의 모든 필드가 보존되는지 확인
        for key, value in original_state.items():
            assert key in result, f"기존 state 필드 '{key}'가 누락되었습니다"
            assert result[key] == value, f"기존 state 필드 '{key}'의 값이 변경되었습니다"
        
        # 새로 추가된 필드들도 있는지 확인
        new_fields = ["intent", "needs_web", "plan", "classification_reasoning"]
        for field in new_fields:
            assert field in result, f"새로운 필드 '{field}'가 누락되었습니다"
        
        print("✅ State 보존 검증 통과")
    
    @patch('graph.nodes.planner._llm_classify_intent')
    def test_llm_fallback_integration(self, mock_llm_classify):
        """LLM 실패 시 fallback 통합 테스트"""
        # LLM 호출 실패 시뮬레이션
        mock_llm_classify.side_effect = Exception("LLM 호출 실패")
        
        state = {"question": "휴대품 관련 조항은 어떻게 돼?"}
        result = planner_node(state)
        
        # fallback이 작동하여 결과가 반환되는지 확인
        assert "intent" in result
        assert "needs_web" in result
        assert "plan" in result
        assert "classification_reasoning" in result
        
        # intent가 유효한 값인지 확인
        valid_intents = ["qa", "summary", "compare", "recommend"]
        assert result["intent"] in valid_intents
        
        print(f"✅ LLM fallback 통합 테스트 통과: {result['intent']}")
    
    def test_edge_cases_integration(self):
        """엣지 케이스 통합 테스트"""
        edge_cases = [
            {
                "question": "",
                "description": "빈 질문"
            },
            {
                "question": "안녕하세요",
                "description": "인사말"
            },
            {
                "question": "보험",
                "description": "단일 키워드"
            },
            {
                "question": "어떻게 어떻게 어떻게",
                "description": "반복 키워드"
            }
        ]
        
        for case in edge_cases:
            state = {"question": case["question"]}
            result = planner_node(state)
            
            # 기본 검증 (예외가 발생하지 않아야 함)
            assert "intent" in result
            assert "needs_web" in result
            assert "plan" in result
            assert "classification_reasoning" in result
            
            # intent가 유효한 값인지 확인
            valid_intents = ["qa", "summary", "compare", "recommend"]
            assert result["intent"] in valid_intents
            
            print(f"✅ {case['description']}: {result['intent']} (웹검색: {result['needs_web']})")
    
    def test_real_world_scenarios(self):
        """실제 사용 시나리오 테스트"""
        test_scenarios = [
            {
                "question": "휴대품 관련 조항은 어떻게 돼?",
                "description": "휴대품 조항 질문"
            },
            {
                "question": "2025년 3월 일본 도쿄 여행에 추천하는 보험은?",
                "description": "지역별 보험 추천"
            },
            {
                "question": "여행자보험 약관을 요약해주세요",
                "description": "약관 요약 요청"
            },
            {
                "question": "보험료는 얼마인가요?",
                "description": "기본 QA 질문"
            }
        ]
        
        for scenario in test_scenarios:
            state = {"question": scenario["question"]}
            result = planner_node(state)
            
            # 기본 검증
            assert "intent" in result
            assert "needs_web" in result
            assert "plan" in result
            assert "classification_reasoning" in result
            
            # intent 유효성 검증
            valid_intents = ["qa", "summary", "compare", "recommend"]
            assert result["intent"] in valid_intents
            
            print(f"✅ {scenario['description']}: {result['intent']} (웹검색: {result['needs_web']})")


@pytest.mark.integration
@pytest.mark.slow
def test_performance_benchmark():
    """성능 벤치마크 테스트"""
    import time
    
    test_questions = [
        "여행자보험 보장 내용이 뭐야?",
        "휴대품 관련 조항은 어떻게 돼?",
        "일본 여행에 추천하는 보험은?",
        "여행자보험 약관을 요약해주세요",
        "DB손해보험과 KB손해보험의 차이점을 비교해주세요"
    ] * 10  # 50개 질문으로 테스트
    
    start_time = time.time()
    
    for question in test_questions:
        state = {"question": question}
        result = planner_node(state)
        assert "intent" in result
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(test_questions)
    
    print(f"\n📊 성능 벤치마크 결과:")
    print(f"   총 질문 수: {len(test_questions)}")
    print(f"   총 소요 시간: {total_time:.2f}초")
    print(f"   평균 처리 시간: {avg_time:.3f}초/질문")
    
    # 성능 기준 검증
    assert avg_time < 0.1, f"평균 처리 시간이 너무 느림: {avg_time:.3f}초"
    print("✅ 성능 기준 통과!")


if __name__ == "__main__":
    # 직접 실행 시 성능 벤치마크 테스트
    test_performance_benchmark()
