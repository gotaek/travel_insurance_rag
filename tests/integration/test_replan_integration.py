"""
Replan 노드 통합 테스트
실제 LLM과 함께 replan_node의 전체 워크플로우를 테스트합니다.
"""

import sys
import os
import pytest
from unittest.mock import patch, Mock
import json
import logging

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from graph.nodes.replan import replan_node, _generate_replan_query, _fallback_replan


@pytest.mark.integration
class TestReplanIntegration:
    """Replan 노드 통합 테스트 클래스"""
    
    @pytest.fixture
    def basic_replan_state(self):
        """기본 재검색 상태"""
        return {
            "question": "여행자보험 보상금액은 얼마인가요?",
            "quality_feedback": "답변이 부족합니다. 더 구체적인 정보가 필요합니다.",
            "replan_query": "여행자보험 보상금액 상세 정보",
            "replan_count": 1,
            "max_replan_attempts": 3,
            "intent": "qa",
            "existing_field": "기존 값"
        }
    
    @pytest.fixture
    def web_search_needed_state(self):
        """웹 검색이 필요한 재검색 상태"""
        return {
            "question": "2024년 여행자보험 최신 가격은?",
            "quality_feedback": "최신 정보가 필요합니다. 실시간 가격 정보를 찾아주세요.",
            "replan_query": "2024년 여행자보험 최신 가격 정보",
            "replan_count": 0,
            "max_replan_attempts": 3,
            "intent": "compare"
        }
    
    @pytest.fixture
    def low_quality_feedback_state(self):
        """저품질 피드백 상태"""
        return {
            "question": "여행자보험 가입조건은?",
            "quality_feedback": "답변이 부정확합니다. 정확한 가입조건 정보가 필요합니다.",
            "replan_query": "여행자보험 가입조건 정확한 정보",
            "replan_count": 2,
            "max_replan_attempts": 3,
            "intent": "qa"
        }
    
    def test_replan_node_integration_basic(self, basic_replan_state):
        """기본 재검색 통합 테스트"""
        result = replan_node(basic_replan_state)
        
        # 기본 결과 검증
        assert "question" in result
        assert "needs_web" in result
        assert "replan_count" in result
        assert "max_replan_attempts" in result
        assert "plan" in result
        
        # 상태 보존 검증
        assert result["existing_field"] == "기존 값"
        assert result["intent"] == "qa"
        
        # 재검색 카운터 증가 확인
        assert result["replan_count"] == 2
        
        # 플랜 구조 확인
        plan = result["plan"]
        assert isinstance(plan, list)
        assert len(plan) == 6
        assert "replan" in plan
        assert "websearch" in plan
        assert "search" in plan
        assert "rank_filter" in plan
        assert "verify_refine" in plan
        assert "answer:qa" in plan
    
    def test_replan_node_integration_web_search(self, web_search_needed_state):
        """웹 검색이 필요한 재검색 통합 테스트"""
        result = replan_node(web_search_needed_state)
        
        # 웹 검색 필요성 확인
        assert "needs_web" in result
        assert isinstance(result["needs_web"], bool)
        
        # 재검색 카운터 확인
        assert result["replan_count"] == 1
        
        # 플랜에 websearch가 포함되어 있는지 확인
        plan = result["plan"]
        assert "websearch" in plan
        assert plan.index("websearch") < plan.index("search")
    
    def test_replan_node_integration_max_attempts(self, low_quality_feedback_state):
        """최대 시도 횟수 통합 테스트"""
        result = replan_node(low_quality_feedback_state)
        
        # 재검색 카운터가 증가하는지 확인
        assert result["replan_count"] == 3
        
        # 최대 시도 횟수 설정 확인
        assert result["max_replan_attempts"] == 3
    
    def test_generate_replan_query_integration_success(self):
        """LLM을 사용한 재검색 질문 생성 성공 테스트"""
        original_question = "여행자보험 보상금액은 얼마인가요?"
        feedback = "답변이 부족합니다. 더 구체적인 정보가 필요합니다."
        suggested_query = "여행자보험 보상금액 상세 정보"
        
        # 실제 LLM 호출을 모킹하지 않고 테스트
        try:
            result = _generate_replan_query(original_question, feedback, suggested_query)
            
            # 결과 구조 검증
            assert "new_question" in result
            assert "needs_web" in result
            assert "reasoning" in result
            
            # 타입 검증
            assert isinstance(result["new_question"], str)
            assert isinstance(result["needs_web"], bool)
            assert isinstance(result["reasoning"], str)
            
            # 내용 검증
            assert len(result["new_question"]) > 0
            assert result["reasoning"] != ""
            
        except Exception as e:
            # LLM 호출 실패 시 fallback이 작동하는지 확인
            assert "new_question" in result
            assert "needs_web" in result
            assert "reasoning" in result
    
    def test_generate_replan_query_integration_fallback(self):
        """LLM 호출 실패 시 fallback 테스트"""
        original_question = "여행자보험 보상금액은 얼마인가요?"
        feedback = "답변이 부족합니다."
        suggested_query = "여행자보험 보상금액 상세 정보"
        
        # LLM 호출을 강제로 실패시키기
        with patch('graph.nodes.replan.get_planner_llm') as mock_get_llm:
            mock_get_llm.side_effect = Exception("LLM 호출 실패")
            
            result = _generate_replan_query(original_question, feedback, suggested_query)
            
            # Fallback 결과 검증
            assert result["new_question"] == suggested_query
            assert isinstance(result["needs_web"], bool)
            assert "Fallback 재검색" in result["reasoning"]
    
    def test_fallback_replan_integration(self):
        """Fallback 재검색 통합 테스트"""
        # 웹 검색이 필요한 키워드 테스트
        web_keywords = ["최신", "현재", "실시간", "뉴스", "2024", "2025", "요즘", "지금"]
        
        for keyword in web_keywords:
            original_question = f"여행자보험 {keyword} 정보"
            suggested_query = f"여행자보험 {keyword} 정보"
            
            result = _fallback_replan(original_question, suggested_query)
            
            assert result["new_question"] == suggested_query
            assert result["needs_web"] == True
            assert "Fallback 재검색" in result["reasoning"]
        
        # 웹 검색이 필요하지 않은 키워드 테스트
        non_web_keywords = ["보상금액", "가입조건", "보장내용", "면책사항"]
        
        for keyword in non_web_keywords:
            original_question = f"여행자보험 {keyword} 정보"
            suggested_query = f"여행자보험 {keyword} 정보"
            
            result = _fallback_replan(original_question, suggested_query)
            
            assert result["new_question"] == suggested_query
            assert result["needs_web"] == False
            assert "Fallback 재검색" in result["reasoning"]
    
    def test_replan_node_integration_state_preservation(self):
        """상태 보존 통합 테스트"""
        state = {
            "question": "원래 질문",
            "quality_feedback": "피드백",
            "replan_query": "재검색 질문",
            "replan_count": 1,
            "intent": "qa",
            "custom_field1": "값1",
            "custom_field2": "값2",
            "nested_field": {
                "key1": "value1",
                "key2": "value2"
            }
        }
        
        result = replan_node(state)
        
        # 기존 상태가 보존되는지 확인
        assert result["intent"] == "qa"
        assert result["custom_field1"] == "값1"
        assert result["custom_field2"] == "값2"
        assert result["nested_field"]["key1"] == "value1"
        assert result["nested_field"]["key2"] == "value2"
        
        # 새로운 필드가 추가되는지 확인
        assert "question" in result
        assert "needs_web" in result
        assert "replan_count" in result
        assert "max_replan_attempts" in result
        assert "plan" in result
    
    def test_replan_node_integration_plan_structure(self):
        """플랜 구조 통합 테스트"""
        state = {
            "question": "테스트 질문",
            "quality_feedback": "피드백",
            "replan_query": "재검색 질문",
            "replan_count": 0
        }
        
        result = replan_node(state)
        
        # 플랜 구조 검증
        plan = result["plan"]
        assert isinstance(plan, list)
        assert len(plan) == 6
        
        # 플랜 순서 검증
        expected_plan = ["replan", "websearch", "search", "rank_filter", "verify_refine", "answer:qa"]
        assert plan == expected_plan
        
        # 각 노드가 올바른 위치에 있는지 확인
        assert plan[0] == "replan"
        assert plan[1] == "websearch"
        assert plan[2] == "search"
        assert plan[3] == "rank_filter"
        assert plan[4] == "verify_refine"
        assert plan[5] == "answer:qa"
    
    def test_replan_node_integration_error_handling(self):
        """에러 처리 통합 테스트"""
        # 잘못된 상태로 테스트
        invalid_state = {
            "question": None,
            "quality_feedback": None,
            "replan_query": None,
            "replan_count": None
        }
        
        # 에러가 발생하지 않는지 확인
        try:
            result = replan_node(invalid_state)
            
            # 기본값들이 설정되는지 확인
            assert "question" in result
            assert "needs_web" in result
            assert "replan_count" in result
            assert "max_replan_attempts" in result
            assert "plan" in result
            
        except Exception as e:
            pytest.fail(f"replan_node가 예상치 못한 에러를 발생시켰습니다: {e}")
    
    def test_replan_node_integration_logging(self, caplog):
        """로깅 통합 테스트"""
        state = {
            "question": "테스트 질문",
            "quality_feedback": "피드백",
            "replan_query": "재검색 질문",
            "replan_count": 0
        }
        
        with caplog.at_level(logging.INFO):
            result = replan_node(state)
        
        # 로그 메시지 확인
        log_messages = [record.message for record in caplog.records]
        
        # 재검색 시작 로그 확인
        assert any("재검색 시작" in msg for msg in log_messages)
        
        # 재검색 완료 로그 확인
        assert any("재검색 질문 생성 완료" in msg for msg in log_messages)
    
    def test_replan_node_integration_performance(self):
        """성능 통합 테스트"""
        import time
        
        state = {
            "question": "성능 테스트 질문",
            "quality_feedback": "피드백",
            "replan_query": "재검색 질문",
            "replan_count": 0
        }
        
        start_time = time.time()
        result = replan_node(state)
        end_time = time.time()
        
        # 실행 시간이 합리적인 범위 내에 있는지 확인 (5초 이내)
        execution_time = end_time - start_time
        assert execution_time < 5.0, f"replan_node 실행 시간이 너무 깁니다: {execution_time:.2f}초"
        
        # 결과가 올바르게 반환되는지 확인
        assert "question" in result
        assert "needs_web" in result
        assert "replan_count" in result
    
    def test_replan_node_integration_concurrent_calls(self):
        """동시 호출 통합 테스트"""
        import threading
        import time
        
        results = []
        errors = []
        
        def run_replan(state):
            try:
                result = replan_node(state)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # 여러 스레드에서 동시에 replan_node 호출
        threads = []
        for i in range(5):
            state = {
                "question": f"동시 테스트 질문 {i}",
                "quality_feedback": f"피드백 {i}",
                "replan_query": f"재검색 질문 {i}",
                "replan_count": 0
            }
            thread = threading.Thread(target=run_replan, args=(state,))
            threads.append(thread)
            thread.start()
        
        # 모든 스레드 완료 대기
        for thread in threads:
            thread.join()
        
        # 에러가 발생하지 않았는지 확인
        assert len(errors) == 0, f"동시 호출 중 에러 발생: {errors}"
        
        # 모든 결과가 올바르게 반환되었는지 확인
        assert len(results) == 5
        for result in results:
            assert "question" in result
            assert "needs_web" in result
            assert "replan_count" in result
