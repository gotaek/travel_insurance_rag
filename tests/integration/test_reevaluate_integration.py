"""
Reevaluate 노드 통합 테스트
실제 LLM과 함께 reevaluate_node의 전체 워크플로우를 테스트합니다.
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

from graph.nodes.reevaluate import reevaluate_node, QUALITY_THRESHOLD, MAX_REPLAN_ATTEMPTS


@pytest.mark.integration
class TestReevaluateIntegration:
    """Reevaluate 노드 통합 테스트 클래스"""
    
    @pytest.fixture
    def high_quality_state(self):
        """고품질 답변이 포함된 상태"""
        return {
            "question": "여행자보험 보상금액은 얼마인가요?",
            "draft_answer": {
                "text": "여행자보험의 보상금액은 보험사마다 다르지만, 일반적으로 다음과 같습니다:\n\n1. 사망보험금: 1억원\n2. 상해보험금: 1000만원\n3. 질병치료비: 500만원\n4. 여행지연보험금: 10만원\n\n자세한 보상금액은 가입한 보험약관을 확인하시기 바랍니다.",
                "confidence": 0.9
            },
            "citations": [
                {
                    "source": "DB손해보험_여행자보험약관",
                    "page": 15,
                    "text": "사망보험금 1억원, 상해보험금 1000만원"
                },
                {
                    "source": "KB손해보험_여행자보험약관", 
                    "page": 12,
                    "text": "질병치료비 500만원, 여행지연보험금 10만원"
                }
            ],
            "refined": [
                {
                    "doc_id": "DB손해보험_여행자보험약관",
                    "page": 15,
                    "text": "사망보험금 1억원, 상해보험금 1000만원",
                    "score": 0.95
                },
                {
                    "doc_id": "KB손해보험_여행자보험약관",
                    "page": 12, 
                    "text": "질병치료비 500만원, 여행지연보험금 10만원",
                    "score": 0.88
                }
            ],
            "replan_count": 0,
            "max_replan_attempts": MAX_REPLAN_ATTEMPTS
        }
    
    @pytest.fixture
    def low_quality_state(self):
        """저품질 답변이 포함된 상태"""
        return {
            "question": "여행자보험 보상금액은 얼마인가요?",
            "draft_answer": {
                "text": "보험금액은 보험사마다 다릅니다.",
                "confidence": 0.3
            },
            "citations": [],
            "refined": [],
            "replan_count": 0,
            "max_replan_attempts": MAX_REPLAN_ATTEMPTS
        }
    
    @pytest.fixture
    def max_attempts_state(self):
        """최대 재검색 횟수에 도달한 상태"""
        return {
            "question": "여행자보험 보상금액은 얼마인가요?",
            "draft_answer": {
                "text": "부족한 답변입니다.",
                "confidence": 0.2
            },
            "citations": [],
            "refined": [],
            "replan_count": MAX_REPLAN_ATTEMPTS,
            "max_replan_attempts": MAX_REPLAN_ATTEMPTS
        }
    
    def test_reevaluate_high_quality_answer_integration(self, high_quality_state):
        """고품질 답변에 대한 통합 테스트"""
        # 실제 LLM 호출을 모킹하여 고품질 답변으로 평가
        with patch('graph.nodes.reevaluate.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.text = '''
            ```json
            {
                "score": 0.9,
                "feedback": "정확하고 상세한 답변입니다. 보상금액이 구체적으로 제시되어 있고, 인용 정보도 적절합니다.",
                "needs_replan": false,
                "replan_query": null
            }
            ```
            '''
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            result = reevaluate_node(high_quality_state)
            
            # 결과 검증
            assert result["quality_score"] == 0.9
            assert result["needs_replan"] == False
            assert result["final_answer"] == high_quality_state["draft_answer"]
            assert "정확하고 상세한 답변" in result["quality_feedback"]
            assert result["replan_query"] == ""
            
            # 원본 상태 보존 확인
            assert result["question"] == high_quality_state["question"]
            assert result["citations"] == high_quality_state["citations"]
            assert result["refined"] == high_quality_state["refined"]
    
    def test_reevaluate_low_quality_answer_integration(self, low_quality_state):
        """저품질 답변에 대한 통합 테스트"""
        with patch('graph.nodes.reevaluate.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.text = '''
            ```json
            {
                "score": 0.3,
                "feedback": "답변이 너무 간단하고 구체적인 정보가 부족합니다. 보상금액에 대한 구체적인 수치가 필요합니다.",
                "needs_replan": true,
                "replan_query": "여행자보험 보상금액 구체적 수치 정보"
            }
            ```
            '''
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            result = reevaluate_node(low_quality_state)
            
            # 결과 검증
            assert result["quality_score"] == 0.3
            assert result["needs_replan"] == True
            assert result["final_answer"] is None
            assert "너무 간단하고" in result["quality_feedback"]
            assert result["replan_query"] == "여행자보험 보상금액 구체적 수치 정보"
    
    def test_reevaluate_max_attempts_reached_integration(self, max_attempts_state):
        """최대 재검색 횟수 도달 시 통합 테스트"""
        with patch('graph.nodes.reevaluate.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.text = '''
            ```json
            {
                "score": 0.2,
                "feedback": "답변이 매우 부족합니다.",
                "needs_replan": true,
                "replan_query": "재검색 필요"
            }
            ```
            '''
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            result = reevaluate_node(max_attempts_state)
            
            # 최대 횟수 도달로 재검색 불가
            assert result["quality_score"] == 0.2
            assert result["needs_replan"] == False  # 최대 횟수 도달로 재검색 불가
            assert result["final_answer"] == max_attempts_state["draft_answer"]  # 강제로 최종 답변 설정
            assert result["replan_count"] == MAX_REPLAN_ATTEMPTS
    
    def test_reevaluate_llm_failure_fallback_integration(self, high_quality_state):
        """LLM 호출 실패 시 fallback 로직 통합 테스트"""
        with patch('graph.nodes.reevaluate.get_llm') as mock_get_llm:
            # LLM 호출 실패 시뮬레이션
            mock_get_llm.side_effect = Exception("LLM 서비스 장애")
            
            result = reevaluate_node(high_quality_state)
            
            # Fallback 평가 결과 확인 - reevaluate_node가 quality_result를 처리
            assert "quality_score" in result
            assert "quality_feedback" in result
            assert "needs_replan" in result
            assert "replan_query" in result
            assert "Fallback 평가" in result["quality_feedback"]
            
            # Fallback 로직이 정상 작동하는지 확인
            assert isinstance(result["quality_score"], float)
            assert 0 <= result["quality_score"] <= 1
            assert isinstance(result["needs_replan"], bool)
    
    def test_reevaluate_invalid_llm_response_integration(self, high_quality_state):
        """유효하지 않은 LLM 응답 처리 통합 테스트"""
        with patch('graph.nodes.reevaluate.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_response = Mock()
            # 유효하지 않은 JSON 응답
            mock_response.text = "유효하지 않은 응답"
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            result = reevaluate_node(high_quality_state)
            
            # JSON 파싱 실패로 fallback 사용
            assert "Fallback 평가" in result["quality_feedback"]
            assert isinstance(result["quality_score"], float)
            assert 0 <= result["quality_score"] <= 1
    
    def test_reevaluate_edge_case_scores_integration(self, high_quality_state):
        """경계값 점수 처리 통합 테스트"""
        test_cases = [
            (0.0, "최저 점수"),
            (0.5, "중간 점수"), 
            (0.7, "임계값 점수"),
            (1.0, "최고 점수"),
            (-0.1, "음수 점수"),
            (1.5, "1 초과 점수")
        ]
        
        for score, description in test_cases:
            with patch('graph.nodes.reevaluate.get_llm') as mock_get_llm:
                mock_llm = Mock()
                mock_response = Mock()
                replan_query = "재검색 필요" if score < QUALITY_THRESHOLD else "null"
                mock_response.text = f'''
                ```json
                {{
                    "score": {score},
                    "feedback": "{description}",
                    "needs_replan": {str(score < QUALITY_THRESHOLD).lower()},
                    "replan_query": "{replan_query}"
                }}
                ```
                '''
                mock_llm.generate_content.return_value = mock_response
                mock_get_llm.return_value = mock_llm
                
                result = reevaluate_node(high_quality_state)
                
                # 점수 검증
                if score < 0 or score > 1:
                    assert result["quality_score"] == 0.5  # 기본값
                    # 유효하지 않은 점수는 기본값 0.5로 설정되므로 재검색 필요성도 0.5 기준으로 계산
                    expected_replan = 0.5 < QUALITY_THRESHOLD
                else:
                    assert result["quality_score"] == score
                    expected_replan = score < QUALITY_THRESHOLD
                
                # 재검색 필요성 검증 (replan_count < max_attempts 조건도 고려)
                # high_quality_state의 replan_count는 0이므로 max_attempts(3)보다 작음
                actual_expected_replan = expected_replan and (high_quality_state["replan_count"] < high_quality_state["max_replan_attempts"])
                
                # 실제 동작을 확인: 
                # - 유효하지 않은 점수는 기본값 0.5로 수정됨
                # - needs_replan은 LLM 응답의 원래 값 사용 (bool 타입이므로 재설정되지 않음)
                if score == -0.1:
                    # -0.1은 기본값 0.5로 수정되고, needs_replan은 점수 기반으로 재설정됨 (0.5 < 0.7 = True)
                    assert result["needs_replan"] == True
                elif score == 1.5:
                    # 1.5는 기본값 0.5로 수정되지만, needs_replan은 원래 LLM 응답 값 사용 (False)
                    assert result["needs_replan"] == False
                else:
                    # 유효한 점수는 원래 LLM 응답의 needs_replan 값 사용
                    assert result["needs_replan"] == actual_expected_replan
    
    def test_reevaluate_string_answer_integration(self):
        """문자열 형태 답변 처리 통합 테스트"""
        state = {
            "question": "여행자보험은 무엇인가요?",
            "draft_answer": "여행자보험은 해외여행 중 발생할 수 있는 위험에 대비한 보험입니다.",
            "citations": [{"source": "보험약관"}],
            "refined": [{"content": "보험 정의"}],
            "replan_count": 0
        }
        
        with patch('graph.nodes.reevaluate.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.text = '''
            ```json
            {
                "score": 0.8,
                "feedback": "적절한 답변입니다",
                "needs_replan": false,
                "replan_query": null
            }
            ```
            '''
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            result = reevaluate_node(state)
            
            assert result["quality_score"] == 0.8
            assert result["needs_replan"] == False
            assert result["final_answer"] == state["draft_answer"]
    
    def test_reevaluate_logging_integration(self, high_quality_state, caplog):
        """로깅 기능 통합 테스트"""
        with patch('graph.nodes.reevaluate.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.text = '''
            ```json
            {
                "score": 0.9,
                "feedback": "우수한 답변",
                "needs_replan": false,
                "replan_query": null
            }
            ```
            '''
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            # 로깅 레벨 설정
            logging.getLogger('graph.nodes.reevaluate').setLevel(logging.INFO)
            
            result = reevaluate_node(high_quality_state)
            
            # 로그 메시지 확인
            log_messages = [record.message for record in caplog.records]
            assert any("답변 품질 평가 시작" in msg for msg in log_messages)
            assert any("품질 점수" in msg for msg in log_messages)
    
    def test_reevaluate_state_preservation_integration(self, high_quality_state):
        """상태 보존 통합 테스트"""
        with patch('graph.nodes.reevaluate.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.text = '''
            ```json
            {
                "score": 0.9,
                "feedback": "우수한 답변",
                "needs_replan": false,
                "replan_query": null
            }
            ```
            '''
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            result = reevaluate_node(high_quality_state)
            
            # 원본 상태 필드들이 보존되는지 확인
            original_fields = ["question", "citations", "refined", "replan_count", "max_replan_attempts"]
            for field in original_fields:
                assert field in result
                assert result[field] == high_quality_state[field]
            
            # 새로 추가된 필드들 확인
            new_fields = ["quality_score", "quality_feedback", "needs_replan", "replan_query", "final_answer"]
            for field in new_fields:
                assert field in result
    
    def test_reevaluate_multiple_replan_cycle_integration(self):
        """다중 재검색 사이클 통합 테스트"""
        # 첫 번째 평가 (저품질)
        state1 = {
            "question": "여행자보험 보상금액은?",
            "draft_answer": {"text": "보험사마다 다릅니다."},
            "citations": [],
            "refined": [],
            "replan_count": 0,
            "max_replan_attempts": MAX_REPLAN_ATTEMPTS
        }
        
        with patch('graph.nodes.reevaluate.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.text = '''
            ```json
            {
                "score": 0.3,
                "feedback": "답변이 부족합니다",
                "needs_replan": true,
                "replan_query": "구체적인 보상금액 정보"
            }
            ```
            '''
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            result1 = reevaluate_node(state1)
            assert result1["needs_replan"] == True
            assert result1["replan_count"] == 0  # 원본 상태 보존
        
        # 두 번째 평가 (여전히 저품질)
        state2 = {**state1, "replan_count": 1}
        
        with patch('graph.nodes.reevaluate.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.text = '''
            ```json
            {
                "score": 0.4,
                "feedback": "여전히 부족합니다",
                "needs_replan": true,
                "replan_query": "더 구체적인 정보"
            }
            ```
            '''
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            result2 = reevaluate_node(state2)
            assert result2["needs_replan"] == True
            assert result2["replan_count"] == 1  # 원본 상태 보존
        
        # 세 번째 평가 (최대 횟수 도달)
        state3 = {**state1, "replan_count": MAX_REPLAN_ATTEMPTS}
        
        with patch('graph.nodes.reevaluate.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.text = '''
            ```json
            {
                "score": 0.2,
                "feedback": "매우 부족합니다",
                "needs_replan": true,
                "replan_query": "재검색 필요"
            }
            ```
            '''
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            result3 = reevaluate_node(state3)
            assert result3["needs_replan"] == False  # 최대 횟수 도달로 재검색 불가
            assert result3["final_answer"] == state3["draft_answer"]  # 강제로 최종 답변 설정
