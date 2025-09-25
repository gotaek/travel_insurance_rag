import pytest
from unittest.mock import Mock, patch, MagicMock
from graph.nodes.reevaluate import (
    reevaluate_node, 
    _evaluate_answer_quality, 
    _fallback_evaluate,
    QUALITY_THRESHOLD,
    MAX_REPLAN_ATTEMPTS
)


class TestReevaluateNode:
    """reevaluate_node 함수 테스트"""
    
    def test_reevaluate_node_with_high_quality_answer(self):
        """고품질 답변에 대한 테스트"""
        state = {
            "question": "여행자보험 보상금액은 얼마인가요?",
            "draft_answer": {"text": "여행자보험 보상금액은 보험사마다 다르지만, 일반적으로 사망시 1억원, 상해시 1000만원까지 보상받을 수 있습니다."},
            "citations": [{"source": "보험약관", "page": 1}],
            "refined": [{"content": "보상금액 정보"}],
            "replan_count": 0,
            "max_replan_attempts": 3
        }
        
        with patch('graph.nodes.reevaluate._evaluate_answer_quality') as mock_evaluate:
            mock_evaluate.return_value = {
                "score": 0.9,
                "feedback": "우수한 답변",
                "needs_replan": False,
                "replan_query": ""
            }
            
            result = reevaluate_node(state)
            
            assert result["quality_score"] == 0.9
            assert result["needs_replan"] == False
            assert result["final_answer"] == state["draft_answer"]
            assert result["replan_count"] == 0
    
    def test_reevaluate_node_with_low_quality_answer(self):
        """저품질 답변에 대한 테스트"""
        state = {
            "question": "여행자보험 보상금액은 얼마인가요?",
            "draft_answer": {"text": "모르겠습니다."},
            "citations": [],
            "refined": [],
            "replan_count": 0,
            "max_replan_attempts": 3
        }
        
        with patch('graph.nodes.reevaluate._evaluate_answer_quality') as mock_evaluate:
            mock_evaluate.return_value = {
                "score": 0.3,
                "feedback": "답변이 부족합니다",
                "needs_replan": True,
                "replan_query": "여행자보험 보상금액 상세 정보"
            }
            
            result = reevaluate_node(state)
            
            assert result["quality_score"] == 0.3
            assert result["needs_replan"] == True
            assert result["final_answer"] is None
            assert result["replan_query"] == "여행자보험 보상금액 상세 정보"
    
    def test_reevaluate_node_max_attempts_reached(self):
        """최대 재검색 횟수 도달 시 테스트"""
        state = {
            "question": "여행자보험 보상금액은 얼마인가요?",
            "draft_answer": {"text": "부족한 답변"},
            "citations": [],
            "refined": [],
            "replan_count": 3,
            "max_replan_attempts": 3
        }
        
        with patch('graph.nodes.reevaluate._evaluate_answer_quality') as mock_evaluate:
            mock_evaluate.return_value = {
                "score": 0.3,
                "feedback": "답변이 부족합니다",
                "needs_replan": True,
                "replan_query": "재검색 필요"
            }
            
            result = reevaluate_node(state)
            
            assert result["needs_replan"] == False  # 최대 횟수 도달로 재검색 불가
            assert result["final_answer"] == state["draft_answer"]  # 강제로 최종 답변 설정
    
    def test_reevaluate_node_with_string_answer(self):
        """문자열 형태 답변에 대한 테스트"""
        state = {
            "question": "여행자보험은 무엇인가요?",
            "draft_answer": "여행자보험은 해외여행 중 발생할 수 있는 위험에 대비한 보험입니다.",
            "citations": [{"source": "보험약관"}],
            "refined": [{"content": "보험 정의"}],
            "replan_count": 0
        }
        
        with patch('graph.nodes.reevaluate._evaluate_answer_quality') as mock_evaluate:
            mock_evaluate.return_value = {
                "score": 0.8,
                "feedback": "적절한 답변",
                "needs_replan": False,
                "replan_query": ""
            }
            
            result = reevaluate_node(state)
            
            assert result["quality_score"] == 0.8
            assert result["needs_replan"] == False


class TestEvaluateAnswerQuality:
    """_evaluate_answer_quality 함수 테스트"""
    
    @patch('graph.nodes.reevaluate.get_llm')
    def test_evaluate_answer_quality_success(self, mock_get_llm):
        """LLM 평가 성공 테스트"""
        # Mock LLM 설정
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = '''
        ```json
        {
            "score": 0.85,
            "feedback": "정확하고 상세한 답변입니다",
            "needs_replan": false,
            "replan_query": null
        }
        ```
        '''
        mock_llm.generate_content.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        
        result = _evaluate_answer_quality(
            "여행자보험 보상금액은?",
            "일반적으로 사망시 1억원까지 보상받을 수 있습니다.",
            [{"source": "보험약관"}],
            [{"content": "보상 정보"}]
        )
        
        assert result["score"] == 0.85
        assert result["feedback"] == "정확하고 상세한 답변입니다"
        assert result["needs_replan"] == False
        assert result["replan_query"] == ""
    
    @patch('graph.nodes.reevaluate.get_llm')
    def test_evaluate_answer_quality_invalid_score(self, mock_get_llm):
        """유효하지 않은 점수 처리 테스트"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = '''
        {
            "score": 1.5,
            "feedback": "테스트",
            "needs_replan": false,
            "replan_query": null
        }
        '''
        mock_llm.generate_content.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        
        result = _evaluate_answer_quality("질문", "답변", [], [])
        
        assert result["score"] == 0.5  # 기본값으로 수정
        assert result["needs_replan"] == False
    
    @patch('graph.nodes.reevaluate.get_llm')
    def test_evaluate_answer_quality_invalid_needs_replan(self, mock_get_llm):
        """유효하지 않은 needs_replan 처리 테스트"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = '''
        {
            "score": 0.6,
            "feedback": "테스트",
            "needs_replan": "invalid",
            "replan_query": null
        }
        '''
        mock_llm.generate_content.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        
        result = _evaluate_answer_quality("질문", "답변", [], [])
        
        assert result["score"] == 0.6
        assert result["needs_replan"] == True  # 점수가 임계값 미만이므로 True
    
    @patch('graph.nodes.reevaluate.get_llm')
    def test_evaluate_answer_quality_llm_failure(self, mock_get_llm):
        """LLM 호출 실패 시 fallback 테스트"""
        mock_get_llm.side_effect = Exception("LLM 호출 실패")
        
        result = _evaluate_answer_quality("질문", "답변", [], [])
        
        # fallback 평가 결과 확인
        assert "score" in result
        assert "feedback" in result
        assert "needs_replan" in result
        assert "replan_query" in result
        assert "Fallback 평가" in result["feedback"]


class TestFallbackEvaluate:
    """_fallback_evaluate 함수 테스트"""
    
    def test_fallback_evaluate_with_good_answer(self):
        """좋은 답변에 대한 fallback 평가 테스트"""
        result = _fallback_evaluate(
            "여행자보험 보상금액은?",
            "일반적으로 사망시 1억원까지 보상받을 수 있습니다. 상해시에는 1000만원까지 보상받을 수 있습니다.",
            [{"source": "보험약관"}],
            [{"content": "보상 정보"}]
        )
        
        assert result["score"] > 0.5
        assert result["needs_replan"] == (result["score"] < QUALITY_THRESHOLD)
        assert "Fallback 평가" in result["feedback"]
    
    def test_fallback_evaluate_with_poor_answer(self):
        """부족한 답변에 대한 fallback 평가 테스트"""
        result = _fallback_evaluate(
            "여행자보험 보상금액은?",
            "모르겠습니다.",
            [],
            []
        )
        
        assert result["score"] <= 0.5
        assert result["needs_replan"] == True
        assert result["replan_query"] == "여행자보험 보상금액은?"
    
    def test_fallback_evaluate_keyword_matching(self):
        """키워드 매칭 테스트"""
        result = _fallback_evaluate(
            "여행자보험 보상금액",
            "여행자보험의 보상금액은 보험사마다 다릅니다",
            [{"source": "보험약관"}],
            [{"content": "보상 정보"}]
        )
        
        # 키워드 매칭으로 점수 증가 확인
        assert result["score"] > 0.5
        # 피드백 메시지 형식 확인
        assert "Fallback 평가" in result["feedback"]
        assert "답변길이" in result["feedback"]
    
    def test_fallback_evaluate_score_cap(self):
        """점수 상한선 테스트"""
        result = _fallback_evaluate(
            "질문",
            "매우 긴 답변입니다. " * 20,  # 긴 답변
            [{"source": "1"}, {"source": "2"}, {"source": "3"}],  # 많은 인용
            [{"content": "1"}, {"content": "2"}, {"content": "3"}]  # 많은 문서
        )
        
        assert result["score"] <= 1.0  # 상한선 확인


class TestConstants:
    """상수 테스트"""
    
    def test_quality_threshold(self):
        """품질 임계값 테스트"""
        assert QUALITY_THRESHOLD == 0.7
    
    def test_max_replan_attempts(self):
        """최대 재검색 횟수 테스트"""
        assert MAX_REPLAN_ATTEMPTS == 3


class TestIntegration:
    """통합 테스트"""
    
    @patch('graph.nodes.reevaluate._evaluate_answer_quality')
    def test_reevaluate_node_integration(self, mock_evaluate):
        """reevaluate_node 통합 테스트"""
        mock_evaluate.return_value = {
            "score": 0.8,
            "feedback": "좋은 답변",
            "needs_replan": False,
            "replan_query": ""
        }
        
        state = {
            "question": "여행자보험은 무엇인가요?",
            "draft_answer": {"text": "여행자보험은 해외여행 중 발생할 수 있는 위험에 대비한 보험입니다."},
            "citations": [{"source": "보험약관"}],
            "refined": [{"content": "보험 정의"}],
            "replan_count": 0
        }
        
        result = reevaluate_node(state)
        
        # 상태 업데이트 확인
        assert result["quality_score"] == 0.8
        assert result["quality_feedback"] == "좋은 답변"
        assert result["needs_replan"] == False
        assert result["final_answer"] == state["draft_answer"]
        
        # 원본 상태 보존 확인
        assert result["question"] == state["question"]
        assert result["citations"] == state["citations"]
        assert result["refined"] == state["refined"]
