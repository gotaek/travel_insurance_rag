import pytest
from unittest.mock import Mock, patch, MagicMock
from graph.nodes.replan import (
    replan_node, 
    _generate_replan_query, 
    _fallback_replan
)


class TestReplanNode:
    """replan_node 함수 테스트"""
    
    def test_replan_node_basic_functionality(self):
        """기본 재검색 기능 테스트"""
        state = {
            "question": "여행자보험 보상금액은 얼마인가요?",
            "quality_feedback": "답변이 부족합니다. 더 구체적인 정보가 필요합니다.",
            "replan_query": "여행자보험 보상금액 상세 정보",
            "replan_count": 1
        }
        
        with patch('graph.nodes.replan._generate_replan_query') as mock_generate:
            mock_generate.return_value = {
                "new_question": "여행자보험 보상금액 상세 정보",
                "needs_web": False,
                "reasoning": "더 구체적인 질문으로 개선"
            }
            
            result = replan_node(state)
            
            assert result["question"] == "여행자보험 보상금액 상세 정보"
            assert result["needs_web"] == False
            assert result["replan_count"] == 2
            assert result["max_replan_attempts"] == 3
            assert "replan" in result["plan"]
            assert "websearch" in result["plan"]
            assert "search" in result["plan"]
            assert "rank_filter" in result["plan"]
            assert "verify_refine" in result["plan"]
            assert "answer:qa" in result["plan"]
    
    def test_replan_node_with_web_search_needed(self):
        """웹 검색이 필요한 재검색 테스트"""
        state = {
            "question": "2024년 여행자보험 최신 가격은?",
            "quality_feedback": "최신 정보가 필요합니다.",
            "replan_query": "2024년 여행자보험 최신 가격 정보",
            "replan_count": 0
        }
        
        with patch('graph.nodes.replan._generate_replan_query') as mock_generate:
            mock_generate.return_value = {
                "new_question": "2024년 여행자보험 최신 가격 정보",
                "needs_web": True,
                "reasoning": "최신 정보가 필요하므로 웹 검색 필요"
            }
            
            result = replan_node(state)
            
            assert result["needs_web"] == True
            assert result["replan_count"] == 1
    
    def test_replan_node_preserves_existing_state(self):
        """기존 상태 보존 테스트"""
        state = {
            "question": "원래 질문",
            "quality_feedback": "피드백",
            "replan_query": "재검색 질문",
            "replan_count": 1,
            "existing_field": "기존 값",
            "intent": "qa"
        }
        
        with patch('graph.nodes.replan._generate_replan_query') as mock_generate:
            mock_generate.return_value = {
                "new_question": "새 질문",
                "needs_web": False,
                "reasoning": "개선된 질문"
            }
            
            result = replan_node(state)
            
            # 기존 상태가 보존되는지 확인
            assert result["existing_field"] == "기존 값"
            assert result["intent"] == "qa"
            assert result["quality_feedback"] == "피드백"
    
    def test_replan_node_with_empty_inputs(self):
        """빈 입력값에 대한 테스트"""
        state = {
            "question": "",
            "quality_feedback": "",
            "replan_query": "",
            "replan_count": 0
        }
        
        with patch('graph.nodes.replan._generate_replan_query') as mock_generate:
            mock_generate.return_value = {
                "new_question": "기본 질문",
                "needs_web": False,
                "reasoning": "기본 처리"
            }
            
            result = replan_node(state)
            
            assert result["question"] == "기본 질문"
            assert result["replan_count"] == 1


class TestGenerateReplanQuery:
    """_generate_replan_query 함수 테스트"""
    
    def test_generate_replan_query_success(self):
        """성공적인 LLM 호출 테스트"""
        original_question = "여행자보험 보상금액은?"
        feedback = "답변이 부족합니다."
        suggested_query = "여행자보험 보상금액 상세 정보"
        
        mock_response = Mock()
        mock_response.text = '''
        {
            "new_question": "여행자보험 보상금액 상세 정보",
            "needs_web": false,
            "reasoning": "더 구체적인 질문으로 개선"
        }
        '''
        
        with patch('graph.nodes.replan.get_planner_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            result = _generate_replan_query(original_question, feedback, suggested_query)
            
            assert result["new_question"] == "여행자보험 보상금액 상세 정보"
            assert result["needs_web"] == False
            assert "reasoning" in result
    
    def test_generate_replan_query_with_json_wrapper(self):
        """JSON 래퍼가 있는 응답 처리 테스트"""
        original_question = "여행자보험 보상금액은?"
        feedback = "답변이 부족합니다."
        suggested_query = "여행자보험 보상금액 상세 정보"
        
        mock_response = Mock()
        mock_response.text = '''
        ```json
        {
            "new_question": "여행자보험 보상금액 상세 정보",
            "needs_web": true,
            "reasoning": "최신 정보가 필요합니다"
        }
        ```
        '''
        
        with patch('graph.nodes.replan.get_planner_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            result = _generate_replan_query(original_question, feedback, suggested_query)
            
            assert result["new_question"] == "여행자보험 보상금액 상세 정보"
            assert result["needs_web"] == True
    
    def test_generate_replan_query_validation(self):
        """유효성 검증 테스트"""
        original_question = "원래 질문"
        feedback = "피드백"
        suggested_query = "제안 질문"
        
        mock_response = Mock()
        mock_response.text = '''
        {
            "new_question": "",
            "needs_web": "invalid",
            "reasoning": "테스트"
        }
        '''
        
        with patch('graph.nodes.replan.get_planner_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            result = _generate_replan_query(original_question, feedback, suggested_query)
            
            # 빈 질문이면 원래 질문 사용
            assert result["new_question"] == original_question
            # 잘못된 needs_web이면 True로 설정
            assert result["needs_web"] == True
    
    def test_generate_replan_query_llm_failure(self):
        """LLM 호출 실패 시 fallback 테스트"""
        original_question = "원래 질문"
        feedback = "피드백"
        suggested_query = "제안 질문"
        
        with patch('graph.nodes.replan.get_planner_llm') as mock_get_llm:
            mock_get_llm.side_effect = Exception("LLM 호출 실패")
            
            with patch('graph.nodes.replan._fallback_replan') as mock_fallback:
                mock_fallback.return_value = {
                    "new_question": "fallback 질문",
                    "needs_web": False,
                    "reasoning": "fallback 처리"
                }
                
                result = _generate_replan_query(original_question, feedback, suggested_query)
                
                mock_fallback.assert_called_once_with(original_question, suggested_query)
                assert result["new_question"] == "fallback 질문"


class TestFallbackReplan:
    """_fallback_replan 함수 테스트"""
    
    def test_fallback_replan_with_suggested_query(self):
        """제안된 질문이 있는 경우 테스트"""
        original_question = "원래 질문"
        suggested_query = "제안된 질문"
        
        result = _fallback_replan(original_question, suggested_query)
        
        assert result["new_question"] == suggested_query
        assert result["reasoning"] == f"Fallback 재검색: {suggested_query}"
    
    def test_fallback_replan_without_suggested_query(self):
        """제안된 질문이 없는 경우 테스트"""
        original_question = "원래 질문"
        suggested_query = ""
        
        result = _fallback_replan(original_question, suggested_query)
        
        assert result["new_question"] == original_question
        assert result["reasoning"] == f"Fallback 재검색: {original_question}"
    
    def test_fallback_replan_web_search_detection(self):
        """웹 검색 필요성 감지 테스트"""
        # 웹 검색이 필요한 키워드들
        web_keywords = ["최신", "현재", "실시간", "뉴스", "2024", "2025", "요즘", "지금"]
        
        for keyword in web_keywords:
            original_question = f"여행자보험 {keyword} 정보"
            suggested_query = f"여행자보험 {keyword} 정보"
            
            result = _fallback_replan(original_question, suggested_query)
            
            assert result["needs_web"] == True
            assert keyword in result["new_question"]
    
    def test_fallback_replan_no_web_search_needed(self):
        """웹 검색이 필요하지 않은 경우 테스트"""
        original_question = "여행자보험 보상금액은 얼마인가요?"
        suggested_query = "여행자보험 보상금액 정보"
        
        result = _fallback_replan(original_question, suggested_query)
        
        assert result["needs_web"] == False
        assert result["new_question"] == suggested_query
    
    def test_fallback_replan_empty_suggested_query(self):
        """빈 제안 질문 처리 테스트"""
        original_question = "원래 질문"
        suggested_query = "   "  # 공백만 있는 경우
        
        result = _fallback_replan(original_question, suggested_query)
        
        assert result["new_question"] == original_question
        assert result["needs_web"] == False


class TestReplanNodeEdgeCases:
    """replan 노드 엣지 케이스 테스트"""
    
    def test_replan_node_missing_fields(self):
        """누락된 필드 처리 테스트"""
        state = {}  # 빈 상태
        
        with patch('graph.nodes.replan._generate_replan_query') as mock_generate:
            mock_generate.return_value = {
                "new_question": "기본 질문",
                "needs_web": False,
                "reasoning": "기본 처리"
            }
            
            result = replan_node(state)
            
            assert result["question"] == "기본 질문"
            assert result["replan_count"] == 1
            assert result["max_replan_attempts"] == 3
    
    def test_replan_node_with_none_values(self):
        """None 값 처리 테스트"""
        state = {
            "question": None,
            "quality_feedback": None,
            "replan_query": None,
            "replan_count": None
        }
        
        with patch('graph.nodes.replan._generate_replan_query') as mock_generate:
            mock_generate.return_value = {
                "new_question": "기본 질문",
                "needs_web": False,
                "reasoning": "기본 처리"
            }
            
            result = replan_node(state)
            
            assert result["question"] == "기본 질문"
            assert result["replan_count"] == 1
    
    def test_replan_node_plan_generation(self):
        """플랜 생성 테스트"""
        state = {
            "question": "테스트 질문",
            "quality_feedback": "피드백",
            "replan_query": "재검색 질문",
            "replan_count": 0
        }
        
        with patch('graph.nodes.replan._generate_replan_query') as mock_generate:
            mock_generate.return_value = {
                "new_question": "새 질문",
                "needs_web": False,
                "reasoning": "개선된 질문"
            }
            
            result = replan_node(state)
            
            # 플랜이 올바르게 생성되는지 확인
            plan = result["plan"]
            assert "replan" in plan
            assert "websearch" in plan
            assert "search" in plan
            assert "rank_filter" in plan
            assert "verify_refine" in plan
            assert "answer:qa" in plan
            
            # 순서 확인
            assert plan[0] == "replan"
            assert plan[1] == "websearch"
            assert plan[2] == "search"
            assert plan[3] == "rank_filter"
            assert plan[4] == "verify_refine"
            assert plan[5] == "answer:qa"
