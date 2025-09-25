"""
Recommend 노드 단위 테스트
recommend_node의 핵심 기능과 에러 처리를 테스트합니다.
"""

import sys
import os
import pytest
from unittest.mock import Mock, patch
import json

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from graph.nodes.answerers.recommend import recommend_node, _format_context, _format_web_results, _parse_llm_response


@pytest.mark.unit
class TestRecommendNode:
    """Recommend 노드 단위 테스트 클래스"""
    
    def test_format_context_empty_passages(self):
        """빈 패시지 리스트 처리 테스트"""
        result = _format_context([])
        assert result == "관련 문서를 찾을 수 없습니다."
        print("✅ 빈 패시지 리스트 처리")
    
    def test_format_context_with_passages(self):
        """패시지 포맷팅 테스트"""
        passages = [
            {
                "doc_id": "DB손해보험_여행자보험약관",
                "page": 15,
                "text": "일본 여행 시 지진 특약이 포함된 해외여행보험을 추천합니다."
            },
            {
                "doc_id": "KB손해보험_여행자보험약관", 
                "page": 12,
                "text": "유럽 여행에 특화된 의료비 보장 특약이 우수합니다."
            }
        ]
        
        result = _format_context(passages)
        
        # 결과에 필요한 정보가 포함되어 있는지 확인
        assert "DB손해보험_여행자보험약관" in result
        assert "페이지 15" in result
        assert "KB손해보험_여행자보험약관" in result
        assert "페이지 12" in result
        assert "일본 여행" in result
        assert "유럽 여행" in result
        print("✅ 패시지 포맷팅")
    
    def test_format_context_text_truncation(self):
        """긴 텍스트 잘림 처리 테스트"""
        long_text = "A" * 1000  # 1000자 텍스트
        passages = [{
            "doc_id": "테스트_문서",
            "page": 1,
            "text": long_text
        }]
        
        result = _format_context(passages)
        # 500자로 제한되어야 함
        assert len(result.split('\n')[1]) <= 500
        print("✅ 텍스트 잘림 처리")
    
    def test_format_web_results_empty(self):
        """빈 웹 검색 결과 처리 테스트"""
        result = _format_web_results([])
        assert result == "실시간 뉴스 정보가 없습니다."
        print("✅ 빈 웹 검색 결과 처리")
    
    def test_format_web_results_with_data(self):
        """웹 검색 결과 포맷팅 테스트"""
        web_results = [
            {
                "title": "일본 여행 보험 가이드 2025",
                "snippet": "일본 여행 시 지진 대비 보험이 중요합니다. 최신 여행 경보를 확인하세요."
            },
            {
                "title": "도쿄 안전 정보",
                "snippet": "도쿄는 안전한 여행지이지만 지진에 대비한 보험 가입을 권장합니다."
            }
        ]
        
        result = _format_web_results(web_results)
        
        assert "일본 여행 보험 가이드 2025" in result
        assert "도쿄 안전 정보" in result
        assert "지진 대비" in result
        print("✅ 웹 검색 결과 포맷팅")
    
    def test_format_web_results_snippet_truncation(self):
        """웹 검색 결과 스니펫 잘림 처리 테스트"""
        long_snippet = "B" * 500  # 500자 스니펫
        web_results = [{
            "title": "테스트 뉴스",
            "snippet": long_snippet
        }]
        
        result = _format_web_results(web_results)
        # 200자로 제한되어야 함
        lines = result.split('\n')
        snippet_line = next((line for line in lines if line.startswith('B')), "")
        assert len(snippet_line) <= 200
        print("✅ 스니펫 잘림 처리")
    
    def test_parse_llm_response_valid_json(self):
        """유효한 JSON 응답 파싱 테스트"""
        valid_response = {
            "conclusion": "일본 여행에 DB손해보험을 추천합니다.",
            "evidence": ["지진 특약이 우수함", "의료비 보장이 충분함"],
            "caveats": ["지진 특약 가입 조건 확인 필요"],
            "quotes": [
                {
                    "text": "일본 여행 시 지진 특약이 포함된...",
                    "source": "DB손해보험_여행자보험약관_페이지15"
                }
            ],
            "recommendations": [
                {
                    "type": "DB손해보험",
                    "name": "DB손해보험",
                    "reason": "지진 특약이 우수함",
                    "coverage": "",
                    "priority": "높음",
                    "category": "보험사"
                }
            ],
            "web_info": {
                "latest_news": "일본 지진 경보 발령",
                "travel_alerts": "도쿄 지역 안전"
            }
        }
        
        response_text = json.dumps(valid_response, ensure_ascii=False)
        result = _parse_llm_response(response_text)
        
        assert result["conclusion"] == "일본 여행에 DB손해보험을 추천합니다."
        assert len(result["evidence"]) == 2
        assert len(result["recommendations"]) == 1
        assert result["recommendations"][0]["category"] == "보험사"
        print("✅ 유효한 JSON 응답 파싱")
    
    def test_parse_llm_response_json_with_markdown(self):
        """마크다운으로 감싸진 JSON 파싱 테스트"""
        valid_response = {
            "conclusion": "유럽 여행에 KB손해보험을 추천합니다.",
            "evidence": ["의료비 보장이 우수함"],
            "caveats": ["유럽 지역 제한 확인 필요"],
            "quotes": [],
            "recommendations": [],
            "web_info": {}
        }
        
        response_text = f"```json\n{json.dumps(valid_response, ensure_ascii=False)}\n```"
        result = _parse_llm_response(response_text)
        
        assert result["conclusion"] == "유럽 여행에 KB손해보험을 추천합니다."
        print("✅ 마크다운 JSON 파싱")
    
    def test_parse_llm_response_invalid_json(self):
        """잘못된 JSON 응답 처리 테스트"""
        invalid_response = "이것은 JSON이 아닙니다."
        result = _parse_llm_response(invalid_response)
        
        assert "오류가 발생했습니다" in result["conclusion"]
        assert result["evidence"] == ["응답 파싱 오류"]
        assert result["caveats"] == ["추가 확인이 필요합니다."]
        assert result["quotes"] == []
        assert result["recommendations"] == []
        assert result["web_info"] == {}
        print("✅ 잘못된 JSON 응답 처리")
    
    def test_parse_llm_response_missing_fields(self):
        """누락된 필드 처리 테스트"""
        incomplete_response = {
            "conclusion": "기본 추천입니다."
            # 다른 필드들 누락
        }
        
        response_text = json.dumps(incomplete_response, ensure_ascii=False)
        result = _parse_llm_response(response_text)
        
        assert result["conclusion"] == "기본 추천입니다."
        assert result["evidence"] == []
        assert result["caveats"] == []
        assert result["quotes"] == []
        assert result["recommendations"] == []
        assert result["web_info"] == {}
        print("✅ 누락된 필드 처리")
    
    @patch('graph.nodes.answerers.recommend.get_llm')
    def test_recommend_node_success(self, mock_get_llm):
        """성공적인 추천 노드 실행 테스트"""
        # Mock LLM 설정
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps({
            "conclusion": "일본 여행에 DB손해보험을 추천합니다.",
            "evidence": ["지진 특약이 우수함"],
            "caveats": ["지진 특약 가입 조건 확인 필요"],
            "quotes": [
                {
                    "text": "일본 여행 시 지진 특약이 포함된...",
                    "source": "DB손해보험_여행자보험약관_페이지15"
                }
            ],
            "recommendations": [
                {
                    "type": "DB손해보험",
                    "name": "DB손해보험",
                    "reason": "지진 특약이 우수함",
                    "coverage": "",
                    "priority": "높음",
                    "category": "보험사"
                }
            ],
            "web_info": {
                "latest_news": "일본 지진 경보 발령",
                "travel_alerts": "도쿄 지역 안전"
            }
        }, ensure_ascii=False)
        mock_llm.generate_content.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        
        # 테스트 상태
        state = {
            "question": "일본 여행에 추천하는 보험은?",
            "passages": [
                {
                    "doc_id": "DB손해보험_여행자보험약관",
                    "page": 15,
                    "text": "일본 여행 시 지진 특약이 포함된 해외여행보험을 추천합니다."
                }
            ],
            "web_results": [
                {
                    "title": "일본 여행 보험 가이드",
                    "snippet": "일본 여행 시 지진 대비 보험이 중요합니다."
                }
            ]
        }
        
        result = recommend_node(state)
        
        # 결과 검증
        assert "draft_answer" in result
        assert "final_answer" in result
        assert result["draft_answer"]["conclusion"] == "일본 여행에 DB손해보험을 추천합니다."
        assert len(result["draft_answer"]["recommendations"]) == 1
        assert result["draft_answer"]["recommendations"][0]["category"] == "보험사"
        print("✅ 성공적인 추천 노드 실행")
    
    @patch('graph.nodes.answerers.recommend.get_llm')
    def test_recommend_node_llm_failure(self, mock_get_llm):
        """LLM 호출 실패 시 fallback 테스트"""
        # Mock LLM이 예외 발생하도록 설정
        mock_get_llm.side_effect = Exception("LLM 호출 실패")
        
        state = {
            "question": "일본 여행에 추천하는 보험은?",
            "passages": [],
            "web_results": []
        }
        
        result = recommend_node(state)
        
        # Fallback 응답 검증
        assert "draft_answer" in result
        assert "final_answer" in result
        assert "오류가 발생했습니다" in result["draft_answer"]["conclusion"]
        assert result["draft_answer"]["evidence"] == ["LLM 호출 중 오류가 발생했습니다."]
        assert result["draft_answer"]["quotes"] == []
        assert result["draft_answer"]["recommendations"] == []
        assert result["draft_answer"]["web_info"] == {}
        print("✅ LLM 호출 실패 시 fallback")
    
    def test_recommend_node_empty_state(self):
        """빈 상태로 노드 실행 테스트"""
        state = {}
        
        with patch('graph.nodes.answerers.recommend.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.text = json.dumps({
                "conclusion": "추천 정보를 생성했습니다.",
                "evidence": [],
                "caveats": [],
                "quotes": [],
                "recommendations": [],
                "web_info": {}
            }, ensure_ascii=False)
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            result = recommend_node(state)
            
            assert "draft_answer" in result
            assert "final_answer" in result
            print("✅ 빈 상태로 노드 실행")
    
    def test_recommend_node_quotes_not_overwritten(self):
        """LLM 응답에 이미 quotes가 있을 때 덮어쓰지 않는지 테스트"""
        with patch('graph.nodes.answerers.recommend.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.text = json.dumps({
                "conclusion": "추천 정보를 생성했습니다.",
                "evidence": [],
                "caveats": [],
                "quotes": [
                    {
                        "text": "LLM이 생성한 인용구",
                        "source": "LLM_생성_인용구"
                    }
                ],
                "recommendations": [],
                "web_info": {}
            }, ensure_ascii=False)
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            state = {
                "question": "테스트 질문",
                "passages": [
                    {
                        "doc_id": "테스트_문서",
                        "page": 1,
                        "text": "테스트 텍스트"
                    }
                ],
                "web_results": []
            }
            
            result = recommend_node(state)
            
            # LLM이 생성한 quotes가 유지되어야 함
            assert len(result["draft_answer"]["quotes"]) == 1
            assert result["draft_answer"]["quotes"][0]["text"] == "LLM이 생성한 인용구"
            print("✅ quotes 덮어쓰기 방지")


@pytest.mark.unit
class TestRecommendNodeIntegration:
    """Recommend 노드 통합 테스트"""
    
    def test_recommend_node_with_real_data_structure(self):
        """실제 데이터 구조로 통합 테스트"""
        with patch('graph.nodes.answerers.recommend.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.text = json.dumps({
                "conclusion": "일본 여행에 맞는 보험을 추천합니다.",
                "evidence": ["지진 특약 필요", "의료비 보장 중요"],
                "caveats": ["지진 특약 가입 조건 확인 필요"],
                "quotes": [],
                "recommendations": [
                    {
                        "type": "DB손해보험",
                        "name": "DB손해보험",
                        "reason": "지진 특약이 우수함",
                        "coverage": "",
                        "priority": "높음",
                        "category": "보험사"
                    },
                    {
                        "type": "지진보험특약",
                        "name": "지진보험특약",
                        "reason": "일본의 지진 위험에 대비",
                        "coverage": "지진으로 인한 여행 중단 시 보상",
                        "priority": "높음",
                        "category": "특약"
                    }
                ],
                "web_info": {
                    "latest_news": "일본 지진 경보 발령",
                    "travel_alerts": "도쿄 지역 안전"
                }
            }, ensure_ascii=False)
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            # 실제와 유사한 상태 데이터
            state = {
                "question": "일본 여행에 추천하는 보험은?",
                "intent": "recommend",
                "passages": [
                    {
                        "doc_id": "DB손해보험_여행자보험약관",
                        "page": 15,
                        "text": "일본 여행 시 지진 특약이 포함된 해외여행보험을 추천합니다.",
                        "score": 0.95
                    },
                    {
                        "doc_id": "KB손해보험_여행자보험약관",
                        "page": 12,
                        "text": "유럽 여행에 특화된 의료비 보장 특약이 우수합니다.",
                        "score": 0.88
                    }
                ],
                "web_results": [
                    {
                        "title": "일본 여행 보험 가이드 2025",
                        "snippet": "일본 여행 시 지진 대비 보험이 중요합니다.",
                        "url": "https://example.com/guide"
                    },
                    {
                        "title": "도쿄 안전 정보",
                        "snippet": "도쿄는 안전한 여행지이지만 지진에 대비한 보험 가입을 권장합니다.",
                        "url": "https://example.com/safety"
                    }
                ]
            }
            
            result = recommend_node(state)
            
            # 결과 검증
            assert "draft_answer" in result
            assert "final_answer" in result
            
            answer = result["draft_answer"]
            assert answer["conclusion"] == "일본 여행에 맞는 보험을 추천합니다."
            assert len(answer["evidence"]) == 2
            assert len(answer["recommendations"]) == 2
            
            # 추천 검증
            recommendations = answer["recommendations"]
            assert recommendations[0]["category"] == "보험사"
            assert recommendations[1]["category"] == "특약"
            
            # 웹 정보 검증
            assert "latest_news" in answer["web_info"]
            assert "travel_alerts" in answer["web_info"]
            
            print("✅ 실제 데이터 구조로 통합 테스트")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
