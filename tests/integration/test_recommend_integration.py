"""
Recommend 노드 통합 테스트
실제 워크플로우에서 recommend 노드의 동작을 테스트합니다.
"""

import sys
import os
import pytest
from unittest.mock import Mock, patch
import json

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from graph.nodes.answerers.recommend import recommend_node


@pytest.mark.integration
class TestRecommendIntegration:
    """Recommend 노드 통합 테스트 클래스"""
    
    @patch('graph.nodes.answerers.recommend.get_llm')
    def test_recommend_node_full_workflow(self, mock_get_llm):
        """전체 워크플로우 테스트"""
        # Mock LLM 설정
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps({
            "conclusion": "일본 여행에 맞는 보험을 추천합니다.",
            "evidence": ["지진 특약 필요", "의료비 보장 중요"],
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
        
        # 실제 워크플로우와 유사한 상태
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
        assert result["draft_answer"] == result["final_answer"]
        
        answer = result["draft_answer"]
        assert answer["conclusion"] == "일본 여행에 맞는 보험을 추천합니다."
        assert len(answer["evidence"]) == 2
        assert len(answer["caveats"]) == 1
        assert len(answer["quotes"]) == 1
        assert len(answer["recommendations"]) == 2
        
        # 추천 검증
        recommendations = answer["recommendations"]
        assert recommendations[0]["category"] == "보험사"
        assert recommendations[1]["category"] == "특약"
        
        # 웹 정보 검증
        assert "latest_news" in answer["web_info"]
        assert "travel_alerts" in answer["web_info"]
        
        print("✅ 전체 워크플로우 테스트 통과")
    
    @patch('graph.nodes.answerers.recommend.get_llm')
    def test_recommend_node_with_different_intents(self, mock_get_llm):
        """다양한 의도에 대한 추천 테스트"""
        test_cases = [
            {
                "question": "유럽 여행에 추천하는 보험은?",
                "expected_keywords": ["유럽", "의료비", "보장"],
                "description": "유럽 여행 추천"
            },
            {
                "question": "미국 여행에 추천하는 보험은?",
                "expected_keywords": ["미국", "의료비", "보장"],
                "description": "미국 여행 추천"
            },
            {
                "question": "동남아 여행에 추천하는 보험은?",
                "expected_keywords": ["동남아", "의료비", "보장"],
                "description": "동남아 여행 추천"
            }
        ]
        
        for case in test_cases:
            # Mock LLM 설정
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.text = json.dumps({
                "conclusion": f"{case['question']}에 대한 추천입니다.",
                "evidence": ["의료비 보장 중요"],
                "caveats": ["추가 확인 필요"],
                "quotes": [],
                "recommendations": [
                    {
                        "type": "테스트보험사",
                        "name": "테스트보험사",
                        "reason": "의료비 보장이 우수함",
                        "coverage": "",
                        "priority": "높음",
                        "category": "보험사"
                    }
                ],
                "web_info": {}
            }, ensure_ascii=False)
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            state = {
                "question": case["question"],
                "intent": "recommend",
                "passages": [
                    {
                        "doc_id": "테스트_문서",
                        "page": 1,
                        "text": "테스트 텍스트",
                        "score": 0.9
                    }
                ],
                "web_results": []
            }
            
            result = recommend_node(state)
            
            # 결과 검증
            assert "draft_answer" in result
            assert result["draft_answer"]["conclusion"] == f"{case['question']}에 대한 추천입니다."
            assert len(result["draft_answer"]["recommendations"]) == 1
            
            print(f"✅ {case['description']} 테스트 통과")
    
    @patch('graph.nodes.answerers.recommend.get_llm')
    def test_recommend_node_with_empty_data(self, mock_get_llm):
        """빈 데이터로 추천 테스트"""
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
        
        state = {
            "question": "추천해주세요",
            "intent": "recommend",
            "passages": [],
            "web_results": []
        }
        
        result = recommend_node(state)
        
        # 결과 검증
        assert "draft_answer" in result
        assert result["draft_answer"]["conclusion"] == "추천 정보를 생성했습니다."
        assert result["draft_answer"]["recommendations"] == []
        
        print("✅ 빈 데이터로 추천 테스트 통과")
    
    @patch('graph.nodes.answerers.recommend.get_llm')
    def test_recommend_node_with_malformed_llm_response(self, mock_get_llm):
        """잘못된 LLM 응답 처리 테스트"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = "이것은 유효한 JSON이 아닙니다."
        mock_llm.generate_content.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        
        state = {
            "question": "추천해주세요",
            "intent": "recommend",
            "passages": [],
            "web_results": []
        }
        
        result = recommend_node(state)
        
        # Fallback 응답 검증
        assert "draft_answer" in result
        assert "오류가 발생했습니다" in result["draft_answer"]["conclusion"]
        assert result["draft_answer"]["evidence"] == ["응답 파싱 오류"]
        assert result["draft_answer"]["quotes"] == []
        assert result["draft_answer"]["recommendations"] == []
        assert result["draft_answer"]["web_info"] == {}
        
        print("✅ 잘못된 LLM 응답 처리 테스트 통과")
    
    @patch('graph.nodes.answerers.recommend.get_llm')
    def test_recommend_node_with_large_data(self, mock_get_llm):
        """대용량 데이터 처리 테스트"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps({
            "conclusion": "대용량 데이터로 추천합니다.",
            "evidence": ["데이터 1", "데이터 2"],
            "caveats": ["주의사항 1"],
            "quotes": [],
            "recommendations": [
                {
                    "type": "보험사1",
                    "name": "보험사1",
                    "reason": "추천 이유 1",
                    "coverage": "",
                    "priority": "높음",
                    "category": "보험사"
                },
                {
                    "type": "보험사2",
                    "name": "보험사2",
                    "reason": "추천 이유 2",
                    "coverage": "",
                    "priority": "보통",
                    "category": "보험사"
                }
            ],
            "web_info": {
                "latest_news": "최신 뉴스",
                "travel_alerts": "여행 경보"
            }
        }, ensure_ascii=False)
        mock_llm.generate_content.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        
        # 대용량 패시지 데이터
        large_passages = []
        for i in range(10):
            large_passages.append({
                "doc_id": f"문서_{i}",
                "page": i + 1,
                "text": f"대용량 텍스트 데이터 {i} " * 100,  # 긴 텍스트
                "score": 0.9 - (i * 0.01)
            })
        
        # 대용량 웹 결과 데이터
        large_web_results = []
        for i in range(5):
            large_web_results.append({
                "title": f"뉴스 제목 {i}",
                "snippet": f"뉴스 내용 {i} " * 50,  # 긴 스니펫
                "url": f"https://example.com/news{i}"
            })
        
        state = {
            "question": "대용량 데이터로 추천해주세요",
            "intent": "recommend",
            "passages": large_passages,
            "web_results": large_web_results
        }
        
        result = recommend_node(state)
        
        # 결과 검증
        assert "draft_answer" in result
        assert result["draft_answer"]["conclusion"] == "대용량 데이터로 추천합니다."
        assert len(result["draft_answer"]["recommendations"]) == 2
        
        print("✅ 대용량 데이터 처리 테스트 통과")
    
    @patch('graph.nodes.answerers.recommend.get_llm')
    def test_recommend_node_performance(self, mock_get_llm):
        """성능 테스트"""
        import time
        
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps({
            "conclusion": "성능 테스트 추천",
            "evidence": ["성능 테스트"],
            "caveats": [],
            "quotes": [],
            "recommendations": [],
            "web_info": {}
        }, ensure_ascii=False)
        mock_llm.generate_content.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        
        state = {
            "question": "성능 테스트",
            "intent": "recommend",
            "passages": [
                {
                    "doc_id": "성능_테스트_문서",
                    "page": 1,
                    "text": "성능 테스트 텍스트",
                    "score": 0.9
                }
            ],
            "web_results": []
        }
        
        start_time = time.time()
        result = recommend_node(state)
        end_time = time.time()
        
        # 성능 검증 (1초 이내 완료)
        assert (end_time - start_time) < 1.0
        assert "draft_answer" in result
        
        print(f"✅ 성능 테스트 통과 (소요시간: {end_time - start_time:.3f}초)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
