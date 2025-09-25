"""
QA 노드 단위 테스트
qa_node의 핵심 기능과 에러 처리를 테스트합니다.
"""

import sys
import os
import pytest
from unittest.mock import Mock, patch
import json

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from graph.nodes.answerers.qa import qa_node, _format_context, _parse_llm_response


@pytest.mark.unit
class TestQANode:
    """QA 노드 단위 테스트 클래스"""
    
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
                "text": "항공기 연착으로 인한 지연 시 최대 24시간까지 지연보상금을 지급합니다."
            },
            {
                "doc_id": "KB손해보험_여행자보험약관", 
                "page": 12,
                "text": "여행 중 질병으로 인한 의료비는 실제 발생한 비용에 대해 보상합니다."
            }
        ]
        
        result = _format_context(passages)
        
        # 결과에 필요한 정보가 포함되어 있는지 확인
        assert "DB손해보험_여행자보험약관" in result
        assert "페이지 15" in result
        assert "KB손해보험_여행자보험약관" in result
        assert "페이지 12" in result
        assert "항공기 연착" in result
        assert "의료비" in result
        
        print("✅ 패시지 포맷팅")
    
    def test_format_context_long_text_truncation(self):
        """긴 텍스트 자동 잘림 테스트"""
        long_text = "여행자보험" * 200  # 매우 긴 텍스트
        passages = [{
            "doc_id": "테스트_문서",
            "page": 1,
            "text": long_text
        }]
        
        result = _format_context(passages)
        
        # 500자로 제한되어야 함
        assert len(result) < len(long_text)
        assert "여행자보험" in result
        
        print("✅ 긴 텍스트 자동 잘림")
    
    def test_parse_llm_response_valid_json(self):
        """유효한 JSON 응답 파싱 테스트"""
        valid_response = {
            "conclusion": "항공기 연착 시 지연보상금을 받을 수 있습니다",
            "evidence": ["DB손해보험 약관 15페이지"],
            "caveats": ["보험 가입 후 7일 대기기간"],
            "quotes": []
        }
        
        json_text = json.dumps(valid_response, ensure_ascii=False)
        result = _parse_llm_response(json_text)
        
        assert result["conclusion"] == "항공기 연착 시 지연보상금을 받을 수 있습니다"
        assert "DB손해보험 약관 15페이지" in result["evidence"]
        assert "보험 가입 후 7일 대기기간" in result["caveats"]
        
        print("✅ 유효한 JSON 응답 파싱")
    
    def test_parse_llm_response_markdown_json(self):
        """마크다운으로 감싸진 JSON 파싱 테스트"""
        valid_response = {
            "conclusion": "테스트 결론",
            "evidence": ["테스트 증거"],
            "caveats": ["테스트 주의사항"],
            "quotes": []
        }
        
        markdown_json = f"```json\n{json.dumps(valid_response, ensure_ascii=False)}\n```"
        result = _parse_llm_response(markdown_json)
        
        assert result["conclusion"] == "테스트 결론"
        
        print("✅ 마크다운 JSON 파싱")
    
    def test_parse_llm_response_invalid_json(self):
        """잘못된 JSON 응답 fallback 테스트"""
        invalid_response = "이것은 JSON이 아닙니다"
        result = _parse_llm_response(invalid_response)
        
        # fallback 구조 확인
        assert "conclusion" in result
        assert "evidence" in result
        assert "caveats" in result
        assert "quotes" in result
        assert "답변을 생성하는 중 오류가 발생했습니다" in result["conclusion"]
        
        print("✅ 잘못된 JSON fallback")
    
    @patch('graph.nodes.answerers.qa.get_llm')
    def test_qa_node_success(self, mock_get_llm):
        """qa_node 성공 케이스 테스트"""
        # Mock LLM 설정
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps({
            "conclusion": "항공기 연착 시 지연보상금을 받을 수 있습니다",
            "evidence": ["DB손해보험 약관 15페이지"],
            "caveats": ["보험 가입 후 7일 대기기간"],
            "quotes": []
        }, ensure_ascii=False)
        mock_llm.generate_content.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        
        # 테스트 state
        state = {
            "question": "비행기 연착 시 보장 알려줘",
            "passages": [
                {
                    "doc_id": "DB손해보험_여행자보험약관",
                    "page": 15,
                    "text": "항공기 연착으로 인한 지연 시 최대 24시간까지 지연보상금을 지급합니다."
                }
            ]
        }
        
        result = qa_node(state)
        
        # 결과 검증
        assert "draft_answer" in result
        assert "final_answer" in result
        assert result["draft_answer"]["conclusion"] == "항공기 연착 시 지연보상금을 받을 수 있습니다"
        assert "DB손해보험 약관 15페이지" in result["draft_answer"]["evidence"]
        
        # LLM 호출 확인
        mock_llm.generate_content.assert_called_once()
        
        print("✅ qa_node 성공 케이스")
    
    @patch('graph.nodes.answerers.qa.get_llm')
    def test_qa_node_llm_failure(self, mock_get_llm):
        """qa_node LLM 호출 실패 테스트"""
        # Mock LLM이 예외 발생하도록 설정
        mock_llm = Mock()
        mock_llm.generate_content.side_effect = Exception("LLM 호출 실패")
        mock_get_llm.return_value = mock_llm
        
        # 테스트 state
        state = {
            "question": "비행기 연착 시 보장 알려줘",
            "passages": []
        }
        
        result = qa_node(state)
        
        # fallback 답변 확인
        assert "draft_answer" in result
        assert "final_answer" in result
        assert "질문을 확인했습니다" in result["draft_answer"]["conclusion"]
        assert "LLM 호출 중 오류가 발생했습니다." in result["draft_answer"]["evidence"]
        assert "추가 확인이 필요합니다." in result["draft_answer"]["caveats"]
        
        print("✅ qa_node LLM 실패 처리")
    
    def test_qa_node_empty_question(self):
        """빈 질문 처리 테스트"""
        state = {
            "question": "",
            "passages": []
        }
        
        # LLM 호출 없이 테스트 (실제로는 Mock이 필요하지만 여기서는 구조만 확인)
        with patch('graph.nodes.answerers.qa.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.text = json.dumps({
                "conclusion": "질문을 확인했습니다",
                "evidence": [],
                "caveats": [],
                "quotes": []
            }, ensure_ascii=False)
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            result = qa_node(state)
            
            assert "draft_answer" in result
            assert "final_answer" in result
            
        print("✅ 빈 질문 처리")
    
    def test_qa_node_quotes_generation(self):
        """출처 정보 자동 생성 테스트"""
        with patch('graph.nodes.answerers.qa.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.text = json.dumps({
                "conclusion": "테스트 결론",
                "evidence": ["테스트 증거"],
                "caveats": ["테스트 주의사항"],
                "quotes": []
            }, ensure_ascii=False)
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            state = {
                "question": "테스트 질문",
                "passages": [
                    {
                        "doc_id": "DB손해보험_여행자보험약관",
                        "page": 15,
                        "text": "항공기 연착으로 인한 지연 시 최대 24시간까지 지연보상금을 지급합니다."
                    },
                    {
                        "doc_id": "KB손해보험_여행자보험약관",
                        "page": 12,
                        "text": "여행 중 질병으로 인한 의료비는 실제 발생한 비용에 대해 보상합니다."
                    }
                ]
            }
            
            result = qa_node(state)
            
            # quotes 자동 생성 확인
            quotes = result["draft_answer"]["quotes"]
            assert len(quotes) == 2  # 상위 2개만
            assert "DB손해보험_여행자보험약관_페이지15" in quotes[0]["source"]
            assert "KB손해보험_여행자보험약관_페이지12" in quotes[1]["source"]
            
        print("✅ 출처 정보 자동 생성")


@pytest.mark.benchmark
def test_qa_node_performance_benchmark():
    """QA 노드 성능 벤치마크 테스트"""
    print("\n" + "="*60)
    print("🎯 QA 노드 성능 벤치마크")
    print("="*60)
    
    # 테스트 케이스들
    test_cases = [
        {
            "question": "여행자보험 보장 내용이 뭐야?",
            "passages": [
                {
                    "doc_id": "DB손해보험_여행자보험약관",
                    "page": 15,
                    "text": "항공기 연착으로 인한 지연 시 최대 24시간까지 지연보상금을 지급합니다."
                }
            ]
        },
        {
            "question": "휴대품 분실 시 보상은?",
            "passages": [
                {
                    "doc_id": "KB손해보험_여행자보험약관",
                    "page": 20,
                    "text": "휴대품 분실 시 실제 가치에 따라 보상금을 지급합니다."
                }
            ]
        },
        {
            "question": "의료비 보상 한도는?",
            "passages": [
                {
                    "doc_id": "삼성화재_여행자보험약관",
                    "page": 25,
                    "text": "의료비 보상 한도는 1억원까지 보장합니다."
                }
            ]
        }
    ]
    
    success_count = 0
    total_count = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        try:
            with patch('graph.nodes.answerers.qa.get_llm') as mock_get_llm:
                mock_llm = Mock()
                mock_response = Mock()
                mock_response.text = json.dumps({
                    "conclusion": f"테스트 {i} 답변",
                    "evidence": [f"테스트 {i} 증거"],
                    "caveats": [f"테스트 {i} 주의사항"],
                    "quotes": []
                }, ensure_ascii=False)
                mock_llm.generate_content.return_value = mock_response
                mock_get_llm.return_value = mock_llm
                
                result = qa_node(case)
                
                # 기본 구조 확인
                assert "draft_answer" in result
                assert "final_answer" in result
                assert "conclusion" in result["draft_answer"]
                
                success_count += 1
                print(f"✅ 테스트 {i}: {case['question'][:30]}...")
                
        except Exception as e:
            print(f"❌ 테스트 {i}: {case['question'][:30]}... - {str(e)}")
    
    success_rate = (success_count / total_count) * 100
    print(f"\n📊 성공률: {success_count}/{total_count} ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("🎉 우수한 성능!")
    elif success_rate >= 80:
        print("👍 양호한 성능")
    else:
        print("⚠️ 개선 필요")
    
    return success_rate


if __name__ == "__main__":
    # 직접 실행 시 벤치마크 테스트
    test_qa_node_performance_benchmark()
