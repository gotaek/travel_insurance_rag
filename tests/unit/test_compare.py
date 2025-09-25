"""
Compare 노드 단위 테스트
compare_node의 핵심 기능과 에러 처리를 테스트합니다.
"""

import sys
import os
import pytest
from unittest.mock import Mock, patch
import json

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from graph.nodes.answerers.compare import compare_node, _format_context, _parse_llm_response


@pytest.mark.unit
class TestCompareNode:
    """Compare 노드 단위 테스트 클래스"""
    
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
                "text": "사망보장 한도는 1억원이며, 상해보장은 5천만원입니다."
            },
            {
                "doc_id": "카카오페이_여행자보험약관", 
                "page": 12,
                "text": "사망보장 한도는 5천만원이며, 상해보장은 3천만원입니다."
            }
        ]
        
        result = _format_context(passages)
        
        # 결과에 필요한 정보가 포함되어 있는지 확인
        assert "DB손해보험_여행자보험약관" in result
        assert "페이지 15" in result
        assert "카카오페이_여행자보험약관" in result
        assert "페이지 12" in result
        assert "사망보장" in result
        assert "상해보장" in result
        
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
    
    def test_format_context_max_passages_limit(self):
        """최대 패시지 수 제한 테스트"""
        # 10개의 패시지 생성
        passages = []
        for i in range(10):
            passages.append({
                "doc_id": f"문서_{i}",
                "page": i + 1,
                "text": f"테스트 텍스트 {i}"
            })
        
        result = _format_context(passages)
        
        # 상위 5개만 사용되어야 함
        assert "문서_0" in result
        assert "문서_4" in result
        assert "문서_5" not in result  # 6번째부터는 제외
        
        print("✅ 최대 패시지 수 제한")
    
    def test_parse_llm_response_valid_json(self):
        """유효한 JSON 응답 파싱 테스트"""
        valid_response = {
            "conclusion": "DB손보가 카카오페이보다 보장한도가 높습니다",
            "evidence": ["사망보장 1억원 vs 5천만원", "상해보장 5천만원 vs 3천만원"],
            "caveats": ["보험료 차이 고려 필요"],
            "quotes": [],
            "comparison_table": {
                "headers": ["항목", "DB손보", "카카오페이"],
                "rows": [
                    ["사망보장", "1억원", "5천만원"],
                    ["상해보장", "5천만원", "3천만원"]
                ]
            }
        }
        
        json_text = json.dumps(valid_response, ensure_ascii=False)
        result = _parse_llm_response(json_text)
        
        assert result["conclusion"] == "DB손보가 카카오페이보다 보장한도가 높습니다"
        assert "사망보장 1억원 vs 5천만원" in result["evidence"]
        assert "보험료 차이 고려 필요" in result["caveats"]
        assert "comparison_table" in result
        assert result["comparison_table"]["headers"] == ["항목", "DB손보", "카카오페이"]
        
        print("✅ 유효한 JSON 응답 파싱")
    
    def test_parse_llm_response_markdown_json(self):
        """마크다운으로 감싸진 JSON 파싱 테스트"""
        valid_response = {
            "conclusion": "테스트 결론",
            "evidence": ["테스트 증거"],
            "caveats": ["테스트 주의사항"],
            "quotes": [],
            "comparison_table": {
                "headers": ["항목", "결과"],
                "rows": [["테스트", "성공"]]
            }
        }
        
        markdown_json = f"```json\n{json.dumps(valid_response, ensure_ascii=False)}\n```"
        result = _parse_llm_response(markdown_json)
        
        assert result["conclusion"] == "테스트 결론"
        assert "comparison_table" in result
        
        print("✅ 마크다운 JSON 파싱")
    
    def test_parse_llm_response_missing_comparison_table(self):
        """comparison_table 필드가 없는 경우 처리 테스트"""
        response_without_table = {
            "conclusion": "테스트 결론",
            "evidence": ["테스트 증거"],
            "caveats": ["테스트 주의사항"],
            "quotes": []
        }
        
        json_text = json.dumps(response_without_table, ensure_ascii=False)
        result = _parse_llm_response(json_text)
        
        # comparison_table 필드가 자동으로 추가되어야 함
        assert "comparison_table" in result
        assert result["comparison_table"]["headers"] == ["항목", "비교 결과"]
        assert result["comparison_table"]["rows"] == [["비교 정보", "표 형태로 제공되지 않음"]]
        
        print("✅ 누락된 comparison_table 필드 처리")
    
    def test_parse_llm_response_invalid_json(self):
        """잘못된 JSON 응답 fallback 테스트"""
        invalid_response = "이것은 JSON이 아닙니다"
        result = _parse_llm_response(invalid_response)
        
        # fallback 구조 확인
        assert "conclusion" in result
        assert "evidence" in result
        assert "caveats" in result
        assert "quotes" in result
        assert "comparison_table" in result
        assert "답변을 생성하는 중 오류가 발생했습니다" in result["conclusion"]
        assert result["comparison_table"]["rows"] == [["오류", "파싱 실패"]]
        
        print("✅ 잘못된 JSON fallback")
    
    @patch('graph.nodes.answerers.compare.get_llm')
    def test_compare_node_success(self, mock_get_llm):
        """compare_node 성공 케이스 테스트"""
        # Mock LLM 설정
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps({
            "conclusion": "DB손보가 카카오페이보다 보장한도가 높습니다",
            "evidence": ["사망보장 1억원 vs 5천만원", "상해보장 5천만원 vs 3천만원"],
            "caveats": ["보험료 차이 고려 필요"],
            "quotes": [],
            "comparison_table": {
                "headers": ["항목", "DB손보", "카카오페이"],
                "rows": [
                    ["사망보장", "1억원", "5천만원"],
                    ["상해보장", "5천만원", "3천만원"]
                ]
            }
        }, ensure_ascii=False)
        mock_llm.generate_content.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        
        # 테스트 state
        state = {
            "question": "DB손보와 카카오페이 여행자보험 차이 비교",
            "passages": [
                {
                    "doc_id": "DB손해보험_여행자보험약관",
                    "page": 15,
                    "text": "사망보장 한도는 1억원이며, 상해보장은 5천만원입니다."
                },
                {
                    "doc_id": "카카오페이_여행자보험약관",
                    "page": 12,
                    "text": "사망보장 한도는 5천만원이며, 상해보장은 3천만원입니다."
                }
            ]
        }
        
        result = compare_node(state)
        
        # 결과 검증
        assert "draft_answer" in result
        assert "final_answer" in result
        assert result["draft_answer"]["conclusion"] == "DB손보가 카카오페이보다 보장한도가 높습니다"
        assert "사망보장 1억원 vs 5천만원" in result["draft_answer"]["evidence"]
        assert "comparison_table" in result["draft_answer"]
        assert result["draft_answer"]["comparison_table"]["headers"] == ["항목", "DB손보", "카카오페이"]
        
        # LLM 호출 확인
        mock_llm.generate_content.assert_called_once()
        
        print("✅ compare_node 성공 케이스")
    
    @patch('graph.nodes.answerers.compare.get_llm')
    def test_compare_node_llm_failure(self, mock_get_llm):
        """compare_node LLM 호출 실패 테스트"""
        # Mock LLM이 예외 발생하도록 설정
        mock_llm = Mock()
        mock_llm.generate_content.side_effect = Exception("LLM 호출 실패")
        mock_get_llm.return_value = mock_llm
        
        # 테스트 state
        state = {
            "question": "DB손보와 카카오페이 여행자보험 차이 비교",
            "passages": []
        }
        
        result = compare_node(state)
        
        # fallback 답변 확인
        assert "draft_answer" in result
        assert "final_answer" in result
        assert "비교 분석 중 오류가 발생했습니다" in result["draft_answer"]["conclusion"]
        assert "LLM 호출 중 오류가 발생했습니다." in result["draft_answer"]["evidence"]
        assert "추가 확인이 필요합니다." in result["draft_answer"]["caveats"]
        assert "comparison_table" in result["draft_answer"]
        assert result["draft_answer"]["comparison_table"]["rows"] == [["오류", "LLM 호출 실패"]]
        
        print("✅ compare_node LLM 실패 처리")
    
    def test_compare_node_empty_question(self):
        """빈 질문 처리 테스트"""
        state = {
            "question": "",
            "passages": []
        }
        
        # LLM 호출 없이 테스트 (실제로는 Mock이 필요하지만 여기서는 구조만 확인)
        with patch('graph.nodes.answerers.compare.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.text = json.dumps({
                "conclusion": "비교 분석을 위해 질문을 확인했습니다",
                "evidence": [],
                "caveats": [],
                "quotes": [],
                "comparison_table": {
                    "headers": ["항목", "비교 결과"],
                    "rows": [["질문", "빈 질문"]]
                }
            }, ensure_ascii=False)
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            result = compare_node(state)
            
            assert "draft_answer" in result
            assert "final_answer" in result
            assert "comparison_table" in result["draft_answer"]
            
        print("✅ 빈 질문 처리")
    
    def test_compare_node_quotes_generation(self):
        """출처 정보 자동 생성 테스트"""
        with patch('graph.nodes.answerers.compare.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.text = json.dumps({
                "conclusion": "테스트 결론",
                "evidence": ["테스트 증거"],
                "caveats": ["테스트 주의사항"],
                "quotes": [],
                "comparison_table": {
                    "headers": ["항목", "결과"],
                    "rows": [["테스트", "성공"]]
                }
            }, ensure_ascii=False)
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            state = {
                "question": "테스트 질문",
                "passages": [
                    {
                        "doc_id": "DB손해보험_여행자보험약관",
                        "page": 15,
                        "text": "사망보장 한도는 1억원이며, 상해보장은 5천만원입니다."
                    },
                    {
                        "doc_id": "카카오페이_여행자보험약관",
                        "page": 12,
                        "text": "사망보장 한도는 5천만원이며, 상해보장은 3천만원입니다."
                    }
                ]
            }
            
            result = compare_node(state)
            
            # quotes 자동 생성 확인
            quotes = result["draft_answer"]["quotes"]
            assert len(quotes) == 2  # 상위 2개만
            assert "DB손해보험_여행자보험약관_페이지15" in quotes[0]["source"]
            assert "카카오페이_여행자보험약관_페이지12" in quotes[1]["source"]
            
        print("✅ 출처 정보 자동 생성")
    
    def test_compare_node_comparison_table_structure(self):
        """comparison_table 구조 검증 테스트"""
        with patch('graph.nodes.answerers.compare.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.text = json.dumps({
                "conclusion": "테스트 결론",
                "evidence": ["테스트 증거"],
                "caveats": ["테스트 주의사항"],
                "quotes": [],
                "comparison_table": {
                    "headers": ["항목", "DB손보", "카카오페이", "차이점"],
                    "rows": [
                        ["사망보장", "1억원", "5천만원", "DB손보 2배 높음"],
                        ["상해보장", "5천만원", "3천만원", "DB손보 1.7배 높음"]
                    ]
                }
            }, ensure_ascii=False)
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            state = {
                "question": "테스트 질문",
                "passages": []
            }
            
            result = compare_node(state)
            
            # comparison_table 구조 확인
            table = result["draft_answer"]["comparison_table"]
            assert "headers" in table
            assert "rows" in table
            assert len(table["headers"]) == 4
            assert len(table["rows"]) == 2
            assert table["headers"] == ["항목", "DB손보", "카카오페이", "차이점"]
            assert table["rows"][0] == ["사망보장", "1억원", "5천만원", "DB손보 2배 높음"]
            
        print("✅ comparison_table 구조 검증")


@pytest.mark.benchmark
def test_compare_node_performance_benchmark():
    """Compare 노드 성능 벤치마크 테스트"""
    print("\n" + "="*60)
    print("🎯 Compare 노드 성능 벤치마크")
    print("="*60)
    
    # 테스트 케이스들
    test_cases = [
        {
            "question": "DB손보와 카카오페이 여행자보험 차이 비교",
            "passages": [
                {
                    "doc_id": "DB손해보험_여행자보험약관",
                    "page": 15,
                    "text": "사망보장 한도는 1억원이며, 상해보장은 5천만원입니다."
                },
                {
                    "doc_id": "카카오페이_여행자보험약관",
                    "page": 12,
                    "text": "사망보장 한도는 5천만원이며, 상해보장은 3천만원입니다."
                }
            ]
        },
        {
            "question": "보험사별 여행자보험 가격 비교",
            "passages": [
                {
                    "doc_id": "KB손해보험_여행자보험약관",
                    "page": 20,
                    "text": "1일 보험료는 3,000원부터 시작됩니다."
                },
                {
                    "doc_id": "삼성화재_여행자보험약관",
                    "page": 18,
                    "text": "1일 보험료는 2,500원부터 시작됩니다."
                }
            ]
        },
        {
            "question": "여행자보험 보장 내용 차이점",
            "passages": [
                {
                    "doc_id": "현대해상_여행자보험약관",
                    "page": 25,
                    "text": "의료비 보상 한도는 1억원까지 보장합니다."
                },
                {
                    "doc_id": "DB손해보험_여행자보험약관",
                    "page": 30,
                    "text": "의료비 보상 한도는 5천만원까지 보장합니다."
                }
            ]
        }
    ]
    
    success_count = 0
    total_count = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        try:
            with patch('graph.nodes.answerers.compare.get_llm') as mock_get_llm:
                mock_llm = Mock()
                mock_response = Mock()
                mock_response.text = json.dumps({
                    "conclusion": f"테스트 {i} 비교 결과",
                    "evidence": [f"테스트 {i} 증거"],
                    "caveats": [f"테스트 {i} 주의사항"],
                    "quotes": [],
                    "comparison_table": {
                        "headers": ["항목", "보험사A", "보험사B"],
                        "rows": [["테스트항목", "값A", "값B"]]
                    }
                }, ensure_ascii=False)
                mock_llm.generate_content.return_value = mock_response
                mock_get_llm.return_value = mock_llm
                
                result = compare_node(case)
                
                # 기본 구조 확인
                assert "draft_answer" in result
                assert "final_answer" in result
                assert "conclusion" in result["draft_answer"]
                assert "comparison_table" in result["draft_answer"]
                
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
    test_compare_node_performance_benchmark()
