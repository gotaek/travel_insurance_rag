"""
Summarize 노드 단위 테스트 파일
summarize_node의 기능과 JSON 파싱, 에러 핸들링을 테스트합니다.
"""

import sys
import os
import pytest
import json
from unittest.mock import patch, MagicMock

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from graph.nodes.answerers.summarize import (
    summarize_node,
    _format_context,
    _parse_llm_response,
    _load_prompt
)


@pytest.mark.unit
class TestSummarizeNode:
    """Summarize 노드 단위 테스트 클래스"""
    
    def test_format_context_empty_passages(self):
        """빈 passages에 대한 컨텍스트 포맷팅 테스트"""
        result = _format_context([])
        assert result == "관련 문서를 찾을 수 없습니다."
    
    def test_format_context_single_passage(self):
        """단일 passage에 대한 컨텍스트 포맷팅 테스트"""
        passages = [
            {
                "doc_id": "DB손해보험",
                "page": 1,
                "text": "여행자보험은 해외여행 중 발생할 수 있는 각종 위험에 대비한 보험입니다."
            }
        ]
        
        result = _format_context(passages)
        
        assert "[문서 1] DB손해보험 (페이지 1)" in result
        assert "여행자보험은 해외여행 중" in result
    
    def test_format_context_multiple_passages(self):
        """여러 passages에 대한 컨텍스트 포맷팅 테스트"""
        passages = [
            {
                "doc_id": "DB손해보험",
                "page": 1,
                "text": "여행자보험은 해외여행 중 발생할 수 있는 각종 위험에 대비한 보험입니다."
            },
            {
                "doc_id": "KB손해보험", 
                "page": 2,
                "text": "보장 내용에는 사망, 상해, 질병, 수하물 등이 포함됩니다."
            }
        ]
        
        result = _format_context(passages)
        
        assert "[문서 1] DB손해보험 (페이지 1)" in result
        assert "[문서 2] KB손해보험 (페이지 2)" in result
        assert "여행자보험은 해외여행 중" in result
        assert "보장 내용에는 사망" in result
    
    def test_format_context_text_truncation(self):
        """긴 텍스트 자르기 테스트 (500자 제한)"""
        long_text = "여행자보험은 " + "매우 긴 텍스트입니다. " * 50  # 500자 이상
        
        passages = [
            {
                "doc_id": "테스트보험",
                "page": 1,
                "text": long_text
            }
        ]
        
        result = _format_context(passages)
        
        # 500자로 제한되었는지 확인
        text_part = result.split("\n")[1]  # 첫 번째 줄 제외
        assert len(text_part) <= 500
    
    def test_format_context_max_passages(self):
        """최대 5개 passages만 사용하는지 테스트"""
        passages = [
            {"doc_id": f"보험{i}", "page": i, "text": f"텍스트{i}"}
            for i in range(10)  # 10개 passages 생성
        ]
        
        result = _format_context(passages)
        
        # 5개만 사용되었는지 확인
        assert result.count("[문서") == 5
        assert "[문서 6]" not in result  # 6번째는 없어야 함


@pytest.mark.unit
class TestParseLLMResponse:
    """LLM 응답 파싱 테스트 클래스"""
    
    def test_parse_valid_json(self):
        """유효한 JSON 응답 파싱 테스트"""
        valid_json = {
            "conclusion": "여행자보험은 해외여행 중 위험에 대비한 보험입니다.",
            "evidence": ["사망보장", "상해보장"],
            "caveats": ["나이 제한 있음"],
            "quotes": []
        }
        
        response_text = json.dumps(valid_json, ensure_ascii=False)
        result = _parse_llm_response(response_text)
        
        assert result["conclusion"] == "여행자보험은 해외여행 중 위험에 대비한 보험입니다."
        assert "사망보장" in result["evidence"]
        assert "나이 제한 있음" in result["caveats"]
    
    def test_parse_json_with_markdown(self):
        """마크다운 코드 블록이 포함된 JSON 파싱 테스트"""
        valid_json = {
            "conclusion": "테스트 결론",
            "evidence": ["테스트 증거"],
            "caveats": ["테스트 주의사항"],
            "quotes": []
        }
        
        response_text = f"```json\n{json.dumps(valid_json, ensure_ascii=False)}\n```"
        result = _parse_llm_response(response_text)
        
        assert result["conclusion"] == "테스트 결론"
        assert "테스트 증거" in result["evidence"]
    
    def test_parse_invalid_json(self):
        """잘못된 JSON 응답 파싱 테스트"""
        invalid_json = "이것은 유효한 JSON이 아닙니다."
        
        result = _parse_llm_response(invalid_json)
        
        # fallback 응답 확인
        assert result["conclusion"] == "답변을 생성하는 중 오류가 발생했습니다."
        assert "응답 파싱 오류" in result["evidence"]
        assert "추가 확인이 필요합니다." in result["caveats"]
        assert result["quotes"] == []
    
    def test_parse_malformed_json(self):
        """형식이 잘못된 JSON 파싱 테스트"""
        malformed_json = '{"conclusion": "테스트", "evidence": ["테스트"], "caveats": ["테스트"], "quotes": [}'  # 마지막 괄호 누락
        
        result = _parse_llm_response(malformed_json)
        
        # fallback 응답 확인
        assert result["conclusion"] == "답변을 생성하는 중 오류가 발생했습니다."
        assert "응답 파싱 오류" in result["evidence"]


@pytest.mark.unit
class TestLoadPrompt:
    """프롬프트 로드 테스트 클래스"""
    
    def test_load_prompt_system_core(self):
        """system_core 프롬프트 로드 테스트"""
        prompt = _load_prompt("system_core")
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # system_core 프롬프트의 특징적인 내용 확인
        assert "여행자보험" in prompt or "보험" in prompt
    
    def test_load_prompt_summarize(self):
        """summarize 프롬프트 로드 테스트"""
        prompt = _load_prompt("summarize")
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # summarize 프롬프트의 특징적인 내용 확인
        assert "요약" in prompt
        assert "JSON" in prompt
        assert "conclusion" in prompt


@pytest.mark.unit
class TestSummarizeNodeIntegration:
    """Summarize 노드 통합 테스트 클래스"""
    
    @patch('graph.nodes.answerers.summarize.get_llm')
    def test_summarize_node_success(self, mock_get_llm):
        """성공적인 요약 노드 실행 테스트"""
        # Mock LLM 응답 설정
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "conclusion": "여행자보험은 해외여행 중 위험에 대비한 보험입니다.",
            "evidence": ["사망보장", "상해보장"],
            "caveats": ["나이 제한 있음"],
            "quotes": []
        }, ensure_ascii=False)
        mock_llm.generate_content.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        
        # 테스트 상태
        state = {
            "question": "여행자보험 약관을 요약해주세요",
            "passages": [
                {
                    "doc_id": "DB손해보험",
                    "doc_name": "여행자보험약관",
                    "page": 1,
                    "text": "여행자보험은 해외여행 중 발생할 수 있는 각종 위험에 대비한 보험입니다."
                }
            ]
        }
        
        result = summarize_node(state)
        
        # 결과 검증
        assert "draft_answer" in result
        assert "final_answer" in result
        assert result["draft_answer"]["conclusion"] == "여행자보험은 해외여행 중 위험에 대비한 보험입니다."
        assert "사망보장" in result["draft_answer"]["evidence"]
        
        # 출처 정보 확인
        assert "quotes" in result["draft_answer"]
        assert len(result["draft_answer"]["quotes"]) == 1
        assert "DB손해보험_여행자보험약관_페이지1" in result["draft_answer"]["quotes"][0]["source"]
    
    @patch('graph.nodes.answerers.summarize.get_llm')
    def test_summarize_node_llm_error(self, mock_get_llm):
        """LLM 호출 실패 시 fallback 테스트"""
        # Mock LLM 에러 설정
        mock_get_llm.side_effect = Exception("LLM 호출 실패")
        
        # 테스트 상태
        state = {
            "question": "여행자보험 약관을 요약해주세요",
            "passages": []
        }
        
        result = summarize_node(state)
        
        # fallback 응답 확인
        assert "draft_answer" in result
        assert "final_answer" in result
        assert "요약을 생성하는 중 오류가 발생했습니다" in result["draft_answer"]["conclusion"]
        assert "LLM 호출 중 오류가 발생했습니다." in result["draft_answer"]["evidence"]
        assert "추가 확인이 필요합니다." in result["draft_answer"]["caveats"]
    
    @patch('graph.nodes.answerers.summarize.get_llm')
    def test_summarize_node_empty_passages(self, mock_get_llm):
        """빈 passages에 대한 요약 노드 테스트"""
        # Mock LLM 응답 설정
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "conclusion": "관련 문서가 없어 요약할 수 없습니다.",
            "evidence": [],
            "caveats": ["문서 부족"],
            "quotes": []
        }, ensure_ascii=False)
        mock_llm.generate_content.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        
        # 테스트 상태 (빈 passages)
        state = {
            "question": "여행자보험 약관을 요약해주세요",
            "passages": []
        }
        
        result = summarize_node(state)
        
        # 결과 검증
        assert "draft_answer" in result
        assert "final_answer" in result
        assert result["draft_answer"]["quotes"] == []  # 빈 passages이므로 quotes도 빈 배열
    
    @patch('graph.nodes.answerers.summarize.get_llm')
    def test_summarize_node_multiple_passages(self, mock_get_llm):
        """여러 passages에 대한 요약 노드 테스트"""
        # Mock LLM 응답 설정
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "conclusion": "여러 보험사의 여행자보험을 요약했습니다.",
            "evidence": ["DB손해보험 보장내용", "KB손해보험 보장내용"],
            "caveats": ["각 보험사별 차이점 있음"],
            "quotes": []
        }, ensure_ascii=False)
        mock_llm.generate_content.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        
        # 테스트 상태 (여러 passages)
        state = {
            "question": "여러 보험사 여행자보험을 요약해주세요",
            "passages": [
                {
                    "doc_id": "DB손해보험",
                    "doc_name": "여행자보험약관",
                    "page": 1,
                    "text": "DB손해보험 여행자보험 내용입니다."
                },
                {
                    "doc_id": "KB손해보험",
                    "doc_name": "여행자보험약관", 
                    "page": 2,
                    "text": "KB손해보험 여행자보험 내용입니다."
                },
                {
                    "doc_id": "삼성화재",
                    "doc_name": "여행자보험약관",
                    "page": 3,
                    "text": "삼성화재 여행자보험 내용입니다."
                }
            ]
        }
        
        result = summarize_node(state)
        
        # 결과 검증
        assert "draft_answer" in result
        assert "final_answer" in result
        
        # 상위 3개 passages에 대한 quotes 생성 확인
        assert len(result["draft_answer"]["quotes"]) == 3
        assert "DB손해보험_여행자보험약관_페이지1" in result["draft_answer"]["quotes"][0]["source"]
        assert "KB손해보험_여행자보험약관_페이지2" in result["draft_answer"]["quotes"][1]["source"]
        assert "삼성화재_여행자보험약관_페이지3" in result["draft_answer"]["quotes"][2]["source"]


@pytest.mark.unit
class TestSummarizeNodeEdgeCases:
    """Summarize 노드 엣지 케이스 테스트 클래스"""
    
    def test_summarize_node_missing_question(self):
        """질문이 없는 상태에서의 요약 노드 테스트"""
        state = {
            "passages": [
                {
                    "doc_id": "테스트보험",
                    "doc_name": "테스트약관",
                    "page": 1,
                    "text": "테스트 내용입니다."
                }
            ]
        }
        
        # 질문이 없어도 에러가 발생하지 않아야 함
        with patch('graph.nodes.answerers.summarize.get_llm') as mock_get_llm:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.text = json.dumps({
                "conclusion": "테스트 요약",
                "evidence": ["테스트 증거"],
                "caveats": ["테스트 주의사항"],
                "quotes": []
            }, ensure_ascii=False)
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            result = summarize_node(state)
            
            assert "draft_answer" in result
            assert "final_answer" in result
    
    def test_summarize_node_missing_passages(self):
        """passages가 없는 상태에서의 요약 노드 테스트"""
        state = {
            "question": "여행자보험 약관을 요약해주세요"
        }
        
        # passages가 없어도 에러가 발생하지 않아야 함
        with patch('graph.nodes.answerers.summarize.get_llm') as mock_get_llm:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.text = json.dumps({
                "conclusion": "관련 문서가 없어 요약할 수 없습니다.",
                "evidence": [],
                "caveats": ["문서 부족"],
                "quotes": []
            }, ensure_ascii=False)
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            result = summarize_node(state)
            
            assert "draft_answer" in result
            assert "final_answer" in result
            assert result["draft_answer"]["quotes"] == []  # 빈 passages이므로 quotes도 빈 배열
    
    def test_summarize_node_incomplete_passage_data(self):
        """불완전한 passage 데이터에 대한 요약 노드 테스트"""
        state = {
            "question": "여행자보험 약관을 요약해주세요",
            "passages": [
                {
                    "doc_id": "테스트보험",
                    # doc_name 누락
                    "page": 1,
                    "text": "테스트 내용입니다."
                }
            ]
        }
        
        with patch('graph.nodes.answerers.summarize.get_llm') as mock_get_llm:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.text = json.dumps({
                "conclusion": "테스트 요약",
                "evidence": ["테스트 증거"],
                "caveats": ["테스트 주의사항"],
                "quotes": []
            }, ensure_ascii=False)
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            result = summarize_node(state)
            
            # doc_name이 누락되어도 기본값 "문서"로 처리되어야 함
            assert "draft_answer" in result
            assert "final_answer" in result
            assert "테스트보험_문서_페이지1" in result["draft_answer"]["quotes"][0]["source"]


@pytest.mark.unit
class TestSummarizeNodePerformance:
    """Summarize 노드 성능 테스트 클래스"""
    
    def test_summarize_node_large_passages(self):
        """대용량 passages에 대한 성능 테스트"""
        # 10개의 passages 생성 (5개만 사용되어야 함)
        passages = [
            {
                "doc_id": f"보험{i}",
                "doc_name": f"약관{i}",
                "page": i,
                "text": f"매우 긴 텍스트입니다. " * 100  # 각각 2000자 이상
            }
            for i in range(10)
        ]
        
        state = {
            "question": "여행자보험 약관을 요약해주세요",
            "passages": passages
        }
        
        with patch('graph.nodes.answerers.summarize.get_llm') as mock_get_llm:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.text = json.dumps({
                "conclusion": "대용량 문서 요약",
                "evidence": ["대용량 처리"],
                "caveats": ["처리 시간 소요"],
                "quotes": []
            }, ensure_ascii=False)
            mock_llm.generate_content.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            result = summarize_node(state)
            
            # 5개만 사용되었는지 확인
            assert "draft_answer" in result
            assert len(result["draft_answer"]["quotes"]) == 3  # 상위 3개만 quotes에 포함
