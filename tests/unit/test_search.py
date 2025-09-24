"""
Search 노드 최적화 테스트
웹 검색 결과를 활용한 개선된 search 노드의 기능을 테스트합니다.
"""

import sys
import os
import pytest
from unittest.mock import Mock, patch, MagicMock

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from graph.nodes.search import (
    search_node,
    _enhance_query_with_web_results,
    _extract_keywords_from_web_results,
    _determine_k_value,
    _enhanced_hybrid_search,
    _add_web_context_to_results,
    _calculate_web_relevance,
    _convert_web_results_to_passages,
    _enhanced_hybrid_search_with_web_weight
)
from retriever.korean_tokenizer import (
    extract_insurance_keywords,
    calculate_keyword_relevance,
    get_keyword_weights
)


@pytest.mark.unit
class TestSearchOptimization:
    """Search 노드 최적화 테스트 클래스"""
    
    def test_enhance_query_with_web_results(self):
        """웹 검색 결과를 활용한 쿼리 확장 테스트"""
        # 테스트 데이터
        original_query = "여행자보험 보장내용"
        web_results = [
            {
                "title": "DB손해보험 여행자보험 특약 보장내용",
                "snippet": "해외여행보험의 상해보장과 질병보장에 대한 상세 정보"
            },
            {
                "title": "여행자보험 보험료 비교",
                "snippet": "카카오페이 여행자보험과 삼성화재 보험료 비교"
            }
        ]
        
        # 쿼리 확장 테스트
        enhanced_query = _enhance_query_with_web_results(original_query, web_results)
        
        # 결과 검증
        assert enhanced_query != original_query
        assert "여행자보험" in enhanced_query
        assert "보장내용" in enhanced_query
        # 웹 결과에서 추출된 키워드가 포함되어야 함
        assert len(enhanced_query.split()) > len(original_query.split())
    
    def test_enhance_query_without_web_results(self):
        """웹 검색 결과가 없을 때 원본 쿼리 반환 테스트"""
        original_query = "여행자보험 보장내용"
        web_results = []
        
        enhanced_query = _enhance_query_with_web_results(original_query, web_results)
        
        assert enhanced_query == original_query
    
    def test_extract_keywords_from_web_results(self):
        """웹 검색 결과에서 키워드 추출 테스트"""
        web_results = [
            {
                "title": "DB손해보험 여행자보험 특약",
                "snippet": "해외여행보험 상해보장 질병보장 의료비"
            },
            {
                "title": "여행자보험 보험료 비교",
                "snippet": "카카오페이 삼성화재 현대해상 보험료"
            }
        ]
        
        keywords = _extract_keywords_from_web_results(web_results)
        
        # 여행자보험 관련 키워드가 추출되어야 함
        assert len(keywords) > 0
        assert any("보험" in keyword for keyword in keywords)
        assert any("여행" in keyword for keyword in keywords)
    
    def test_determine_k_value(self):
        """동적 k 값 조정 테스트"""
        # 웹 결과가 있을 때
        query = "여행자보험 보장내용이 뭐야?"
        web_results = [{"title": "테스트", "snippet": "테스트"}]
        
        k_with_web = _determine_k_value(query, web_results)
        assert k_with_web > 5  # 웹 결과가 있으면 k 값이 증가해야 함
        
        # 웹 결과가 없을 때
        k_without_web = _determine_k_value(query, [])
        assert k_without_web == 5  # 기본값
        
        # 긴 질문일 때
        long_query = "여행자보험의 보장내용과 특약에 대한 상세한 정보를 알고 싶습니다"
        k_long_query = _determine_k_value(long_query, [])
        assert k_long_query > 5  # 긴 질문이면 k 값이 증가해야 함
    
    def test_add_web_context_to_results(self):
        """로컬 검색 결과에 웹 컨텍스트 추가 테스트"""
        local_results = [
            {
                "text": "여행자보험 보장내용에 대한 정보",
                "doc_id": "doc1",
                "score": 0.8
            }
        ]
        web_results = [
            {
                "title": "여행자보험 보장내용",
                "snippet": "해외여행보험의 상세한 보장내용"
            }
        ]
        
        enhanced_results = _add_web_context_to_results(local_results, web_results)
        
        # 웹 컨텍스트 정보가 추가되어야 함
        assert len(enhanced_results) == 1
        assert "web_context" in enhanced_results[0]
        assert enhanced_results[0]["web_context"]["has_web_info"] is True
        assert enhanced_results[0]["web_context"]["web_sources_count"] > 0
    
    def test_calculate_web_relevance(self):
        """웹 관련성 계산 테스트"""
        local_result = {
            "text": "여행자보험의 상해보장과 질병보장에 대한 정보"
        }
        web_results = [
            {
                "title": "여행자보험 보장내용",
                "snippet": "해외여행보험 상해보장 질병보장 의료비"
            }
        ]
        
        relevance = _calculate_web_relevance(local_result, web_results)
        
        # 관련성이 계산되어야 함
        assert 0.0 <= relevance <= 1.0
        assert relevance > 0.0  # 관련성이 있어야 함
    
    def test_calculate_web_relevance_no_web_results(self):
        """웹 결과가 없을 때 관련성 계산 테스트"""
        local_result = {"text": "테스트"}
        web_results = []
        
        relevance = _calculate_web_relevance(local_result, web_results)
        
        assert relevance == 0.0
    
    def test_convert_web_results_to_passages(self):
        """웹 검색 결과를 패시지로 변환하는 테스트"""
        web_results = [
            {
                "title": "DB손해보험 여행자보험 보장내용",
                "snippet": "해외여행보험의 상세한 보장내용과 특약 정보",
                "url": "https://example.com",
                "score_web": 0.8,
                "relevance_score": 0.7
            },
            {
                "title": "여행자보험 보험료 비교",
                "snippet": "카카오페이와 삼성화재 여행자보험 보험료 비교",
                "url": "https://example2.com",
                "score_web": 0.6,
                "relevance_score": 0.5
            }
        ]
        
        passages = _convert_web_results_to_passages(web_results)
        
        # 결과 검증
        assert len(passages) == 2
        assert passages[0]["source"] == "web"
        assert passages[0]["url"] == "https://example.com"
        assert passages[0]["title"] == "DB손해보험 여행자보험 보장내용"
        assert passages[0]["score_web"] == 0.8
        assert passages[0]["web_relevance_score"] == 0.7
        assert passages[0]["doc_id"].startswith("web_")
    
    def test_convert_web_results_to_passages_empty(self):
        """빈 웹 검색 결과를 패시지로 변환하는 테스트"""
        passages = _convert_web_results_to_passages([])
        assert passages == []
    
    def test_enhanced_hybrid_search_with_web_weight(self):
        """웹 가중치를 반영한 향상된 하이브리드 검색 테스트"""
        query = "여행자보험 보장내용"
        vector_results = [
            {"text": "여행자보험 보장내용", "doc_id": "doc1", "score": 0.8}
        ]
        keyword_results = [
            {"text": "여행자보험 보장내용", "doc_id": "doc1", "score": 0.7}
        ]
        web_passages = [
            {
                "text": "웹 검색 결과",
                "source": "web",
                "score_web": 0.6,
                "web_relevance_score": 0.5
            }
        ]
        
        result = _enhanced_hybrid_search_with_web_weight(
            query, vector_results, keyword_results, web_passages, k=3
        )
        
        # 결과 검증
        assert len(result) <= 3
        assert all("score" in item for item in result)
        assert all(0.0 <= item["score"] <= 1.0 for item in result)
        
        # 점수 기준 정렬 확인
        scores = [item["score"] for item in result]
        assert scores == sorted(scores, reverse=True)
    
    def test_enhanced_hybrid_search_with_web_weight_no_web(self):
        """웹 패시지가 없을 때 향상된 하이브리드 검색 테스트"""
        query = "여행자보험 보장내용"
        vector_results = [
            {"text": "여행자보험 보장내용", "doc_id": "doc1", "score": 0.8}
        ]
        keyword_results = [
            {"text": "여행자보험 보장내용", "doc_id": "doc1", "score": 0.7}
        ]
        web_passages = []
        
        result = _enhanced_hybrid_search_with_web_weight(
            query, vector_results, keyword_results, web_passages, k=2
        )
        
        # 결과 검증
        assert len(result) <= 2
        assert all("score" in item for item in result)
        assert all(0.0 <= item["score"] <= 1.0 for item in result)


@pytest.mark.unit
class TestSearchEdgeCases:
    """Search 노드 엣지 케이스 테스트"""
    
    def test_search_node_empty_question(self):
        """빈 질문에 대한 search 노드 테스트"""
        state = {
            "question": "",
            "web_results": []
        }
        
        result = search_node(state)
        
        # 결과 검증
        assert "passages" in result
        assert len(result["passages"]) == 0
        assert result["search_meta"]["reason"] == "empty_question"
        assert result["search_meta"]["k_value"] == 0
        assert result["search_meta"]["candidates_count"] == 0
    
    def test_search_node_whitespace_question(self):
        """공백만 있는 질문에 대한 search 노드 테스트"""
        state = {
            "question": "   ",
            "web_results": []
        }
        
        result = search_node(state)
        
        # 결과 검증
        assert "passages" in result
        assert len(result["passages"]) == 0
        assert result["search_meta"]["reason"] == "empty_question"
    
    @patch('graph.nodes.search.vector_search')
    @patch('graph.nodes.search.keyword_search_full_corpus')
    def test_search_node_no_results(self, mock_keyword, mock_vector):
        """검색 결과가 없을 때 search 노드 테스트"""
        # 모킹 설정 - 빈 결과 반환
        mock_vector.return_value = []
        mock_keyword.return_value = []
        
        state = {
            "question": "존재하지 않는 질문",
            "web_results": []
        }
        
        result = search_node(state)
        
        # 결과 검증
        assert "passages" in result
        assert len(result["passages"]) == 0
        assert result["search_meta"]["reason"] == "no_search_results"
        assert result["search_meta"]["candidates_count"] == 0
    
    @patch('graph.nodes.search.vector_search')
    @patch('graph.nodes.search.keyword_search_full_corpus')
    def test_search_node_exception_handling(self, mock_keyword, mock_vector):
        """예외 발생 시 search 노드 테스트"""
        # 모킹 설정 - 예외 발생
        mock_vector.side_effect = Exception("벡터 검색 오류")
        mock_keyword.side_effect = Exception("키워드 검색 오류")
        
        state = {
            "question": "여행자보험 보장내용",
            "web_results": []
        }
        
        result = search_node(state)
        
        # 결과 검증
        assert "passages" in result
        assert len(result["passages"]) == 0
        assert "search_error" in result["search_meta"]["reason"]
        assert result["search_meta"]["candidates_count"] == 0


@pytest.mark.unit
class TestKoreanTokenizer:
    """한국어 토크나이저 테스트 클래스"""
    
    def test_extract_insurance_keywords(self):
        """여행자보험 도메인 키워드 추출 테스트"""
        text = "DB손해보험 여행자보험의 상해보장과 질병보장 특약에 대한 정보"
        
        keywords = extract_insurance_keywords(text, min_frequency=1)
        
        # 여행자보험 관련 키워드가 추출되어야 함
        assert len(keywords) > 0
        assert any("보험" in keyword for keyword in keywords)
        assert any("여행" in keyword for keyword in keywords)
    
    def test_calculate_keyword_relevance(self):
        """키워드 관련성 계산 테스트"""
        text1 = "여행자보험의 상해보장과 질병보장"
        text2 = "해외여행보험 상해보장 질병보장 의료비"
        
        relevance = calculate_keyword_relevance(text1, [text2])
        
        assert 0.0 <= relevance <= 1.0
        assert relevance > 0.0  # 관련성이 있어야 함
    
    def test_get_keyword_weights(self):
        """키워드 가중치 계산 테스트"""
        keywords = ["보험", "여행", "보험", "여행", "보장"]
        
        weights = get_keyword_weights(keywords)
        
        # 가중치가 계산되어야 함
        assert len(weights) > 0
        assert "보험" in weights
        assert "여행" in weights
        assert "보장" in weights
        
        # 빈도에 따른 가중치 확인
        assert weights["보험"] > weights["보장"]  # "보험"이 더 많이 나타남


@pytest.mark.unit
class TestSearchNodeIntegration:
    """Search 노드 통합 테스트"""
    
    @patch('graph.nodes.search.vector_search')
    @patch('graph.nodes.search.keyword_search_full_corpus')
    @patch('graph.nodes.search.hybrid_search')
    def test_search_node_with_web_results(self, mock_hybrid, mock_keyword, mock_vector):
        """웹 검색 결과가 있을 때 search 노드 테스트"""
        # 모킹 설정
        mock_vector.return_value = [
            {"text": "여행자보험 보장내용", "doc_id": "doc1", "score_vec": 0.8}
        ]
        mock_keyword.return_value = [
            {"text": "여행자보험 보장내용", "doc_id": "doc1", "score_kw": 0.7}
        ]
        mock_hybrid.return_value = [
            {"text": "여행자보험 보장내용", "doc_id": "doc1", "score": 0.75}
        ]
        
        # 테스트 상태
        state = {
            "question": "여행자보험 보장내용이 뭐야?",
            "web_results": [
                {
                    "title": "여행자보험 보장내용",
                    "snippet": "해외여행보험의 상세한 보장내용"
                }
            ]
        }
        
        # Search 노드 실행
        result = search_node(state)
        
        # 결과 검증
        assert "passages" in result
        assert len(result["passages"]) > 0
        assert "search_meta" in result
        assert result["search_meta"]["web_keywords"] is not None
        
        # 웹 결과가 활용되었는지 확인
        mock_vector.assert_called_once()
        mock_keyword.assert_called_once()
        mock_hybrid.assert_called_once()
    
    @patch('graph.nodes.search.vector_search')
    @patch('graph.nodes.search.keyword_search_full_corpus')
    @patch('graph.nodes.search.hybrid_search')
    def test_search_node_without_web_results(self, mock_hybrid, mock_keyword, mock_vector):
        """웹 검색 결과가 없을 때 search 노드 테스트"""
        # 모킹 설정
        mock_vector.return_value = [
            {"text": "여행자보험 보장내용", "doc_id": "doc1", "score_vec": 0.8}
        ]
        mock_keyword.return_value = [
            {"text": "여행자보험 보장내용", "doc_id": "doc1", "score_kw": 0.7}
        ]
        mock_hybrid.return_value = [
            {"text": "여행자보험 보장내용", "doc_id": "doc1", "score": 0.75}
        ]
        
        # 테스트 상태 (웹 결과 없음)
        state = {
            "question": "여행자보험 보장내용이 뭐야?",
            "web_results": []
        }
        
        # Search 노드 실행
        result = search_node(state)
        
        # 결과 검증
        assert "passages" in result
        assert len(result["passages"]) > 0
        
        # 기본 검색이 수행되었는지 확인
        mock_vector.assert_called_once()
        mock_keyword.assert_called_once()
        mock_hybrid.assert_called_once()


@pytest.mark.unit
class TestSearchPerformance:
    """Search 노드 성능 테스트"""
    
    def test_keyword_extraction_performance(self):
        """키워드 추출 성능 테스트"""
        import time
        
        # 대용량 텍스트 생성
        large_text = "여행자보험 " * 1000 + "해외여행 " * 500 + "보장내용 " * 300
        
        start_time = time.time()
        keywords = extract_insurance_keywords(large_text, min_frequency=1)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # 성능 기준: 1초 이내
        assert execution_time < 1.0, f"키워드 추출 시간이 너무 김: {execution_time}초"
        assert len(keywords) > 0, "키워드가 추출되지 않음"
    
    def test_relevance_calculation_performance(self):
        """관련성 계산 성능 테스트"""
        import time
        
        text1 = "여행자보험의 상해보장과 질병보장에 대한 상세한 정보"
        text2 = "해외여행보험 상해보장 질병보장 의료비 치료비"
        
        start_time = time.time()
        relevance = calculate_keyword_relevance(text1, [text2])
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # 성능 기준: 0.1초 이내
        assert execution_time < 0.1, f"관련성 계산 시간이 너무 김: {execution_time}초"
        assert 0.0 <= relevance <= 1.0, "관련성 점수가 범위를 벗어남"


if __name__ == "__main__":
    # 직접 실행 시 성능 테스트
    pytest.main([__file__, "-v"])
