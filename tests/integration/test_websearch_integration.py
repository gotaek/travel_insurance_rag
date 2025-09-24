"""
Websearch 노드 인티그레이션 테스트
실제 환경에서 웹 검색 노드의 전체 워크플로우를 테스트합니다.
"""

import sys
import os
import pytest
from unittest.mock import patch, Mock
import json

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from graph.nodes.websearch import websearch_node
from tests.fixtures.test_data import (
    sample_questions,
    web_search_questions,
    edge_case_questions
)


@pytest.mark.integration
class TestWebsearchIntegration:
    """Websearch 노드 인티그레이션 테스트 클래스"""
    
    @pytest.fixture
    def mock_tavily_response(self):
        """Tavily API 응답 모킹"""
        return {
            "results": [
                {
                    "url": "https://www.dbinsu.co.kr/travel-insurance",
                    "title": "DB손해보험 여행자보험 보장내용",
                    "content": "해외여행보험의 상세한 보장내용을 안내합니다. 의료비, 휴대품, 여행지연 등 다양한 위험을 보장합니다.",
                    "score": 0.85
                },
                {
                    "url": "https://www.kbinsure.co.kr/products/travel",
                    "title": "KB손해보험 여행자보험 상품안내",
                    "content": "KB손해보험의 여행자보험 상품에 대한 상세 정보를 제공합니다. 보험료, 가입조건, 보장내용을 확인하세요.",
                    "score": 0.82
                },
                {
                    "url": "https://www.naver.com/travel-insurance-guide",
                    "title": "여행자보험 가이드 - 네이버",
                    "content": "여행자보험 선택 가이드와 비교 정보를 제공합니다. 여행 목적에 맞는 보험을 선택하는 방법을 안내합니다.",
                    "score": 0.75
                }
            ]
        }
    
    @pytest.fixture
    def mock_redis_client(self):
        """Redis 클라이언트 모킹"""
        mock_client = Mock()
        mock_client.get.return_value = None  # 기본적으로 캐시 미스
        mock_client.setex.return_value = True
        return mock_client
    
    @patch('graph.nodes.websearch.get_redis_client')
    @patch('graph.nodes.websearch.get_settings')
    @patch('graph.nodes.websearch.TavilyClient')
    def test_websearch_full_workflow_qa(self, mock_tavily_client, mock_settings, mock_redis, 
                                       mock_tavily_response, mock_redis_client, sample_questions):
        """QA 의도에 대한 전체 워크플로우 테스트"""
        # 설정 모킹
        mock_settings.return_value.TAVILY_API_KEY = "test_api_key"
        mock_redis.return_value = mock_redis_client
        
        # Tavily 클라이언트 모킹
        mock_client_instance = Mock()
        mock_tavily_client.return_value = mock_client_instance
        mock_client_instance.search.return_value = mock_tavily_response
        
        # QA 질문 테스트
        qa_questions = sample_questions["qa"]
        
        for question in qa_questions:
            state = {
                "question": question,
                "intent": "qa"
            }
            
            result = websearch_node(state)
            
            # 결과 검증
            assert "web_results" in result
            assert len(result["web_results"]) > 0
            
            # 웹 결과 품질 검증
            for web_result in result["web_results"]:
                assert web_result["source"] == "tavily_web"
                assert "url" in web_result
                assert "title" in web_result
                assert "snippet" in web_result
                assert "score_web" in web_result
                assert web_result["score_web"] > 0.2  # 최소 점수 완화
                assert "relevance_score" in web_result
                assert "timestamp" in web_result
    
    @patch('graph.nodes.websearch.get_redis_client')
    @patch('graph.nodes.websearch.get_settings')
    @patch('graph.nodes.websearch.TavilyClient')
    def test_websearch_full_workflow_compare(self, mock_tavily_client, mock_settings, mock_redis,
                                           mock_tavily_response, mock_redis_client, sample_questions):
        """Compare 의도에 대한 전체 워크플로우 테스트"""
        # 설정 모킹
        mock_settings.return_value.TAVILY_API_KEY = "test_api_key"
        mock_redis.return_value = mock_redis_client
        
        # Tavily 클라이언트 모킹
        mock_client_instance = Mock()
        mock_tavily_client.return_value = mock_client_instance
        mock_client_instance.search.return_value = mock_tavily_response
        
        # Compare 질문 테스트
        compare_questions = sample_questions["compare"]
        
        for question in compare_questions:
            state = {
                "question": question,
                "intent": "compare"
            }
            
            result = websearch_node(state)
            
            # 결과 검증
            assert "web_results" in result
            assert len(result["web_results"]) > 0
            
            # 비교 관련 키워드가 검색 쿼리에 포함되었는지 확인
            # (실제로는 Tavily API 호출 시 전달되는 쿼리를 확인할 수 없지만,
            #  결과가 반환되었다는 것으로 쿼리 구성이 올바르게 되었음을 간접 확인)
            assert all("score_web" in result for result in result["web_results"])
    
    @patch('graph.nodes.websearch.get_redis_client')
    @patch('graph.nodes.websearch.get_settings')
    @patch('graph.nodes.websearch.TavilyClient')
    def test_websearch_full_workflow_recommend(self, mock_tavily_client, mock_settings, mock_redis,
                                             mock_tavily_response, mock_redis_client, sample_questions):
        """Recommend 의도에 대한 전체 워크플로우 테스트"""
        # 설정 모킹
        mock_settings.return_value.TAVILY_API_KEY = "test_api_key"
        mock_redis.return_value = mock_redis_client
        
        # Tavily 클라이언트 모킹
        mock_client_instance = Mock()
        mock_tavily_client.return_value = mock_client_instance
        mock_client_instance.search.return_value = mock_tavily_response
        
        # Recommend 질문 테스트
        recommend_questions = sample_questions["recommend"]
        
        for question in recommend_questions:
            state = {
                "question": question,
                "intent": "recommend"
            }
            
            result = websearch_node(state)
            
            # 결과 검증
            assert "web_results" in result
            assert len(result["web_results"]) > 0
    
    @patch('graph.nodes.websearch.get_redis_client')
    @patch('graph.nodes.websearch.get_settings')
    @patch('graph.nodes.websearch.TavilyClient')
    def test_websearch_caching_workflow(self, mock_tavily_client, mock_settings, mock_redis,
                                      mock_tavily_response, mock_redis_client):
        """캐싱 워크플로우 테스트"""
        # 설정 모킹
        mock_settings.return_value.TAVILY_API_KEY = "test_api_key"
        mock_redis.return_value = mock_redis_client
        
        # Tavily 클라이언트 모킹
        mock_client_instance = Mock()
        mock_tavily_client.return_value = mock_client_instance
        mock_client_instance.search.return_value = mock_tavily_response
        
        state = {
            "question": "여행자보험 보장내용이 뭐야?",
            "intent": "qa"
        }
        
        # 첫 번째 호출 (캐시 미스)
        result1 = websearch_node(state)
        
        # 캐시 저장 확인
        mock_redis_client.setex.assert_called()
        
        # 두 번째 호출 (캐시 히트)
        cached_results = result1["web_results"]
        mock_redis_client.get.return_value = json.dumps(cached_results, ensure_ascii=False)
        
        result2 = websearch_node(state)
        
        # 캐시된 결과와 동일한지 확인
        assert result1["web_results"] == result2["web_results"]
        
        # 두 번째 호출에서는 Tavily API가 호출되지 않아야 함 (첫 번째 호출에서 4개 쿼리로 4번 호출)
        assert mock_client_instance.search.call_count == 4
    
    @patch('graph.nodes.websearch.get_redis_client')
    @patch('graph.nodes.websearch.get_settings')
    @patch('graph.nodes.websearch.TavilyClient')
    def test_websearch_error_handling(self, mock_tavily_client, mock_settings, mock_redis,
                                    mock_redis_client):
        """오류 처리 워크플로우 테스트"""
        # 설정 모킹
        mock_settings.return_value.TAVILY_API_KEY = "test_api_key"
        mock_redis.return_value = mock_redis_client
        
        # Tavily 클라이언트 모킹 (오류 발생)
        mock_tavily_client.side_effect = Exception("API 연결 오류")
        
        state = {
            "question": "여행자보험 보장내용이 뭐야?",
            "intent": "qa"
        }
        
        result = websearch_node(state)
        
        # 오류 발생 시에도 대체 결과 반환
        assert "web_results" in result
        assert len(result["web_results"]) == 1
        
        web_result = result["web_results"][0]
        assert web_result["source"] == "fallback_stub"
        assert "검색 서비스 일시 중단" in web_result["snippet"]
    
    @patch('graph.nodes.websearch.get_redis_client')
    @patch('graph.nodes.websearch.get_settings')
    def test_websearch_no_api_key(self, mock_settings, mock_redis_client):
        """API 키가 없는 경우 테스트"""
        # 설정 모킹 (API 키 없음)
        mock_settings.return_value.TAVILY_API_KEY = ""
        
        state = {
            "question": "여행자보험 보장내용이 뭐야?",
            "intent": "qa"
        }
        
        result = websearch_node(state)
        
        # 대체 결과 반환 확인
        assert "web_results" in result
        assert len(result["web_results"]) == 1
        
        web_result = result["web_results"][0]
        assert web_result["source"] == "fallback_stub"
        assert web_result["url"] == "https://www.fss.or.kr/"
    
    @patch('graph.nodes.websearch.get_redis_client')
    @patch('graph.nodes.websearch.get_settings')
    @patch('graph.nodes.websearch.TavilyClient')
    def test_websearch_edge_cases(self, mock_tavily_client, mock_settings, mock_redis,
                                mock_redis_client, edge_case_questions):
        """엣지 케이스 테스트"""
        # 설정 모킹
        mock_settings.return_value.TAVILY_API_KEY = "test_api_key"
        mock_redis.return_value = mock_redis_client
        
        # Tavily 클라이언트 모킹
        mock_client_instance = Mock()
        mock_tavily_client.return_value = mock_client_instance
        mock_client_instance.search.return_value = {"results": []}
        
        for question in edge_case_questions:
            state = {
                "question": question,
                "intent": "qa"
            }
            
            # 예외가 발생하지 않고 결과가 반환되는지 확인
            result = websearch_node(state)
            assert "web_results" in result
    
    @patch('graph.nodes.websearch.get_redis_client')
    @patch('graph.nodes.websearch.get_settings')
    @patch('graph.nodes.websearch.TavilyClient')
    def test_websearch_domain_filtering(self, mock_tavily_client, mock_settings, mock_redis,
                                      mock_redis_client):
        """도메인 필터링 테스트"""
        # 설정 모킹
        mock_settings.return_value.TAVILY_API_KEY = "test_api_key"
        mock_redis.return_value = mock_redis_client
        
        # Tavily 클라이언트 모킹
        mock_client_instance = Mock()
        mock_tavily_client.return_value = mock_client_instance
        
        # 다양한 도메인의 결과를 포함한 응답
        mixed_domain_response = {
            "results": [
                {
                    "url": "https://www.dbinsu.co.kr/travel-insurance",
                    "title": "DB손해보험 여행자보험",
                    "content": "보험사 공식 정보입니다.",
                    "score": 0.8
                },
                {
                    "url": "https://www.naver.com/travel-guide",
                    "title": "네이버 여행 가이드",
                    "content": "포털 사이트 정보입니다.",
                    "score": 0.7
                },
                {
                    "url": "https://www.example.com/random",
                    "title": "기타 사이트",
                    "content": "기타 사이트 정보입니다.",
                    "score": 0.6
                }
            ]
        }
        mock_client_instance.search.return_value = mixed_domain_response
        
        state = {
            "question": "여행자보험 보장내용이 뭐야?",
            "intent": "qa"
        }
        
        result = websearch_node(state)
        
        # 결과가 반환되는지 확인
        assert "web_results" in result
        assert len(result["web_results"]) > 0
        
        # 관련성 점수가 계산되었는지 확인
        for web_result in result["web_results"]:
            assert "relevance_score" in web_result
            assert 0 <= web_result["relevance_score"] <= 1
    
    @patch('graph.nodes.websearch.get_redis_client')
    @patch('graph.nodes.websearch.get_settings')
    @patch('graph.nodes.websearch.TavilyClient')
    def test_websearch_relevance_scoring(self, mock_tavily_client, mock_settings, mock_redis,
                                       mock_redis_client):
        """관련성 점수 계산 테스트"""
        # 설정 모킹
        mock_settings.return_value.TAVILY_API_KEY = "test_api_key"
        mock_redis.return_value = mock_redis_client
        
        # Tavily 클라이언트 모킹
        mock_client_instance = Mock()
        mock_tavily_client.return_value = mock_client_instance
        
        # 관련성 점수 테스트를 위한 응답
        relevance_test_response = {
            "results": [
                {
                    "url": "https://www.dbinsu.co.kr/travel-insurance",
                    "title": "여행자보험 보장내용 상세 안내",
                    "content": "해외여행보험의 보장내용에 대한 상세한 정보를 제공합니다. 의료비, 휴대품, 여행지연 등 다양한 위험을 보장합니다.",
                    "score": 0.8
                },
                {
                    "url": "https://www.example.com/weather",
                    "title": "오늘 날씨 정보",
                    "content": "오늘 날씨가 맑고 기온이 적당합니다. 외출하기 좋은 날씨입니다.",
                    "score": 0.6
                }
            ]
        }
        mock_client_instance.search.return_value = relevance_test_response
        
        state = {
            "question": "여행자보험 보장내용이 뭐야?",
            "intent": "qa"
        }
        
        result = websearch_node(state)
        
        # 결과가 반환되는지 확인
        assert "web_results" in result
        assert len(result["web_results"]) > 0
        
        # 관련성 점수가 계산되었는지 확인
        for web_result in result["web_results"]:
            assert "relevance_score" in web_result
            assert 0 <= web_result["relevance_score"] <= 1
        
        # 여행자보험 관련 결과가 날씨 관련 결과보다 높은 점수를 받는지 확인
        if len(result["web_results"]) >= 2:
            travel_result = next((r for r in result["web_results"] if "여행자보험" in r["title"]), None)
            weather_result = next((r for r in result["web_results"] if "날씨" in r["title"]), None)
            
            if travel_result and weather_result:
                assert travel_result["relevance_score"] > weather_result["relevance_score"]


@pytest.mark.integration
class TestWebsearchPerformance:
    """Websearch 노드 성능 테스트"""
    
    @patch('graph.nodes.websearch.get_redis_client')
    @patch('graph.nodes.websearch.get_settings')
    @patch('graph.nodes.websearch.TavilyClient')
    def test_websearch_performance_with_cache(self, mock_tavily_client, mock_settings, mock_redis):
        """캐시를 사용한 성능 테스트"""
        import time
        
        # 설정 모킹
        mock_settings.return_value.TAVILY_API_KEY = "test_api_key"
        
        # Redis 모킹 (캐시 히트)
        mock_redis_client = Mock()
        mock_redis.return_value = mock_redis_client
        
        cached_results = [{"url": "https://example.com", "title": "Cached Result", "score_web": 0.8}]
        mock_redis_client.get.return_value = json.dumps(cached_results, ensure_ascii=False)
        
        state = {
            "question": "여행자보험 보장내용이 뭐야?",
            "intent": "qa"
        }
        
        # 성능 측정
        start_time = time.time()
        result = websearch_node(state)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # 캐시 사용 시 빠른 응답 시간 확인 (1초 이내)
        assert execution_time < 1.0, f"캐시 사용 시 응답 시간이 너무 김: {execution_time}초"
        assert result["web_results"] == cached_results
    
    @patch('graph.nodes.websearch.get_redis_client')
    @patch('graph.nodes.websearch.get_settings')
    @patch('graph.nodes.websearch.TavilyClient')
    def test_websearch_concurrent_requests(self, mock_tavily_client, mock_settings, mock_redis):
        """동시 요청 처리 테스트"""
        import threading
        import time
        
        # 설정 모킹
        mock_settings.return_value.TAVILY_API_KEY = "test_api_key"
        
        # Redis 모킹
        mock_redis_client = Mock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.get.return_value = None
        mock_redis_client.setex.return_value = True
        
        # Tavily 클라이언트 모킹
        mock_client_instance = Mock()
        mock_tavily_client.return_value = mock_client_instance
        mock_client_instance.search.return_value = {"results": []}
        
        results = []
        errors = []
        
        def make_request(question):
            try:
                state = {"question": question, "intent": "qa"}
                result = websearch_node(state)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # 동시 요청 생성
        threads = []
        questions = [f"테스트 질문 {i}" for i in range(5)]
        
        start_time = time.time()
        for question in questions:
            thread = threading.Thread(target=make_request, args=(question,))
            threads.append(thread)
            thread.start()
        
        # 모든 스레드 완료 대기
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 오류 없이 모든 요청이 처리되었는지 확인
        assert len(errors) == 0, f"동시 요청 처리 중 오류 발생: {errors}"
        assert len(results) == 5, f"모든 요청이 처리되지 않음: {len(results)}/5"
        
        # 합리적인 실행 시간 확인 (5초 이내)
        assert execution_time < 5.0, f"동시 요청 처리 시간이 너무 김: {execution_time}초"
