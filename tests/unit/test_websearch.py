"""
Websearch 노드 단위 테스트
웹 검색 노드의 각 기능을 개별적으로 테스트합니다.
"""

import sys
import os
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from graph.nodes.websearch import (
    websearch_node,
    _build_search_queries,
    _process_search_results,
    _calculate_relevance_score,
    _deduplicate_and_rank,
    _check_cache,
    _save_to_cache,
    _generate_cache_key,
    _get_fallback_results,
    TRUSTED_DOMAINS,
    EXCLUDED_DOMAINS
)


@pytest.mark.unit
class TestWebsearchNode:
    """Websearch 노드 단위 테스트 클래스"""
    
    def test_build_search_queries_qa_intent(self):
        """QA 의도에 대한 검색 쿼리 구성 테스트"""
        state = {
            "question": "보험료는 얼마인가요?",
            "intent": "qa"
        }
        
        queries = _build_search_queries(state)
        
        assert len(queries) == 3
        assert "여행자보험 보험료는 얼마인가요?" in queries
        assert "여행자보험 보장내용 보험료는 얼마인가요?" in queries
        assert "여행자보험 가입조건 보험료는 얼마인가요?" in queries
    
    def test_build_search_queries_compare_intent(self):
        """Compare 의도에 대한 검색 쿼리 구성 테스트"""
        state = {
            "question": "DB손해보험과 KB손해보험 비교",
            "intent": "compare"
        }
        
        queries = _build_search_queries(state)
        
        assert len(queries) == 3
        assert "여행자보험 DB손해보험과 KB손해보험 비교" in queries
        assert "여행자보험 비교 DB손해보험과 KB손해보험 비교" in queries
        assert "보험상품 비교 DB손해보험과 KB손해보험 비교" in queries
    
    def test_build_search_queries_recommend_intent(self):
        """Recommend 의도에 대한 검색 쿼리 구성 테스트"""
        state = {
            "question": "일본 여행 추천",
            "intent": "recommend"
        }
        
        queries = _build_search_queries(state)
        
        assert len(queries) == 3
        assert "여행자보험 일본 여행 추천" in queries
        assert "여행자보험 추천 일본 여행 추천" in queries
        assert "여행지별 보험 일본 여행 추천" in queries
    
    def test_build_search_queries_summary_intent(self):
        """Summary 의도에 대한 검색 쿼리 구성 테스트"""
        state = {
            "question": "약관 요약",
            "intent": "summary"
        }
        
        queries = _build_search_queries(state)
        
        assert len(queries) == 3
        assert "여행자보험 약관 요약" in queries
        assert "여행자보험 약관 요약 약관 요약" in queries
        assert "보험상품 정리 약관 요약" in queries
    
    def test_calculate_relevance_score_special_keywords(self):
        """특별 키워드 보너스 점수 테스트"""
        # 이스라엘 관련 키워드 테스트
        title = "이스라엘 여행자보험 특별조항 안내"
        content = "이스라엘 여행을 위한 특별한 보험 조항과 특약에 대한 정보"
        question = "이스라엘 여행자보험 특약"
        
        score = _calculate_relevance_score(title, content, question)
        assert score > 0.8, f"특별 키워드 보너스가 적용되지 않음: {score}"
    
    def test_calculate_relevance_score_insurance_keywords(self):
        """보험 관련 키워드 점수 테스트"""
        title = "해외여행보험 보장내용 및 보험료"
        content = "여행자보험의 보장내용, 보험료, 특약, 가입조건에 대한 상세 정보"
        question = "여행자보험 보장내용"
        
        score = _calculate_relevance_score(title, content, question)
        assert score > 0.6, f"보험 관련 키워드 점수가 낮음: {score}"
    
    def test_calculate_relevance_score_travel_keywords(self):
        """여행 관련 키워드 점수 테스트"""
        title = "해외여행 가이드 및 관광정보"
        content = "일본 여행을 위한 준비사항, 항공, 호텔, 여행사 정보"
        question = "일본 여행 준비"
        
        score = _calculate_relevance_score(title, content, question)
        assert 0.2 <= score <= 0.7, f"여행 관련 점수가 예상 범위를 벗어남: {score}"
    
    def test_calculate_relevance_score_high_relevance(self):
        """높은 관련성 점수 계산 테스트"""
        title = "여행자보험 보장내용 및 보험료 안내"
        content = "해외여행보험의 보장내용과 보험료에 대한 상세 정보를 제공합니다."
        question = "여행자보험 보장내용이 뭐야?"
        
        score = _calculate_relevance_score(title, content, question)
        assert score > 0.5, f"높은 관련성 점수가 예상보다 낮음: {score}"
    
    def test_calculate_relevance_score_low_relevance(self):
        """낮은 관련성 점수 계산 테스트"""
        title = "일반 뉴스 기사"
        content = "오늘 날씨가 맑습니다. 경제 뉴스입니다."
        question = "여행자보험 보장내용이 뭐야?"
        
        score = _calculate_relevance_score(title, content, question)
        assert score < 0.3, f"낮은 관련성 점수가 예상보다 높음: {score}"
    
    
    def test_process_search_results_quality_filtering(self):
        """검색 결과 품질 필터링 테스트"""
        mock_results = [
            {
                "url": "https://www.dbinsu.co.kr/travel-insurance",
                "title": "여행자보험 보장내용",
                "content": "해외여행보험의 상세한 보장내용을 안내합니다.",
                "score": 0.8
            },
            {
                "url": "https://www.example.com/random",
                "title": "일반 뉴스",
                "content": "오늘 날씨가 좋습니다.",
                "score": 0.2
            }
        ]
        
        state = {"question": "여행자보험 보장내용이 뭐야?"}
        processed = _process_search_results(mock_results, state)
        
        # 고품질 결과만 포함되어야 함 (최소 점수 0.2로 완화)
        assert len(processed) == 1
        assert processed[0]["url"] == "https://www.dbinsu.co.kr/travel-insurance"
        assert processed[0]["score_web"] > 0.2
    
    def test_process_search_results_snippet_length_limit(self):
        """검색 결과 스니펫 길이 제한 테스트"""
        long_content = "여행자보험에 대한 상세한 설명입니다. " * 100  # 매우 긴 내용
        
        mock_results = [
            {
                "url": "https://www.dbinsu.co.kr/travel-insurance",
                "title": "여행자보험 안내",
                "content": long_content,
                "score": 0.8
            }
        ]
        
        state = {"question": "여행자보험 보장내용이 뭐야?"}
        processed = _process_search_results(mock_results, state)
        
        assert len(processed) == 1
        assert len(processed[0]["snippet"]) <= 503  # 500자 + "..."
        assert processed[0]["snippet"].endswith("...")
    
    def test_deduplicate_and_rank(self):
        """중복 제거 및 정렬 테스트"""
        results = [
            {
                "url": "https://www.example.com/page1",
                "title": "Page 1",
                "score_web": 0.5
            },
            {
                "url": "https://www.example.com/page1",  # 중복 URL
                "title": "Page 1 Duplicate",
                "score_web": 0.7
            },
            {
                "url": "https://www.example.com/page2",
                "title": "Page 2",
                "score_web": 0.9
            },
            {
                "url": "https://www.example.com/page3",
                "title": "Page 3",
                "score_web": 0.3
            }
        ]
        
        deduplicated = _deduplicate_and_rank(results)
        
        # 중복 제거 확인
        urls = [r["url"] for r in deduplicated]
        assert len(set(urls)) == len(urls), "중복 URL이 제거되지 않음"
        
        # 정렬 확인 (점수 내림차순)
        scores = [r["score_web"] for r in deduplicated]
        assert scores == sorted(scores, reverse=True), "점수 정렬이 올바르지 않음"
        
        # 상위 5개만 반환 확인
        assert len(deduplicated) <= 5
    
    def test_generate_cache_key(self):
        """캐시 키 생성 테스트"""
        state1 = {"question": "여행자보험 보장내용", "intent": "qa"}
        state2 = {"question": "여행자보험 보장내용", "intent": "qa"}
        state3 = {"question": "여행자보험 보장내용", "intent": "compare"}
        
        key1 = _generate_cache_key(state1)
        key2 = _generate_cache_key(state2)
        key3 = _generate_cache_key(state3)
        
        # 동일한 상태는 동일한 키 생성
        assert key1 == key2, "동일한 상태의 캐시 키가 다름"
        
        # 다른 의도는 다른 키 생성
        assert key1 != key3, "다른 의도의 캐시 키가 동일함"
        
        # 키 형식 확인 (MD5 해시는 32자리 16진수 문자열)
        assert len(key1) == 32, "캐시 키 길이가 올바르지 않음"
        assert all(c in '0123456789abcdef' for c in key1), "캐시 키가 유효한 MD5 해시가 아님"
    
    def test_get_fallback_results_qa_intent(self):
        """QA 의도에 대한 대체 결과 테스트"""
        state = {"question": "보험료는 얼마인가요?", "intent": "qa"}
        
        result = _get_fallback_results(state)
        
        assert "web_results" in result
        assert len(result["web_results"]) == 1
        
        web_result = result["web_results"][0]
        assert web_result["source"] == "fallback_stub"
        assert "보험사 고객센터" in web_result["snippet"]
        assert web_result["score_web"] == 0.5
    
    def test_get_fallback_results_compare_intent(self):
        """Compare 의도에 대한 대체 결과 테스트"""
        state = {"question": "보험 비교", "intent": "compare"}
        
        result = _get_fallback_results(state)
        
        web_result = result["web_results"][0]
        assert "보험사 고객센터나 공식 홈페이지" in web_result["snippet"]
    
    def test_get_fallback_results_recommend_intent(self):
        """Recommend 의도에 대한 대체 결과 테스트"""
        state = {"question": "보험 추천", "intent": "recommend"}
        
        result = _get_fallback_results(state)
        
        web_result = result["web_results"][0]
        assert "전문가 상담을 권장" in web_result["snippet"]
    
    @patch('graph.nodes.websearch.get_redis_client')
    def test_check_cache_hit(self, mock_redis):
        """캐시 히트 테스트"""
        # Redis 모킹
        mock_redis_client = Mock()
        mock_redis.return_value = mock_redis_client
        
        cached_data = [{"url": "https://example.com", "title": "Cached Result"}]
        mock_redis_client.get.return_value = json.dumps(cached_data, ensure_ascii=False)
        
        state = {"question": "테스트 질문", "intent": "qa"}
        result = _check_cache(state)
        
        assert result == cached_data
        mock_redis_client.get.assert_called_once()
    
    @patch('graph.nodes.websearch.get_redis_client')
    def test_check_cache_miss(self, mock_redis):
        """캐시 미스 테스트"""
        # Redis 모킹
        mock_redis_client = Mock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.get.return_value = None
        
        state = {"question": "테스트 질문", "intent": "qa"}
        result = _check_cache(state)
        
        assert result is None
    
    @patch('graph.nodes.websearch.get_redis_client')
    def test_save_to_cache(self, mock_redis):
        """캐시 저장 테스트"""
        # Redis 모킹
        mock_redis_client = Mock()
        mock_redis.return_value = mock_redis_client
        
        state = {"question": "테스트 질문", "intent": "qa"}
        results = [{"url": "https://example.com", "title": "Test Result"}]
        
        _save_to_cache(state, results)
        
        mock_redis_client.setex.assert_called_once()
        call_args = mock_redis_client.setex.call_args
        assert call_args[0][1] == 1800  # TTL 30분
    
    def test_trusted_domains_configuration(self):
        """신뢰할 수 있는 도메인 설정 테스트"""
        expected_domains = [
            "naver.com", "daum.net", "kakao.com",
            "dbinsu.co.kr", "kbinsure.co.kr", "samsungfire.com", "hyundai.com",
            "fss.or.kr", "kdi.re.kr",
            "korea.kr", "visitkorea.or.kr",
            "tripadvisor.co.kr", "agoda.com", "booking.com"
        ]
        
        for domain in expected_domains:
            assert domain in TRUSTED_DOMAINS, f"신뢰할 수 있는 도메인에 {domain}이 없음"
    
    def test_excluded_domains_configuration(self):
        """제외할 도메인 설정 테스트"""
        expected_excluded = [
            "wikipedia.org",
            "pornhub.com", "xvideos.com"
        ]
        
        for domain in expected_excluded:
            assert domain in EXCLUDED_DOMAINS, f"제외할 도메인에 {domain}이 없음"


@pytest.mark.unit
class TestWebsearchNodeIntegration:
    """Websearch 노드 통합 테스트 (모킹 사용)"""
    
    @patch('graph.nodes.websearch.get_redis_client')
    @patch('graph.nodes.websearch.get_settings')
    @patch('graph.nodes.websearch.TavilyClient')
    def test_websearch_node_success(self, mock_tavily_client, mock_settings, mock_redis):
        """웹 검색 노드 성공 케이스 테스트"""
        # 설정 모킹
        mock_settings.return_value.TAVILY_API_KEY = "test_api_key"
        
        # Redis 모킹 (캐시 미스)
        mock_redis_client = Mock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.get.return_value = None
        
        # Tavily 클라이언트 모킹
        mock_client_instance = Mock()
        mock_tavily_client.return_value = mock_client_instance
        
        mock_search_response = {
            "results": [
                {
                    "url": "https://www.dbinsu.co.kr/travel-insurance",
                    "title": "여행자보험 보장내용",
                    "content": "해외여행보험의 상세한 보장내용을 안내합니다.",
                    "score": 0.8
                }
            ]
        }
        mock_client_instance.search.return_value = mock_search_response
        
        # 테스트 실행
        state = {
            "question": "여행자보험 보장내용이 뭐야?",
            "intent": "qa"
        }
        
        result = websearch_node(state)
        
        # 결과 검증
        assert "web_results" in result
        assert len(result["web_results"]) > 0
        
        web_result = result["web_results"][0]
        assert web_result["source"] == "tavily_web"
        assert web_result["url"] == "https://www.dbinsu.co.kr/travel-insurance"
        assert web_result["score_web"] > 0.2
    
    @patch('graph.nodes.websearch.get_redis_client')
    @patch('graph.nodes.websearch.get_settings')
    def test_websearch_node_no_api_key(self, mock_settings, mock_redis):
        """API 키가 없는 경우 테스트"""
        # 설정 모킹 (API 키 없음)
        mock_settings.return_value.TAVILY_API_KEY = ""
        
        # Redis 모킹 (캐시 미스)
        mock_redis_client = Mock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.get.return_value = None
        
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
        assert "검색 서비스 일시 중단" in web_result["snippet"]
    
    @patch('graph.nodes.websearch.get_redis_client')
    @patch('graph.nodes.websearch.get_settings')
    @patch('graph.nodes.websearch.TavilyClient')
    def test_websearch_node_with_cache(self, mock_tavily_client, mock_settings, mock_redis):
        """캐시가 있는 경우 테스트"""
        # 설정 모킹
        mock_settings.return_value.TAVILY_API_KEY = "test_api_key"
        
        # Redis 모킹 (캐시 히트)
        mock_redis_client = Mock()
        mock_redis.return_value = mock_redis_client
        
        cached_results = [{"url": "https://example.com", "title": "Cached Result", "score_web": 0.8}]
        mock_redis_client.get.return_value = '[{"url": "https://example.com", "title": "Cached Result", "score_web": 0.8}]'
        
        state = {
            "question": "여행자보험 보장내용이 뭐야?",
            "intent": "qa"
        }
        
        result = websearch_node(state)
        
        # 캐시된 결과 반환 확인
        assert "web_results" in result
        assert result["web_results"] == cached_results
        
        # Tavily API 호출되지 않음 확인
        mock_tavily_client.assert_not_called()
    
    @patch('graph.nodes.websearch.get_redis_client')
    @patch('graph.nodes.websearch.get_settings')
    @patch('graph.nodes.websearch.TavilyClient')
    def test_websearch_node_api_error(self, mock_tavily_client, mock_settings, mock_redis):
        """API 오류 발생 시 테스트"""
        # 설정 모킹
        mock_settings.return_value.TAVILY_API_KEY = "test_api_key"
        
        # Redis 모킹 (캐시 미스)
        mock_redis_client = Mock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.get.return_value = None
        
        # Tavily 클라이언트 모킹 (오류 발생)
        mock_tavily_client.side_effect = Exception("API 오류")
        
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
