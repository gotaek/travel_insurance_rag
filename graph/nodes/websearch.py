# graph/nodes/websearch.py
from __future__ import annotations
from typing import Dict, Any, List
import logging
from tavily import TavilyClient
from app.deps import get_settings

# 로깅 설정
logger = logging.getLogger(__name__)

def websearch_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tavily API를 사용한 웹 검색 노드.
    - planner.needs_web == True 일 때만 실행.
    - 여행자보험 관련 최신 정보를 검색하여 제공.
    """
    try:
        # Tavily API 키 확인
        settings = get_settings()
        api_key = settings.TAVILY_API_KEY
        if not api_key:
            logger.warning("TAVILY_API_KEY가 설정되지 않음. 스텁 결과 반환")
            return _get_fallback_results(state)
        
        # Tavily 클라이언트 초기화
        client = TavilyClient(api_key=api_key)
        
        # 검색 쿼리 구성
        question = state.get("question", "")
        search_query = f"여행자보험 {question}"
        
        # Tavily API로 검색 실행
        response = client.search(
            query=search_query,
            search_depth="basic",  # basic 또는 advanced
            max_results=5,  # 최대 5개 결과
            include_domains=["naver.com", "daum.net"],  # 한국 사이트 우선
            exclude_domains=["wikipedia.org"]  # 위키피디아 제외
        )
        
        # 결과를 시스템 형식으로 변환
        web_results = []
        for result in response.get("results", []):
            web_results.append({
                "source": "tavily_web",
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "snippet": result.get("content", ""),
                "score_web": result.get("score", 0.0),
            })
        
        logger.info(f"Tavily 검색 완료: {len(web_results)}개 결과")
        return {**state, "web_results": web_results}
        
    except Exception as e:
        logger.error(f"Tavily 검색 중 오류 발생: {str(e)}")
        return _get_fallback_results(state)

def _get_fallback_results(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tavily API 사용 불가 시 대체 결과 반환.
    """
    question = state.get("question", "")
    fallback_hits: List[Dict[str, Any]] = [
        {
            "source": "fallback_stub",
            "url": "https://example.com/travel-insurance-info",
            "title": "여행자보험 관련 정보",
            "snippet": f"검색 서비스 일시 중단. '{question}' 관련 여행자보험 정보는 고객센터에 문의하세요.",
            "score_web": 0.5,
        }
    ]
    return {**state, "web_results": fallback_hits}

__all__ = ["websearch_node"]