# graph/nodes/websearch.py
from __future__ import annotations
from typing import Dict, Any, List

def websearch_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stub WebSearch node.
    - planner.needs_web == True 일 때만 실행.
    - 나중에 실제 검색 API(Tavily/Serper 등)로 교체 예정.
    """
    q = state.get("question", "")
    fake_hits: List[Dict[str, Any]] = [
        {
            "source": "news_stub",
            "url": "https://news.example.com/travel-insurance",
            "title": "최근 항공기 연착 사고 발생",
            "snippet": f"검색 스텁: '{q}' 관련 최근 연착/여행 위험 이슈",
            "score_web": 1.0,
        }
    ]
    return {**state, "web_results": fake_hits}

__all__ = ["websearch_node"]