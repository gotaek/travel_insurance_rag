from typing import Dict, Any
import re

def _extract_trip_info(q: str) -> Dict[str, str]:
    # 매우 단순한 스텁 추출: 날짜 yyyy-mm-dd, 도시 단어
    date = re.findall(r"\d{4}-\d{2}-\d{2}", q)
    city = None
    if any(x in q.lower() for x in ["la", "los angeles", "엘에이", "로스앤젤레스"]):
        city = "Los Angeles"
    return {"dates": ", ".join(date) if date else "", "city": city or ""}

def recommend_node(state: Dict[str, Any]) -> Dict[str, Any]:
    refined = state.get("refined", [])
    citations = state.get("citations", [])
    warnings = state.get("warnings", []) or []
    policy_disclaimer = state.get("policy_disclaimer")
    trip = _extract_trip_info(state.get("question", ""))

    # 간단 추천 스텁: 연착/수하물/의료 3종을 기본 제안
    recommended = ["항공기(연착/결항) 특약", "수하물 지연/분실 특약", "해외의료비/긴급의료이송"]
    if trip.get("city"):
        recommended.append(f"{trip['city']} 지역 특화 위험도 확인(웹검색 연동 예정)")
    if trip.get("dates"):
        recommended.append(f"여행일정({trip['dates']}) 기준 최신 이슈 체크(웹검색 연동 예정)")

    caveats = ["추천은 현재 룰 기반 스텁입니다. 실시간 뉴스/현지 위험 데이터는 다음 커밋에서 연결됩니다."]
    caveats += warnings
    if policy_disclaimer:
        caveats.append(policy_disclaimer)

    answer = {
        "conclusion": "추천 특약(초안): " + ", ".join(recommended[:3]),
        "evidence": [p.get("text") for p in refined[:2]],
        "caveats": caveats,
        "quotes": citations,
    }
    return {**state, "draft_answer": answer}