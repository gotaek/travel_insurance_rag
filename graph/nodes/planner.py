from typing import Dict, Any
import re

INTENTS = ["summary", "compare", "qa", "recommend"]

SUMMARY_KEYS = ["요약", "정리", "summary"]
COMPARE_KEYS = ["비교", "차이", "다른 점", "compare"]
RECOMMEND_KEYS = ["추천", "특약", "권장", "recommend"]

def _guess_intent(q: str) -> str:
    ql = q.lower()
    if any(k in q for k in SUMMARY_KEYS):
        return "summary"
    if any(k in q for k in COMPARE_KEYS):
        return "compare"
    if any(k in q for k in RECOMMEND_KEYS):
        return "recommend"
    return "qa"

def _needs_web(q: str, intent: str) -> bool:
    if intent != "recommend":
        return False
    # 날짜/도시/뉴스 키워드가 있으면 최신성 필요
    has_date = bool(re.search(r"\d{4}-\d{2}-\d{2}", q))
    has_city = any(x in q.lower() for x in ["la", "los angeles", "엘에이", "로스앤젤레스", "도쿄", "뉴욕", "파리"])
    has_live = any(x in q for x in ["뉴스", "현지", "실시간"])
    return has_date or has_city or has_live

def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    q = state.get("question", "")
    intent = _guess_intent(q)
    plan = ["planner", "search", "rank_filter", "verify_refine", f"answer:{intent}"]
    needs_web = _needs_web(q, intent)
    return {**state, "intent": intent, "needs_web": needs_web, "plan": plan}