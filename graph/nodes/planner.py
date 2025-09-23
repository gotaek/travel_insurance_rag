from typing import Dict, Any
import json
import re
from app.deps import get_llm

INTENTS = ["summary", "compare", "qa", "recommend"]

def _llm_classify_intent(question: str) -> Dict[str, Any]:
    """
    LLM을 사용하여 질문의 intent와 needs_web을 분류
    """
    prompt = f"""
다음 질문을 분석하여 여행자보험 RAG 시스템에서 적절한 처리 방식을 결정해주세요.

질문: "{question}"

다음 중 하나의 intent를 선택하세요:
- "qa": 일반적인 질문-답변 (보장 내용, 가입 조건, 보험료 등)
- "summary": 문서 요약 (약관 요약, 상품 정리 등)
- "compare": 비교 분석 (보험 상품 간 비교, 차이점 분석 등)
- "recommend": 추천 및 권장 (특약 추천, 여행지별 보험 추천 등)

또한 다음 조건을 확인하여 needs_web을 결정하세요:
- 최신 뉴스나 실시간 정보가 필요한가?
- 특정 날짜나 지역의 현재 상황이 필요한가?
- 여행지의 현재 안전 상황이나 규제가 필요한가?

반드시 다음 JSON 형식으로만 답변하세요:
{{
    "intent": "qa|summary|compare|recommend",
    "needs_web": true|false,
    "reasoning": "분류 근거를 간단히 설명"
}}
"""

    try:
        llm = get_llm()
        response = llm.generate_content(prompt, request_options={"timeout": 30})
        
        # JSON 파싱
        response_text = response.text.strip()
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        else:
            json_text = response_text
        
        result = json.loads(json_text)
        
        # 유효성 검증
        if result.get("intent") not in INTENTS:
            result["intent"] = "qa"
        if not isinstance(result.get("needs_web"), bool):
            result["needs_web"] = False
            
        return result
        
    except Exception as e:
        # LLM 호출 실패 시 fallback (디버깅 정보 포함)
        print(f"⚠️ LLM 분류 실패, fallback 사용: {str(e)}")
        return _fallback_classify(question)

def _fallback_classify(question: str) -> Dict[str, Any]:
    """
    LLM 호출 실패 시 사용하는 향상된 키워드 기반 fallback
    """
    ql = question.lower()
    
    # Intent 분류 (더 정교한 패턴 매칭)
    intent = "qa"  # 기본값
    
    # Summary 키워드 (우선순위 높음)
    summary_keywords = ["요약", "정리", "summary", "약관 요약", "상품 요약", "핵심 내용"]
    if any(k in question for k in summary_keywords):
        intent = "summary"
    
    # Compare 키워드
    elif any(k in question for k in ["비교", "차이", "다른 점", "compare", "vs", "대비", "구분"]):
        intent = "compare"
    
    # Recommend 키워드
    elif any(k in question for k in ["추천", "특약", "권장", "recommend", "어떤", "선택", "가장 좋은"]):
        intent = "recommend"
    
    # Web 검색 필요성 (더 정교한 판단)
    needs_web = False
    
    # 날짜 패턴 (2024, 2025, 3월, 12월 등)
    date_patterns = [
        r"\d{4}년", r"\d{4}-\d{2}", r"\d{4}/\d{2}",
        r"\d{1,2}월", r"내년", r"올해", r"다음 달"
    ]
    has_date = any(re.search(pattern, question) for pattern in date_patterns)
    
    # 지역 키워드 (확장)
    city_keywords = [
        "la", "los angeles", "엘에이", "로스앤젤레스", "도쿄", "뉴욕", "파리",
        "런던", "시드니", "싱가포르", "홍콩", "베이징", "상하이", "방콕",
        "유럽", "아시아", "미국", "일본", "중국", "태국", "베트남"
    ]
    has_city = any(x in ql for x in city_keywords)
    
    # 실시간 정보 키워드
    live_keywords = ["뉴스", "현지", "실시간", "최신", "현재", "지금", "요즘"]
    has_live = any(x in question for x in live_keywords)
    
    # Recommend intent이면서 날짜/지역/실시간 정보가 있는 경우
    if intent == "recommend" and (has_date or has_city or has_live):
        needs_web = True
    
    # 일반적인 실시간 정보 요청
    if has_live and any(x in question for x in ["상황", "정보", "뉴스", "현재"]):
        needs_web = True
    
    return {
        "intent": intent,
        "needs_web": needs_web,
        "reasoning": f"Enhanced fallback: {intent} (date:{has_date}, city:{has_city}, live:{has_live})"
    }

def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM 기반 질문 분석 및 분기 결정
    """
    q = state.get("question", "")
    
    # LLM을 사용한 분류
    classification = _llm_classify_intent(q)
    
    intent = classification["intent"]
    needs_web = classification["needs_web"]
    reasoning = classification.get("reasoning", "")
    
    # 실행 계획 생성
    plan = ["planner", "search", "rank_filter", "verify_refine", f"answer:{intent}"]
    if needs_web:
        plan.insert(1, "websearch")
    
    return {
        **state, 
        "intent": intent, 
        "needs_web": needs_web, 
        "plan": plan,
        "classification_reasoning": reasoning
    }