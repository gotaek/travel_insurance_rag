from typing import Dict, Any
import json
import re
import logging
from app.deps import get_llm
from app.langsmith_llm import get_llm_with_tracing
from graph.models import PlannerResponse

# 로깅 설정
logger = logging.getLogger(__name__)

INTENTS = ["summary", "compare", "qa", "recommend"]

def _llm_classify_intent(question: str) -> Dict[str, Any]:
    """
    LLM을 사용하여 질문의 intent와 needs_web을 분류 (structured output 사용)
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
- 가격 비교가 필요한가?
"""

    try:
        logger.debug("LLM을 사용한 의도 분류 시작 (structured output)")
        llm = get_llm_with_tracing()
        
        # structured output 사용
        structured_llm = llm.with_structured_output(PlannerResponse)
        response = structured_llm.generate_content(prompt, request_options={"timeout": 10})
        
        logger.debug(f"Structured LLM 응답: {response}")
        
        # 유효성 검증
        intent = response.intent
        if intent not in INTENTS:
            logger.warning(f"유효하지 않은 의도: {intent}, 기본값 'qa' 사용")
            intent = "qa"
            
        needs_web = response.needs_web
        if not isinstance(needs_web, bool):
            needs_web = _determine_web_search_need(question, intent)
            logger.warning(f"유효하지 않은 needs_web: {needs_web}, 휴리스틱으로 재판단")
            
        return {
            "intent": intent,
            "needs_web": needs_web,
            "reasoning": response.reasoning
        }
        
    except Exception as e:
        logger.error(f"LLM 의도 분류 실패, fallback 사용: {str(e)}")
        return _fallback_classify(question)

def _fallback_classify(question: str) -> Dict[str, Any]:
    """
    LLM 호출 실패 시 사용하는 가중치 기반 향상된 fallback 분류기
    """
    ql = question.lower()
    
    # 각 intent별 가중치 점수 계산
    intent_scores = {
        "summary": 0,
        "compare": 0, 
        "recommend": 0,
        "qa": 0
    }
    
    # Summary 키워드 (가중치: 높음) - 명시적 요약 요청만
    summary_keywords = {
        "요약": 10, "정리": 8, "summary": 10, "약관 요약": 12, 
        "상품 요약": 12, "핵심 내용": 9, "간단히": 6, "줄여서": 7,
        "한눈에": 8, "개요": 9, "요점": 8, "총정리": 10,
        "요약해주세요": 12, "정리해주세요": 10, "간단히 설명": 8  # 명시적 요약 요청
    }
    for keyword, weight in summary_keywords.items():
        if keyword in question:
            intent_scores["summary"] += weight
    
    # Compare 키워드 (가중치: 높음)
    compare_keywords = {
        "비교": 10, "차이": 9, "다른 점": 10, "compare": 10, 
        "vs": 8, "대비": 8, "구분": 7, "어떤 차이": 12,
        "차이점": 10, "비교해": 9, "대조": 7, "상이": 6,
        "다르다": 8, "구별": 7, "구분하다": 7
    }
    for keyword, weight in compare_keywords.items():
        if keyword in question:
            intent_scores["compare"] += weight
    
    # Recommend 키워드 (가중치: 높음)
    recommend_keywords = {
        "추천": 10, "특약": 9, "권장": 9, "recommend": 10,
        "어떤": 7, "선택": 8, "가장 좋은": 11, "최고": 8,
        "추천해": 9, "추천해주": 9, "어떤 게": 8, "어떤 것이": 8,
        "선택해야": 9, "고르다": 7, "결정": 6, "추천받": 9,
        "도움": 6, "조언": 7, "가이드": 6, "어떤 걸": 8,
        "가장 좋을까": 12, "어떤 것이 좋을까": 12, "어떤 게 좋을까": 12,
        "어떤 보험이": 15, "어떤 상품이": 15, "어떤 것이 좋을까요": 15,  # 추가 recommend 패턴
        "가장 좋을까요": 15, "어떤 게 좋을까요": 15
    }
    for keyword, weight in recommend_keywords.items():
        if keyword in question:
            intent_scores["recommend"] += weight
    
    # QA 키워드 (가중치: 중간)
    qa_keywords = {
        "무엇": 6, "어떻게": 6, "언제": 6, "어디서": 6, "왜": 6,
        "얼마": 6, "몇": 6, "어느": 6, "무슨": 6, "어떤": 5,
        "보장": 8, "가입": 7, "보험료": 8, "조건": 7, "내용": 6,
        "혜택": 7, "지급": 7, "배상": 7, "면책": 6, "제외": 6,
        "포함": 6, "적용": 6, "기간": 6, "범위": 6, "한도": 7,
        "조항": 6, "규정": 6, "정책": 6, "약관": 7, "보상": 5,  # 보상 가중치 감소
        "뭐야": 6, "되나요": 6, "인가요": 6, "인지": 6  # 추가 QA 질문어
    }
    for keyword, weight in qa_keywords.items():
        if keyword in question:
            intent_scores["qa"] += weight
    
    # 문맥 분석을 통한 추가 점수
    context_boost = _analyze_question_context(question)
    for intent, boost in context_boost.items():
        intent_scores[intent] += boost
    
    # 가장 높은 점수의 intent 선택
    intent = max(intent_scores, key=intent_scores.get)
    
    # 점수가 너무 낮으면 기본값인 qa로 설정
    if intent_scores[intent] < 5:
        intent = "qa"
    
    # 웹 검색 필요성 판단 (개선된 로직)
    needs_web = _determine_web_search_need(question, intent)
    
    # 분류 결과 로깅 (디버깅용)
    print(f"🔍 Fallback 분류 결과: {intent} (점수: {intent_scores[intent]}, 웹검색: {needs_web})")
    
    return {
        "intent": intent,
        "needs_web": needs_web,
        "reasoning": f"Enhanced fallback: {intent} (score: {intent_scores[intent]}, web: {needs_web})"
    }

def _analyze_question_context(question: str) -> Dict[str, int]:
    """
    질문의 문맥을 분석하여 intent별 추가 점수를 부여
    """
    context_boost = {"summary": 0, "compare": 0, "recommend": 0, "qa": 0}
    
    # 질문 형태 분석
    if question.endswith("?"):
        context_boost["qa"] += 3
    
    # 복수 비교 키워드가 있으면 compare 점수 증가
    compare_indicators = ["여러", "몇 개", "여러 개", "다양한", "각각", "모든"]
    if any(indicator in question for indicator in compare_indicators):
        context_boost["compare"] += 5
    
    # 요약 관련 문맥 키워드 (명시적 요약 요청만)
    summary_context = ["전체", "모든", "종합", "포괄", "총", "전반"]
    if any(ctx in question for ctx in summary_context):
        context_boost["summary"] += 4
    
    # 추천 관련 문맥 키워드
    recommend_context = ["나에게", "내가", "저에게", "제가", "적합한", "맞는", "좋은"]
    if any(ctx in question for ctx in recommend_context):
        context_boost["recommend"] += 4
    
    # 보험 관련 전문 용어가 많으면 QA 점수 증가
    insurance_terms = ["보험료", "보장", "면책", "지급", "배상", "가입", "해지", "갱신"]
    term_count = sum(1 for term in insurance_terms if term in question)
    context_boost["qa"] += min(term_count * 2, 8)  # 최대 8점
    
    # 보험 조항/규정 관련 질문은 자동으로 compare intent로 분류
    # 단, "요약" 키워드가 있으면 summary 우선
    clause_keywords = ["조항", "규정", "정책", "약관", "보상", "보상 규정"]
    summary_keywords = ["요약", "정리", "개요", "핵심", "주요"]
    
    # 조항/규정 키워드가 있으면 관련 intent 점수를 부드럽게 가중치 부여 (강제성 완화)
    if any(keyword in question for keyword in clause_keywords):
        # "요약" 관련 키워드가 있으면 summary에 높은 가중치
        if any(summary_kw in question for summary_kw in summary_keywords):
            context_boost["summary"] += 8  # summary 우선, 기존보다 낮은 점수
        # "어떻게", "뭐야" 등 질문어가 있으면 qa에 가중치
        elif any(q_word in question for q_word in ["어떻게", "뭐야", "무엇", "무슨", "어떤", "되나요"]):
            context_boost["qa"] += 7  # 기존보다 낮은 점수

    
    return context_boost

def _determine_web_search_need(question: str, intent: str) -> bool:
    """
    웹 검색 필요성을 정교하게 판단
    """
    ql = question.lower()
    
    # 날짜 패턴 (확장)
    date_patterns = [
        r"\d{4}년", r"\d{4}-\d{2}", r"\d{4}/\d{2}", r"\d{4}\.\d{2}",
        r"\d{1,2}월", r"내년", r"올해", r"다음 달", r"이번 달",
        r"현재", r"지금", r"요즘", r"최근", r"최신"
    ]
    has_date = any(re.search(pattern, question) for pattern in date_patterns)
    
    # 지역 키워드 (핵심 도시만 선별)
    key_cities = [
        # 주요 여행지
        "도쿄", "오사카", "파리", "런던", "뉴욕", "로스앤젤레스",
        "방콕", "싱가포르", "홍콩", "시드니", "멜버른", "두바이",
        "베를린", "로마", "마드리드", "암스테르담", "취리히",
        "상하이", "베이징", "호치민", "하노이", "자카르타", "발리",
        "마닐라", "세부", "프놈펜", "시엠립", "비엔티안", "양곤",
        "뭄바이", "델리", "방갈로르", "아부다비", "도하", "리야드",
        "브리즈번", "퍼스", "오클랜드", "토론토", "밴쿠버", "몬트리올",
        "상파울루", "리우데자네이루", "부에노스아이레스", "산티아고",
        "케이프타운", "요하네스버그", "모스크바", "키예프", "바르샤바",
        
        # 국가명
        "미국", "일본", "중국", "태국", "베트남", "싱가포르", "말레이시아",
        "인도네시아", "필리핀", "인도", "호주", "뉴질랜드", "캐나다",
        "영국", "프랑스", "독일", "이탈리아", "스페인", "네덜란드",
        "스위스", "오스트리아", "벨기에", "덴마크", "스웨덴", "노르웨이",
        "핀란드", "러시아", "터키", "그리스", "포르투갈", "아일랜드",
        "폴란드", "체코", "헝가리", "루마니아", "불가리아", "크로아티아",
        "세르비아", "우크라이나", "벨라루스", "리투아니아", "라트비아",
        "에스토니아", "유럽", "아시아", "아메리카", "북미", "남미",
        "오세아니아", "아프리카", "중동", "동남아시아", "동아시아"
    ]
    has_city = any(city in ql for city in key_cities)
    
    # 실시간 정보 키워드 (확장)
    live_keywords = [
        "뉴스", "현지", "실시간", "최신", "현재", "지금", "요즘",
        "상황", "정보", "현황", "동향", "트렌드", "변화", "업데이트",
        "최근", "새로운", "변경", "수정", "발표", "공지"
    ]
    has_live = any(keyword in question for keyword in live_keywords)
    
    # 안전/보안 관련 키워드
    safety_keywords = [
        "안전", "보안", "위험", "주의", "경고", "금지", "제한",
        "테러", "사고", "재난", "재해", "감염", "질병", "전염병",
        "코로나", "covid", "백신", "검역", "격리", "봉쇄"
    ]
    has_safety = any(keyword in question for keyword in safety_keywords)
    
    # 가격/비용 비교 관련 키워드 (웹 검색 필요)
    price_keywords = [
        "가격", "비용", "요금", "보험료", "비교", "차이", "얼마",
        "저렴", "비싸", "경쟁", "시장", "현재 가격", "최신 가격",
        "가격 비교", "비용 비교", "요금 비교", "보험료 비교",
        "가장 저렴", "가장 비싼", "순서", "순위", "랭킹", "현재",
        "실시간", "최신", "업데이트", "변동", "시세"
    ]
    
    # 혜택/이벤트 관련 키워드 (웹 검색 필요 - 최신 정보)
    benefit_keywords = [
        "페이백", "캐시백", "리워드", "적립", "할인", "혜택", "이벤트",
        "프로모션", "특가", "세일", "쿠폰", "포인트", "적립금", "현금",
        "현금화", "지급", "지원", "보상", "인센티브", "추가혜택",
        "신규고객", "첫가입", "가입혜택", "신규혜택", "특별혜택"
    ]
    has_price = any(keyword in question for keyword in price_keywords)
    has_benefit = any(keyword in question for keyword in benefit_keywords)
    
    # 웹 검색 필요성 판단 로직
    web_score = 0
    
    # 날짜 정보가 있으면 +3
    if has_date:
        web_score += 3
    
    # 지역 정보가 있으면 +3
    if has_city:
        web_score += 3
    
    # 실시간 정보 키워드가 있으면 +4
    if has_live:
        web_score += 4
    
    # 안전 관련 키워드가 있으면 +3
    if has_safety:
        web_score += 3
    
    # 가격/비용 비교 키워드가 있으면 +4 (실시간 가격 정보 필요)
    if has_price:
        web_score += 4
    
    # 혜택/이벤트 키워드가 있으면 +5 (최신 혜택 정보 필요)
    if has_benefit:
        web_score += 5
    
    # Recommend intent이면서 지역/날짜/실시간 정보가 있으면 +2
    if intent == "recommend" and (has_city or has_date or has_live):
        web_score += 2
    
    # Compare intent이면서 가격 비교 키워드가 있으면 +3
    if intent == "compare" and has_price:
        web_score += 3
    
    # 특정 패턴들
    if any(pattern in question for pattern in ["어떤 보험이", "어떤 상품이", "추천해주세요"]):
        if has_city or has_date:
            web_score += 3
    
    # 웹 검색 필요성 임계값 (5점 이상이면 웹 검색 필요)
    return web_score >= 5

def _needs_llm_classification(question: str) -> bool:
    """
    복잡한 케이스인지 판단하여 LLM 분류가 필요한지 결정
    """
    # 복잡한 패턴들 (LLM이 더 정확할 수 있는 경우)
    complex_patterns = [
        # 모호한 질문
        "어떤", "어느", "무엇", "뭐", "어떻게", "왜", "언제", "어디서",
        # 복합 질문
        "그리고", "또한", "또는", "그런데", "하지만", "그러나",
        # 비교 관련
        "차이", "다른", "비교", "대비", "vs", "대조",
        # 추천 관련
        "추천", "권장", "어떤 게", "어떤 것이", "선택",
        # 요약 관련
        "요약", "정리", "핵심", "주요", "개요"
    ]
    
    # 복잡한 키워드가 2개 이상 있으면 LLM 사용
    complex_count = sum(1 for pattern in complex_patterns if pattern in question)
    return complex_count >= 2

def _is_llm_result_better(fallback_result: Dict[str, Any], llm_result: Dict[str, Any]) -> bool:
    """
    LLM 결과가 fallback 결과보다 더 정확한지 판단
    """
    # LLM 결과가 더 구체적인 reasoning을 제공하면 우선
    if len(llm_result.get("reasoning", "")) > len(fallback_result.get("reasoning", "")):
        return True
    
    # LLM이 더 구체적인 intent를 제공하면 우선
    llm_intent = llm_result.get("intent", "")
    fallback_intent = fallback_result.get("intent", "")
    
    # 특정 intent에 대한 우선순위
    intent_priority = {"recommend": 4, "compare": 3, "summary": 2, "qa": 1}
    
    llm_priority = intent_priority.get(llm_intent, 0)
    fallback_priority = intent_priority.get(fallback_intent, 0)
    
    return llm_priority > fallback_priority

def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM 기반 질문 분석 및 분기 결정 (성능 최적화: fallback 우선 사용)
    """
    q = state.get("question", "")
    replan_count = state.get("replan_count", 0)
    
    # 재검색 횟수 로깅
    if replan_count > 0:
        logger.info(f"재검색으로 인한 planner 재실행 - 재검색 횟수: {replan_count}")
    
    # 성능 최적화: fallback 분류 우선 사용
    logger.debug("빠른 fallback 분류 사용")
    classification = _fallback_classify(q)
    
    # 복잡한 케이스에만 LLM 사용 (선택적)
    if _needs_llm_classification(q):
        logger.debug("복잡한 케이스로 LLM 분류 사용")
        try:
            llm_classification = _llm_classify_intent(q)
            # LLM 결과가 더 정확하면 사용
            if _is_llm_result_better(classification, llm_classification):
                classification = llm_classification
                logger.debug("LLM 분류 결과 사용")
        except Exception as e:
            logger.warning(f"LLM 분류 실패, fallback 결과 유지: {str(e)}")
    
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
        "classification_reasoning": reasoning,
        # replan_count는 명시적으로 유지 (초기화하지 않음)
        "replan_count": replan_count,
        "max_replan_attempts": state.get("max_replan_attempts", 3)
    }