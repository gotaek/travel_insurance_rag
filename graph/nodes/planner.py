from typing import Dict, Any, List, Optional
import json
import re
import logging
from app.deps import get_planner_llm
from graph.models import PlannerResponse
from graph.config_manager import get_system_config

# 로깅 설정
logger = logging.getLogger(__name__)

INTENTS = ["summary", "compare", "qa", "recommend"]

# 보유 보험사 목록
OWNED_INSURERS = ["삼성화재", "카카오페이", "현대해상", "DB손해보험", "KB손해보험"]

# 상단에 보조 맵 추가
ALIAS_MAP = {
    # 보유 5사 (정식명 → 동일, 약칭/영문 → 정식명)
    "삼성화재": "삼성화재", "삼성": "삼성화재", "samsung fire": "삼성화재", 
    "카카오페이": "카카오페이", "카카오": "카카오페이", "kakaopay": "카카오페이", "kakao": "카카오페이",
    "현대해상": "현대해상", "현대": "현대해상", "hyundai marine": "현대해상",
    "db손해보험": "DB손해보험", "db": "DB손해보험", "동부화재": "DB손해보험", "동부": "DB손해보험",
    "kb손해보험": "KB손해보험", "kb": "KB손해보험", "kb손해": "KB손해보험",

    # 비보유 예시(확장 가능) — 정식명/약칭을 동일 canonical로
    "한화손해보험": "한화손해보험", "한화": "한화손해보험",
    "메리츠화재": "메리츠화재", "메리츠": "메리츠화재",
    "롯데손해보험": "롯데손해보험", "롯데": "롯데손해보험",
    "nh손해보험": "NH손해보험", "nh": "NH손해보험",
    "흥국화재": "흥국화재", "흥국": "흥국화재",
    "axa손해보험": "AXA손해보험", "axa": "AXA손해보험",
    "mg손해보험": "MG손해보험", "mg": "MG손해보험",
    "신한손해보험": "신한손해보험", "신한": "신한손해보험",
    "하나손해보험": "하나손해보험", "하나": "하나손해보험",
}

# 정규식 패턴용 후보(길이 내림차순으로 중복/겹침 방지)
ALIAS_SORTED = sorted(ALIAS_MAP.keys(), key=len, reverse=True)

def _extract_insurers_from_question(question: str) -> List[str]:
    """
    질문에서 보험사 엔티티(정식명 canonical)를 추출합니다.
    - 긴 별칭 우선 매칭
    - 약칭/영문 별칭을 정식명으로 정규화
    - 한국어 조사/문장부호 경계 허용
    - 중복 제거, 겹침 방지
    """
    q = question.lower()
    found = []
    used_spans = []  # (start, end)로 겹침 방지

    for alias in ALIAS_SORTED:
        # 간단한 단어 경계 검색
        if alias.lower() in q:
            # 겹침 방지
            start_pos = q.find(alias.lower())
            end_pos = start_pos + len(alias.lower())
            
            # 겹치는지 확인
            if any(not (end_pos <= s or e <= start_pos) for s, e in used_spans):
                continue
                
            canon = ALIAS_MAP[alias]
            found.append(canon)
            used_spans.append((start_pos, end_pos))

    # 순서 보존 중복 제거
    seen = set()
    dedup = []
    for c in found:
        if c not in seen:
            dedup.append(c)
            seen.add(c)
    return dedup

def _determine_insurer_filter_and_web_need(question: str) -> Dict[str, Any]:
    """
    추출 결과를 보유/비보유로 나누고, filter/needs_web을 결정
    """
    extracted = _extract_insurers_from_question(question)

    if not extracted:
        return {
            "insurer_filter": None,
            "needs_web": False,
            "extracted_insurers": [],
            "owned_insurers": [],
            "non_owned_insurers": []
        }

    # 보유/비보유 분리
    owned = [c for c in extracted if c in OWNED_INSURERS]
    non_owned = [c for c in extracted if c not in OWNED_INSURERS]

    # 하나라도 비보유가 있으면 웹 필요
    needs_web = len(non_owned) > 0

    # 보유사가 하나라도 있으면 filter, 없으면 None
    insurer_filter = owned if owned else None

    return {
        "insurer_filter": insurer_filter,
        "needs_web": needs_web,
        "extracted_insurers": extracted,
        "owned_insurers": owned,
        "non_owned_insurers": non_owned
    }

def _llm_classify_intent(question: str) -> Dict[str, Any]:
    """
    LLM을 사용하여 질문의 도메인 관련성, intent와 needs_web을 분류 (structured output 사용)
    """
    prompt = f"""
다음 질문을 분석하여 여행자보험 RAG 시스템에서 적절한 처리 방식을 결정해주세요.

질문: "{question}"

먼저 이 질문이 여행자 보험 도메인과 관련된지 판단하세요:
- 여행자 보험, 해외여행보험, 보험료, 보험가입, 보험상품, 보험약관, 보험보장, 보험혜택
- 보험금, 보험지급, 보험배상, 보험면책, 보험제외, 보험조건, 보험기간, 보험범위, 보험한도
- 해외여행, 해외출장, 해외관광, 해외방문, 해외체류, 여행, 출장, 관광, 방문, 체류
- 삼성화재, 카카오페이, 현대해상, DB손해보험, KB손해보험 등 보험사
- 특약, 의료비, 치료비, 항공기지연, 수하물지연, 개인배상, 여행중단, 긴급의료 등
- 보장내용, 가입조건, 해지, 갱신, 보험금청구 등

만약 여행자 보험 도메인과 관련되지 않은 질문이라면:
- is_domain_related: false
- intent: "qa" (일반 LLM 답변)
- needs_web: false

만약 여행자 보험 도메인과 관련된 질문이라면:
- is_domain_related: true
- 다음 중 하나의 intent를 선택하세요:
  * "qa": 일반적인 질문-답변 (보장 내용, 가입 조건, 보험료 등)
  * "summary": 문서 요약 (약관 요약, 상품 정리 등)
  * "compare": 비교 분석 (보험 상품 간 비교, 차이점 분석 등)
  * "recommend": 추천 및 권장 (특약 추천, 여행지별 보험 추천 등)

- 다음 조건을 확인하여 needs_web을 결정하세요:
  * 최신 뉴스나 실시간 정보가 필요한가?
  * 특정 날짜나 지역의 현재 상황이 필요한가?
  * 여행지의 현재 안전 상황이나 규제가 필요한가?
  * 가격 비교가 필요한가?
  * 리워드 정보가 필요한가?
"""

    try:
        logger.debug("LLM을 사용한 의도 분류 시작 (structured output)")
        llm = get_planner_llm()
        
        # structured output 사용
        structured_llm = llm.with_structured_output(PlannerResponse)
        response = structured_llm.generate_content(prompt)
        
        logger.debug(f"Structured LLM 응답: {response}")
        
        # 유효성 검증
        is_domain_related = response.is_domain_related
        if not isinstance(is_domain_related, bool):
            is_domain_related = _is_travel_insurance_domain(question)
            logger.warning(f"유효하지 않은 is_domain_related: {is_domain_related}, 휴리스틱으로 재판단")
            
        intent = response.intent
        if intent not in INTENTS:
            logger.warning(f"유효하지 않은 의도: {intent}, 기본값 'qa' 사용")
            intent = "qa"
            
        needs_web = response.needs_web
        if not isinstance(needs_web, bool):
            needs_web = _determine_web_search_need(question, intent)
            logger.warning(f"유효하지 않은 needs_web: {needs_web}, 휴리스틱으로 재판단")
            
        return {
            "is_domain_related": is_domain_related,
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
    
    # 도메인 관련성 검사
    is_domain_related = _is_travel_insurance_domain(question)
    
    return {
        "is_domain_related": is_domain_related,
        "intent": intent,
        "needs_web": needs_web,
        "reasoning": f"Enhanced fallback: {intent} (score: {intent_scores[intent]}, web: {needs_web}, domain: {is_domain_related})"
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

def _is_travel_insurance_domain(question: str) -> bool:
    """
    질문이 여행자 보험 도메인과 관련된지 판단합니다.
    """
    ql = question.lower()
    
    # 여행자 보험 관련 핵심 키워드
    travel_insurance_keywords = [
        # 보험 관련
        "여행자보험", "여행보험", "해외여행보험", "해외보험", "여행자보험료", "여행보험료",
        "보험료", "보험가입", "보험상품", "보험약관", "보험보장", "보험혜택",
        "보험금", "보험지급", "보험배상", "보험면책", "보험제외", "보험조건",
        "보험기간", "보험범위", "보험한도", "보험조항", "보험규정", "보험정책",
        "보험보상", "보험금지급", "보험금배상", "보험금면책", "보험금제외",
        
        # 여행 관련
        "해외여행", "해외출장", "해외관광", "해외방문", "해외체류", "해외휴가",
        "여행", "출장", "관광", "방문", "체류", "휴가", "여행지", "여행국가",
        "여행기간", "여행목적", "여행일정", "여행계획", "여행준비",
        
        # 보험사 관련
        "삼성화재", "카카오페이", "현대해상", "db손해보험", "kb손해보험",
        "삼성", "카카오", "현대", "db", "kb", "동부화재", "동부", "kb손해",
        
        # 특약 관련
        "특약", "선택특약", "기본특약", "추가특약", "특별약관", "특별조항",
        "의료비", "치료비", "병원비", "의료보장", "치료보장", "병원보장",
        "항공기지연", "항공기결항", "항공기취소", "항공기지연보장", "항공기결항보장",
        "수하물지연", "수하물분실", "수하물손상", "수하물보장", "수하물지연보장",
        "개인배상", "개인배상책임", "배상책임", "배상보장", "책임보장",
        "여행중단", "여행취소", "여행지연", "여행중단보장", "여행취소보장",
        "긴급의료", "긴급송환", "긴급의료송환", "긴급의료보장", "긴급송환보장",
        
        # 보장 내용 관련
        "보장내용", "보장항목", "보장범위", "보장한도", "보장조건", "보장기간",
        "지급조건", "지급한도", "지급범위", "지급기준", "지급절차",
        "면책조항", "면책사항", "면책기간", "면책범위", "면책조건",
        "제외조항", "제외사항", "제외기간", "제외범위", "제외조건",
        
        # 가입 관련
        "가입조건", "가입자격", "가입절차", "가입방법", "가입신청", "가입서류",
        "가입비용", "가입요금", "가입수수료", "가입보험료", "가입금액",
        "가입기간", "가입일", "가입시기", "가입시점", "가입시점",
        
        # 해지/갱신 관련
        "해지", "해지조건", "해지절차", "해지방법", "해지시기", "해지시점",
        "갱신", "갱신조건", "갱신절차", "갱신방법", "갱신시기", "갱신시점",
        "자동갱신", "수동갱신", "갱신보험료", "갱신금액", "갱신기간",
        
        # 보험금 청구 관련
        "보험금청구", "보험금신청", "보험금지급", "보험금배상", "보험금처리",
        "청구절차", "청구방법", "청구서류", "청구조건", "청구기간",
        "지급절차", "지급방법", "지급서류", "지급조건", "지급기간",
        
        # 비교/추천 관련
        "보험비교", "보험상품비교", "보험료비교", "보험보장비교", "보험혜택비교",
        "보험추천", "보험상품추천", "보험선택", "보험가이드", "보험상담",
        "어떤보험", "어떤상품", "어떤보험이", "어떤상품이", "어떤것이",
        
        # 요약 관련
        "보험약관", "보험조항", "보험규정", "보험정책", "보험내용",
        "약관요약", "조항요약", "규정요약", "정책요약", "내용요약",
        "보험정리", "보험개요", "보험핵심", "보험주요", "보험총정리"
    ]
    
    # 여행자 보험 관련 키워드가 있는지 확인
    domain_score = 0
    for keyword in travel_insurance_keywords:
        if keyword in ql:
            domain_score += 1
    
    # 여행자 보험 도메인 관련 키워드가 1개 이상 있으면 도메인 관련으로 판단
    is_domain_related = domain_score > 0
    
    # 디버깅을 위한 로그
    logger.debug(f"도메인 관련성 점수: {domain_score}점, 도메인 관련: {is_domain_related}")
    
    return is_domain_related

def _determine_web_search_need(question: str, intent: str) -> bool:
    """
    웹 검색 필요성을 정교하게 판단
    """
    ql = question.lower()
    
    # 날짜 패턴 (확장)
    date_patterns = [
        # 연도 패턴
        r"\d{4}년", r"\d{4}-\d{2}", r"\d{4}/\d{2}", r"\d{4}\.\d{2}",
        r"\d{4}년\s*\d{1,2}월", r"\d{4}-\d{2}-\d{2}", r"\d{4}/\d{2}/\d{2}",
        
        # 월 패턴
        r"\d{1,2}월", r"\d{1,2}월\s*\d{1,2}일", r"\d{1,2}월\s*\d{1,2}일",
        r"1월", r"2월", r"3월", r"4월", r"5월", r"6월",
        r"7월", r"8월", r"9월", r"10월", r"11월", r"12월",
        r"일월", r"이월", r"삼월", r"사월", r"오월", r"유월",
        r"칠월", r"팔월", r"구월", r"시월", r"십일월", r"십이월",
        
        # 계절 패턴
        r"봄", r"여름", r"가을", r"겨울", r"봄철", r"여름철", r"가을철", r"겨울철",
        r"봄여행", r"여름여행", r"가을여행", r"겨울여행",
        r"봄휴가", r"여름휴가", r"가을휴가", r"겨울휴가",
        
        # 주 단위 패턴
        r"다음주", r"이번 주", r"이번주", r"다음 주", r"이번주말", r"다음주말",
        r"주말", r"주중", r"평일", r"휴일", r"공휴일",
        r"일주일", r"2주일", r"3주일", r"한 주", r"두 주", r"세 주",
        r"일주일 후", r"2주일 후", r"3주일 후", r"한 주 후", r"두 주 후",
        r"일주일 이내", r"2주일 이내", r"3주일 이내", r"한 주 이내", r"두 주 이내",
        
        # 일 단위 패턴
        r"내일", r"오늘", r"모레", r"글피", r"어제", r"그제",
        r"오늘부터", r"내일부터", r"모레부터", r"오늘부터\s*\d+일",
        r"내일부터\s*\d+일", r"모레부터\s*\d+일",
        r"\d+일 후", r"\d+일 뒤", r"\d+일 뒤에", r"\d+일 후에",
        r"며칠 후", r"며칠 뒤", r"며칠 뒤에", r"며칠 후에",
        r"하루", r"이틀", r"사흘", r"나흘", r"닷새", r"엿새", r"이레",
        r"하루 후", r"이틀 후", r"사흘 후", r"나흘 후", r"닷새 후",
        
        # 월 단위 패턴
        r"다음 달", r"이번 달", r"이번달", r"다음달", r"다음 달", r"이번 달",
        r"한 달", r"두 달", r"세 달", r"1개월", r"2개월", r"3개월",
        r"한 달 후", r"두 달 후", r"세 달 후", r"1개월 후", r"2개월 후",
        r"한 달 이내", r"두 달 이내", r"세 달 이내", r"1개월 이내", r"2개월 이내",
        
        # 연도 패턴
        r"내년", r"올해", r"작년", r"내후년", r"재작년",
        r"2024년", r"2025년", r"2026년", r"2027년", r"2028년",
        r"내년 여름", r"내년 겨울", r"올해 여름", r"올해 겨울",
        r"내년 봄", r"내년 가을", r"올해 봄", r"올해 가을",
        
        # 시간 관련 패턴
        r"현재", r"지금", r"요즘", r"최근", r"최신", r"요새", r"요즈음",
        r"최근에", r"요즘에", r"지금까지", r"현재까지", r"요즘까지",
        r"최근 몇", r"요즘 몇", r"지금 몇", r"현재 몇",
        r"최근 몇일", r"요즘 몇일", r"지금 몇일", r"현재 몇일",
        r"최근 몇주", r"요즘 몇주", r"지금 몇주", r"현재 몇주",
        r"최근 몇개월", r"요즘 몇개월", r"지금 몇개월", r"현재 몇개월",
        
        # 특별한 날짜 패턴
        r"설날", r"추석", r"어린이날", r"어버이날", r"스승의날",
        r"현충일", r"광복절", r"개천절", r"한글날", r"크리스마스",
        r"신정", r"구정", r"부처님오신날", r"어린이날", r"어버이날",
        r"설날 연휴", r"추석 연휴", r"어린이날 연휴", r"현충일 연휴",
        r"광복절 연휴", r"개천절 연휴", r"한글날 연휴", r"크리스마스 연휴",
        
        # 휴가/여행 관련 패턴
        r"휴가", r"여행", r"출장", r"관광", r"방문", r"체류",
        r"휴가철", r"여행철", r"출장철", r"관광철", r"방문철",
        r"휴가 기간", r"여행 기간", r"출장 기간", r"관광 기간", r"방문 기간",
        r"휴가 때", r"여행 때", r"출장 때", r"관광 때", r"방문 때",
        r"휴가 중", r"여행 중", r"출장 중", r"관광 중", r"방문 중",
        
        # 기간 표현 패턴
        r"기간", r"동안", r"사이", r"중에", r"때", r"중",
        r"부터", r"까지", r"~", r"-", r"에서", r"로",
        r"이내", r"안에", r"내에", r"후", r"뒤", r"뒤에", r"후에",
        r"전에", r"앞에", r"앞서", r"이전", r"이후", r"이후에",
        
        # 숫자 + 단위 패턴
        r"\d+일", r"\d+주", r"\d+개월", r"\d+년",
        r"\d+일간", r"\d+주간", r"\d+개월간", r"\d+년간",
        r"\d+일 동안", r"\d+주 동안", r"\d+개월 동안", r"\d+년 동안",
        r"\d+일째", r"\d+주째", r"\d+개월째", r"\d+년째",
        
        # 상대적 시간 표현
        r"곧", r"가까운", r"가까운 시일", r"가까운 장래", r"가까운 미래",
        r"조만간", r"얼마 안", r"얼마 안에", r"얼마 안 되어", r"얼마 안 돼서",
        r"금방", r"바로", r"즉시", r"당장", r"지금 당장", r"지금 바로",
        r"언제든", r"언제든지", r"언제나", r"항상", r"계속", r"지속적으로"
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
        "실시간", "최신", "업데이트", "변동", "시세", "비교해주세요",
        "비교해", "비교해줘", "비교해주시고", "비교해주세요"
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
    
    # 디버깅을 위한 로그 추가
    logger.debug(f"웹 검색 점수 계산: {web_score}점 (날짜:{has_date}, 지역:{has_city}, 실시간:{has_live}, 안전:{has_safety}, 가격:{has_price}, 혜택:{has_benefit}, intent:{intent})")
    
    # 웹 검색 필요성 임계값 (5점 이상이면 웹 검색 필요)
    return web_score >= 5

def _needs_llm_classification(question: str) -> bool:
    """
    복잡한 케이스인지 판단하여 LLM 분류가 필요한지 결정
    """
    # 시스템 설정에서 임계값 가져오기
    config = get_system_config()
    threshold = config.get_complex_case_threshold()
    
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
    
    # 복잡한 키워드가 임계값 이상 있으면 LLM 사용
    complex_count = sum(1 for pattern in complex_patterns if pattern in question)
    return complex_count >= threshold

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
    LLM 기반 질문 분석 및 분기 결정 (정확도 향상: LLM 분류 우선 사용)
    보험사 엔티티 추출 및 필터링 로직 포함
    여행자 보험 도메인 관련 질문이 아닌 경우 바로 qa 노드로 라우팅
    """
    q = state.get("question", "")
    replan_count = state.get("replan_count", 0)
    
    # 재검색 횟수 로깅
    if replan_count > 0:
        logger.info(f"재검색으로 인한 planner 재실행 - 재검색 횟수: {replan_count}")
    
    # LLM 분류 우선 사용 (도메인 관련성 포함)
    logger.debug("LLM 분류 우선 사용 (도메인 관련성 포함)")
    try:
        classification = _llm_classify_intent(q)
        logger.debug("LLM 분류 성공")
    except Exception as e:
        logger.warning(f"LLM 분류 실패, fallback 사용: {str(e)}")
        classification = _fallback_classify(q)
    
    # LLM 분류 결과에서 도메인 관련성 확인
    is_domain_related = classification["is_domain_related"]
    logger.info(f"도메인 관련성 검사: {is_domain_related}")
    
    # 여행자 보험 도메인과 관련되지 않은 질문인 경우 바로 qa 노드로 라우팅
    if not is_domain_related:
        logger.info(f"🚫 비도메인 질문 감지 - 바로 qa 노드로 라우팅: '{q}'")
        return {
            **state,
            "intent": "qa",
            "classification_reasoning": classification.get("reasoning", "여행자 보험 도메인과 관련되지 않은 질문으로 일반 LLM 답변 제공"),
            "is_domain_related": False,
            "insurer_filter": None,
            "extracted_insurers": [],
            "owned_insurers": [],
            "non_owned_insurers": [],
            "replan_count": replan_count,
            "max_replan_attempts": state.get("max_replan_attempts", 3)
        }
    
    # 여행자 보험 도메인 관련 질문인 경우 기존 로직 수행
    logger.info(f"✅ 도메인 관련 질문 - RAG 파이프라인 실행: '{q}'")
    
    # 보험사 엔티티 추출 및 needs_web 결정
    insurer_info = _determine_insurer_filter_and_web_need(q)
    logger.info(f"보험사 추출 결과: {insurer_info}")
    
    # 디버깅을 위한 상세 로그
    logger.info(f"질문: '{q}'")
    logger.info(f"추출된 보험사: {insurer_info['extracted_insurers']}")
    logger.info(f"보유 보험사: {insurer_info['owned_insurers']}")
    logger.info(f"비보유 보험사: {insurer_info['non_owned_insurers']}")
    logger.info(f"보험사 기반 needs_web: {insurer_info['needs_web']}")
    
    # 이미 위에서 분류 결과를 받았으므로 재사용
    intent = classification["intent"]
    
    # 2번째 사이클에서는 무조건 needs_web을 True로 설정
    if replan_count >= 1:
        needs_web = True
        logger.info(f"🔄 2번째 사이클 이상 - 무조건 웹 검색 활성화 (재검색 횟수: {replan_count})")
    else:
        # 보험사 정보와 키워드 기반 needs_web을 OR 조건으로 결합
        # 둘 중 하나라도 True면 웹 검색 필요
        insurer_based_web = insurer_info["needs_web"]
        keyword_based_web = classification["needs_web"]
        needs_web = insurer_based_web or keyword_based_web
        
        logger.info(f"웹 검색 필요성 결정:")
        logger.info(f"  보험사 기반: {insurer_based_web} (추출된 보험사: {insurer_info['extracted_insurers']})")
        logger.info(f"  키워드 기반: {keyword_based_web} (intent: {intent})")
        logger.info(f"  최종 결정: {needs_web} (OR 조건)")
    
    reasoning = classification.get("reasoning", "")
    
    return {
        **state, 
        "intent": intent, 
        "needs_web": needs_web, 
        "classification_reasoning": reasoning,
        "is_domain_related": True,  # 도메인 관련 질문임을 명시
        # 보험사 필터 정보 추가
        "insurer_filter": insurer_info["insurer_filter"],
        "extracted_insurers": insurer_info["extracted_insurers"],
        "owned_insurers": insurer_info["owned_insurers"],
        "non_owned_insurers": insurer_info["non_owned_insurers"],
        # replan_count는 명시적으로 유지 (초기화하지 않음)
        "replan_count": replan_count,
        "max_replan_attempts": state.get("max_replan_attempts", 3)  # 기본값 3으로 설정
    }