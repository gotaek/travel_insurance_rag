from __future__ import annotations
from typing import Dict, Any, List, Optional
import logging
import hashlib
import json
from datetime import datetime
from tavily import TavilyClient
from app.deps import get_settings, get_redis_client

# 로깅 설정
logger = logging.getLogger(__name__)

# 여행자보험 관련 신뢰할 수 있는 도메인 목록 (보강)
TRUSTED_DOMAINS = [
    # 포털
    "naver.com", "daum.net", "kakao.com", "google.com",
    # 보험사 공식 사이트
    "dbinsu.co.kr", "kbinsure.co.kr", "samsungfire.com", "hyundai.com",
    "chubb.com", "aig.com", "allianz.com",
    # 정부/공공/보건
    "mofa.go.kr", "cdc.gov", "who.int", "travel.state.gov", "gov.uk",
    "smartraveller.gov.au", "ecdc.europa.eu",
    "fss.or.kr", "kdi.re.kr", "korea.kr", "visitkorea.or.kr",
    # 여행 관련 사이트
    "tripadvisor.co.kr", "agoda.com", "booking.com",
    # 전문 정보 사이트/커뮤니티(최소)
    "reddit.com", "stonewellfinancial.com", "insuranceforisrael.com"
]

# 제외할 도메인 목록 (최소화)
EXCLUDED_DOMAINS = [
    "wikipedia.org",  # 위키 사이트만 제외
    "pornhub.com", "xvideos.com"  # 성인 사이트만 제외
]


def websearch_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tavily API를 사용한 여행자보험 도메인 특화 웹 검색 노드.
    여행지가 감지되면 위험 정보를 수집하여 특약을 추천하고,
    그 외에는 일반 검색을 수행합니다.

    Args:
        state: RAG 상태 딕셔너리

    Returns:
        web_results(검색결과) 및 risk_assessment(위험/특약 추천)가 추가된 상태 딕셔너리
    """
    try:
        # 캐시 확인
        cached_results = _check_cache(state)
        if cached_results:
            logger.info("캐시된 웹 검색 결과 사용")
            # 캐시에 risk_assessment는 없을 수 있으니 최소한 web_results만 반영
            return {**state, "web_results": cached_results}

        # Tavily API 키 확인
        settings = get_settings()
        api_key = settings.TAVILY_API_KEY
        if not api_key:
            logger.warning("TAVILY_API_KEY가 설정되지 않음. 스텁 결과 반환")
            return _get_fallback_results(state)

        # Tavily 클라이언트 초기화
        client = TavilyClient(api_key=api_key)

        # 도메인 특화 검색 쿼리 구성
        search_queries = _build_search_queries(state)

        # 시간 기반 검색 매개변수 설정
        search_params = _get_optimized_search_params(state)

        # 성능 최적화: 단일 쿼리로 검색 (가장 관련성 높은 쿼리 하나만 사용)
        all_results: List[Dict[str, Any]] = []
        try:
            primary_query = search_queries[0] if search_queries else f"여행자보험 {state.get('question', '')}"

            logger.debug(f"웹 검색 쿼리: {primary_query}")
            logger.debug(f"검색 매개변수: {search_params}")

            response = client.search(
                query=primary_query,
                search_depth="basic",
                max_results=8,
                exclude_domains=EXCLUDED_DOMAINS,
                **search_params
            )

            processed_results = _process_search_results(response.get("results", []), state)
            all_results.extend(processed_results)

            # [ADD] 여행지 감지 → 위험 정보 보강 검색
            destinations = _extract_destinations(state.get("question", ""))
            if destinations:
                risk_queries = _build_risk_queries(destinations, state.get("question", ""))
                for rq in risk_queries:
                    try:
                        r = client.search(
                            query=rq,
                            search_depth="basic",
                            max_results=5,
                            exclude_domains=EXCLUDED_DOMAINS,
                            topic="news",
                            days=90,
                        )
                        risk_proc = _process_search_results(r.get("results", []), state)
                        all_results.extend(risk_proc)
                    except Exception:
                        continue
            
            # [ADD] 보험사명 감지 → 보험사별 특화 검색
            insurance_companies = _extract_insurance_companies(state.get("question", ""))
            if insurance_companies:
                # 질문 그대로를 쿼리로 사용하여 네이버에서 정보 수집
                try:
                    c = client.search(
                        query=state.get("question", ""),
                        search_depth="basic",
                        max_results=4,
                        exclude_domains=EXCLUDED_DOMAINS,
                        # topic과 days 제거하여 기사가 아닌 일반 정보 수집
                    )
                    company_proc = _process_company_results(c.get("results", []), state, insurance_companies[0])
                    all_results.extend(company_proc)
                except Exception:
                    pass

            # 뉴스 검색이 필요한 경우 추가 검색 수행
            if search_params.get("topic") == "news" and len(processed_results) < 3:
                logger.debug("뉴스 검색 결과 부족, 추가 검색 수행")
                try:
                    news_response = client.search(
                        query=f"여행자보험 뉴스 {state.get('question', '')}",
                        search_depth="basic",
                        max_results=5,
                        topic="news",
                        days=7,
                        exclude_domains=EXCLUDED_DOMAINS,
                    )
                    news_results = _process_search_results(news_response.get("results", []), state)
                    all_results.extend(news_results)
                except Exception as news_e:
                    logger.warning(f"뉴스 검색 중 오류: {str(news_e)}")

        except Exception as e:
            logger.warning(f"웹 검색 실행 중 오류: {str(e)}")
            return _get_fallback_results(state)

        # 결과 정렬 및 중복 제거
        web_results = _deduplicate_and_rank(all_results)

        # [ADD] 위험 신호 집계 및 특약 추천
        agg: Dict[str, float] = {}
        for it in web_results:
            rs = it.get("risk_signals", {}) or {}
            for k, v in rs.items():
                agg[k] = max(agg.get(k, 0.0), float(v))  # max pooling

        suggested = _map_risks_to_riders(agg) if agg else []

        def _risk_summary(a: Dict[str, float]) -> str:
            if not a:
                return "특별한 위험 징후가 두드러지지 않습니다. 일반형 구성으로도 충분해 보입니다."
            top = sorted(a.items(), key=lambda x: x[1], reverse=True)[:3]
            label_kr = {
                "natural_disaster": "자연재해",
                "disease": "감염병",
                "crime_theft": "치안/도난",
                "terror_civil": "테러/소요",
                "transport_strike": "교통/파업",
                "weather_extreme": "극한기상",
                "activity_risk": "레저활동 위험",
                "geo_political": "정세/입국제한",
            }
            tops = [f"{label_kr.get(k, k)}(점수 {round(v, 2)})" for k, v in top]
            return "주요 위험 요인: " + ", ".join(tops)

        risk_assessment = {
            "destinations": _extract_destinations(state.get("question", "")),
            "risk_summary": _risk_summary(agg),
            "risk_scores": agg,
            "suggested_riders": suggested,
            "evidence": [{"title": r.get("title"), "url": r.get("url")} for r in web_results[:6]],
        }

        # state에 주입
        state_out = {**state, "web_results": web_results, "risk_assessment": risk_assessment}

        # 캐시 저장 (web_results 기준)
        _save_to_cache(state_out, web_results)

        logger.info(
            f"웹 검색 완료: {len(web_results)}개 결과, 위험요인: {list(agg.keys()) if agg else '없음'}"
        )
        return state_out

    except Exception as e:
        logger.error(f"웹 검색 중 오류 발생: {str(e)}")
        return _get_fallback_results(state)


# ---------------------------------------------------------------------------
# Query Builder & Params
# ---------------------------------------------------------------------------

def _build_search_queries(state: Dict[str, Any]) -> List[str]:
    """
    질문과 의도에 따라 검색 쿼리를 구성합니다. (400자 제한 고려)
    """
    question = state.get("question", "")
    intent = state.get("intent", "qa")

    max_question_length = 300
    if len(question) > max_question_length:
        question = question[:max_question_length] + "..."

    base_query = f"여행자보험 {question}"
    queries = [base_query]

    if intent == "compare":
        queries.extend([
            f"여행자보험 비교 {question[:100]}",
            f"보험상품 비교 {question[:100]}",
            f"여행자보험 가격비교 {question[:100]}",
        ])
    elif intent == "recommend":
        queries.extend([
            f"여행자보험 추천 {question[:100]}",
            f"여행지별 보험 {question[:100]}",
            f"여행자보험 특약 추천 {question[:100]}",
        ])
    elif intent == "summary":
        queries.extend([
            f"여행자보험 약관 요약 {question[:100]}",
            f"보험상품 정리 {question[:100]}",
        ])
    else:  # qa
        queries.extend([
            f"여행자보험 보장내용 {question[:100]}",
            f"여행자보험 가입조건 {question[:100]}",
            f"여행자보험 보험료 {question[:100]}",
        ])

    limited_queries: List[str] = []
    for query in queries[:3]:  # 최대 3개만 후보화
        limited_queries.append(query[:400])
    return limited_queries


def _get_optimized_search_params(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    질문 유형과 의도에 따라 최적화된 검색 매개변수를 설정합니다.
    Tavily API의 시간 관련 매개변수를 활용합니다.
    """
    question = state.get("question", "").lower()
    intent = state.get("intent", "qa")

    params: Dict[str, Any] = {}

    news_keywords = ["뉴스", "최신", "시장", "동향", "트렌드", "변화", "업데이트", "발표", "공지"]
    is_news_query = any(keyword in question for keyword in news_keywords)

    if intent == "compare" or is_news_query:
        params["topic"] = "news"
        params["days"] = 30
        params["time_range"] = "month"
    elif intent == "recommend":
        params["topic"] = "news"
        params["days"] = 14
        params["time_range"] = "week"
    elif intent == "summary":
        params["time_range"] = "year"
    else:  # qa
        params["time_range"] = "month"

    if "2025" in question:
        params["start_date"] = "2025-01-01"
        params["end_date"] = "2025-12-31"
    elif "2024" in question:
        params["start_date"] = "2024-01-01"
        params["end_date"] = "2024-12-31"

    current_month = datetime.now().month
    if current_month in [6, 7, 8]:
        if "여름" in question or "휴가" in question:
            params["time_range"] = "month"
            params["days"] = 30
    elif current_month in [12, 1, 2]:
        if "겨울" in question or "설날" in question or "연말" in question:
            params["time_range"] = "month"
            params["days"] = 30

    logger.debug(f"설정된 검색 매개변수: {params}")
    return params


# ---------------------------------------------------------------------------
# Result Processing & Scoring
# ---------------------------------------------------------------------------

def _process_search_results(results: List[Dict], state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    검색 결과를 처리하고 품질을 평가합니다. 시간 정보를 활용하여 최신성 점수를 추가.
    위험 신호를 추출해 결과에 태깅합니다.
    """
    processed_results: List[Dict[str, Any]] = []
    question = state.get("question", "").lower()
    intent = state.get("intent", "qa")

    for result in results:
        url = result.get("url", "")
        title = result.get("title", "")
        content = result.get("content", "")
        score = result.get("score", 0.0)
        published_date = result.get("published_date", "")

        relevance_score = _calculate_relevance_score(title, content, question)
        freshness_score = _calculate_freshness_score(published_date, intent)
        final_score = (score * 0.5) + (relevance_score * 0.3) + (freshness_score * 0.2)

        if final_score >= 0.2:
            risk_signals = _extract_risk_signals(f"{title} {content}")
            processed_results.append({
                "source": "tavily_web",
                "url": url,
                "title": title,
                "snippet": content[:500] + "..." if len(content) > 500 else content,
                "score_web": final_score,
                "relevance_score": relevance_score,
                "freshness_score": freshness_score,
                "published_date": published_date,
                "timestamp": datetime.now().isoformat(),
                "risk_signals": risk_signals,
            })

    return processed_results


def _calculate_relevance_score(title: str, content: str, question: str) -> float:
    """제목과 내용의 여행자보험 관련성을 점수화 (저비용 키워드 기반)"""
    text = f"{title} {content}".lower()
    core_keywords = ["여행자보험", "여행보험", "보험", "여행"]
    keyword_count = sum(1 for keyword in core_keywords if keyword in text)
    base_score = keyword_count / len(core_keywords)
    if "여행자보험" in text or "여행보험" in text:
        base_score += 0.3
    return min(base_score, 1.0)


def _calculate_freshness_score(published_date: str, intent: str) -> float:
    """게시일 기반 신선도 점수"""
    if not published_date:
        return 0.5
    try:
        pub_date = datetime.strptime(published_date, "%Y-%m-%d")
        current_date = datetime.now()
        days_diff = (current_date - pub_date).days

        if intent in ("compare", "recommend"):
            if days_diff <= 7:
                return 1.0
            elif days_diff <= 30:
                return 0.8
            elif days_diff <= 90:
                return 0.6
            else:
                return 0.3
        else:
            if days_diff <= 30:
                return 1.0
            elif days_diff <= 180:
                return 0.8
            elif days_diff <= 365:
                return 0.6
            else:
                return 0.4
    except ValueError:
        return 0.5


def _deduplicate_and_rank(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """URL 기반 중복 제거 및 우선순위 기반 정렬 후 상위 N개 반환"""
    seen_urls = set()
    unique_results: List[Dict[str, Any]] = []
    
    for result in results:
        url = result.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)
    
    # 우선순위 기반 정렬 함수
    def get_priority_score(result):
        base_score = result.get("score_web", 0)
        
        # 보험사별 정보 우선순위 부여 (가장 높음)
        if result.get("company_focus", False):
            base_score *= 1.5
        
        # 지역 정보가 포함된 결과 우선순위 부여
        if result.get("location"):
            base_score *= 1.2
        
        # 위험 신호가 있는 결과 우선순위 부여
        risk_signals = result.get("risk_signals", {})
        if risk_signals and any(score > 0 for score in risk_signals.values()):
            base_score *= 1.1
        
        return base_score
    
    # 우선순위 점수 기반 정렬 (내림차순)
    unique_results.sort(key=get_priority_score, reverse=True)
    
    return unique_results[:5]  # 결과 수를 5개로 증가 (보험사별 정보 포함)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _check_cache(state: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Redis에서 캐시된 검색 결과를 확인"""
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return None
        cache_key = _generate_cache_key(state)
        cached_data = redis_client.get(cache_key)
        if cached_data:
            return json.loads(cached_data)
    except Exception as e:
        logger.warning(f"캐시 확인 중 오류: {str(e)}")
    return None


def _save_to_cache(state: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    """검색 결과를 Redis에 캐시 (10분 TTL)"""
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return
        cache_key = _generate_cache_key(state)
        redis_client.setex(cache_key, 600, json.dumps(results, ensure_ascii=False))
        logger.debug(f"검색 결과 캐시 저장: {cache_key}")
    except Exception as e:
        logger.warning(f"캐시 저장 중 오류: {str(e)}")


def _generate_cache_key(state: Dict[str, Any]) -> str:
    """질문/의도/시간/목적지를 반영하여 캐시 키 생성"""
    question = state.get("question", "")
    intent = state.get("intent", "qa")
    search_params = _get_optimized_search_params(state)
    time_info = f"{search_params.get('time_range', 'default')}:{search_params.get('days', 'default')}"
    dests = _extract_destinations(question)
    dest_key = ",".join(dests) if dests else "none"
    key_data = f"websearch:{intent}:{time_info}:{dest_key}:{question}"
    return hashlib.md5(key_data.encode('utf-8')).hexdigest()


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def _get_fallback_results(state: Dict[str, Any]) -> Dict[str, Any]:
    """Tavily 사용 불가 시 대체 결과 반환"""
    question = state.get("question", "")
    intent = state.get("intent", "qa")

    if intent == "compare":
        message = f"'{question}' 관련 여행자보험 비교 정보는 보험사 고객센터나 공식 홈페이지에서 확인하세요."
    elif intent == "recommend":
        message = f"""'{question}' 관련 여행자보험 추천은 여행 목적과 예산에 따라 다르므로 전문가 상담을 권장합니다."""
    else:
        message = f"'{question}' 관련 여행자보험 정보는 보험사 고객센터에 문의하시기 바랍니다."

    fallback_hits: List[Dict[str, Any]] = [
        {
            "source": "fallback_stub",
            "url": "https://www.fss.or.kr/",
            "title": "여행자보험 관련 정보 안내",
            "snippet": f"검색 서비스 일시 중단. {message}",
            "score_web": 0.5,
            "relevance_score": 0.3,
            "timestamp": datetime.now().isoformat(),
        }
    ]
    return {**state, "web_results": fallback_hits}


# ---------------------------------------------------------------------------
# Destination Extraction & Risk Intelligence
# ---------------------------------------------------------------------------

def _extract_destinations(question: str) -> List[str]:
    """질문에서 국가/도시 후보를 단순 추출 (저비용 룰 기반)"""
    if not question:
        return []
    q = question.lower()

    countries = [
        "일본", "japan", "미국", "usa", "united states", "영국", "uk", "france", "프랑스", "italy", "이탈리아",
        "spain", "스페인", "태국", "thailand", "vietnam", "베트남", "필리핀", "philippines", "india", "인도",
        "이스라엘", "israel", "turkey", "튀르키예", "turkiye", "australia", "호주", "canada", "캐나다",
        "china", "중국", "taiwan", "대만", "hong kong", "홍콩", "singapore", "싱가포르", "malaysia", "말레이시아",
        "indonesia", "인도네시아", "uae", "아랍에미리트", "dubai", "두바이", "qatar", "카타르", "egypt", "이집트",
        "morocco", "모로코", "peru", "페루", "mexico", "멕시코", "brazil", "브라질", "argentina", "아르헨티나",
        "kenya", "케냐", "south africa", "남아공", "south korea", "대한민국", "korea"
    ]

    cities = [
        "도쿄", "tokyo", "오사카", "osaka", "삿포로", "sapporo", "후쿠오카", "fukuoka", "교토", "kyoto",
        "파리", "paris", "런던", "london", "로마", "rome", "바르셀로나", "barcelona", "마드리드", "madrid",
        "방콕", "bangkok", "치앙마이", "chiang mai", "다낭", "danang", "하노이", "hanoi", "나트랑", "nha trang",
        "세부", "cebu", "마닐라", "manila", "발리", "bali", "자카르타", "jakarta", "싱가포르", "singapore",
        "상하이", "shanghai", "베이징", "beijing", "타이베이", "taipei", "홍콩", "hong kong", "두바이", "dubai",
        "이스탄불", "istanbul", "카이로", "cairo", "케이프타운", "cape town", "멜버른", "melbourne", "시드니", "sydney",
        "뉴욕", "new york", "la", "los angeles", "밴쿠버", "vancouver", "토론토", "toronto", "멕시코시티", "mexico city",
        "텔아비브", "tel aviv", "예루살렘", "jerusalem", "리우", "rio", "상파울루", "sao paulo"
    ]

    found: List[str] = []
    for token in countries + cities:
        if token in q:
            found.append(token)
    return list(dict.fromkeys(found))[:3]


def _extract_insurance_companies(question: str) -> List[str]:
    """질문에서 보험사명을 추출"""
    if not question:
        return []
    q = question.lower()
    
    # 보험사명 목록 (다양한 표현 포함)
    insurance_companies = [
        # 카카오페이 (다양한 표현)
        "카카오페이", "kakao pay", "카카오페이보험", "kakao pay insurance",
        
        # 주요 보험사
        "db손해보험", "db손보", "db insurance", "db손해",
        "kb손해보험", "kb손보", "kb insurance", "kb손해",
        "삼성화재", "samsung fire", "삼성화재보험",
        "현대해상", "hyundai marine", "현대해상보험",
        "하나손해보험", "하나손보", "hana insurance",
        "메리츠화재", "meritz fire", "메리츠화재보험",
        "동양화재", "dongyang fire", "동양화재보험",
        "한화손해보험", "hanwha insurance", "한화손보",
        "흥국화재", "heungkuk fire", "흥국화재보험",
        "lgu+손해보험", "lgu+손보", "lgu insurance",
        
        # 해외 보험사
        "aig", "allianz", "chubb", "axa", "zurich"
    ]
    
    found: List[str] = []
    for company in insurance_companies:
        if company in q:
            found.append(company)
    
    return list(dict.fromkeys(found))[:2]  # 최대 2개 보험사




def _process_company_results(results: List[Dict], state: Dict[str, Any], company: str) -> List[Dict[str, Any]]:
    """보험사별 검색 결과 처리 및 가중치 부여"""
    processed_results: List[Dict[str, Any]] = []
    question = state.get("question", "").lower()
    
    for result in results:
        url = result.get("url", "")
        title = result.get("title", "")
        content = result.get("content", "")
        score = result.get("score", 0.0)
        published_date = result.get("published_date", "")
        
        # 보험사별 관련성 점수 계산
        company_relevance_score = _calculate_company_relevance_score(title, content, question, company)
        
        # 시간 기반 신선도 점수 계산
        freshness_score = _calculate_freshness_score(published_date, "recommend")
        
        # 보험사별 가중치 적용
        company_weight = 1.5 if company.lower() in f"{title} {content}".lower() else 1.0
        
        # 최종 점수 계산 (보험사별 정보는 높은 가중치)
        final_score = ((score * 0.4) + (company_relevance_score * 0.4) + (freshness_score * 0.2)) * company_weight
        
        # 최소 점수 완화 (보험사별 정보는 더 관대하게)
        if final_score >= 0.15:
            processed_results.append({
                "source": "tavily_company",
                "url": url,
                "title": title,
                "snippet": content[:500] + "..." if len(content) > 500 else content,
                "score_web": final_score,
                "relevance_score": company_relevance_score,
                "freshness_score": freshness_score,
                "published_date": published_date,
                "company": company,
                "company_focus": True,  # 보험사별 정보임을 표시
                "timestamp": datetime.now().isoformat(),
                "risk_signals": _extract_risk_signals(f"{title} {content}"),
            })
    
    return processed_results


def _calculate_company_relevance_score(title: str, content: str, question: str, company: str) -> float:
    """보험사별 정보의 관련성을 점수화"""
    text = f"{title} {content}".lower()
    question_lower = question.lower()
    company_lower = company.lower()
    
    # 보험사 관련성 확인
    company_relevance = 1.0 if company_lower in text else 0.3
    
    # 혜택/리워드 관련 키워드
    benefit_keywords = [
        "리워드", "혜택", "이벤트", "할인", "캐시백", "적립", "포인트", "프로모션", "특가", "세일",
        "보너스", "증정", "사은품", "쿠폰", "할인쿠폰", "무료", "free"
    ]
    
    # 여행자보험 관련 키워드
    insurance_keywords = [
        "여행자보험", "여행보험", "해외여행보험", "여행자보험료", "여행보험료"
    ]
    
    # 보험사 키워드 매칭 점수
    company_count = 1 if company_lower in text else 0
    company_score = company_count * 0.5
    
    # 혜택 키워드 매칭 점수
    benefit_count = sum(1 for keyword in benefit_keywords if keyword in text)
    benefit_score = min(benefit_count / len(benefit_keywords) * 3, 1.0)
    
    # 여행자보험 키워드 매칭 점수
    insurance_count = sum(1 for keyword in insurance_keywords if keyword in text)
    insurance_score = min(insurance_count / len(insurance_keywords) * 2, 1.0)
    
    # 질문과의 관련성
    question_relevance = 0.5
    if any(keyword in question_lower for keyword in ["리워드", "혜택", "이벤트", "할인"]):
        question_relevance = 0.8
    elif company_lower in question_lower:
        question_relevance = 0.7
    
    # 최종 점수 계산
    final_score = (company_score * 0.3) + (benefit_score * 0.3) + (insurance_score * 0.2) + (question_relevance * 0.2)
    
    return min(final_score, 1.0)


def _build_risk_queries(destinations: List[str], question: str) -> List[str]:
    """여행지별 위험/경보/질병/치안/자연재해/항공/파업 이슈 전용 쿼리"""
    base: List[str] = []
    for loc in destinations:
        base.extend([
            f"{loc} 여행 경보 site:mofa.go.kr",
            f"{loc} travel advisory site:travel.state.gov",
            f"{loc} travel advice site:gov.uk",
            f"{loc} 감염병 위험 site:cdc.gov",
            f"{loc} 감염병 안내 site:who.int",
            f"{loc} 치안 위험 여행자 site:naver.com",
            f"{loc} 자연재해 태풍 지진 홍수",
            f"{loc} 파업 운항 결항 공항 지연",
        ])
    activity_keywords = ["스키", "스노보드", "트레킹", "등산", "다이빙", "스쿠버", "패러글라이딩", "번지점프", "서핑", "모터스포츠"]
    if any(k in question for k in activity_keywords):
        base.append("레저 스포츠 안전 위험도 해외 여행자")
    return base[:6]


def _extract_risk_signals(text: str) -> Dict[str, float]:
    """문장에서 위험 키워드를 점수화 (0~1)"""
    t = (text or "").lower()
    buckets: Dict[str, List[str]] = {
        "terror_civil": ["테러", "terror", "내전", "coup", "폭동", "시위", "unrest", "충돌"],
        "crime_theft": ["소매치기", "pickpocket", "절도", "theft", "강도", "robbery", "사기", "scam"],
        "natural_disaster": ["지진", "earthquake", "태풍", "typhoon", "허리케인", "hurricane", "홍수", "flood", "산불", "wildfire", "화산", "volcano", "폭우", "heavy rain"],
        "disease": ["감염병", "outbreak", "covid", "뎅기", "dengue", "말라리아", "malaria", "콜레라", "chikungunya", "measles"],
        "transport_strike": ["파업", "strike", "결항", "cancellation", "지연", "delay", "운항중단", "shutdown"],
        "weather_extreme": ["폭염", "heatwave", "한파", "cold wave", "blizzard", "폭설", "heavy snow"],
        "activity_risk": ["스키", "스노보드", "다이빙", "스쿠버", "패러글라이딩", "paragliding", "번지", "서핑", "rafting", "모터스포츠"],
        "geo_political": ["분쟁", "sanction", "border", "국경폐쇄", "입국제한", "visa suspension"],
    }
    scores: Dict[str, float] = {}
    for k, words in buckets.items():
        hit = sum(1 for w in words if w in t)
        scores[k] = min(hit / 3.0, 1.0)  # 간단 로그형 완화
    return scores


def _map_risks_to_riders(risk_scores: Dict[str, float]) -> List[Dict[str, Any]]:
    """위험 카테고리 → 대표 특약 매핑 및 가중치 순 추천"""
    mapping = [
        ("natural_disaster", "여행중단/여행취소(천재지변)", "태풍·지진·홍수 등으로 일정 변경/취소 가능성"),
        ("disease", "해외의료실비/질병치료비 + 긴급의료후송", "감염병 유행 시 치료·후송 비용 대비"),
        ("crime_theft", "휴대품 손해/도난 특약", "소매치기·절도 등 개인 소지품 도난 위험"),
        ("terror_civil", "테러/정치적 소요로 인한 여행중단·중지", "치안 불안/소요 발생시 보호"),
        ("transport_strike", "출발지연/수하물지연·분실", "항공 파업·결항·지연 및 수하물 문제 대비"),
        ("weather_extreme", "항공편 지연/결항 + 일정변경 비용", "폭염/폭설 등 극한기상으로 인한 결항 가능"),
        ("activity_risk", "레저·익스트림 스포츠 상해 특약", "스키·다이빙 등 고위험 활동 보장"),
        ("geo_political", "여행중지/귀국비용 특약", "입국 제한·국경폐쇄 등 긴급 귀국 대비"),
    ]

    recs: List[Dict[str, Any]] = []
    for key, rider, why in mapping:
        s = float(risk_scores.get(key, 0.0))
        if s > 0:
            recs.append({"rider": rider, "score": round(s, 3), "why": why})
    return sorted(recs, key=lambda x: x["score"], reverse=True)[:5]


__all__ = ["websearch_node"]
