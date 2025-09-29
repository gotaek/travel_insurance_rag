# graph/nodes/websearch.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import logging
import hashlib
import json
from datetime import datetime, timedelta
from tavily import TavilyClient
from app.deps import get_settings, get_redis_client

# 로깅 설정
logger = logging.getLogger(__name__)

# 여행자보험 관련 신뢰할 수 있는 도메인 목록 (완화된 필터)
TRUSTED_DOMAINS = [
    # 포털 사이트
    "naver.com", "daum.net", "kakao.com", "google.com",
    # 보험사 공식 사이트
    "dbinsu.co.kr", "kbinsure.co.kr", "samsungfire.com", "hyundai.com",
    "chubb.com", "aig.com", "allianz.com",  # 해외 보험사
    # 정부/공공기관
    "fss.or.kr", "kdi.re.kr", "korea.kr", "visitkorea.or.kr", "mofa.go.kr",
    # 여행 관련 사이트
    "tripadvisor.co.kr", "agoda.com", "booking.com",
    # 전문 정보 사이트
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
    시간 관련 매개변수를 활용하여 최적화된 검색을 수행합니다.
    
    Args:
        state: RAG 상태 딕셔너리
        
    Returns:
        web_results가 추가된 상태 딕셔너리
    """
    try:
        # 캐시 확인
        cached_results = _check_cache(state)
        if cached_results:
            logger.info("캐시된 웹 검색 결과 사용")
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
        
        # 성능 최적화: 단일 쿼리로 검색 (다중 쿼리 제거)
        all_results = []
        try:
            # 가장 관련성 높은 쿼리 하나만 사용
            primary_query = search_queries[0] if search_queries else f"여행자보험 {state.get('question', '')}"
            
            logger.debug(f"웹 검색 쿼리: {primary_query}")
            logger.debug(f"검색 매개변수: {search_params}")
            
            # 최적화된 API 호출
            response = client.search(
                query=primary_query,
                search_depth="basic",
                max_results=8,  # 결과 수 증가로 품질 향상
                exclude_domains=EXCLUDED_DOMAINS,
                **search_params  # 시간 관련 매개변수 포함
            )
            
            # 결과 처리 및 품질 평가
            processed_results = _process_search_results(response.get("results", []), state)
            all_results.extend(processed_results)
            
            # 뉴스 검색이 필요한 경우 추가 검색 수행
            if search_params.get("topic") == "news" and len(processed_results) < 3:
                logger.debug("뉴스 검색 결과 부족, 추가 검색 수행")
                try:
                    # 뉴스 전용 검색 수행
                    news_response = client.search(
                        query=f"여행자보험 뉴스 {state.get('question', '')}",
                        search_depth="basic",
                        max_results=5,
                        topic="news",
                        days=7,  # 최근 7일간의 뉴스
                        exclude_domains=EXCLUDED_DOMAINS
                    )
                    
                    news_results = _process_search_results(news_response.get("results", []), state)
                    all_results.extend(news_results)
                    
                except Exception as news_e:
                    logger.warning(f"뉴스 검색 중 오류: {str(news_e)}")
            
        except Exception as e:
            logger.warning(f"웹 검색 실행 중 오류: {str(e)}")
            # 실패 시 fallback 결과 반환
            return _get_fallback_results(state)
        
        # 결과 정렬 및 중복 제거
        web_results = _deduplicate_and_rank(all_results)
        
        # 캐시 저장
        _save_to_cache(state, web_results)
        
        logger.info(f"웹 검색 완료: {len(web_results)}개 결과")
        return {**state, "web_results": web_results}
        
    except Exception as e:
        logger.error(f"웹 검색 중 오류 발생: {str(e)}")
        return _get_fallback_results(state)

def _build_search_queries(state: Dict[str, Any]) -> List[str]:
    """
    질문과 의도에 따라 검색 쿼리를 구성합니다.
    400자 제한을 고려하여 쿼리를 단축합니다.
    
    Args:
        state: RAG 상태 딕셔너리
        
    Returns:
        검색 쿼리 리스트 (400자 제한)
    """
    question = state.get("question", "")
    intent = state.get("intent", "qa")
    
    # 질문 길이 제한 (400자 제한 고려)
    max_question_length = 300  # 여유를 두고 300자로 제한
    if len(question) > max_question_length:
        question = question[:max_question_length] + "..."
    
    # 기본 쿼리
    base_query = f"여행자보험 {question}"
    queries = [base_query]
    
    # 의도별 특화 쿼리 추가 (각 쿼리도 400자 제한)
    if intent == "compare":
        queries.extend([
            f"여행자보험 비교 {question[:100]}",
            f"보험상품 비교 {question[:100]}",
            f"여행자보험 가격비교 {question[:100]}"
        ])
    elif intent == "recommend":
        queries.extend([
            f"여행자보험 추천 {question[:100]}",
            f"여행지별 보험 {question[:100]}",
            f"여행자보험 특약 추천 {question[:100]}"
        ])
    elif intent == "summary":
        queries.extend([
            f"여행자보험 약관 요약 {question[:100]}",
            f"보험상품 정리 {question[:100]}"
        ])
    else:  # qa
        queries.extend([
            f"여행자보험 보장내용 {question[:100]}",
            f"여행자보험 가입조건 {question[:100]}",
            f"여행자보험 보험료 {question[:100]}"
        ])
    
    # 각 쿼리가 400자를 초과하지 않도록 제한
    limited_queries = []
    for query in queries[:3]:  # 최대 3개 쿼리만 사용
        if len(query) > 400:
            query = query[:400]
        limited_queries.append(query)
    
    return limited_queries

def _get_optimized_search_params(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    질문 유형과 의도에 따라 최적화된 검색 매개변수를 설정합니다.
    Tavily API의 시간 관련 매개변수를 활용합니다.
    
    Args:
        state: RAG 상태 딕셔너리
        
    Returns:
        최적화된 검색 매개변수 딕셔너리
    """
    question = state.get("question", "").lower()
    intent = state.get("intent", "qa")
    
    # 기본 매개변수
    params = {}
    
    # 뉴스 관련 키워드 감지
    news_keywords = ["뉴스", "최신", "시장", "동향", "트렌드", "변화", "업데이트", "발표", "공지"]
    is_news_query = any(keyword in question for keyword in news_keywords)
    
    # 의도별 시간 범위 설정
    if intent == "compare" or is_news_query:
        # 비교나 뉴스 관련 질문은 최근 정보가 중요
        params["topic"] = "news"
        params["days"] = 30  # 최근 30일간의 뉴스
        params["time_range"] = "month"  # 최근 한 달
    elif intent == "recommend":
        # 추천은 최신 상품 정보가 중요
        params["topic"] = "news"
        params["days"] = 14  # 최근 2주간의 정보
        params["time_range"] = "week"  # 최근 일주일
    elif intent == "summary":
        # 요약은 일반적인 정보도 포함
        params["time_range"] = "year"  # 최근 1년
    else:  # qa
        # 일반 질문은 최근 정보 우선
        params["time_range"] = "month"  # 최근 한 달
    
    # 특정 날짜 범위가 필요한 경우 (예: 특정 기간의 보험 상품 비교)
    if "2024" in question or "2025" in question:
        # 특정 연도 언급 시 해당 연도 범위 설정
        if "2025" in question:
            params["start_date"] = "2025-01-01"
            params["end_date"] = "2025-12-31"
        elif "2024" in question:
            params["start_date"] = "2024-01-01"
            params["end_date"] = "2024-12-31"
    
    # 여행 시즌별 최적화
    current_month = datetime.now().month
    if current_month in [6, 7, 8]:  # 여름 휴가철
        if "여름" in question or "휴가" in question:
            params["time_range"] = "month"
            params["days"] = 30
    elif current_month in [12, 1, 2]:  # 겨울 휴가철
        if "겨울" in question or "설날" in question or "연말" in question:
            params["time_range"] = "month"
            params["days"] = 30
    
    logger.debug(f"설정된 검색 매개변수: {params}")
    return params

def _process_search_results(results: List[Dict], state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    검색 결과를 처리하고 품질을 평가합니다.
    시간 정보를 활용하여 최신성 점수를 추가합니다.
    
    Args:
        results: Tavily API 검색 결과
        state: RAG 상태 딕셔너리
        
    Returns:
        처리된 검색 결과 리스트
    """
    processed_results = []
    question = state.get("question", "").lower()
    intent = state.get("intent", "qa")
    
    for result in results:
        # 기본 정보 추출
        url = result.get("url", "")
        title = result.get("title", "")
        content = result.get("content", "")
        score = result.get("score", 0.0)
        published_date = result.get("published_date", "")
        
        # 여행자보험 관련성 점수 계산
        relevance_score = _calculate_relevance_score(title, content, question)
        
        # 시간 기반 신선도 점수 계산
        freshness_score = _calculate_freshness_score(published_date, intent)
        
        # 최종 점수 계산 (Tavily 점수 50% + 관련성 점수 30% + 신선도 점수 20%)
        final_score = (score * 0.5) + (relevance_score * 0.3) + (freshness_score * 0.2)
        
        # 최소 점수 완화 (0.3 → 0.2)
        if final_score >= 0.2:
            processed_results.append({
                "source": "tavily_web",
                "url": url,
                "title": title,
                "snippet": content[:500] + "..." if len(content) > 500 else content,  # 스니펫 길이 제한
                "score_web": final_score,
                "relevance_score": relevance_score,
                "freshness_score": freshness_score,
                "published_date": published_date,
                "timestamp": datetime.now().isoformat()
            })
    
    return processed_results

# 도메인 가중치 함수 제거 - 단순화된 점수 계산 사용

def _calculate_relevance_score(title: str, content: str, question: str) -> float:
    """
    제목과 내용의 여행자보험 관련성을 점수화합니다. (성능 최적화)
    
    Args:
        title: 검색 결과 제목
        content: 검색 결과 내용
        question: 사용자 질문
        
    Returns:
        관련성 점수 (0.0 ~ 1.0)
    """
    text = f"{title} {content}".lower()
    
    # 핵심 키워드만 사용하여 성능 최적화
    core_keywords = ["여행자보험", "여행보험", "보험", "여행"]
    
    # 간단한 키워드 매칭 점수 계산
    keyword_count = sum(1 for keyword in core_keywords if keyword in text)
    base_score = keyword_count / len(core_keywords)
    
    # 여행자보험 키워드 보너스
    if "여행자보험" in text or "여행보험" in text:
        base_score += 0.3
    
    return min(base_score, 1.0)

def _calculate_freshness_score(published_date: str, intent: str) -> float:
    """
    게시일을 기반으로 신선도 점수를 계산합니다.
    
    Args:
        published_date: 게시일 문자열 (YYYY-MM-DD 형식)
        intent: 사용자 의도
        
    Returns:
        신선도 점수 (0.0 ~ 1.0)
    """
    if not published_date:
        return 0.5  # 날짜 정보가 없으면 중간 점수
    
    try:
        # 날짜 파싱
        pub_date = datetime.strptime(published_date, "%Y-%m-%d")
        current_date = datetime.now()
        days_diff = (current_date - pub_date).days
        
        # 의도별 신선도 가중치
        if intent == "compare" or intent == "recommend":
            # 비교나 추천은 최신 정보가 중요
            if days_diff <= 7:  # 1주일 이내
                return 1.0
            elif days_diff <= 30:  # 1개월 이내
                return 0.8
            elif days_diff <= 90:  # 3개월 이내
                return 0.6
            else:
                return 0.3
        else:  # qa, summary
            # 일반 질문은 상대적으로 오래된 정보도 유용
            if days_diff <= 30:  # 1개월 이내
                return 1.0
            elif days_diff <= 180:  # 6개월 이내
                return 0.8
            elif days_diff <= 365:  # 1년 이내
                return 0.6
            else:
                return 0.4
                
    except ValueError:
        # 날짜 파싱 실패 시 중간 점수 반환
        return 0.5

def _deduplicate_and_rank(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    중복 제거 및 점수 기반 정렬을 수행합니다.
    
    Args:
        results: 검색 결과 리스트
        
    Returns:
        정렬된 검색 결과 리스트
    """
    # URL 기반 중복 제거
    seen_urls = set()
    unique_results = []
    
    for result in results:
        url = result.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)
    
    # 점수 기반 정렬 (내림차순)
    unique_results.sort(key=lambda x: x.get("score_web", 0), reverse=True)
    
    return unique_results[:3]  # 성능 최적화: 상위 3개 결과만 반환

def _check_cache(state: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    Redis에서 캐시된 검색 결과를 확인합니다.
    
    Args:
        state: RAG 상태 딕셔너리
        
    Returns:
        캐시된 결과 또는 None
    """
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return None
        
        # 캐시 키 생성
        cache_key = _generate_cache_key(state)
        cached_data = redis_client.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        
    except Exception as e:
        logger.warning(f"캐시 확인 중 오류: {str(e)}")
    
    return None

def _save_to_cache(state: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    """
    검색 결과를 Redis에 캐시합니다.
    
    Args:
        state: RAG 상태 딕셔너리
        results: 검색 결과 리스트
    """
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return
        
        # 캐시 키 생성
        cache_key = _generate_cache_key(state)
        
        # 10분 TTL로 캐시 저장 (성능 최적화)
        redis_client.setex(
            cache_key,
            600,  # 10분
            json.dumps(results, ensure_ascii=False)
        )
        
        logger.debug(f"검색 결과 캐시 저장: {cache_key}")
        
    except Exception as e:
        logger.warning(f"캐시 저장 중 오류: {str(e)}")

def _generate_cache_key(state: Dict[str, Any]) -> str:
    """
    상태 정보를 기반으로 캐시 키를 생성합니다.
    시간 매개변수도 고려하여 캐시 키를 생성합니다.
    
    Args:
        state: RAG 상태 딕셔너리
        
    Returns:
        캐시 키 문자열
    """
    question = state.get("question", "")
    intent = state.get("intent", "qa")
    
    # 시간 매개변수 정보 추가
    search_params = _get_optimized_search_params(state)
    time_info = f"{search_params.get('time_range', 'default')}:{search_params.get('days', 'default')}"
    
    # 질문, 의도, 시간 정보를 해시화하여 캐시 키 생성
    key_data = f"websearch:{intent}:{time_info}:{question}"
    return hashlib.md5(key_data.encode('utf-8')).hexdigest()

def _get_fallback_results(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tavily API 사용 불가 시 대체 결과 반환.
    
    Args:
        state: RAG 상태 딕셔너리
        
    Returns:
        대체 결과가 포함된 상태 딕셔너리
    """
    question = state.get("question", "")
    intent = state.get("intent", "qa")
    
    # 의도별 맞춤형 대체 메시지
    if intent == "compare":
        message = f"'{question}' 관련 여행자보험 비교 정보는 보험사 고객센터나 공식 홈페이지에서 확인하세요."
    elif intent == "recommend":
        message = f"'{question}' 관련 여행자보험 추천은 여행 목적과 예산에 따라 다르므로 전문가 상담을 권장합니다."
    else:
        message = f"'{question}' 관련 여행자보험 정보는 보험사 고객센터에 문의하시기 바랍니다."
    
    fallback_hits: List[Dict[str, Any]] = [
        {
            "source": "fallback_stub",
            "url": "https://www.fss.or.kr/",  # 금융감독원 공식 사이트
            "title": "여행자보험 관련 정보 안내",
            "snippet": f"검색 서비스 일시 중단. {message}",
            "score_web": 0.5,
            "relevance_score": 0.3,
            "timestamp": datetime.now().isoformat()
        }
    ]
    return {**state, "web_results": fallback_hits}

__all__ = ["websearch_node"]