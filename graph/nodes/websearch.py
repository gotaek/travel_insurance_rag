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
        
        # 여러 쿼리로 검색 실행
        all_results = []
        for query in search_queries:
            try:
                # 도메인 필터 완화: include_domains 제거, exclude_domains만 유지
                response = client.search(
                    query=query,
                    search_depth="basic",
                    max_results=5,  # 쿼리당 5개씩 증가
                    exclude_domains=EXCLUDED_DOMAINS
                )
                
                # 결과 처리 및 품질 평가
                processed_results = _process_search_results(response.get("results", []), state)
                all_results.extend(processed_results)
                
            except Exception as e:
                logger.warning(f"검색 쿼리 '{query}' 실행 중 오류: {str(e)}")
                continue
        
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
    
    Args:
        state: RAG 상태 딕셔너리
        
    Returns:
        검색 쿼리 리스트
    """
    question = state.get("question", "")
    intent = state.get("intent", "qa")
    
    # 기본 쿼리
    base_query = f"여행자보험 {question}"
    queries = [base_query]
    
    # 의도별 특화 쿼리 추가
    if intent == "compare":
        queries.extend([
            f"여행자보험 비교 {question}",
            f"보험상품 비교 {question}",
            f"여행자보험 가격비교 {question}"
        ])
    elif intent == "recommend":
        queries.extend([
            f"여행자보험 추천 {question}",
            f"여행지별 보험 {question}",
            f"여행자보험 특약 추천 {question}"
        ])
    elif intent == "summary":
        queries.extend([
            f"여행자보험 약관 요약 {question}",
            f"보험상품 정리 {question}"
        ])
    else:  # qa
        queries.extend([
            f"여행자보험 보장내용 {question}",
            f"여행자보험 가입조건 {question}",
            f"여행자보험 보험료 {question}"
        ])
    
    return queries[:4]  # 최대 4개 쿼리만 사용

def _process_search_results(results: List[Dict], state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    검색 결과를 처리하고 품질을 평가합니다.
    
    Args:
        results: Tavily API 검색 결과
        state: RAG 상태 딕셔너리
        
    Returns:
        처리된 검색 결과 리스트
    """
    processed_results = []
    question = state.get("question", "").lower()
    
    for result in results:
        # 기본 정보 추출
        url = result.get("url", "")
        title = result.get("title", "")
        content = result.get("content", "")
        score = result.get("score", 0.0)
        
        # 여행자보험 관련성 점수 계산
        relevance_score = _calculate_relevance_score(title, content, question)
        
        # 최종 점수 계산 (Tavily 점수 70% + 관련성 점수 30%)
        final_score = (score * 0.7) + (relevance_score * 0.3)
        
        # 최소 점수 완화 (0.3 → 0.2)
        if final_score >= 0.2:
            processed_results.append({
                "source": "tavily_web",
                "url": url,
                "title": title,
                "snippet": content[:500] + "..." if len(content) > 500 else content,  # 스니펫 길이 제한
                "score_web": final_score,
                "relevance_score": relevance_score,
                "timestamp": datetime.now().isoformat()
            })
    
    return processed_results

# 도메인 가중치 함수 제거 - 단순화된 점수 계산 사용

def _calculate_relevance_score(title: str, content: str, question: str) -> float:
    """
    제목과 내용의 여행자보험 관련성을 점수화합니다.
    
    Args:
        title: 검색 결과 제목
        content: 검색 결과 내용
        question: 사용자 질문
        
    Returns:
        관련성 점수 (0.0 ~ 1.0)
    """
    text = f"{title} {content}".lower()
    question_lower = question.lower()
    
    # 여행자보험 관련 키워드 (확장)
    insurance_keywords = [
        "여행자보험", "여행보험", "해외여행보험", "해외여행자보험",
        "보험료", "보장내용", "보험금", "특약", "가입조건",
        "보험사", "손해보험", "화재보험", "생명보험", "보험",
        "의료비", "상해", "질병", "휴대품", "여행지연", "여행취소"
    ]
    
    # 여행 관련 키워드 (확장)
    travel_keywords = [
        "여행", "해외여행", "국내여행", "여행지", "관광",
        "항공", "호텔", "여행사", "여행상품", "출국", "입국",
        "해외", "외국"
    ]
    
    # 키워드 매칭 점수 계산
    insurance_score = sum(1 for keyword in insurance_keywords if keyword in text) / len(insurance_keywords)
    travel_score = sum(1 for keyword in travel_keywords if keyword in text) / len(travel_keywords)
    question_score = sum(1 for word in question_lower.split() if word in text) / max(len(question_lower.split()), 1)
    
    # 가중 평균으로 최종 점수 계산 (여행자보험 키워드에 더 높은 가중치)
    final_score = (insurance_score * 0.6) + (travel_score * 0.3) + (question_score * 0.1)
    
    # 특별 키워드 보너스
    special_bonus = 0
    if "여행자보험" in text or "여행보험" in text:
        special_bonus += 0.3
    if "특별조항" in text or "특약" in text:
        special_bonus += 0.2
    
    final_score = min(final_score + special_bonus, 1.0)
    return final_score

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
    
    return unique_results[:5]  # 상위 5개 결과만 반환

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
        
        # 30분 TTL로 캐시 저장
        redis_client.setex(
            cache_key,
            1800,  # 30분
            json.dumps(results, ensure_ascii=False)
        )
        
        logger.debug(f"검색 결과 캐시 저장: {cache_key}")
        
    except Exception as e:
        logger.warning(f"캐시 저장 중 오류: {str(e)}")

def _generate_cache_key(state: Dict[str, Any]) -> str:
    """
    상태 정보를 기반으로 캐시 키를 생성합니다.
    
    Args:
        state: RAG 상태 딕셔너리
        
    Returns:
        캐시 키 문자열
    """
    question = state.get("question", "")
    intent = state.get("intent", "qa")
    
    # 질문과 의도를 해시화하여 캐시 키 생성
    key_data = f"websearch:{intent}:{question}"
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