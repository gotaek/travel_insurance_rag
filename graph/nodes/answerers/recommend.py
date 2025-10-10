from typing import Dict, Any
import json
import logging
import time
from app.deps import get_answerer_llm
from graph.models import RecommendResponse, EvidenceInfo, CaveatInfo
from graph.prompts.utils import get_simple_fallback_response
from graph.cache_manager import cache_manager
from .common import (
    get_system_prompt, get_prompt_cached, format_context_optimized,
    process_verify_refine_data, create_optimized_prompt, 
    handle_llm_error_optimized, log_performance
)

logger = logging.getLogger(__name__)

def _format_web_results(web_results: list) -> str:
    """웹 검색 결과를 포맷팅"""
    if not web_results:
        return "실시간 뉴스 정보가 없습니다."
    
    web_parts = []
    for i, result in enumerate(web_results[:3], 1):  # 상위 3개만 사용
        title = result.get("title", "제목 없음")
        snippet = result.get("snippet", "")[:200]  # 200자로 제한
        web_parts.append(f"[뉴스 {i}] {title}\n{snippet}\n")
    
    return "\n".join(web_parts)

def _parse_llm_response_fallback(llm, prompt: str, question: str) -> Dict[str, Any]:
    """structured output 실패 시 간단한 fallback"""
    try:
        logger.debug("Recommend 노드 fallback 파싱 시도")
        response = llm.generate_content(prompt)
        response_text = response.text
        
        return {
            "conclusion": response_text[:500] if response_text else "추천을 생성했습니다.",
            "evidence": [EvidenceInfo(text="Fallback 파싱으로 생성된 답변", source="Fallback 시스템")],
            "caveats": [CaveatInfo(text="원본 structured output이 실패하여 일반 파싱을 사용했습니다.", source="Fallback 시스템")],
            "web_quotes": [],
            "recommendations": [],
            "web_info": {}
        }
        
    except Exception as fallback_error:
        logger.error(f"Recommend 노드 fallback도 실패: {str(fallback_error)}")
        return get_simple_fallback_response(question, "Recommend")

def _parse_llm_response_structured(llm, prompt: str, emergency_fallback: bool = False) -> Dict[str, Any]:
    """LLM 응답을 structured output으로 파싱"""
    try:
        # structured output 사용 (긴급 탈출 모드 지원)
        structured_llm = llm.with_structured_output(RecommendResponse, emergency_fallback=emergency_fallback)
        response = structured_llm.generate_content(prompt)
        
        return {
            "conclusion": response.conclusion,
            "evidence": response.evidence,
            "caveats": response.caveats,
            "web_quotes": response.web_quotes,
            "recommendations": response.recommendations,
            "web_info": response.web_info
        }
    except Exception as e:
        # structured output 실패 시 기본 구조로 fallback
        error_str = str(e).lower()
        logger.error(f"Recommend 노드 structured output 실패: {str(e)}")
        
        if "quota" in error_str or "limit" in error_str or "429" in error_str or "exceeded" in error_str:
            return {
                "conclusion": "죄송합니다. 현재 API 할당량이 초과되어 답변을 생성할 수 없습니다.",
                "evidence": [EvidenceInfo(text="Gemini API 일일 할당량 초과", source="API 시스템")],
                "caveats": [
                    CaveatInfo(text="잠시 후 다시 시도해주세요.", source="API 시스템"),
                    CaveatInfo(text="API 할당량이 복구되면 정상적으로 답변을 제공할 수 있습니다.", source="API 시스템"),
                    CaveatInfo(text="오류 코드: 429 (Quota Exceeded)", source="API 시스템")
                ],
                "recommendations": [],
                "web_info": {}
            }
        elif "404" in error_str or "publisher" in error_str or "model" in error_str:
            return {
                "conclusion": "모델 설정 오류로 인해 답변을 생성할 수 없습니다.",
                "evidence": [EvidenceInfo(text="Gemini 모델 설정 오류", source="API 시스템")],
                "caveats": [
                    CaveatInfo(text="모델 이름을 확인해주세요.", source="API 시스템"),
                    CaveatInfo(text="잠시 후 다시 시도해주세요.", source="API 시스템")
                ],
                "recommendations": [],
                "web_info": {}
            }
        else:
            # structured output 실패 시 fallback 파싱 시도
            return _parse_llm_response_fallback(llm, prompt, "질문")

def recommend_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    추천 에이전트: 여행 일정, 지역, 최신 뉴스를 고려하여 맞춤 특약 추천 (최적화된 버전)
    """
    start_time = time.time()
    question = state.get("question", "")
    refined = state.get("refined", [])
    web_results = state.get("web_results", [])
    
    # 성능 로깅
    log_performance("Recommend 시작", start_time, 
                   question_length=len(question), refined_count=len(refined), 
                   web_results_count=len(web_results))
    
    # 컨텍스트 포맷팅 (최적화된 함수 사용)
    context = format_context_optimized(refined)
    web_info = _format_web_results(web_results)
    
    # 캐시된 프롬프트 로드 (모듈 레벨 캐시 사용)
    system_prompt = get_system_prompt()
    recommend_prompt = get_prompt_cached("recommend")
    
    # 최적화된 프롬프트 생성
    full_prompt = f"""{system_prompt}

{recommend_prompt}

## 질문
{question}

## 참고 문서
{context}

## 실시간 뉴스/정보
{web_info}

위 정보를 참고하여 맞춤 추천을 해주세요."""
    
    try:
        # LLM 응답 캐시 확인
        prompt_hash = cache_manager.generate_prompt_hash(full_prompt)
        cached_response = cache_manager.get_cached_llm_response(prompt_hash)
        if cached_response:
            logger.info("🔍 [Recommend] LLM 응답 캐시 히트!")
            answer = cached_response
        else:
            # Answerer 전용 LLM 사용 (Gemini 2.5 Flash)
            llm = get_answerer_llm()
            
            # 간소화된 structured output 사용
            try:
                answer = _parse_llm_response_structured(llm, full_prompt, emergency_fallback=False)
                
                # LLM 응답 캐싱
                cache_manager.cache_llm_response(prompt_hash, answer)
                logger.info("🔍 [Recommend] LLM 응답 캐시 저장 완료")
            except Exception as e:
                logger.warning(f"Structured output 실패, fallback 사용: {e}")
                answer = get_simple_fallback_response(question, "Recommend")
        
        # verify_refine 데이터 처리 (최적화된 함수 사용)
        answer = process_verify_refine_data(state, answer)
        
        # 웹 검색 결과를 web_quotes에 추가 (웹 검색 결과가 있을 때만)
        if web_results and not answer.get("web_quotes"):
            answer["web_quotes"] = [
                {
                    "text": result.get("snippet", "")[:200] + "...",
                    "source": f"웹검색_{result.get('title', '제목 없음')}_{result.get('url', 'URL 없음')}"
                }
                for result in web_results[:3]  # 상위 3개만
            ]
        
        # web_info 필드 처리
        if isinstance(answer.get("web_info"), dict):
            web_info_dict = answer["web_info"]
            answer["web_info"] = {
                "latest_news": web_info_dict.get("latest_news", ""),
                "travel_alerts": web_info_dict.get("travel_alerts", "")
            }
        elif not answer.get("web_info"):
            answer["web_info"] = {
                "latest_news": "",
                "travel_alerts": ""
            }
        
        # 성능 로깅
        log_performance("Recommend 완료", start_time, 
                       conclusion_length=len(answer.get('conclusion', '')))
        
        return {
            **state, 
            "draft_answer": answer, 
            "final_answer": answer
        }
        
    except Exception as e:
        # 최적화된 오류 처리
        logger.error(f"Recommend LLM 호출 실패: {str(e)}")
        fallback_answer = handle_llm_error_optimized(e, question, "Recommend")
        return {**state, "draft_answer": fallback_answer, "final_answer": fallback_answer}