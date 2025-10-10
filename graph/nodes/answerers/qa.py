from typing import Dict, Any
import json
import logging
import time
from app.deps import get_answerer_llm
from graph.models import AnswerResponse, EvidenceInfo, CaveatInfo
from graph.prompts.utils import get_simple_fallback_response
from graph.cache_manager import cache_manager
from .common import (
    get_system_prompt, get_prompt_cached, format_context_optimized,
    process_verify_refine_data, create_optimized_prompt, 
    handle_llm_error_optimized, log_performance
)

logger = logging.getLogger(__name__)

def _parse_llm_response_fallback(llm, prompt: str, question: str) -> Dict[str, Any]:
    """structured output 실패 시 간단한 fallback"""
    try:
        logger.debug("QA 노드 fallback 파싱 시도")
        response = llm.generate_content(prompt)
        response_text = response.text
        
        return {
            "conclusion": response_text[:500] if response_text else "답변을 생성했습니다.",
            "evidence": [EvidenceInfo(text="Fallback 파싱으로 생성된 답변", source="Fallback 시스템")],
            "caveats": [CaveatInfo(text="원본 structured output이 실패하여 일반 파싱을 사용했습니다.", source="Fallback 시스템")],
            "web_quotes": [],
            "web_info": {}
        }
        
    except Exception as fallback_error:
        logger.error(f"QA 노드 fallback도 실패: {str(fallback_error)}")
        return get_simple_fallback_response(question, "QA")

def _parse_llm_response_structured(llm, prompt: str, emergency_fallback: bool = False) -> Dict[str, Any]:
    """LLM 응답을 structured output으로 파싱"""
    try:
        # structured output 사용 (긴급 탈출 모드 지원)
        structured_llm = llm.with_structured_output(AnswerResponse, emergency_fallback=emergency_fallback)
        response = structured_llm.generate_content(prompt)
        
        return {
            "conclusion": response.conclusion,
            "evidence": response.evidence,
            "caveats": response.caveats,
            "web_quotes": [],
            "web_info": {}
        }
    except Exception as e:
        # structured output 실패 시 기본 구조로 fallback
        error_str = str(e).lower()
        logger.error(f"QA 노드 structured output 실패: {str(e)}")
        
        if "quota" in error_str or "limit" in error_str or "429" in error_str or "exceeded" in error_str:
            return {
                "conclusion": "죄송합니다. 현재 API 할당량이 초과되어 답변을 생성할 수 없습니다.",
                "evidence": [EvidenceInfo(text="Gemini API 일일 할당량 초과", source="API 시스템")],
                "caveats": [
                    CaveatInfo(text="잠시 후 다시 시도해주세요.", source="API 시스템"),
                    CaveatInfo(text="API 할당량이 복구되면 정상적으로 답변을 제공할 수 있습니다.", source="API 시스템"),
                    CaveatInfo(text="오류 코드: 429 (Quota Exceeded)", source="API 시스템")
                ],
            }
        else:
            # structured output 실패 시 fallback 파싱 시도
            return _parse_llm_response_fallback(llm, prompt, "질문")

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

def qa_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    QA 에이전트: 질문에 대한 직접적인 답변 생성 (최적화된 버전)
    """
    start_time = time.time()
    question = state.get("question", "")
    refined = state.get("refined", [])
    web_results = state.get("web_results", [])
    
    # 성능 로깅
    log_performance("QA 시작", start_time, 
                   question_length=len(question), refined_count=len(refined),
                   web_results_count=len(web_results))
    
    # 컨텍스트 포맷팅 (최적화된 함수 사용)
    context = format_context_optimized(refined)
    web_info = _format_web_results(web_results)
    
    # 디버깅 로그 추가
    logger.info(f"🔍 [QA] 컨텍스트 길이: {len(context)}자, 웹 정보 길이: {len(web_info)}자")
    logger.info(f"🔍 [QA] refined 문서 수: {len(refined)}")
    
    # 캐시된 프롬프트 로드 (모듈 레벨 캐시 사용)
    system_prompt = get_system_prompt()
    qa_prompt = get_prompt_cached("qa")
    
    logger.info(f"🔍 [QA] 시스템 프롬프트 길이: {len(system_prompt)}자, QA 프롬프트 길이: {len(qa_prompt)}자")
    
    # 웹 정보를 포함한 프롬프트 생성
    full_prompt = f"""{system_prompt}

{qa_prompt}

## 질문
{question}

## 참고 문서
{context}

## 실시간 뉴스/정보
{web_info}

위 정보를 참고하여 답변해주세요."""
    
    try:
        # LLM 응답 캐시 확인
        prompt_hash = cache_manager.generate_prompt_hash(full_prompt)
        cached_response = cache_manager.get_cached_llm_response(prompt_hash)
        if cached_response:
            logger.info("🔍 [QA] LLM 응답 캐시 히트!")
            answer = cached_response
        else:
            # Answerer 전용 LLM 사용 (Gemini 2.5 Flash)
            llm = get_answerer_llm()
            logger.info(f"🔍 [QA] LLM 초기화 완료, 프롬프트 총 길이: {len(full_prompt)}자")
            
            # 간소화된 structured output 사용
            try:
                logger.info("🔍 [QA] Structured output 시도 중...")
                answer = _parse_llm_response_structured(llm, full_prompt, emergency_fallback=False)
                logger.info(f"🔍 [QA] Structured output 성공 - 답변 길이: {len(answer.get('conclusion', ''))}자")
                
                # LLM 응답 캐싱
                cache_manager.cache_llm_response(prompt_hash, answer)
                logger.info("🔍 [QA] LLM 응답 캐시 저장 완료")
            except Exception as e:
                logger.warning(f"🔍 [QA] Structured output 실패, fallback 사용: {e}")
                answer = get_simple_fallback_response(question, "QA")
                logger.info(f"🔍 [QA] Fallback 답변 생성 완료 - 답변 길이: {len(answer.get('conclusion', ''))}자")
        
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
        log_performance("QA 완료", start_time, 
                       conclusion_length=len(answer.get('conclusion', '')))
        
        return {
            **state, 
            "draft_answer": answer, 
            "final_answer": answer
        }
        
    except Exception as e:
        # 최적화된 오류 처리
        logger.error(f"QA LLM 호출 실패: {str(e)}")
        fallback_answer = handle_llm_error_optimized(e, question, "QA")
        return {**state, "draft_answer": fallback_answer, "final_answer": fallback_answer}