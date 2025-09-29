from typing import Dict, Any
import json
import logging
import time
from app.deps import get_answerer_llm
from graph.models import AnswerResponse, EvidenceInfo, CaveatInfo
from graph.prompts.utils import get_simple_fallback_response
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
            return _parse_llm_response_fallback(llm, prompt)

def qa_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    QA 에이전트: 질문에 대한 직접적인 답변 생성 (최적화된 버전)
    """
    start_time = time.time()
    question = state.get("question", "")
    refined = state.get("refined", [])
    
    # 성능 로깅
    log_performance("QA 시작", start_time, 
                   question_length=len(question), refined_count=len(refined))
    
    # 컨텍스트 포맷팅 (최적화된 함수 사용)
    context = format_context_optimized(refined)
    
    # 캐시된 프롬프트 로드 (모듈 레벨 캐시 사용)
    system_prompt = get_system_prompt()
    qa_prompt = get_prompt_cached("qa")
    
    # 최적화된 프롬프트 생성
    full_prompt = create_optimized_prompt(system_prompt, qa_prompt, question, context)
    
    try:
        # Answerer 전용 LLM 사용 (Gemini 2.5 Flash)
        llm = get_answerer_llm()
        
        # 간소화된 structured output 사용
        try:
            answer = _parse_llm_response_structured(llm, full_prompt, emergency_fallback=False)
        except Exception as e:
            logger.warning(f"Structured output 실패, fallback 사용: {e}")
            answer = get_simple_fallback_response(question, "QA")
        
        # verify_refine 데이터 처리 (최적화된 함수 사용)
        answer = process_verify_refine_data(state, answer)
        
        # web_info 기본값 설정
        if not answer.get("web_info"):
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