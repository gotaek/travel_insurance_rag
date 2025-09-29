from typing import Dict, Any
import json
import logging
import time
from app.deps import get_answerer_llm
from graph.models import CompareResponse, EvidenceInfo, CaveatInfo
from graph.prompts.utils import get_simple_fallback_response
from .common import (
    get_system_prompt, get_prompt_cached, format_context_optimized,
    process_verify_refine_data, create_optimized_prompt, 
    handle_llm_error_optimized, log_performance
)

logger = logging.getLogger(__name__)

def _parse_llm_response_structured(llm, prompt: str, emergency_fallback: bool = False) -> Dict[str, Any]:
    """LLM 응답을 structured output으로 파싱"""
    try:
        # structured output 사용 (긴급 탈출 모드 지원)
        structured_llm = llm.with_structured_output(CompareResponse, emergency_fallback=emergency_fallback)
        response = structured_llm.generate_content(prompt)
        
        return {
            "conclusion": response.conclusion,
            "evidence": response.evidence,
            "caveats": response.caveats,
            "web_quotes": [],
            "web_info": {},
            "comparison_table": {
                "headers": response.comparison_table.headers,
                "rows": response.comparison_table.rows
            }
        }
    except Exception as e:
        # structured output 실패 시 기본 구조로 fallback
        error_str = str(e).lower()
        logger.error(f"Compare 노드 structured output 실패: {str(e)}")
        
        if "quota" in error_str or "limit" in error_str or "429" in error_str or "exceeded" in error_str:
            return {
                "conclusion": "죄송합니다. 현재 API 할당량이 초과되어 답변을 생성할 수 없습니다.",
                "evidence": [EvidenceInfo(text="Gemini API 일일 할당량 초과", source="API 시스템")],
                "caveats": [
                    CaveatInfo(text="잠시 후 다시 시도해주세요.", source="API 시스템"),
                    CaveatInfo(text="API 할당량이 복구되면 정상적으로 답변을 제공할 수 있습니다.", source="API 시스템"),
                    CaveatInfo(text="오류 코드: 429 (Quota Exceeded)", source="API 시스템")
                ],
                "web_quotes": [],
                "web_info": {},
                "comparison_table": {
                    "headers": ["항목", "비교 결과"],
                    "rows": [["API 할당량 초과", "서비스 일시 중단 (429)"]]
                }
            }
        elif "404" in error_str or "publisher" in error_str or "model" in error_str:
            return {
                "conclusion": "모델 설정 오류로 인해 답변을 생성할 수 없습니다.",
                "evidence": [EvidenceInfo(text="Gemini 모델 설정 오류", source="API 시스템")],
                "caveats": [
                    CaveatInfo(text="모델 이름을 확인해주세요.", source="API 시스템"),
                    CaveatInfo(text="잠시 후 다시 시도해주세요.", source="API 시스템")
                ],
                "web_quotes": [],
                "web_info": {},
                "comparison_table": {
                    "headers": ["항목", "비교 결과"],
                    "rows": [["모델 오류", "설정 확인 필요 (404)"]]
                }
            }
        else:
            return {
                "conclusion": "답변을 생성하는 중 오류가 발생했습니다.",
                "evidence": [EvidenceInfo(text=f"응답 파싱 오류: {str(e)[:100]}", source="시스템 오류")],
                "caveats": [
                    CaveatInfo(text=f"상세 오류: {str(e)}", source="시스템 오류"),
                    CaveatInfo(text="추가 확인이 필요합니다.", source="시스템 오류")
                ],
                "web_quotes": [],
                "web_info": {},
                "comparison_table": {
                    "headers": ["항목", "비교 결과"],
                    "rows": [["오류", f"파싱 실패: {str(e)[:50]}"]]
                }
            }

def compare_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    비교 에이전트: 보험사별 차이점을 표로 정리하여 비교 (최적화된 버전)
    """
    start_time = time.time()
    question = state.get("question", "")
    refined = state.get("refined", [])
    
    # 성능 로깅
    log_performance("Compare 시작", start_time, 
                   question_length=len(question), refined_count=len(refined))
    
    # 컨텍스트 포맷팅 (최적화된 함수 사용)
    context = format_context_optimized(refined)
    
    # 캐시된 프롬프트 로드 (모듈 레벨 캐시 사용)
    system_prompt = get_system_prompt()
    compare_prompt = get_prompt_cached("compare")
    
    # 최적화된 프롬프트 생성
    full_prompt = create_optimized_prompt(system_prompt, compare_prompt, question, context)
    
    try:
        # Answerer 전용 LLM 사용 (Gemini 2.5 Flash)
        llm = get_answerer_llm()
        
        # 간소화된 structured output 사용
        try:
            answer = _parse_llm_response_structured(llm, full_prompt, emergency_fallback=False)
        except Exception as e:
            logger.warning(f"Structured output 실패, fallback 사용: {e}")
            answer = get_simple_fallback_response(question, "Compare")
        
        # verify_refine 데이터 처리 (최적화된 함수 사용)
        answer = process_verify_refine_data(state, answer)
        
        # web_info 기본값 설정
        if not answer.get("web_info"):
            answer["web_info"] = {
                "latest_news": "",
                "travel_alerts": ""
            }
        
        # 성능 로깅
        log_performance("Compare 완료", start_time, 
                       conclusion_length=len(answer.get('conclusion', '')))
        
        return {
            **state, 
            "draft_answer": answer, 
            "final_answer": answer
        }
        
    except Exception as e:
        # 최적화된 오류 처리
        logger.error(f"Compare LLM 호출 실패: {str(e)}")
        fallback_answer = handle_llm_error_optimized(e, question, "Compare")
        return {**state, "draft_answer": fallback_answer, "final_answer": fallback_answer}