from typing import Dict, Any
import json
import logging
from app.deps import get_llm
from graph.models import ReplanResponse
from graph.config_manager import get_system_config

# 로깅 설정
logger = logging.getLogger(__name__)

def replan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    재검색을 위한 새로운 질문 생성 및 웹 검색 필요성 판단 (무한루프 방지 포함)
    """
    # 시스템 설정 가져오기
    config = get_system_config()
    
    original_question = state.get("question", "") or ""
    quality_feedback = state.get("quality_feedback", "") or ""
    replan_query = state.get("replan_query", "") or ""
    replan_count = state.get("replan_count", 0) or 0
    max_attempts = state.get("max_replan_attempts", config.get_max_replan_attempts())
    max_structured_failures = config.get_max_structured_failures()
    
    logger.info(f"재검색 시작 - 원래 질문: {original_question[:50] if original_question else 'None'}..., 재검색 횟수: {replan_count}/{max_attempts}")
    logger.debug(f"품질 피드백: {quality_feedback}")
    logger.debug(f"제안된 재검색 질문: {replan_query}")
    
    # 긴급 탈출 로직: 연속 구조화 실패 감지
    structured_failure_count = state.get("structured_failure_count", 0)
    emergency_fallback_used = state.get("emergency_fallback_used", False)
    
    # 최대 시도 횟수 체크
    if replan_count >= max_attempts:
        logger.warning(f"🚨 최대 재검색 횟수({max_attempts})에 도달하여 재검색을 중단합니다.")
        print(f"🚨 replan에서 강제 완료 - replan_count: {replan_count}, max_attempts: {max_attempts}")
        return {
            **state,
            "replan_count": replan_count + 1,
            "needs_replan": False,
            "final_answer": state.get("draft_answer", {"conclusion": "재검색 횟수 초과로 답변을 완료합니다."})
        }
    
    # 긴급 탈출: 연속 구조화 실패가 임계값에 도달한 경우
    if structured_failure_count >= max_structured_failures or emergency_fallback_used:
        logger.warning(f"🚨 연속 구조화 실패 임계값 도달({structured_failure_count}/{max_structured_failures}) - 재검색 중단")
        print(f"🚨 replan에서 긴급 탈출 - 구조화 실패: {structured_failure_count}/{max_structured_failures}")
        return {
            **state,
            "replan_count": replan_count + 1,
            "needs_replan": False,
            "final_answer": state.get("draft_answer", {"conclusion": "연속 구조화 실패로 인한 긴급 탈출"}),
            "emergency_fallback_used": True
        }
    
    # LLM을 사용하여 재검색 질문 생성
    replan_result = _generate_replan_query(original_question, quality_feedback, replan_query)
    
    # 2번째 사이클에서는 무조건 needs_web을 True로 설정
    if replan_count >= 1:
        replan_result["needs_web"] = True
        logger.info(f"🔄 2번째 사이클 이상 - 무조건 웹 검색 활성화 (재검색 횟수: {replan_count})")
    
    logger.info(f"재검색 질문 생성 완료 - 새 질문: {replan_result['new_question'][:50]}..., 웹 검색 필요: {replan_result['needs_web']}")
    logger.debug(f"재검색 근거: {replan_result.get('reasoning', 'N/A')}")
    
    return {
        **state,
        "question": replan_result["new_question"],
        "needs_web": replan_result["needs_web"],
        "replan_count": replan_count + 1,  # 재검색 횟수 증가
        "max_replan_attempts": max_attempts,  # 기존 설정 유지
        # plan은 planner가 다시 생성하도록 제거
    }

def _generate_replan_query(original_question: str, feedback: str, suggested_query: str) -> Dict[str, Any]:
    """
    LLM을 사용하여 재검색을 위한 새로운 질문 생성
    """
    prompt = f"""
다음은 여행자보험 RAG 시스템에서 답변 품질이 낮아 재검색이 필요한 상황입니다.

원래 질문: "{original_question}"
품질 피드백: "{feedback}"
제안된 재검색 질문: "{suggested_query}"

**중요**: 원래 질문의 핵심 의도와 비교 대상은 반드시 유지해야 합니다.

다음 기준으로 새로운 검색 질문을 생성해주세요:

1. **의도 유지**: 원래 질문의 핵심 의도(비교/추천/질문/요약)를 그대로 유지
2. **비교 대상 유지**: 비교 질문인 경우 모든 비교 대상 포함 (예: "A와 B 비교" → "A와 B 비교" 유지)
3. **구체성**: 더 구체적이고 명확한 질문으로 개선
4. **키워드**: 여행자보험 관련 핵심 키워드 포함
5. **범위**: 너무 넓지도 좁지도 않은 적절한 범위
6. **웹 검색 필요성**: 실시간 정보나 최신 정보가 필요한지 판단

**비교 질문의 경우**: 모든 비교 대상의 정보를 포함하여 검색하도록 질문을 개선하세요.
**단일 대상 질문의 경우**: 해당 대상에 대한 더 구체적인 정보를 요청하도록 개선하세요.

다음 정보를 제공해주세요:
- new_question: 개선된 검색 질문 (원래 의도와 비교 대상 유지)
- needs_web: 웹 검색 필요 여부 (true/false)
- reasoning: 재검색 질문 개선 근거
"""

    try:
        logger.debug("LLM을 사용한 재검색 질문 생성 시작 (structured output)")
        llm = get_llm()
        
        # structured output 사용
        structured_llm = llm.with_structured_output(ReplanResponse)
        response = structured_llm.generate_content(prompt)
        
        logger.debug(f"Structured LLM 응답: {response}")
        
        # 유효성 검증
        new_question = response.new_question
        if not new_question or new_question.strip() == "":
            logger.warning("빈 질문 생성됨, 원래 질문 사용")
            new_question = original_question
            
        needs_web = response.needs_web
        if not isinstance(needs_web, bool):
            logger.warning(f"유효하지 않은 needs_web 값: {needs_web}, 기본값 True 사용")
            needs_web = True
            
        logger.info(f"LLM 재검색 질문 생성 성공 - 새 질문: {new_question[:50]}..., 웹 검색 필요: {needs_web}")
        return {
            "new_question": new_question,
            "needs_web": needs_web,
            "reasoning": response.reasoning
        }
        
    except Exception as e:
        logger.error(f"LLM 재검색 질문 생성 실패, fallback 사용: {str(e)}")
        return _fallback_replan(original_question, suggested_query)

def _fallback_replan(original_question: str, suggested_query: str) -> Dict[str, Any]:
    """
    LLM 호출 실패 시 사용하는 간단한 재검색 질문 생성
    """
    logger.info("Fallback 재검색 질문 생성 시작")
    
    # 원래 질문의 핵심 의도 유지하면서 개선
    if suggested_query and suggested_query.strip():
        # 제안된 질문이 있으면 사용하되, 원래 질문의 핵심 의도 확인
        new_question = suggested_query
        logger.debug(f"제안된 질문 사용: {new_question[:50]}...")
    else:
        # 원래 질문을 기반으로 간단한 개선
        new_question = original_question
        logger.debug(f"원래 질문 사용: {new_question[:50]}...")
    
    # 비교 질문인 경우 모든 대상이 포함되었는지 확인
    if "비교" in original_question or "vs" in original_question.lower() or "와" in original_question:
        # 비교 질문의 경우 모든 대상이 포함되었는지 확인
        if "db" in original_question.lower() and "카카오" in original_question.lower():
            if "db" not in new_question.lower() or "카카오" not in new_question.lower():
                # 비교 대상이 누락된 경우 원래 질문 사용
                new_question = original_question
                logger.warning("비교 대상 누락 감지 - 원래 질문 사용")
    
    logger.debug(f"Fallback 질문 선택: {new_question[:50]}...")
    
    # 웹 검색 필요성 간단 판단
    web_keywords = ["최신", "현재", "실시간", "뉴스", "2024", "2025", "요즘", "지금"]
    needs_web = any(keyword in new_question.lower() for keyword in web_keywords)
    
    logger.info(f"Fallback 재검색 완료 - 새 질문: {new_question[:50]}..., 웹 검색 필요: {needs_web}")
    logger.debug(f"웹 검색 키워드 매칭: {[kw for kw in web_keywords if kw in new_question.lower()]}")
    
    return {
        "new_question": new_question,
        "needs_web": needs_web,
        "reasoning": f"Fallback 재검색: {new_question}"
    }
