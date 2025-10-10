from typing import Dict, Any, List
import json
import logging
from app.deps import get_reevaluate_llm
from graph.models import QualityEvaluationResponse
from graph.config_manager import get_system_config

# 로깅 설정
logger = logging.getLogger(__name__)

# 상수 정의 (기본값, 실제로는 설정에서 가져옴)
QUALITY_THRESHOLD = 0.7
MAX_REPLAN_ATTEMPTS = 2

def reevaluate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM 기반 답변 품질 평가 및 재검색 필요성 판단 (무한루프 방지 포함)
    """
    # 시스템 설정 가져오기
    config = get_system_config()
    
    question = state.get("question", "")
    answer = state.get("draft_answer", {})
    citations = state.get("citations", [])
    passages = state.get("refined", [])
    replan_count = state.get("replan_count", 0)
    max_attempts = state.get("max_replan_attempts", config.get_max_replan_attempts())
    quality_threshold = config.get_quality_threshold()
    max_structured_failures = config.get_max_structured_failures()
    is_domain_related = state.get("is_domain_related", True)  # 도메인 관련성 확인
    
    logger.info(f"🔍 [Reevaluate] 시작 - 재검색 횟수: {replan_count}/{max_attempts}")
    logger.info(f"🔍 [Reevaluate] 질문: '{question[:100]}...'")
    logger.info(f"🔍 [Reevaluate] 답변 타입: {type(answer)}, 인용 수: {len(citations)}, 패시지 수: {len(passages)}")
    logger.info(f"🔍 [Reevaluate] 도메인 관련성: {is_domain_related}")
    
    # 비도메인 질문인 경우 품질 평가를 건너뛰고 바로 통과
    if not is_domain_related:
        logger.info(f"🔍 [Reevaluate] 비도메인 질문 - 품질 평가 건너뛰고 바로 통과")
        return {
            **state,
            "quality_score": 1.0,  # 최고 점수로 설정
            "quality_feedback": "비도메인 질문으로 품질 평가를 건너뛰었습니다.",
            "needs_replan": False,  # 재검색 불필요
            "replan_query": "",
            "final_answer": answer  # 현재 답변을 최종 답변으로 설정
        }
    
    # 답변 텍스트 추출 - 다양한 답변 구조 지원
    if isinstance(answer, dict):
        # conclusion 필드가 있으면 사용 (summarize, qa, compare, recommend 노드)
        answer_text = answer.get("conclusion", "")
        # conclusion이 없으면 text 필드 사용 (기존 호환성)
        if not answer_text:
            answer_text = answer.get("text", "")
        # 둘 다 없으면 전체 답변을 문자열로 변환
        if not answer_text:
            answer_text = str(answer)
    else:
        answer_text = str(answer)
    
    # 긴급 탈출 로직: 연속 구조화 실패 감지
    structured_failure_count = state.get("structured_failure_count", 0)
    emergency_fallback_used = state.get("emergency_fallback_used", False)
    
    # 성능 최적화: 최대 재검색 횟수에 도달하면 품질 평가 없이 바로 답변 제공
    if replan_count >= max_attempts:
        logger.warning(f"🚨 [Reevaluate] 재검색 횟수가 {replan_count}회에 도달 - 품질 평가 없이 답변 완료")
        return {
            **state,
            "needs_replan": False,
            "final_answer": answer,
            "quality_feedback": f"재검색 횟수({replan_count}회) 초과로 답변을 완료합니다.",
            "replan_count": replan_count
        }
    
    # 긴급 탈출: 연속 구조화 실패가 임계값에 도달한 경우
    if structured_failure_count >= max_structured_failures or emergency_fallback_used:
        logger.warning(f"🚨 [Reevaluate] 연속 구조화 실패 임계값 도달({structured_failure_count}/{max_structured_failures}) - 긴급 탈출")
        return {
            **state,
            "needs_replan": False,
            "final_answer": answer,
            "quality_feedback": f"연속 구조화 실패({structured_failure_count}회)로 인한 긴급 탈출",
            "replan_count": replan_count,
            "emergency_fallback_used": True
        }
    
    # LLM을 사용한 품질 평가 (긴급 탈출 임계값 이전에만)
    logger.info(f"🔍 [Reevaluate] 답변 품질 평가 시작 - 질문: {question[:50]}... (재검색 횟수: {replan_count})")
    logger.debug(f"🔍 [Reevaluate] 추출된 답변 텍스트: {answer_text[:100]}..." if answer_text else "답변 텍스트가 비어있음")
    logger.debug(f"🔍 [Reevaluate] 답변 원본 타입: {type(answer)}, 내용: {str(answer)[:100]}...")
    quality_result = _evaluate_answer_quality(question, answer_text, citations, passages, quality_threshold)
    
    # 재검색 횟수 체크 및 무한루프 방지
    needs_replan = quality_result["needs_replan"] and replan_count < max_attempts
    
    # 무한루프 방지: 최대 시도 횟수에 도달하면 강제로 답변 완료 (3번째 사이클 이후)
    if replan_count >= max_attempts:
        needs_replan = False
        logger.warning(f"🚨 [Reevaluate] 최대 재검색 횟수({max_attempts})에 도달하여 답변을 완료합니다.")
        quality_result["score"] = max(quality_result["score"], 0.5)  # 최소 0.5점 보장
        print(f"🚨 reevaluate에서 강제 완료 - replan_count: {replan_count}, max_attempts: {max_attempts}")
    
    # 답변이 실제로 존재하는 경우 최소 점수 보장
    if answer_text and answer_text.strip() and quality_result["score"] < 0.3:
        logger.warning(f"🔍 [Reevaluate] 답변이 존재하지만 낮은 점수({quality_result['score']:.2f}) - 최소 점수 0.3으로 조정")
        quality_result["score"] = 0.3
        if quality_result["score"] >= quality_threshold:
            needs_replan = False
    
    # 2번째 사이클에서는 무조건 답변 제공 (관대한 평가)
    if replan_count >= 1 and answer_text and answer_text.strip():
        logger.warning(f"🔍 [Reevaluate] 2번째 사이클 이상 - 무조건 답변 제공 (재검색 횟수: {replan_count})")
        quality_result["score"] = max(quality_result["score"], 0.8)  # 높은 점수로 설정
        needs_replan = False  # 재검색 중단
        logger.info(f"🔍 [Reevaluate] 2번째 사이클 답변 제공 - 점수: {quality_result['score']:.2f}")
    
    logger.info(f"🔍 [Reevaluate] 품질 점수: {quality_result['score']:.2f}, 재검색 필요: {needs_replan}, 재검색 횟수: {replan_count}/{max_attempts}")
    
    return {
        **state,
        "quality_score": quality_result["score"],
        "quality_feedback": quality_result["feedback"],
        "needs_replan": needs_replan,
        "replan_query": quality_result["replan_query"],
        "final_answer": answer if quality_result["score"] >= quality_threshold or replan_count >= max_attempts else None
    }

def _evaluate_answer_quality(question: str, answer: str, citations: List[Dict[str, Any]], passages: List[Dict[str, Any]], quality_threshold: float = 0.7) -> Dict[str, Any]:
    """
    LLM을 사용하여 답변 품질을 평가하고 재검색 필요성을 판단
    """
    prompt = f"""
다음은 여행자보험 RAG 시스템의 질문-답변 쌍입니다. 답변의 품질을 평가하고 재검색이 필요한지 판단해주세요.

질문: "{question}"

답변: "{answer}"

인용 정보: {len(citations)}개
검색된 문서: {len(passages)}개

**평가 시 주의사항**:
- 답변이 완벽하지 않아도, 질문에 부분적으로라도 관련된 내용을 담고 있으면 점수를 주세요.
- 답변이 비어있지 않다면 기본적으로 0.5점 이상을 부여해주세요.
- 인용이 부족하거나 완전성이 떨어져도, 답변이 일정 부분 유용하면 재검색 없이 그대로 인정할 수 있습니다.

평가 기준 (각 0-1):
1. **정확성**: 질문에 어느 정도라도 정확히 답하고 있는가?
2. **완전성**: 답변이 충분히 상세하거나, 최소한 핵심은 전달되는가?
3. **관련성**: 여행자보험 도메인과 관련된 답변인가?
4. **인용 품질**: 적절한 인용이 있는가? (없어도 감점은 하되 0점은 아님)

총 점수는 0-1 사이 값으로, 0.5 이상이면 기본적으로 "수용 가능", 0.7 이상이면 "양호"로 간주합니다.

**재검색이 필요한 경우** (더 완화된 기준):
- 답변이 완전히 비어 있거나 무의미한 경우
- 답변이 질문과 전혀 무관한 경우
- 답변이 지나치게 모호하거나 오해를 불러올 정도로 불완전한 경우
- 반드시 최신 정보(예: 여행지 현황, 뉴스 등)가 필요한 질문인데 최신성이 없는 경우

출력 형식(JSON):
- score: 0.0~1.0 사이 품질 점수
- feedback: 품질 평가 상세 설명
- needs_replan: true/false
- replan_query: 재검색이 필요한 경우 새로운 검색 질문 (없으면 null)
"""

    try:
        logger.debug("LLM을 사용한 품질 평가 시작 (structured output)")
        llm = get_reevaluate_llm()  # Reevaluate 전용 LLM 사용 (Gemini 2.5 Flash-Lite)
        
        # structured output 사용
        structured_llm = llm.with_structured_output(QualityEvaluationResponse)
        response = structured_llm.generate_content(prompt)
        
        logger.debug(f"Structured LLM 응답: {response}")
        
        # 유효성 검증
        score = response.score
        if not isinstance(score, (int, float)) or score < 0 or score > 1:
            logger.warning(f"유효하지 않은 점수: {score}, 기본값 0.5 사용")
            score = 0.5
            
        needs_replan = response.needs_replan
        if not isinstance(needs_replan, bool):
            needs_replan = score < quality_threshold
            logger.warning(f"유효하지 않은 needs_replan: {needs_replan}, 점수 기반으로 설정")
            
        replan_query = response.replan_query
        if replan_query == "null" or replan_query is None:
            replan_query = ""
            
        return {
            "score": float(score),
            "feedback": response.feedback,
            "needs_replan": needs_replan,
            "replan_query": replan_query
        }
        
    except Exception as e:
        logger.error(f"LLM 품질 평가 실패, fallback 사용: {str(e)}")
        return _fallback_evaluate(question, answer, citations, passages, quality_threshold)

def _fallback_evaluate(question: str, answer: str, citations: List[Dict[str, Any]], passages: List[Dict[str, Any]], quality_threshold: float = 0.7) -> Dict[str, Any]:
    """
    LLM 호출 실패 시 사용하는 간단한 휴리스틱 평가
    """
    # 답변이 실제로 비어있는지 체크
    if not answer or answer.strip() == "":
        logger.warning("답변이 비어있음 - Fallback 평가에서 0점 처리")
        return {
            "score": 0.0,
            "feedback": "답변이 비어있어 정확성, 완전성, 관련성 모두 0점입니다.",
            "needs_replan": True,
            "replan_query": question
        }
    
    # 기본 점수 계산 (답변이 있으면 최소 0.3점)
    score = 0.3
    
    # 답변 길이 체크
    if len(answer) > 50:
        score += 0.2
    elif len(answer) > 20:
        score += 0.1
    
    # 인용 정보 체크
    if len(citations) > 0:
        score += 0.2
    
    # 검색된 문서 체크
    if len(passages) > 0:
        score += 0.1
    
    # 키워드 매칭 체크
    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())
    if len(question_words.intersection(answer_words)) > 0:
        score += 0.2
    
    # 점수 제한
    score = min(score, 1.0)
    
    needs_replan = score < quality_threshold
    replan_query = question if needs_replan else ""
    
    logger.info(f"Fallback 평가 완료 - 점수: {score:.2f}, 재검색 필요: {needs_replan}")
    
    return {
        "score": score,
        "feedback": f"Fallback 평가: 답변길이({len(answer)}), 인용({len(citations)}), 문서({len(passages)})",
        "needs_replan": needs_replan,
        "replan_query": replan_query
    }
