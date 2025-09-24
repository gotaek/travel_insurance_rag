from typing import Dict, Any
import json
from app.deps import get_llm

def reevaluate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM 기반 답변 품질 평가 및 재검색 필요성 판단
    """
    question = state.get("question", "")
    answer = state.get("draft_answer", {})
    citations = state.get("citations", [])
    passages = state.get("refined", [])
    
    # 답변 텍스트 추출
    answer_text = answer.get("text", "") if isinstance(answer, dict) else str(answer)
    
    # LLM을 사용한 품질 평가
    quality_result = _evaluate_answer_quality(question, answer_text, citations, passages)
    
    return {
        **state,
        "quality_score": quality_result["score"],
        "quality_feedback": quality_result["feedback"],
        "needs_replan": quality_result["needs_replan"],
        "replan_query": quality_result["replan_query"],
        "final_answer": answer if quality_result["score"] >= 0.7 else None
    }

def _evaluate_answer_quality(question: str, answer: str, citations: list, passages: list) -> Dict[str, Any]:
    """
    LLM을 사용하여 답변 품질을 평가하고 재검색 필요성을 판단
    """
    prompt = f"""
다음은 여행자보험 RAG 시스템의 질문-답변 쌍입니다. 답변의 품질을 평가하고 재검색이 필요한지 판단해주세요.

질문: "{question}"

답변: "{answer}"

인용 정보: {len(citations)}개
검색된 문서: {len(passages)}개

다음 기준으로 평가해주세요:

1. **정확성 (0-1)**: 답변이 질문에 정확히 답하고 있는가?
2. **완전성 (0-1)**: 답변이 충분히 상세하고 완전한가?
3. **관련성 (0-1)**: 답변이 여행자보험과 관련된 내용인가?
4. **인용 품질 (0-1)**: 적절한 인용이 포함되어 있는가?

총 점수는 0-1 사이의 값으로, 0.7 이상이면 양호한 답변으로 간주합니다.

또한 다음 경우에 재검색이 필요합니다:
- 답변이 불완전하거나 부정확한 경우
- 더 최신 정보가 필요한 경우
- 추가 문서 검색이 필요한 경우

반드시 다음 JSON 형식으로만 답변하세요:
{{
    "score": 0.0-1.0,
    "feedback": "품질 평가 상세 설명",
    "needs_replan": true|false,
    "replan_query": "재검색이 필요한 경우 새로운 검색 질문 (없으면 null)"
}}
"""

    try:
        llm = get_llm()
        response = llm.generate_content(prompt, request_options={"timeout": 30})
        
        # JSON 파싱
        response_text = response.text.strip()
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        else:
            json_text = response_text
        
        result = json.loads(json_text)
        
        # 유효성 검증
        score = result.get("score", 0.5)
        if not isinstance(score, (int, float)) or score < 0 or score > 1:
            score = 0.5
            
        needs_replan = result.get("needs_replan", False)
        if not isinstance(needs_replan, bool):
            needs_replan = score < 0.7
            
        replan_query = result.get("replan_query")
        if replan_query == "null" or replan_query is None:
            replan_query = ""
            
        return {
            "score": float(score),
            "feedback": result.get("feedback", "품질 평가 완료"),
            "needs_replan": needs_replan,
            "replan_query": replan_query
        }
        
    except Exception as e:
        print(f"⚠️ 품질 평가 실패, fallback 사용: {str(e)}")
        return _fallback_evaluate(question, answer, citations, passages)

def _fallback_evaluate(question: str, answer: str, citations: list, passages: list) -> Dict[str, Any]:
    """
    LLM 호출 실패 시 사용하는 간단한 휴리스틱 평가
    """
    # 기본 점수 계산
    score = 0.5
    
    # 답변 길이 체크
    if len(answer) > 50:
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
        score += 0.1
    
    # 점수 제한
    score = min(score, 1.0)
    
    needs_replan = score < 0.7
    replan_query = question if needs_replan else ""
    
    return {
        "score": score,
        "feedback": f"Fallback 평가: 답변길이({len(answer)}), 인용({len(citations)}), 문서({len(passages)})",
        "needs_replan": needs_replan,
        "replan_query": replan_query
    }
