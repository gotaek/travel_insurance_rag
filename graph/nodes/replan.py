from typing import Dict, Any
import json
from app.deps import get_llm

def replan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    재검색을 위한 새로운 질문 생성 및 웹 검색 필요성 판단
    """
    original_question = state.get("question", "")
    quality_feedback = state.get("quality_feedback", "")
    replan_query = state.get("replan_query", "")
    
    # LLM을 사용하여 재검색 질문 생성
    replan_result = _generate_replan_query(original_question, quality_feedback, replan_query)
    
    return {
        **state,
        "question": replan_result["new_question"],
        "needs_web": replan_result["needs_web"],
        "plan": ["replan", "websearch", "search", "rank_filter", "verify_refine", "answer:qa"]
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

다음 기준으로 새로운 검색 질문을 생성해주세요:

1. **구체성**: 더 구체적이고 명확한 질문으로 개선
2. **키워드**: 여행자보험 관련 핵심 키워드 포함
3. **범위**: 너무 넓지도 좁지도 않은 적절한 범위
4. **웹 검색 필요성**: 실시간 정보나 최신 정보가 필요한지 판단

반드시 다음 JSON 형식으로만 답변하세요:
{{
    "new_question": "개선된 검색 질문",
    "needs_web": true|false,
    "reasoning": "재검색 질문 개선 근거"
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
        new_question = result.get("new_question", original_question)
        if not new_question or new_question.strip() == "":
            new_question = original_question
            
        needs_web = result.get("needs_web", True)
        if not isinstance(needs_web, bool):
            needs_web = True
            
        return {
            "new_question": new_question,
            "needs_web": needs_web,
            "reasoning": result.get("reasoning", "재검색 질문 생성")
        }
        
    except Exception as e:
        print(f"⚠️ 재검색 질문 생성 실패, fallback 사용: {str(e)}")
        return _fallback_replan(original_question, suggested_query)

def _fallback_replan(original_question: str, suggested_query: str) -> Dict[str, Any]:
    """
    LLM 호출 실패 시 사용하는 간단한 재검색 질문 생성
    """
    # 제안된 질문이 있으면 사용, 없으면 원래 질문 사용
    new_question = suggested_query if suggested_query and suggested_query.strip() else original_question
    
    # 웹 검색 필요성 간단 판단
    needs_web = any(keyword in new_question.lower() for keyword in [
        "최신", "현재", "실시간", "뉴스", "2024", "2025", "요즘", "지금"
    ])
    
    return {
        "new_question": new_question,
        "needs_web": needs_web,
        "reasoning": f"Fallback 재검색: {new_question}"
    }
