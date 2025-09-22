from typing import Dict, Any
import json
from pathlib import Path
from app.deps import get_llm

def _load_prompt(prompt_name: str) -> str:
    """프롬프트 파일 로드"""
    prompt_path = Path("/app/graph/prompts") / f"{prompt_name}.md"
    return prompt_path.read_text(encoding="utf-8")

def _format_context(passages: list) -> str:
    """검색된 문서를 컨텍스트로 포맷팅"""
    if not passages:
        return "관련 문서를 찾을 수 없습니다."
    
    context_parts = []
    for i, passage in enumerate(passages[:5], 1):  # 상위 5개만 사용
        doc_id = passage.get("doc_id", "알 수 없음")
        page = passage.get("page", "알 수 없음")
        text = passage.get("text", "")[:500]  # 500자로 제한
        context_parts.append(f"[문서 {i}] {doc_id} (페이지 {page})\n{text}\n")
    
    return "\n".join(context_parts)

def _parse_llm_response(response_text: str) -> Dict[str, Any]:
    """LLM 응답을 JSON으로 파싱"""
    try:
        # JSON 부분만 추출 (```json ... ``` 형태일 수 있음)
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        else:
            json_text = response_text.strip()
        
        return json.loads(json_text)
    except (json.JSONDecodeError, ValueError) as e:
        # JSON 파싱 실패 시 기본 구조로 fallback
        return {
            "conclusion": "답변을 생성하는 중 오류가 발생했습니다.",
            "evidence": ["응답 파싱 오류"],
            "caveats": ["추가 확인이 필요합니다."],
            "quotes": []
        }

def compare_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    비교 에이전트: 보험사별 차이점을 표로 정리하여 비교
    """
    question = state.get("question", "")
    passages = state.get("passages", [])
    
    # 컨텍스트 포맷팅
    context = _format_context(passages)
    
    # 프롬프트 로드
    system_prompt = _load_prompt("system_core")
    compare_prompt = _load_prompt("compare")
    
    # 최종 프롬프트 구성
    full_prompt = f"""
{system_prompt}

{compare_prompt}

## 질문
{question}

## 참고 문서
{context}

위 문서를 참고하여 비교 분석해주세요. 반드시 JSON 형식으로 답변하세요.
"""
    
    try:
        # LLM 호출
        llm = get_llm()
        response = llm.generate_content(full_prompt)
        
        # 응답 파싱
        answer = _parse_llm_response(response.text)
        
        # 출처 정보 추가
        if passages:
            answer["quotes"] = [
                {
                    "text": p.get("text", "")[:200] + "...",
                    "source": f"{p.get('doc_id', '알 수 없음')}_페이지{p.get('page', '?')}"
                }
                for p in passages[:3]  # 상위 3개만
            ]
        
        return {**state, "draft_answer": answer}
        
    except Exception as e:
        # LLM 호출 실패 시 fallback
        fallback_answer = {
            "conclusion": f"비교 분석 중 오류가 발생했습니다: '{question}'",
            "evidence": ["LLM 호출 중 오류가 발생했습니다."],
            "caveats": ["추가 확인이 필요합니다.", f"오류: {str(e)}"],
            "quotes": []
        }
        return {**state, "draft_answer": fallback_answer}