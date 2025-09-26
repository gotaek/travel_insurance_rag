from typing import Dict, Any
import json
from pathlib import Path
from app.deps import get_llm
from graph.models import CompareResponse

def _load_prompt(prompt_name: str) -> str:
    """프롬프트 파일 로드"""
    # 현재 작업 디렉토리 기준으로 경로 설정
    current_dir = Path(__file__).parent.parent.parent
    prompt_path = current_dir / "prompts" / f"{prompt_name}.md"
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

def _parse_llm_response_structured(llm, prompt: str) -> Dict[str, Any]:
    """LLM 응답을 structured output으로 파싱"""
    try:
        # structured output 사용
        structured_llm = llm.with_structured_output(CompareResponse)
        response = structured_llm.generate_content(prompt, request_options={"timeout": 45})
        
        return {
            "conclusion": response.conclusion,
            "evidence": response.evidence,
            "caveats": response.caveats,
            "quotes": response.quotes,
            "comparison_table": response.comparison_table
        }
    except Exception as e:
        # structured output 실패 시 기본 구조로 fallback
        error_str = str(e).lower()
        print(f"❌ Compare 노드 structured output 실패: {str(e)}")
        
        if "quota" in error_str or "limit" in error_str or "429" in error_str or "exceeded" in error_str:
            return {
                "conclusion": "죄송합니다. 현재 API 할당량이 초과되어 답변을 생성할 수 없습니다.",
                "evidence": ["Gemini API 일일 할당량 초과"],
                "caveats": [
                    "잠시 후 다시 시도해주세요.",
                    "API 할당량이 복구되면 정상적으로 답변을 제공할 수 있습니다.",
                    "오류 코드: 429 (Quota Exceeded)"
                ],
                "quotes": [],
                "comparison_table": {
                    "headers": ["항목", "비교 결과"],
                    "rows": [["API 할당량 초과", "서비스 일시 중단 (429)"]]
                }
            }
        elif "404" in error_str or "publisher" in error_str or "model" in error_str:
            return {
                "conclusion": "모델 설정 오류로 인해 답변을 생성할 수 없습니다.",
                "evidence": ["Gemini 모델 설정 오류"],
                "caveats": [
                    "모델 이름을 확인해주세요.",
                    "잠시 후 다시 시도해주세요."
                ],
                "quotes": [],
                "comparison_table": {
                    "headers": ["항목", "비교 결과"],
                    "rows": [["모델 오류", "설정 확인 필요 (404)"]]
                }
            }
        else:
            return {
                "conclusion": "답변을 생성하는 중 오류가 발생했습니다.",
                "evidence": [f"응답 파싱 오류: {str(e)[:100]}"],
                "caveats": [f"상세 오류: {str(e)}", "추가 확인이 필요합니다."],
                "quotes": [],
                "comparison_table": {
                    "headers": ["항목", "비교 결과"],
                    "rows": [["오류", f"파싱 실패: {str(e)[:50]}"]]
                }
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

위 문서를 참고하여 비교 분석해주세요.
"""
    
    try:
        # LLM 호출
        llm = get_llm()
        
        # structured output 사용
        answer = _parse_llm_response_structured(llm, full_prompt)
        
        # 출처 정보 추가
        if passages:
            answer["quotes"] = [
                {
                    "text": p.get("text", "")[:200] + "...",
                    "source": f"{p.get('doc_id', '알 수 없음')}_페이지{p.get('page', '?')}"
                }
                for p in passages[:3]  # 상위 3개만
            ]
        
        return {**state, "draft_answer": answer, "final_answer": answer}
        
    except Exception as e:
        # LLM 호출 실패 시 fallback
        fallback_answer = {
            "conclusion": f"비교 분석 중 오류가 발생했습니다: '{question}'",
            "evidence": ["LLM 호출 중 오류가 발생했습니다."],
            "caveats": ["추가 확인이 필요합니다.", f"오류: {str(e)}"],
            "quotes": [],
            "comparison_table": {
                "headers": ["항목", "비교 결과"],
                "rows": [["오류", "LLM 호출 실패"]]
            }
        }
        return {**state, "draft_answer": fallback_answer, "final_answer": fallback_answer}