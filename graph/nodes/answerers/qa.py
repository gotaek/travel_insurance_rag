from typing import Dict, Any
import json
from pathlib import Path
from app.deps import get_llm
from app.langsmith_llm import get_llm_with_tracing
from graph.models import AnswerResponse

def _load_prompt(prompt_name: str) -> str:
    """프롬프트 파일 로드"""
    # 현재 작업 디렉토리 기준으로 경로 설정
    current_dir = Path(__file__).parent.parent.parent
    prompt_path = current_dir / "prompts" / f"{prompt_name}.md"
    return prompt_path.read_text(encoding="utf-8")

def _format_context(passages: list) -> str:
    """검색된 문서를 컨텍스트로 포맷팅"""
    if not passages or passages is None:
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
        structured_llm = llm.with_structured_output(AnswerResponse)
        response = structured_llm.generate_content(prompt, request_options={"timeout": 45})
        
        return {
            "conclusion": response.conclusion,
            "evidence": response.evidence,
            "caveats": response.caveats,
            "quotes": response.quotes
        }
    except Exception as e:
        # structured output 실패 시 기본 구조로 fallback
        return {
            "conclusion": "답변을 생성하는 중 오류가 발생했습니다.",
            "evidence": ["응답 파싱 오류"],
            "caveats": ["추가 확인이 필요합니다."],
            "quotes": []
        }

def qa_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    QA 에이전트: 질문에 대한 직접적인 답변 생성
    """
    question = state.get("question", "")
    passages = state.get("passages", [])
    
    # 컨텍스트 포맷팅
    context = _format_context(passages)
    
    # 프롬프트 로드
    system_prompt = _load_prompt("system_core")
    qa_prompt = _load_prompt("qa")
    
    # 최종 프롬프트 구성
    full_prompt = f"""
        {system_prompt}

        {qa_prompt}

        ## 질문
        {question}

        ## 참고 문서
        {context}

        위 문서를 참고하여 질문에 답변해주세요.
    """
    
    try:
        # LLM 호출 (타임아웃 설정)
        llm = get_llm_with_tracing()
        
        # structured output 사용
        answer = _parse_llm_response_structured(llm, full_prompt)
        
        # 출처 정보 추가 (quotes가 비어있을 때만)
        if passages and not answer.get("quotes"):
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
            "conclusion": f"질문을 확인했습니다: '{question}'",
            "evidence": ["LLM 호출 중 오류가 발생했습니다."],
            "caveats": ["추가 확인이 필요합니다.", f"오류: {str(e)}"],
            "quotes": []
        }
        return {**state, "draft_answer": fallback_answer, "final_answer": fallback_answer}