from typing import Dict, Any
import json
from pathlib import Path
from app.deps import get_llm
from graph.models import RecommendResponse

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

def _parse_llm_response_fallback(llm, prompt: str) -> Dict[str, Any]:
    """structured output 실패 시 일반 LLM 호출로 fallback"""
    try:
        print("🔄 Recommend 노드 fallback 파싱 시도...")
        response = llm.generate_content(prompt, request_options={"timeout": 45})
        response_text = response.text
        
        # JSON 부분 추출 시도
        import json
        import re
        
        # JSON 패턴 찾기
        json_pattern = r'\{.*\}'
        json_match = re.search(json_pattern, response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group()
            try:
                parsed = json.loads(json_str)
                return {
                    "conclusion": parsed.get("conclusion", "추천을 생성했습니다."),
                    "evidence": parsed.get("evidence", []),
                    "caveats": parsed.get("caveats", []),
                    "quotes": parsed.get("quotes", []),
                    "recommendations": parsed.get("recommendations", []),
                    "web_info": parsed.get("web_info", {})
                }
            except json.JSONDecodeError:
                pass
        
        # JSON 파싱 실패 시 텍스트에서 정보 추출
        return {
            "conclusion": response_text[:500] if response_text else "추천을 생성했습니다.",
            "evidence": ["Fallback 파싱으로 생성된 답변"],
            "caveats": ["원본 structured output이 실패하여 일반 파싱을 사용했습니다."],
            "quotes": [],
            "recommendations": [],
            "web_info": {}
        }
        
    except Exception as fallback_error:
        print(f"❌ Recommend 노드 fallback도 실패: {str(fallback_error)}")
        return {
            "conclusion": "답변을 생성하는 중 오류가 발생했습니다.",
            "evidence": [f"Fallback 파싱도 실패: {str(fallback_error)[:100]}"],
            "caveats": [f"상세 오류: {str(fallback_error)}", "추가 확인이 필요합니다."],
            "quotes": [],
            "recommendations": [],
            "web_info": {}
        }

def _parse_llm_response_structured(llm, prompt: str) -> Dict[str, Any]:
    """LLM 응답을 structured output으로 파싱"""
    try:
        # structured output 사용
        structured_llm = llm.with_structured_output(RecommendResponse)
        response = structured_llm.generate_content(prompt, request_options={"timeout": 45})
        
        return {
            "conclusion": response.conclusion,
            "evidence": response.evidence,
            "caveats": response.caveats,
            "quotes": response.quotes,
            "recommendations": response.recommendations,
            "web_info": response.web_info
        }
    except Exception as e:
        # structured output 실패 시 기본 구조로 fallback
        error_str = str(e).lower()
        print(f"❌ Recommend 노드 structured output 실패: {str(e)}")
        
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
                "recommendations": [],
                "web_info": {}
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
                "recommendations": [],
                "web_info": {}
            }
        else:
            # structured output 실패 시 fallback 파싱 시도
            return _parse_llm_response_fallback(llm, prompt)

def recommend_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    추천 에이전트: 여행 일정, 지역, 최신 뉴스를 고려하여 맞춤 특약 추천
    """
    question = state.get("question", "")
    passages = state.get("passages", [])
    web_results = state.get("web_results", [])
    
    # 컨텍스트 포맷팅
    context = _format_context(passages)
    web_info = _format_web_results(web_results)
    
    # 프롬프트 로드
    system_prompt = _load_prompt("system_core")
    recommend_prompt = _load_prompt("recommend")
    
    # 최종 프롬프트 구성
    full_prompt = f"""
        {system_prompt}

        {recommend_prompt}

        ## 질문
        {question}

        ## 참고 문서
        {context}

        ## 실시간 뉴스/정보
        {web_info}

        위 정보를 참고하여 맞춤 추천을 해주세요.
        """
    
    try:
        # LLM 호출
        llm = get_llm()
        
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
            "conclusion": f"추천 생성 중 오류가 발생했습니다: '{question}'",
            "evidence": ["LLM 호출 중 오류가 발생했습니다."],
            "caveats": ["추가 확인이 필요합니다.", f"오류: {str(e)}"],
            "quotes": [],
            "recommendations": [],
            "web_info": {}
        }
        return {**state, "draft_answer": fallback_answer, "final_answer": fallback_answer}