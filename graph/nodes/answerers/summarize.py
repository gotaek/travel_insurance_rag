from typing import Dict, Any
import json
from pathlib import Path
from app.deps import get_llm
from graph.models import AnswerResponse

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

def _parse_llm_response_structured(llm, prompt: str, emergency_fallback: bool = False) -> Dict[str, Any]:
    """LLM 응답을 structured output으로 파싱"""
    try:
        print(f"🔍 [Summarize] structured output 시작")
        print(f"🔍 [Summarize] 프롬프트 길이: {len(prompt)}자")
        print(f"🔍 [Summarize] 프롬프트 미리보기: {prompt[:200]}...")
        print(f"🔍 [Summarize] 긴급 탈출 모드: {emergency_fallback}")
        
        # structured output 사용 (긴급 탈출 모드 지원)
        structured_llm = llm.with_structured_output(AnswerResponse, emergency_fallback=emergency_fallback)
        print(f"🔍 [Summarize] structured_llm 생성 완료")
        
        response = structured_llm.generate_content(prompt)
        print(f"🔍 [Summarize] LLM 응답 수신 완료")
        print(f"🔍 [Summarize] 응답 객체 타입: {type(response)}")
        print(f"🔍 [Summarize] 응답 객체 속성들: {dir(response)}")
        
        # 각 필드별로 상세 로그
        conclusion = response.conclusion
        evidence = response.evidence
        caveats = response.caveats
        quotes = response.quotes
        
        print(f"🔍 [Summarize] conclusion: '{conclusion}' (타입: {type(conclusion)}, 길이: {len(str(conclusion))})")
        print(f"🔍 [Summarize] evidence: {evidence} (타입: {type(evidence)}, 리스트 길이: {len(evidence) if evidence else 0})")
        print(f"🔍 [Summarize] caveats: {caveats} (타입: {type(caveats)}, 길이: {len(caveats) if caveats else 0})")
        print(f"🔍 [Summarize] quotes: {quotes} (타입: {type(quotes)}, 길이: {len(quotes) if quotes else 0})")
        
        result = {
            "conclusion": conclusion,
            "evidence": evidence,
            "caveats": caveats,
            "quotes": quotes
        }
        
        print(f"🔍 [Summarize] 최종 결과: {result}")
        print(f"🔍 [Summarize] 최종 결과 타입: {type(result)}")
        
        return result
    except Exception as e:
        # structured output 실패 시 기본 구조로 fallback
        error_str = str(e).lower()
        print(f"❌ Summarize 노드 structured output 실패: {str(e)}")
        
        if "quota" in error_str or "limit" in error_str or "429" in error_str or "exceeded" in error_str:
            return {
                "conclusion": "죄송합니다. 현재 API 할당량이 초과되어 답변을 생성할 수 없습니다.",
                "evidence": ["Gemini API 일일 할당량 초과"],
                "caveats": [
                    "잠시 후 다시 시도해주세요.",
                    "API 할당량이 복구되면 정상적으로 답변을 제공할 수 있습니다.",
                    "오류 코드: 429 (Quota Exceeded)"
                ],
                "quotes": []
            }
        elif "404" in error_str or "publisher" in error_str or "model" in error_str:
            return {
                "conclusion": "모델 설정 오류로 인해 답변을 생성할 수 없습니다.",
                "evidence": ["Gemini 모델 설정 오류"],
                "caveats": [
                    "모델 이름을 확인해주세요.",
                    "잠시 후 다시 시도해주세요."
                ],
                "quotes": []
            }
        else:
            return {
                "conclusion": "답변을 생성하는 중 오류가 발생했습니다.",
                "evidence": [f"응답 파싱 오류: {str(e)[:100]}"],
                "caveats": [f"상세 오류: {str(e)}", "추가 확인이 필요합니다."],
                "quotes": []
            }

def summarize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    요약 에이전트: 약관/문서를 쉽게 이해할 수 있도록 요약
    """
    question = state.get("question", "")
    passages = state.get("passages", [])
    
    # 긴급 탈출 로직: 연속 구조화 실패 감지
    structured_failure_count = state.get("structured_failure_count", 0)
    max_structured_failures = state.get("max_structured_failures", 2)
    emergency_fallback_used = state.get("emergency_fallback_used", False)
    
    # 컨텍스트 포맷팅
    context = _format_context(passages)
    
    # 프롬프트 로드
    system_prompt = _load_prompt("system_core")
    summarize_prompt = _load_prompt("summarize")
    
    # 최종 프롬프트 구성
    full_prompt = f"""
{system_prompt}

{summarize_prompt}

## 질문
{question}

## 참고 문서
{context}

위 문서를 참고하여 요약해주세요.
"""
    
    try:
        # LLM 호출
        llm = get_llm()
        print(f"🔍 [Summarize Node] LLM 객체 획득 완료")
        
        # 긴급 탈출 모드 결정
        use_emergency_fallback = (structured_failure_count >= max_structured_failures) or emergency_fallback_used
        
        if use_emergency_fallback:
            print(f"🚨 [Summarize Node] 긴급 탈출 모드 활성화 - 구조화 실패 횟수: {structured_failure_count}/{max_structured_failures}")
        
        # structured output 사용 (긴급 탈출 모드 지원)
        answer = _parse_llm_response_structured(llm, full_prompt, emergency_fallback=use_emergency_fallback)
        print(f"🔍 [Summarize Node] structured output 완료")
        print(f"🔍 [Summarize Node] 받은 answer: {answer}")
        print(f"🔍 [Summarize Node] answer 타입: {type(answer)}")
        print(f"🔍 [Summarize Node] answer keys: {answer.keys() if isinstance(answer, dict) else 'Not a dict'}")
        
        # 구조화 실패 감지 및 카운터 업데이트
        is_empty_result = (
            not answer.get("conclusion") or 
            answer.get("conclusion", "").strip() == "" or
            answer.get("conclusion", "").strip() == "답변을 생성할 수 없습니다."
        )
        
        if is_empty_result and not use_emergency_fallback:
            # 구조화 실패 카운터 증가
            new_failure_count = structured_failure_count + 1
            print(f"⚠️ [Summarize Node] 구조화 실패 감지 - 카운터: {new_failure_count}/{max_structured_failures}")
            
            # 연속 실패가 임계값에 도달하면 긴급 탈출 모드로 재시도
            if new_failure_count >= max_structured_failures:
                print(f"🚨 [Summarize Node] 연속 구조화 실패 임계값 도달 - 긴급 탈출 모드로 재시도")
                answer = _parse_llm_response_structured(llm, full_prompt, emergency_fallback=True)
                return {
                    **state, 
                    "draft_answer": answer, 
                    "final_answer": answer,
                    "structured_failure_count": new_failure_count,
                    "emergency_fallback_used": True
                }
            else:
                return {
                    **state, 
                    "draft_answer": answer, 
                    "final_answer": answer,
                    "structured_failure_count": new_failure_count
                }
        
        # 각 필드별 상세 검사
        for key, value in answer.items():
            print(f"🔍 [Summarize Node] {key}: '{value}' (타입: {type(value)}, 길이: {len(str(value)) if value else 0})")
        
        # 출처 정보 추가
        if passages:
            print(f"🔍 [Summarize Node] 출처 정보 추가 시작 - passages 수: {len(passages)}")
            answer["quotes"] = [
                {
                    "text": p.get("text", "")[:200] + "...",
                    "source": f"{p.get('doc_id', '알 수 없음')}_{p.get('doc_name', '문서')}_페이지{p.get('page', '?')}"
                }
                for p in passages[:3]  # 상위 3개만
            ]
            print(f"🔍 [Summarize Node] 출처 정보 추가 완료 - quotes 수: {len(answer.get('quotes', []))}")
        
        # 성공 시 구조화 실패 카운터 리셋
        final_result = {
            **state, 
            "draft_answer": answer, 
            "final_answer": answer,
            "structured_failure_count": 0,
            "emergency_fallback_used": False
        }
        print(f"🔍 [Summarize Node] 최종 결과 생성 완료")
        print(f"🔍 [Summarize Node] final_result keys: {final_result.keys()}")
        print(f"🔍 [Summarize Node] draft_answer: {final_result.get('draft_answer')}")
        print(f"🔍 [Summarize Node] final_answer: {final_result.get('final_answer')}")
        
        return final_result
        
    except Exception as e:
        # LLM 호출 실패 시 fallback
        fallback_answer = {
            "conclusion": f"요약을 생성하는 중 오류가 발생했습니다: '{question}'",
            "evidence": ["LLM 호출 중 오류가 발생했습니다."],
            "caveats": ["추가 확인이 필요합니다.", f"오류: {str(e)}"],
            "quotes": []
        }
        return {**state, "draft_answer": fallback_answer, "final_answer": fallback_answer}