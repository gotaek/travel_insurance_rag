from typing import Dict, Any
import json
from pathlib import Path
from app.deps import get_llm
from graph.models import RecommendResponse, EvidenceInfo, CaveatInfo

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
        response = llm.generate_content(prompt)
        response_text = response.text
        
        # 간단한 텍스트 기반 fallback
        return {
            "conclusion": response_text[:500] if response_text else "추천을 생성했습니다.",
            "evidence": [EvidenceInfo(text="Fallback 파싱으로 생성된 답변", source="Fallback 시스템")],
            "caveats": [CaveatInfo(text="원본 structured output이 실패하여 일반 파싱을 사용했습니다.", source="Fallback 시스템")],
            "quotes": [],
            "web_quotes": [],
            "recommendations": [],
            "web_info": {}
        }
        
    except Exception as fallback_error:
        print(f"❌ Recommend 노드 fallback도 실패: {str(fallback_error)}")
        return {
            "conclusion": "답변을 생성하는 중 오류가 발생했습니다.",
            "evidence": [EvidenceInfo(text=f"Fallback 파싱도 실패: {str(fallback_error)[:100]}", source="Fallback 시스템")],
            "caveats": [
                CaveatInfo(text=f"상세 오류: {str(fallback_error)}", source="Fallback 시스템"),
                CaveatInfo(text="추가 확인이 필요합니다.", source="Fallback 시스템")
            ],
            "quotes": [],
            "web_quotes": [],
            "recommendations": [],
            "web_info": {}
        }

def _parse_llm_response_structured(llm, prompt: str, emergency_fallback: bool = False) -> Dict[str, Any]:
    """LLM 응답을 structured output으로 파싱"""
    try:
        # structured output 사용 (긴급 탈출 모드 지원)
        structured_llm = llm.with_structured_output(RecommendResponse, emergency_fallback=emergency_fallback)
        response = structured_llm.generate_content(prompt)
        
        return {
            "conclusion": response.conclusion,
            "evidence": response.evidence,
            "caveats": response.caveats,
            "quotes": response.quotes,
            "web_quotes": response.web_quotes,
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
                "evidence": [EvidenceInfo(text="Gemini API 일일 할당량 초과", source="API 시스템")],
                "caveats": [
                    CaveatInfo(text="잠시 후 다시 시도해주세요.", source="API 시스템"),
                    CaveatInfo(text="API 할당량이 복구되면 정상적으로 답변을 제공할 수 있습니다.", source="API 시스템"),
                    CaveatInfo(text="오류 코드: 429 (Quota Exceeded)", source="API 시스템")
                ],
                "quotes": [],
                "recommendations": [],
                "web_info": {}
            }
        elif "404" in error_str or "publisher" in error_str or "model" in error_str:
            return {
                "conclusion": "모델 설정 오류로 인해 답변을 생성할 수 없습니다.",
                "evidence": [EvidenceInfo(text="Gemini 모델 설정 오류", source="API 시스템")],
                "caveats": [
                    CaveatInfo(text="모델 이름을 확인해주세요.", source="API 시스템"),
                    CaveatInfo(text="잠시 후 다시 시도해주세요.", source="API 시스템")
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
    추천 에이전트: 여행 일정, 지역, 최신 뉴스를 고려하여 맞춤 특약 추천 (verify_refine 정보 활용)
    """
    question = state.get("question", "")
    refined = state.get("refined", [])
    web_results = state.get("web_results", [])
    
    # verify_refine에서 생성된 정보들 활용
    citations = state.get("citations", [])
    warnings = state.get("warnings", [])
    verification_status = state.get("verification_status", "pass")
    policy_disclaimer = state.get("policy_disclaimer", "")
    metrics = state.get("metrics", {})
    
    print(f"🔍 [Recommend Node] 검증 상태: {verification_status}, 경고 수: {len(warnings)}, 인용 수: {len(citations)}")
    
    # 긴급 탈출 로직: 연속 구조화 실패 감지
    structured_failure_count = state.get("structured_failure_count", 0)
    max_structured_failures = state.get("max_structured_failures", 2)
    emergency_fallback_used = state.get("emergency_fallback_used", False)
    
    # 컨텍스트 포맷팅 (refined 사용)
    context = _format_context(refined)
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
        
        # 긴급 탈출 모드 결정
        use_emergency_fallback = (structured_failure_count >= max_structured_failures) or emergency_fallback_used
        
        if use_emergency_fallback:
            print(f"🚨 [Recommend Node] 긴급 탈출 모드 활성화 - 구조화 실패 횟수: {structured_failure_count}/{max_structured_failures}")
        
        # structured output 사용 (긴급 탈출 모드 지원)
        answer = _parse_llm_response_structured(llm, full_prompt, emergency_fallback=use_emergency_fallback)
        
        # 구조화 실패 감지 및 카운터 업데이트
        is_empty_result = (
            not answer.get("conclusion") or 
            answer.get("conclusion", "").strip() == "" or
            answer.get("conclusion", "").strip() == "추천 정보를 제공할 수 없습니다."
        )
        
        if is_empty_result and not use_emergency_fallback:
            # 구조화 실패 카운터 증가
            new_failure_count = structured_failure_count + 1
            print(f"⚠️ [Recommend Node] 구조화 실패 감지 - 카운터: {new_failure_count}/{max_structured_failures}")
            
            # 연속 실패가 임계값에 도달하면 긴급 탈출 모드로 재시도
            if new_failure_count >= max_structured_failures:
                print(f"🚨 [Recommend Node] 연속 구조화 실패 임계값 도달 - 긴급 탈출 모드로 재시도")
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
        
        # verify_refine의 citations 활용 (우선순위)
        if citations and not answer.get("quotes"):
            answer["quotes"] = [
                {
                    "text": c.get("snippet", "")[:200] + "...",
                    "source": f"{c.get('insurer', '')}_{c.get('doc_id', '알 수 없음')}_페이지{c.get('page', '?')}"
                }
                for c in citations[:3]  # 상위 3개만
            ]
            print(f"🔍 [Recommend Node] 표준화된 인용 정보 추가 - {len(answer['quotes'])}개")
        elif refined and not answer.get("quotes"):
            # fallback: refined에서 직접 생성
            answer["quotes"] = [
                {
                    "text": p.get("text", "")[:200] + "...",
                    "source": f"{p.get('insurer', '알 수 없음')}_{p.get('doc_id', '알 수 없음')}_페이지{p.get('page', '?')}"
                }
                for p in refined[:3]  # 상위 3개만
            ]
            print(f"🔍 [Recommend Node] fallback 출처 정보 추가 - {len(answer['quotes'])}개")
        
        # verify_refine의 warnings를 caveats에 반영
        if warnings:
            warning_caveats = [CaveatInfo(text=f"⚠️ {warning}", source="검증 시스템") for warning in warnings[:2]]  # 상위 2개 경고만
            answer["caveats"].extend(warning_caveats)
            print(f"🔍 [Recommend Node] 검증 경고 반영 - {len(warning_caveats)}개")
        
        # policy_disclaimer를 caveats에 추가
        if policy_disclaimer:
            answer["caveats"].append(CaveatInfo(text=f"📋 {policy_disclaimer}", source="법적 면책 조항"))
            print(f"🔍 [Recommend Node] 법적 면책 조항 추가")
        
        # verification_status에 따른 답변 조정
        if verification_status == "fail":
            answer["conclusion"] = "충분한 정보를 찾을 수 없어 정확한 추천을 제공하기 어렵습니다."
            answer["caveats"].append(CaveatInfo(text="추가 검색이 필요할 수 있습니다.", source="검증 시스템"))
            print(f"🔍 [Recommend Node] 검증 실패로 인한 답변 조정")
        elif verification_status == "warn":
            answer["caveats"].append(CaveatInfo(text="일부 정보가 부족하거나 상충될 수 있습니다.", source="검증 시스템"))
            print(f"🔍 [Recommend Node] 검증 경고로 인한 주의사항 추가")
        
        # 웹 검색 결과를 web_quotes에 추가 (웹 검색 결과가 있을 때만)
        if web_results and not answer.get("web_quotes"):
            answer["web_quotes"] = [
                {
                    "text": result.get("snippet", "")[:200] + "...",
                    "source": f"웹검색_{result.get('title', '제목 없음')}_{result.get('url', 'URL 없음')}"
                }
                for result in web_results[:3]  # 상위 3개만
            ]
        
        # web_info 필드가 Dict 타입인 경우 WebInfo 객체로 변환
        if isinstance(answer.get("web_info"), dict):
            web_info_dict = answer["web_info"]
            answer["web_info"] = {
                "latest_news": web_info_dict.get("latest_news", ""),
                "travel_alerts": web_info_dict.get("travel_alerts", "")
            }
        elif not answer.get("web_info"):
            # web_info가 없는 경우 기본값 설정
            answer["web_info"] = {
                "latest_news": "",
                "travel_alerts": ""
            }
        
        # 성공 시 구조화 실패 카운터 리셋
        return {
            **state, 
            "draft_answer": answer, 
            "final_answer": answer,
            "structured_failure_count": 0,
            "emergency_fallback_used": False
        }
        
    except Exception as e:
        # LLM 호출 실패 시 fallback
        fallback_answer = {
            "conclusion": f"추천 생성 중 오류가 발생했습니다: '{question}'",
            "evidence": [EvidenceInfo(text="LLM 호출 중 오류가 발생했습니다.", source="시스템 오류")],
            "caveats": [
                CaveatInfo(text="추가 확인이 필요합니다.", source="시스템 오류"),
                CaveatInfo(text=f"오류: {str(e)}", source="시스템 오류")
            ],
            "quotes": [],
            "web_quotes": [],
            "recommendations": [],
            "web_info": {}
        }
        return {**state, "draft_answer": fallback_answer, "final_answer": fallback_answer}