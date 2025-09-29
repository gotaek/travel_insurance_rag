# common.py — answerer 공통 유틸리티 함수
"""
성능 최적화를 위한 공통 함수들
- 중복 코드 제거
- 빠른 응답 처리
- 효율적인 로깅
"""

from typing import Dict, Any, List
import logging
from graph.models import EvidenceInfo, CaveatInfo
from graph.prompts.utils import get_cached_prompt, get_simple_fallback_response

logger = logging.getLogger(__name__)

# 전역 프롬프트 캐시 (모듈 레벨)
_system_prompt = None
_prompt_cache = {}

def get_system_prompt() -> str:
    """시스템 프롬프트 캐시 (모듈 레벨)"""
    global _system_prompt
    if _system_prompt is None:
        _system_prompt = get_cached_prompt("system_core")
    return _system_prompt

def get_prompt_cached(prompt_name: str) -> str:
    """프롬프트 캐시 (모듈 레벨)"""
    if prompt_name not in _prompt_cache:
        _prompt_cache[prompt_name] = get_cached_prompt(prompt_name)
    return _prompt_cache[prompt_name]

def format_context_optimized(passages: List[Dict]) -> str:
    """최적화된 컨텍스트 포맷팅"""
    if not passages:
        return "관련 문서를 찾을 수 없습니다."
    
    # 상위 5개만 처리, 텍스트 길이 제한
    context_parts = []
    for i, passage in enumerate(passages[:5], 1):
        doc_id = passage.get("doc_id", "알 수 없음")
        page = passage.get("page", "알 수 없음")
        text = passage.get("text", "")[:500]  # 500자로 제한
        context_parts.append(f"[문서 {i}] {doc_id} (페이지 {page})\n{text}\n")
    
    return "\n".join(context_parts)

def process_verify_refine_data(state: Dict[str, Any], answer: Dict[str, Any]) -> Dict[str, Any]:
    """verify_refine 데이터를 효율적으로 처리 (evidence 개수 제한)"""
    # verify_refine에서 생성된 정보들
    citations = state.get("citations", [])
    warnings = state.get("warnings", [])
    verification_status = state.get("verification_status", "pass")
    policy_disclaimer = state.get("policy_disclaimer", "")
    
    # citations를 evidence에 추가 (quotes 대신 evidence 사용)
    if citations and not answer.get("evidence"):
        citation_evidence = [
            EvidenceInfo(
                text=c.get("snippet", "")[:200] + "...",
                source=f"{c.get('insurer', '')}_{c.get('doc_id', '알 수 없음')}_페이지{c.get('page', '?')}"
            )
            for c in citations[:3]  # 상위 3개만
        ]
        answer["evidence"] = citation_evidence
    
    # warnings를 caveats에 반영
    if warnings:
        warning_caveats = [
            CaveatInfo(text=f"⚠️ {warning}", source="검증 시스템") 
            for warning in warnings[:2]  # 상위 2개 경고만
        ]
        answer["caveats"].extend(warning_caveats)
    
    # policy_disclaimer를 caveats에 추가
    if policy_disclaimer:
        answer["caveats"].append(CaveatInfo(text=f"📋 {policy_disclaimer}", source="법적 면책 조항"))
    
    # verification_status에 따른 답변 조정
    if verification_status == "fail":
        answer["conclusion"] = "충분한 정보를 찾을 수 없어 정확한 답변을 제공하기 어렵습니다."
        answer["caveats"].append(CaveatInfo(text="추가 검색이 필요할 수 있습니다.", source="검증 시스템"))
    elif verification_status == "warn":
        answer["caveats"].append(CaveatInfo(text="일부 정보가 부족하거나 상충될 수 있습니다.", source="검증 시스템"))
    
    # evidence 개수를 5개로 제한 (성능 최적화)
    if "evidence" in answer and len(answer["evidence"]) > 5:
        try:
            # score 기준으로 정렬하여 상위 5개만 선택
            sorted_evidence = sorted(
                answer["evidence"], 
                key=lambda x: getattr(x, 'score', 0) if hasattr(x, 'score') else 0, 
                reverse=True
            )
            answer["evidence"] = sorted_evidence[:5]
        except Exception:
            # score가 없는 경우 단순히 앞의 5개만 선택
            answer["evidence"] = answer["evidence"][:5]
    
    return answer

def create_optimized_prompt(system_prompt: str, task_prompt: str, question: str, context: str) -> str:
    """최적화된 프롬프트 생성"""
    return f"""{system_prompt}

{task_prompt}

## 질문
{question}

## 참고 문서
{context}

위 문서를 참고하여 답변해주세요."""

def handle_llm_error_optimized(error: Exception, question: str, node_type: str) -> Dict[str, Any]:
    """최적화된 LLM 오류 처리"""
    error_str = str(error).lower()
    
    if "quota" in error_str or "limit" in error_str or "429" in error_str or "exceeded" in error_str:
        return {
            "conclusion": "죄송합니다. 현재 API 할당량이 초과되어 답변을 생성할 수 없습니다.",
            "evidence": [EvidenceInfo(text="Gemini API 일일 할당량 초과", source="API 시스템")],
            "caveats": [
                CaveatInfo(text="잠시 후 다시 시도해주세요.", source="API 시스템"),
                CaveatInfo(text="API 할당량이 복구되면 정상적으로 답변을 제공할 수 있습니다.", source="API 시스템")
            ],
        }
    elif "404" in error_str or "publisher" in error_str or "model" in error_str:
        return {
            "conclusion": "모델 설정 오류로 인해 답변을 생성할 수 없습니다.",
            "evidence": [EvidenceInfo(text="Gemini 모델 설정 오류", source="API 시스템")],
            "caveats": [
                CaveatInfo(text="모델 이름을 확인해주세요.", source="API 시스템"),
                CaveatInfo(text="잠시 후 다시 시도해주세요.", source="API 시스템")
            ],
        }
    else:
        return get_simple_fallback_response(question, node_type)

def log_performance(operation: str, start_time: float, **kwargs):
    """성능 로깅 (디버그 모드에서만)"""
    if logger.isEnabledFor(logging.DEBUG):
        import time
        duration = time.time() - start_time
        logger.debug(f"{operation} 완료 - 소요시간: {duration:.2f}초, {kwargs}")
