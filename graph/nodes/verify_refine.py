from typing import Dict, Any, List
import os
import yaml

POLICY_PATH = os.getenv("POLICY_PATH", "config/policies.yaml")

def _load_policies() -> Dict[str, Any]:
    if os.path.exists(POLICY_PATH):
        with open(POLICY_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

def _build_citations(refined: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """refined passage에서 문서ID/페이지/버전 등을 뽑아 인용 리스트 생성"""
    cites = []
    for p in refined:
        cites.append({
            "doc_id": p.get("doc_id"),
            "page": p.get("page"),
            "insurer": p.get("insurer"),
            "version": p.get("version"),
            "snippet": p.get("text", "")[:120]
        })
    return cites

def verify_refine_node(state: Dict[str, Any]) -> Dict[str, Any]:
    policies = _load_policies()
    refined = state.get("refined", []) or []
    warnings = state.get("warnings", []) or []

    # 최소 컨텍스트/인용 기준 체크
    min_ctx = policies.get("answer", {}).get("min_context", 1)
    min_cite = policies.get("answer", {}).get("min_citations", 1)

    if len(refined) < min_ctx:
        warnings.append("검색 컨텍스트가 충분하지 않습니다. 추가 확인이 필요합니다.")

    citations = _build_citations(refined)
    if len(citations) < min_cite:
        warnings.append("인용이 부족합니다. 최신 약관/증권을 확인하세요.")

    # 법적 문구
    disclaimer = policies.get("legal", {}).get("disclaimer")
    # state에 결과 반영
    out = {**state, "citations": citations, "warnings": warnings}
    if disclaimer:
        # caveat로 넘길 수 있도록 별도 필드로 추가
        out["policy_disclaimer"] = disclaimer
    return out