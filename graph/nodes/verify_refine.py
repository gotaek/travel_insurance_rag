from typing import Dict, Any, List, Optional, Tuple
import os
import yaml
import hashlib
import re
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

POLICY_PATH = os.getenv("POLICY_PATH", "config/policies.yaml")

# 정책 캐시
_policy_cache = None
_cache_timestamp = None

def _load_policies() -> Dict[str, Any]:
    """정책 파일 로드 및 캐시 관리"""
    global _policy_cache, _cache_timestamp
    
    current_time = datetime.now()
    
    # 캐시가 없거나 5분 이상 지났으면 재로드
    if _policy_cache is None or _cache_timestamp is None or (current_time - _cache_timestamp).seconds > 300:
        try:
            if os.path.exists(POLICY_PATH):
                with open(POLICY_PATH, "r", encoding="utf-8") as f:
                    _policy_cache = yaml.safe_load(f) or {}
                _cache_timestamp = current_time
            else:
                _policy_cache = {}
                logger.warning(f"정책 파일을 찾을 수 없습니다: {POLICY_PATH}")
        except Exception as e:
            logger.error(f"정책 파일 로드 실패: {e}")
            _policy_cache = {}
    
    return _policy_cache

def _validate_policy_schema(policies: Dict[str, Any]) -> List[str]:
    """정책 스키마 검증"""
    warnings = []
    required_keys = ["legal", "answer"]
    
    for key in required_keys:
        if key not in policies:
            warnings.append(f"필수 정책 키 누락: {key}")
    
    # answer 섹션 필수 키 검증
    answer_section = policies.get("answer", {})
    answer_required = ["min_citations", "min_context"]
    for key in answer_required:
        if key not in answer_section:
            warnings.append(f"answer 섹션 필수 키 누락: {key}")
    
    return warnings

def _get_intent_based_requirements(intent: str, policies: Dict[str, Any]) -> Dict[str, int]:
    """의도별 기준 적용"""
    base_requirements = {
        "qa": {"min_context": 1, "min_citations": 1, "min_insurers": 1},
        "compare": {"min_context": 3, "min_citations": 3, "min_insurers": 2},
        "summary": {"min_context": 2, "min_citations": 2, "min_insurers": 1},
        "recommend": {"min_context": 3, "min_citations": 3, "min_insurers": 2}
    }
    
    # 정책에서 오버라이드 가능
    policy_requirements = policies.get("intent_requirements", {}).get(intent, {})
    
    requirements = base_requirements.get(intent, base_requirements["qa"])
    requirements.update(policy_requirements)
    
    return requirements

def _check_score_and_freshness(refined: List[Dict[str, Any]], policies: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """스코어 및 신선도 임계치 검증"""
    warnings = []
    needs_more_search = False
    
    # 스코어 임계치
    min_score = policies.get("quality", {}).get("min_score", 0.3)
    # 신선도 임계치 (일 단위)
    max_age_days = policies.get("quality", {}).get("max_age_days", 365)
    
    low_score_count = 0
    old_doc_count = 0
    
    for doc in refined:
        score = doc.get("score", 0.0)
        if score < min_score:
            low_score_count += 1
        
        # 버전 날짜 파싱
        version_date = doc.get("version_date")
        if version_date:
            try:
                if isinstance(version_date, str):
                    doc_date = datetime.strptime(version_date, "%Y-%m-%d")
                else:
                    doc_date = version_date
                
                age_days = (datetime.now() - doc_date).days
                if age_days > max_age_days:
                    old_doc_count += 1
            except:
                warnings.append(f"날짜 파싱 실패: {version_date}")
    
    if low_score_count > 0:
        warnings.append(f"낮은 스코어 문서 {low_score_count}개 발견 (임계치: {min_score})")
        needs_more_search = True
    
    if old_doc_count > 0:
        warnings.append(f"오래된 문서 {old_doc_count}개 발견 (최대 {max_age_days}일)")
        needs_more_search = True
    
    return needs_more_search, warnings

def _remove_duplicates_and_validate_sources(refined: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """중복 제거 및 출처 품질 검증"""
    warnings = []
    
    # 중복 제거: (doc_id, page, version) 기준
    seen = set()
    unique_docs = []
    
    for doc in refined:
        key = (doc.get("doc_id"), doc.get("page"), doc.get("version"))
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
        else:
            warnings.append(f"중복 문서 제거: {key}")
    
    # 출처 신뢰도 가중치 적용
    source_weights = {
        "공식약관": 1.0,
        "공지": 0.9,
        "안내": 0.8,
        "기타": 0.7
    }
    
    for doc in unique_docs:
        doc_type = doc.get("doc_type", "기타")
        weight = source_weights.get(doc_type, 0.7)
        doc["source_weight"] = weight
    
    return unique_docs, warnings

def _detect_conflicts(refined: List[Dict[str, Any]]) -> List[str]:
    """상충 탐지: 동일 보장 항목에 서로 다른 한도/면책"""
    warnings = []
    
    # 보장 항목별 정보 수집
    coverage_info = defaultdict(list)
    
    for doc in refined:
        text = doc.get("text", "").lower()
        insurer = doc.get("insurer", "")
        
        # 간단한 금액 패턴 매칭 (천만원 포함)
        amount_pattern = r'(\d+(?:,\d+)*)(천만원|억원|만원|천원|원)'
        matches = re.findall(amount_pattern, text)
        
        for amount, unit in matches:
            full_amount = f"{amount}{unit}"
            normalized_value = _normalize_amount(full_amount)
            coverage_info[f"한도_{normalized_value}"].append({
                "insurer": insurer,
                "value": full_amount,
                "normalized": normalized_value,
                "doc_id": doc.get("doc_id")
            })
    
    # 상충 검사 - 모든 보장 항목을 하나의 그룹으로 보고 상충 검사
    all_coverage_items = []
    for coverage_type, info_list in coverage_info.items():
        all_coverage_items.extend(info_list)
    
    if len(all_coverage_items) > 1:
        # 보험사별로 그룹핑
        insurer_groups = defaultdict(list)
        for item in all_coverage_items:
            insurer_groups[item["insurer"]].append(item)
        
        # 보험사별로 다른 한도가 있는지 확인
        if len(insurer_groups) > 1:
            unique_values = set(item["normalized"] for item in all_coverage_items)
            if len(unique_values) > 1:
                insurers = list(insurer_groups.keys())
                warnings.append(f"상충 탐지: 보험사별 다른 한도 ({insurers})")
    
    return warnings

def _normalize_amount(amount_str: str) -> str:
    """금액 문자열을 정규화 (천원 단위로 통일)"""
    original = amount_str
    
    if "천만원" in original:
        # 천만원을 천원으로 변환
        num = int(original.replace("천만원", "").replace(",", ""))
        return f"{num * 1000}천원"
    elif "억원" in original:
        # 억원을 천원으로 변환
        num = int(original.replace("억원", "").replace(",", ""))
        return f"{num * 10000}천원"
    elif "만원" in original:
        # 만원을 천원으로 변환
        num = int(original.replace("만원", "").replace(",", ""))
        return f"{num * 10}천원"
    elif "천원" in original:
        return original
    elif "원" in original:
        # 원을 천원으로 변환
        num = int(original.replace("원", "").replace(",", ""))
        return f"{num // 1000}천원"
    return original

def _build_standardized_citations(refined: List[Dict[str, Any]], insurer_filter: List[str] = None) -> List[Dict[str, Any]]:
    """표준화된 인용 구조 생성 (보험사 우선순위 적용)"""
    citations = []
    seen_hashes = set()
    
    # 보험사 필터가 있으면 우선순위 정렬
    if insurer_filter:
        refined = _prioritize_insurer_documents(refined, insurer_filter)
    
    for doc in refined:
        # 텍스트 해시 생성 (중복 제거용)
        text_hash = hashlib.md5(doc.get("text", "").encode()).hexdigest()[:8]
        
        if text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)
        
        # PII 마스킹 (선택적)
        snippet = doc.get("text", "")[:120]
        # 간단한 PII 패턴 마스킹
        snippet = re.sub(r'\d{3}-\d{4}-\d{4}', 'XXX-XXXX-XXXX', snippet)  # 전화번호
        snippet = re.sub(r'\d{6}-\d{7}', 'XXXXXX-XXXXXXX', snippet)  # 주민번호
        
        # 보험사 매칭 여부 확인
        is_insurer_match = False
        if insurer_filter:
            doc_insurer = doc.get("insurer", "").lower()
            for filter_insurer in insurer_filter:
                if filter_insurer.lower() in doc_insurer or doc_insurer in filter_insurer.lower():
                    is_insurer_match = True
                    break
        
        citation = {
            "doc_id": doc.get("doc_id"),
            "page": doc.get("page"),
            "version": doc.get("version"),
            "insurer": doc.get("insurer"),
            "url": doc.get("url", ""),
            "hash": text_hash,
            "snippet": snippet,
            "score": doc.get("score", 0.0),
            "version_date": doc.get("version_date"),
            "doc_type": doc.get("doc_type", "기타"),
            "source_weight": doc.get("source_weight", 1.0),
            "is_insurer_match": is_insurer_match  # 보험사 매칭 여부 추가
        }
        citations.append(citation)
    
    return citations

def _prioritize_insurer_documents(refined: List[Dict[str, Any]], insurer_filter: List[str]) -> List[Dict[str, Any]]:
    """
    보험사 매칭 문서를 우선적으로 정렬합니다.
    
    Args:
        refined: 검색 결과 문서 리스트
        insurer_filter: 우선순위를 부여할 보험사 리스트
        
    Returns:
        보험사 우선순위가 적용된 문서 리스트
    """
    if not insurer_filter:
        return refined
    
    # 보험사 매칭 문서와 비매칭 문서 분리
    insurer_matched = []
    other_docs = []
    
    for doc in refined:
        doc_insurer = doc.get("insurer", "").lower()
        is_matched = False
        
        for filter_insurer in insurer_filter:
            if filter_insurer.lower() in doc_insurer or doc_insurer in filter_insurer.lower():
                is_matched = True
                break
        
        if is_matched:
            insurer_matched.append(doc)
        else:
            other_docs.append(doc)
    
    # 보험사 매칭 문서를 먼저 배치하고, 각 그룹 내에서는 점수 순으로 정렬
    insurer_matched.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    other_docs.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    
    return insurer_matched + other_docs

def _determine_verification_status(warnings: List[str], requirements: Dict[str, int], 
                                 refined: List[Dict[str, Any]], citations: List[Dict[str, Any]]) -> Tuple[str, str]:
    """검증 상태 및 다음 액션 결정"""
    
    # 심각한 경고가 있으면 fail
    critical_warnings = [w for w in warnings if any(keyword in w for keyword in ["상충", "오래된", "낮은 스코어"])]
    if critical_warnings:
        return "fail", "broaden_search"
    
    # 요구사항 미달 체크
    context_insufficient = len(refined) < requirements["min_context"]
    citations_insufficient = len(citations) < requirements["min_citations"]
    
    # 보험사 다양성 체크
    unique_insurers = len(set(doc.get("insurer", "") for doc in refined))
    insurers_insufficient = unique_insurers < requirements.get("min_insurers", 1)
    
    # 요구사항 미달이 있으면 warn
    if context_insufficient or citations_insufficient or insurers_insufficient:
        return "warn", "broaden_search"
    
    # 경고가 있으면 warn, 없으면 pass
    if warnings:
        return "warn", "proceed"
    else:
        return "pass", "proceed"

def _generate_metrics(warnings: List[str], refined: List[Dict[str, Any]]) -> Dict[str, Any]:
    """로깅/메트릭 생성"""
    metrics = {
        "total_documents": len(refined),
        "unique_insurers": len(set(doc.get("insurer", "") for doc in refined)),
        "avg_score": sum(doc.get("score", 0.0) for doc in refined) / len(refined) if refined else 0.0,
        "warning_counts": {}
    }
    
    # 경고 코드화
    warning_codes = {
        "상충": "coverage_conflict",
        "중복": "duplicate_document", 
        "낮은 스코어": "low_score",
        "오래된": "outdated_document",
        "부족": "insufficient_context"
    }
    
    for warning in warnings:
        for keyword, code in warning_codes.items():
            if keyword in warning:
                metrics["warning_counts"][code] = metrics["warning_counts"].get(code, 0) + 1
                break
    
    return metrics

def verify_refine_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """개선된 검증 및 정제 노드 (보험사 우선순위 적용)"""
    import logging
    logger = logging.getLogger(__name__)
    
    # 입력 데이터
    refined = state.get("refined", []) or []
    warnings = state.get("warnings", []) or []
    intent = state.get("intent", "qa")
    insurer_filter = state.get("insurer_filter", None)  # Planner에서 전달된 보험사 필터
    
    logger.info(f"🔍 [VerifyRefine] 시작 - 의도: {intent}, 보험사 필터: {insurer_filter}")
    logger.info(f"🔍 [VerifyRefine] 입력 패시지 수: {len(refined)}")
    
    # 정책 로드 및 스키마 검증
    policies = _load_policies()
    schema_warnings = _validate_policy_schema(policies)
    logger.info(f"🔍 [VerifyRefine] 정책 로드 완료, 스키마 경고: {len(schema_warnings)}개")
    
    # 스키마 경고 추가
    warnings.extend(schema_warnings)
    
    # 의도별 기준 적용
    requirements = _get_intent_based_requirements(intent, policies)
    logger.info(f"🔍 [VerifyRefine] 의도별 기준 적용: {requirements}")
    
    # 스코어 및 신선도 검증
    needs_more_search, quality_warnings = _check_score_and_freshness(refined, policies)
    warnings.extend(quality_warnings)
    logger.info(f"🔍 [VerifyRefine] 품질 검증 완료 - 추가 검색 필요: {needs_more_search}, 경고: {len(quality_warnings)}개")
    
    # 중복 제거 및 출처 품질 검증
    unique_refined, dedup_warnings = _remove_duplicates_and_validate_sources(refined)
    warnings.extend(dedup_warnings)
    logger.info(f"🔍 [VerifyRefine] 중복 제거 완료 - {len(refined)} → {len(unique_refined)}개, 경고: {len(dedup_warnings)}개")
    
    # 상충 탐지
    conflict_warnings = _detect_conflicts(unique_refined)
    warnings.extend(conflict_warnings)
    logger.info(f"🔍 [VerifyRefine] 상충 탐지 완료 - 경고: {len(conflict_warnings)}개")
    
    # intent에 따른 동적 문서 수 제한
    if intent == "compare":
        doc_limit = 8  # 비교 질문은 더 많은 문서 필요
        logger.info(f"🔍 [VerifyRefine] Compare intent - 문서 8개로 제한")
    else:
        doc_limit = 5  # 기본 문서 수
        logger.info(f"🔍 [VerifyRefine] {intent} intent - 문서 5개로 제한")
    
    unique_refined.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    unique_refined = unique_refined[:doc_limit]
    logger.info(f"🔍 [VerifyRefine] 문서 {doc_limit}개로 제한 완료")
    
    # 표준화된 인용 생성 (보험사 우선순위 적용)
    citations = _build_standardized_citations(unique_refined, insurer_filter)
    logger.info(f"🔍 [VerifyRefine] 인용 생성 완료 - {len(citations)}개")
    
    # 검증 상태 결정
    verification_status, next_action = _determine_verification_status(
        warnings, requirements, unique_refined, citations
    )
    logger.info(f"🔍 [VerifyRefine] 검증 상태: {verification_status}, 다음 액션: {next_action}")
    
    # 메트릭 생성
    metrics = _generate_metrics(warnings, unique_refined)
    logger.info(f"🔍 [VerifyRefine] 메트릭 생성 완료 - 총 경고: {len(warnings)}개")
    
    # 법적 면책 조항 (기본값 보장)
    disclaimer = policies.get("legal", {}).get("disclaimer", 
        "본 답변은 참고용 정보이며, 실제 보상/보장 여부는 보험증권과 최신 약관에 따릅니다.")
    
    # 결과 반환
    out = {
        **state,
        "refined": unique_refined,
        "citations": citations,
        "warnings": warnings,
        "verification_status": verification_status,
        "next_action": next_action,
        "needs_more_search": needs_more_search,
        "requirements": requirements,
        "metrics": metrics,
        "policy_disclaimer": disclaimer,
        "insurer_filter": insurer_filter  # 보험사 필터 정보 유지
    }
    
    logger.info(f"검증 완료: {verification_status}, 다음 액션: {next_action}, 경고: {len(warnings)}개, 보험사 필터: {insurer_filter}")
    
    return out