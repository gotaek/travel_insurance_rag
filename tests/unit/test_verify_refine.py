import pytest
from unittest.mock import patch, mock_open
from graph.nodes.verify_refine import (
    _get_intent_based_requirements,
    _check_score_and_freshness,
    _remove_duplicates_and_validate_sources,
    _detect_conflicts,
    _build_standardized_citations,
    _determine_verification_status,
    _generate_metrics,
    verify_refine_node
)
from datetime import datetime, timedelta


class TestIntentBasedRequirements:
    """의도별 기준 적용 테스트"""
    
    def test_qa_requirements(self):
        """QA 의도 기본 요구사항 테스트"""
        policies = {}
        requirements = _get_intent_based_requirements("qa", policies)
        
        assert requirements["min_context"] == 1
        assert requirements["min_citations"] == 1
        assert requirements["min_insurers"] == 1
    
    def test_compare_requirements(self):
        """비교 의도 요구사항 테스트"""
        policies = {}
        requirements = _get_intent_based_requirements("compare", policies)
        
        assert requirements["min_context"] == 3
        assert requirements["min_citations"] == 3
        assert requirements["min_insurers"] == 2
    
    def test_policy_override(self):
        """정책 파일 오버라이드 테스트"""
        policies = {
            "intent_requirements": {
                "qa": {"min_context": 5, "min_citations": 3}
            }
        }
        requirements = _get_intent_based_requirements("qa", policies)
        
        assert requirements["min_context"] == 5
        assert requirements["min_citations"] == 3
        assert requirements["min_insurers"] == 1  # 기본값 유지


class TestScoreAndFreshness:
    """스코어 및 신선도 검증 테스트"""
    
    def test_low_score_detection(self):
        """낮은 스코어 문서 탐지 테스트"""
        refined = [
            {"score": 0.1, "doc_id": "doc1"},
            {"score": 0.5, "doc_id": "doc2"}
        ]
        policies = {"quality": {"min_score": 0.3, "max_age_days": 365}}
        
        needs_more_search, warnings = _check_score_and_freshness(refined, policies)
        
        assert needs_more_search is True
        assert any("낮은 스코어" in w for w in warnings)
    
    def test_outdated_document_detection(self):
        """오래된 문서 탐지 테스트"""
        old_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
        refined = [
            {"score": 0.8, "version_date": old_date, "doc_id": "doc1"},
            {"score": 0.9, "version_date": "2025-01-01", "doc_id": "doc2"}
        ]
        policies = {"quality": {"min_score": 0.3, "max_age_days": 365}}
        
        needs_more_search, warnings = _check_score_and_freshness(refined, policies)
        
        assert needs_more_search is True
        assert any("오래된 문서" in w for w in warnings)
    
    def test_good_quality_documents(self):
        """양질의 문서 테스트"""
        recent_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        refined = [
            {"score": 0.8, "version_date": recent_date, "doc_id": "doc1"},
            {"score": 0.9, "version_date": recent_date, "doc_id": "doc2"}
        ]
        policies = {"quality": {"min_score": 0.3, "max_age_days": 365}}
        
        needs_more_search, warnings = _check_score_and_freshness(refined, policies)
        
        assert needs_more_search is False
        assert len(warnings) == 0


class TestDuplicateRemoval:
    """중복 제거 및 출처 품질 검증 테스트"""
    
    def test_duplicate_removal(self):
        """중복 문서 제거 테스트"""
        refined = [
            {"doc_id": "doc1", "page": 1, "version": "v1", "insurer": "A보험"},
            {"doc_id": "doc1", "page": 1, "version": "v1", "insurer": "A보험"},  # 중복
            {"doc_id": "doc2", "page": 2, "version": "v1", "insurer": "B보험"}
        ]
        
        unique_docs, warnings = _remove_duplicates_and_validate_sources(refined)
        
        assert len(unique_docs) == 2
        assert any("중복 문서 제거" in w for w in warnings)
    
    def test_source_weight_assignment(self):
        """출처 신뢰도 가중치 할당 테스트"""
        refined = [
            {"doc_id": "doc1", "doc_type": "공식약관", "insurer": "A보험"},
            {"doc_id": "doc2", "doc_type": "안내", "insurer": "B보험"}
        ]
        
        unique_docs, _ = _remove_duplicates_and_validate_sources(refined)
        
        assert unique_docs[0]["source_weight"] == 1.0  # 공식약관
        assert unique_docs[1]["source_weight"] == 0.8   # 안내


class TestConflictDetection:
    """상충 탐지 테스트"""
    
    def test_coverage_limit_conflict(self):
        """보장 한도 상충 탐지 테스트"""
        refined = [
            {
                "text": "의료비 보장 한도는 1억원입니다",
                "insurer": "A보험",
                "doc_id": "doc1"
            },
            {
                "text": "의료비 보장 한도는 5천만원입니다", 
                "insurer": "B보험",
                "doc_id": "doc2"
            }
        ]
        
        warnings = _detect_conflicts(refined)
        
        assert any("상충 탐지" in w for w in warnings)
        assert any("한도" in w for w in warnings)
    
    def test_no_conflict_same_limits(self):
        """동일 한도로 상충 없음 테스트"""
        refined = [
            {
                "text": "의료비 보장 한도는 1억원입니다",
                "insurer": "A보험",
                "doc_id": "doc1"
            },
            {
                "text": "의료비 보장 한도는 1억원입니다",
                "insurer": "B보험", 
                "doc_id": "doc2"
            }
        ]
        
        warnings = _detect_conflicts(refined)
        
        assert len(warnings) == 0


class TestStandardizedCitations:
    """표준화된 인용 생성 테스트"""
    
    def test_citation_structure(self):
        """인용 구조 표준화 테스트"""
        refined = [
            {
                "doc_id": "doc1",
                "page": 1,
                "version": "v1",
                "insurer": "A보험",
                "url": "https://example.com",
                "text": "보험 내용입니다",
                "score": 0.8,
                "version_date": "2025-01-01",
                "doc_type": "공식약관"
            }
        ]
        
        citations = _build_standardized_citations(refined)
        
        assert len(citations) == 1
        citation = citations[0]
        
        # 필수 필드 확인
        required_fields = ["doc_id", "page", "version", "insurer", "url", "hash", 
                          "snippet", "score", "version_date", "doc_type", "source_weight"]
        for field in required_fields:
            assert field in citation
    
    def test_pii_masking(self):
        """PII 마스킹 테스트"""
        refined = [
            {
                "text": "고객 전화번호 010-1234-5678과 주민번호 123456-1234567입니다",
                "doc_id": "doc1"
            }
        ]
        
        citations = _build_standardized_citations(refined)
        
        snippet = citations[0]["snippet"]
        assert "XXX-XXXX-XXXX" in snippet
        assert "XXXXXX-XXXXXXX" in snippet
        assert "010-1234-5678" not in snippet
        assert "123456-1234567" not in snippet


class TestVerificationStatus:
    """검증 상태 결정 테스트"""
    
    def test_pass_status(self):
        """통과 상태 테스트"""
        warnings = []
        requirements = {"min_context": 2, "min_citations": 2, "min_insurers": 1}
        refined = [{"insurer": "A보험"}, {"insurer": "B보험"}]
        citations = [{"insurer": "A보험"}, {"insurer": "B보험"}]
        
        status, action = _determine_verification_status(warnings, requirements, refined, citations)
        
        assert status == "pass"
        assert action == "proceed"
    
    def test_fail_status_with_conflicts(self):
        """상충으로 인한 실패 상태 테스트"""
        warnings = ["상충 탐지: 한도_1억원 - 보험사별 다른 한도"]
        requirements = {"min_context": 1, "min_citations": 1, "min_insurers": 1}
        refined = [{"insurer": "A보험"}]
        citations = [{"insurer": "A보험"}]
        
        status, action = _determine_verification_status(warnings, requirements, refined, citations)
        
        assert status == "fail"
        assert action == "broaden_search"
    
    def test_warn_status_insufficient_context(self):
        """컨텍스트 부족으로 인한 경고 상태 테스트"""
        warnings = []
        requirements = {"min_context": 3, "min_citations": 3, "min_insurers": 1}
        refined = [{"insurer": "A보험"}]  # 부족
        citations = [{"insurer": "A보험"}]
        
        status, action = _determine_verification_status(warnings, requirements, refined, citations)
        
        assert status == "warn"
        assert action == "broaden_search"


class TestMetrics:
    """메트릭 생성 테스트"""
    
    def test_metrics_generation(self):
        """메트릭 생성 테스트"""
        warnings = ["상충 탐지", "낮은 스코어 문서 2개 발견"]
        refined = [
            {"insurer": "A보험", "score": 0.8},
            {"insurer": "B보험", "score": 0.6}
        ]
        
        metrics = _generate_metrics(warnings, refined)
        
        assert metrics["total_documents"] == 2
        assert metrics["unique_insurers"] == 2
        assert metrics["avg_score"] == 0.7
        assert metrics["warning_counts"]["coverage_conflict"] == 1
        assert metrics["warning_counts"]["low_score"] == 1


class TestVerifyRefineNode:
    """메인 노드 통합 테스트"""
    
    @patch('graph.nodes.verify_refine._load_policies')
    def test_successful_verification(self, mock_load_policies):
        """성공적인 검증 테스트"""
        mock_load_policies.return_value = {
            "legal": {"disclaimer": "테스트 면책조항"},
            "quality": {"min_score": 0.3, "max_age_days": 365},
            "answer": {"min_citations": 1, "min_context": 1}
        }
        
        state = {
            "refined": [
                {
                    "doc_id": "doc1",
                    "page": 1,
                    "version": "v1",
                    "insurer": "A보험",
                    "text": "보험 내용",
                    "score": 0.8,
                    "version_date": "2025-01-01",
                    "doc_type": "공식약관"
                }
            ],
            "intent": "qa",
            "warnings": []
        }
        
        result = verify_refine_node(state)
        
        assert result["verification_status"] == "pass"
        assert result["next_action"] == "proceed"
        assert len(result["citations"]) == 1
        assert "policy_disclaimer" in result
    
    @patch('graph.nodes.verify_refine._load_policies')
    def test_failed_verification_with_conflicts(self, mock_load_policies):
        """상충으로 인한 검증 실패 테스트"""
        mock_load_policies.return_value = {
            "legal": {"disclaimer": "테스트 면책조항"},
            "quality": {"min_score": 0.3, "max_age_days": 365}
        }
        
        state = {
            "refined": [
                {
                    "doc_id": "doc1",
                    "text": "의료비 보장 한도는 1억원입니다",
                    "insurer": "A보험",
                    "score": 0.8
                },
                {
                    "doc_id": "doc2", 
                    "text": "의료비 보장 한도는 5천만원입니다",
                    "insurer": "B보험",
                    "score": 0.8
                }
            ],
            "intent": "compare",
            "warnings": []
        }
        
        result = verify_refine_node(state)
        
        assert result["verification_status"] == "fail"
        assert result["next_action"] == "broaden_search"
        assert any("상충 탐지" in w for w in result["warnings"])
    
    @patch('graph.nodes.verify_refine._load_policies')
    def test_insufficient_insurers_for_compare(self, mock_load_policies):
        """비교 의도에서 보험사 부족 테스트"""
        mock_load_policies.return_value = {
            "legal": {"disclaimer": "테스트 면책조항"},
            "quality": {"min_score": 0.3, "max_age_days": 365}
        }
        
        state = {
            "refined": [
                {
                    "doc_id": "doc1",
                    "insurer": "A보험",
                    "score": 0.8
                }
            ],
            "intent": "compare",  # 최소 2개 보험사 필요
            "warnings": []
        }
        
        result = verify_refine_node(state)
        
        assert result["verification_status"] == "warn"
        assert result["next_action"] == "broaden_search"
        assert result["requirements"]["min_insurers"] == 2
