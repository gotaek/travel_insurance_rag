"""
Verify Refine 통합 테스트
실제 환경에서 verify_refine 노드의 전체 기능을 테스트합니다.
"""

import sys
import os
import pytest
import time
import tempfile
import yaml
from unittest.mock import patch, Mock
from datetime import datetime, timedelta

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from graph.nodes.verify_refine import verify_refine_node


@pytest.mark.integration
class TestVerifyRefineIntegration:
    """Verify Refine 통합 테스트 클래스"""
    
    def setup_method(self):
        """각 테스트 전에 실행되는 설정"""
        # 임시 정책 파일 생성
        self.temp_policy_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        self.temp_policy_path = self.temp_policy_file.name
        
        # 기본 정책 설정
        policy_config = {
            "legal": {
                "disclaimer": "본 답변은 참고용 정보입니다."
            },
            "answer": {
                "min_citations": 1,
                "min_context": 1
            },
            "intent_requirements": {
                "qa": {"min_context": 1, "min_citations": 1, "min_insurers": 1},
                "compare": {"min_context": 3, "min_citations": 3, "min_insurers": 2},
                "summary": {"min_context": 2, "min_citations": 2, "min_insurers": 1},
                "recommend": {"min_context": 3, "min_citations": 3, "min_insurers": 2}
            },
            "quality": {
                "min_score": 0.3,
                "max_age_days": 365
            },
            "source_weights": {
                "공식약관": 1.0,
                "공지": 0.9,
                "안내": 0.8,
                "기타": 0.7
            }
        }
        
        yaml.dump(policy_config, self.temp_policy_file, default_flow_style=False, allow_unicode=True)
        self.temp_policy_file.close()
    
    def teardown_method(self):
        """각 테스트 후에 실행되는 정리"""
        if os.path.exists(self.temp_policy_path):
            os.unlink(self.temp_policy_path)
    
    @patch('graph.nodes.verify_refine._load_policies')
    def test_qa_intent_successful_verification(self, mock_load_policies):
        """QA 의도 성공적인 검증 테스트"""
        mock_load_policies.return_value = {
            "legal": {"disclaimer": "본 답변은 참고용 정보입니다."},
            "answer": {"min_citations": 1, "min_context": 1},
            "intent_requirements": {
                "qa": {"min_context": 1, "min_citations": 1, "min_insurers": 1}
            },
            "quality": {"min_score": 0.3, "max_age_days": 365},
            "source_weights": {"공식약관": 1.0, "공지": 0.9, "안내": 0.8, "기타": 0.7}
        }
        
        state = {
            "refined": [
                {
                    "doc_id": "doc1",
                    "page": 1,
                    "version": "v1",
                    "insurer": "DB손해보험",
                    "text": "여행자보험의 보장내용은 상해보장, 질병보장, 휴대품보장을 포함합니다.",
                    "score": 0.8,
                    "version_date": "2025-01-01",
                    "doc_type": "공식약관",
                    "url": "https://dbins.co.kr/terms"
                }
            ],
            "intent": "qa",
            "warnings": []
        }
        
        result = verify_refine_node(state)
        
        # 검증 결과 확인
        assert result["verification_status"] == "pass"
        assert result["next_action"] == "proceed"
        assert len(result["citations"]) == 1
        assert result["citations"][0]["insurer"] == "DB손해보험"
        assert result["citations"][0]["score"] == 0.8
        assert "policy_disclaimer" in result
        assert result["requirements"]["min_context"] == 1
        assert result["requirements"]["min_citations"] == 1
        assert result["requirements"]["min_insurers"] == 1
    
    @patch('graph.nodes.verify_refine._load_policies')
    def test_compare_intent_insufficient_insurers(self, mock_load_policies):
        """비교 의도에서 보험사 부족 테스트"""
        mock_load_policies.return_value = {
            "legal": {"disclaimer": "본 답변은 참고용 정보입니다."},
            "answer": {"min_citations": 1, "min_context": 1},
            "intent_requirements": {
                "compare": {"min_context": 3, "min_citations": 3, "min_insurers": 2}
            },
            "quality": {"min_score": 0.3, "max_age_days": 365},
            "source_weights": {"공식약관": 1.0, "공지": 0.9, "안내": 0.8, "기타": 0.7}
        }
        
        state = {
            "refined": [
                {
                    "doc_id": "doc1",
                    "page": 1,
                    "version": "v1",
                    "insurer": "DB손해보험",
                    "text": "DB손해보험 여행자보험 보장내용",
                    "score": 0.8,
                    "version_date": "2025-01-01",
                    "doc_type": "공식약관"
                }
            ],
            "intent": "compare",  # 최소 2개 보험사 필요
            "warnings": []
        }
        
        result = verify_refine_node(state)
        
        # 검증 결과 확인
        assert result["verification_status"] == "warn"
        assert result["next_action"] == "broaden_search"
        assert result["requirements"]["min_insurers"] == 2
        # 보험사 부족으로 인한 warn 상태 확인 (경고 메시지는 없지만 상태는 warn)
        assert result["verification_status"] == "warn"
    
    @patch('graph.nodes.verify_refine._load_policies')
    def test_conflict_detection_integration(self, mock_load_policies):
        """상충 탐지 통합 테스트"""
        mock_load_policies.return_value = {
            "legal": {"disclaimer": "본 답변은 참고용 정보입니다."},
            "answer": {"min_citations": 1, "min_context": 1},
            "intent_requirements": {
                "compare": {"min_context": 3, "min_citations": 3, "min_insurers": 2}
            },
            "quality": {"min_score": 0.3, "max_age_days": 365},
            "source_weights": {"공식약관": 1.0, "공지": 0.9, "안내": 0.8, "기타": 0.7}
        }
        
        state = {
            "refined": [
                {
                    "doc_id": "doc1",
                    "page": 1,
                    "version": "v1",
                    "insurer": "DB손해보험",
                    "text": "의료비 보장 한도는 1억원입니다",
                    "score": 0.8,
                    "version_date": "2025-01-01",
                    "doc_type": "공식약관"
                },
                {
                    "doc_id": "doc2",
                    "page": 1,
                    "version": "v1",
                    "insurer": "삼성화재",
                    "text": "의료비 보장 한도는 5천만원입니다",
                    "score": 0.8,
                    "version_date": "2025-01-01",
                    "doc_type": "공식약관"
                }
            ],
            "intent": "compare",
            "warnings": []
        }
        
        result = verify_refine_node(state)
        
        # 상충 탐지 확인
        assert result["verification_status"] == "fail"
        assert result["next_action"] == "broaden_search"
        assert any("상충 탐지" in w for w in result["warnings"])
        assert any("보험사별 다른 한도" in w for w in result["warnings"])
    
    @patch('graph.nodes.verify_refine._load_policies')
    def test_duplicate_removal_integration(self, mock_load_policies):
        """중복 제거 통합 테스트"""
        mock_load_policies.return_value = {
            "legal": {"disclaimer": "본 답변은 참고용 정보입니다."},
            "answer": {"min_citations": 1, "min_context": 1},
            "intent_requirements": {
                "qa": {"min_context": 1, "min_citations": 1, "min_insurers": 1}
            },
            "quality": {"min_score": 0.3, "max_age_days": 365},
            "source_weights": {"공식약관": 1.0, "공지": 0.9, "안내": 0.8, "기타": 0.7}
        }
        
        state = {
            "refined": [
                {
                    "doc_id": "doc1",
                    "page": 1,
                    "version": "v1",
                    "insurer": "DB손해보험",
                    "text": "여행자보험 보장내용",
                    "score": 0.8,
                    "version_date": "2025-01-01",
                    "doc_type": "공식약관"
                },
                {
                    "doc_id": "doc1",  # 중복
                    "page": 1,
                    "version": "v1",
                    "insurer": "DB손해보험",
                    "text": "여행자보험 보장내용",
                    "score": 0.8,
                    "version_date": "2025-01-01",
                    "doc_type": "공식약관"
                },
                {
                    "doc_id": "doc2",
                    "page": 2,
                    "version": "v1",
                    "insurer": "삼성화재",
                    "text": "여행자보험 보험료",
                    "score": 0.7,
                    "version_date": "2025-01-01",
                    "doc_type": "공식약관"
                }
            ],
            "intent": "qa",
            "warnings": []
        }
        
        result = verify_refine_node(state)
        
        # 중복 제거 확인
        assert len(result["refined"]) == 2  # 중복 제거됨
        assert len(result["citations"]) == 2
        assert any("중복 문서 제거" in w for w in result["warnings"])
    
    @patch('graph.nodes.verify_refine._load_policies')
    def test_low_score_detection_integration(self, mock_load_policies):
        """낮은 스코어 탐지 통합 테스트"""
        mock_load_policies.return_value = {
            "legal": {"disclaimer": "본 답변은 참고용 정보입니다."},
            "answer": {"min_citations": 1, "min_context": 1},
            "intent_requirements": {
                "qa": {"min_context": 1, "min_citations": 1, "min_insurers": 1}
            },
            "quality": {"min_score": 0.3, "max_age_days": 365},
            "source_weights": {"공식약관": 1.0, "공지": 0.9, "안내": 0.8, "기타": 0.7}
        }
        
        state = {
            "refined": [
                {
                    "doc_id": "doc1",
                    "page": 1,
                    "version": "v1",
                    "insurer": "DB손해보험",
                    "text": "여행자보험 보장내용",
                    "score": 0.1,  # 낮은 스코어
                    "version_date": "2025-01-01",
                    "doc_type": "공식약관"
                },
                {
                    "doc_id": "doc2",
                    "page": 1,
                    "version": "v1",
                    "insurer": "삼성화재",
                    "text": "여행자보험 보험료",
                    "score": 0.2,  # 낮은 스코어
                    "version_date": "2025-01-01",
                    "doc_type": "공식약관"
                }
            ],
            "intent": "qa",
            "warnings": []
        }
        
        result = verify_refine_node(state)
        
        # 낮은 스코어 탐지 확인
        assert result["needs_more_search"] is True
        assert any("낮은 스코어" in w for w in result["warnings"])
        assert result["verification_status"] == "fail"
        assert result["next_action"] == "broaden_search"
    
    @patch('graph.nodes.verify_refine._load_policies')
    def test_outdated_document_detection_integration(self, mock_load_policies):
        """오래된 문서 탐지 통합 테스트"""
        mock_load_policies.return_value = {
            "legal": {"disclaimer": "본 답변은 참고용 정보입니다."},
            "answer": {"min_citations": 1, "min_context": 1},
            "intent_requirements": {
                "qa": {"min_context": 1, "min_citations": 1, "min_insurers": 1}
            },
            "quality": {"min_score": 0.3, "max_age_days": 365},
            "source_weights": {"공식약관": 1.0, "공지": 0.9, "안내": 0.8, "기타": 0.7}
        }
        
        # 2년 전 날짜
        old_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        
        state = {
            "refined": [
                {
                    "doc_id": "doc1",
                    "page": 1,
                    "version": "v1",
                    "insurer": "DB손해보험",
                    "text": "여행자보험 보장내용",
                    "score": 0.8,
                    "version_date": old_date,  # 오래된 문서
                    "doc_type": "공식약관"
                }
            ],
            "intent": "qa",
            "warnings": []
        }
        
        result = verify_refine_node(state)
        
        # 오래된 문서 탐지 확인
        assert result["needs_more_search"] is True
        assert any("오래된 문서" in w for w in result["warnings"])
        assert result["verification_status"] == "fail"
        assert result["next_action"] == "broaden_search"
    
    @patch('graph.nodes.verify_refine._load_policies')
    def test_source_weight_assignment_integration(self, mock_load_policies):
        """출처 신뢰도 가중치 할당 통합 테스트"""
        mock_load_policies.return_value = {
            "legal": {"disclaimer": "본 답변은 참고용 정보입니다."},
            "answer": {"min_citations": 1, "min_context": 1},
            "intent_requirements": {
                "qa": {"min_context": 1, "min_citations": 1, "min_insurers": 1}
            },
            "quality": {"min_score": 0.3, "max_age_days": 365},
            "source_weights": {"공식약관": 1.0, "공지": 0.9, "안내": 0.8, "기타": 0.7}
        }
        
        state = {
            "refined": [
                {
                    "doc_id": "doc1",
                    "page": 1,
                    "version": "v1",
                    "insurer": "DB손해보험",
                    "text": "여행자보험 보장내용",
                    "score": 0.8,
                    "version_date": "2025-01-01",
                    "doc_type": "공식약관"
                },
                {
                    "doc_id": "doc2",
                    "page": 1,
                    "version": "v1",
                    "insurer": "삼성화재",
                    "text": "여행자보험 보험료",
                    "score": 0.7,
                    "version_date": "2025-01-01",
                    "doc_type": "안내"
                }
            ],
            "intent": "qa",
            "warnings": []
        }
        
        result = verify_refine_node(state)
        
        # 출처 신뢰도 가중치 확인
        assert len(result["refined"]) == 2
        assert result["refined"][0]["source_weight"] == 1.0  # 공식약관
        assert result["refined"][1]["source_weight"] == 0.8  # 안내
        
        # 인용에서도 가중치 확인
        assert result["citations"][0]["source_weight"] == 1.0
        assert result["citations"][1]["source_weight"] == 0.8
    
    @patch('graph.nodes.verify_refine._load_policies')
    def test_pii_masking_integration(self, mock_load_policies):
        """PII 마스킹 통합 테스트"""
        mock_load_policies.return_value = {
            "legal": {"disclaimer": "본 답변은 참고용 정보입니다."},
            "answer": {"min_citations": 1, "min_context": 1},
            "intent_requirements": {
                "qa": {"min_context": 1, "min_citations": 1, "min_insurers": 1}
            },
            "quality": {"min_score": 0.3, "max_age_days": 365},
            "source_weights": {"공식약관": 1.0, "공지": 0.9, "안내": 0.8, "기타": 0.7}
        }
        
        state = {
            "refined": [
                {
                    "doc_id": "doc1",
                    "page": 1,
                    "version": "v1",
                    "insurer": "DB손해보험",
                    "text": "고객 전화번호 010-1234-5678과 주민번호 123456-1234567입니다",
                    "score": 0.8,
                    "version_date": "2025-01-01",
                    "doc_type": "공식약관"
                }
            ],
            "intent": "qa",
            "warnings": []
        }
        
        result = verify_refine_node(state)
        
        # PII 마스킹 확인
        snippet = result["citations"][0]["snippet"]
        assert "XXX-XXXX-XXXX" in snippet
        assert "XXXXXX-XXXXXXX" in snippet
        assert "010-1234-5678" not in snippet
        assert "123456-1234567" not in snippet
    
    @patch('graph.nodes.verify_refine._load_policies')
    def test_metrics_generation_integration(self, mock_load_policies):
        """메트릭 생성 통합 테스트"""
        mock_load_policies.return_value = {
            "legal": {"disclaimer": "본 답변은 참고용 정보입니다."},
            "answer": {"min_citations": 1, "min_context": 1},
            "intent_requirements": {
                "qa": {"min_context": 1, "min_citations": 1, "min_insurers": 1}
            },
            "quality": {"min_score": 0.3, "max_age_days": 365},
            "source_weights": {"공식약관": 1.0, "공지": 0.9, "안내": 0.8, "기타": 0.7}
        }
        
        state = {
            "refined": [
                {
                    "doc_id": "doc1",
                    "page": 1,
                    "version": "v1",
                    "insurer": "DB손해보험",
                    "text": "여행자보험 보장내용",
                    "score": 0.8,
                    "version_date": "2025-01-01",
                    "doc_type": "공식약관"
                },
                {
                    "doc_id": "doc2",
                    "page": 1,
                    "version": "v1",
                    "insurer": "삼성화재",
                    "text": "여행자보험 보험료",
                    "score": 0.7,
                    "version_date": "2025-01-01",
                    "doc_type": "공식약관"
                }
            ],
            "intent": "qa",
            "warnings": []
        }
        
        result = verify_refine_node(state)
        
        # 메트릭 확인
        metrics = result["metrics"]
        assert metrics["total_documents"] == 2
        assert metrics["unique_insurers"] == 2
        assert metrics["avg_score"] == 0.75
        assert "warning_counts" in metrics
    
    @patch('graph.nodes.verify_refine._load_policies')
    def test_policy_schema_validation_integration(self, mock_load_policies):
        """정책 스키마 검증 통합 테스트"""
        # 잘못된 정책 설정 (필수 키 누락)
        mock_load_policies.return_value = {
            "legal": {"disclaimer": "테스트"}
            # answer 섹션 누락
        }
        
        state = {
            "refined": [
                {
                    "doc_id": "doc1",
                    "page": 1,
                    "version": "v1",
                    "insurer": "DB손해보험",
                    "text": "여행자보험 보장내용",
                    "score": 0.8,
                    "version_date": "2025-01-01",
                    "doc_type": "공식약관"
                }
            ],
            "intent": "qa",
            "warnings": []
        }
        
        result = verify_refine_node(state)
        
        # 스키마 검증 경고 확인
        assert any("필수 정책 키 누락" in w for w in result["warnings"])
        assert any("answer 섹션 필수 키 누락" in w for w in result["warnings"])
    
    @patch('graph.nodes.verify_refine._load_policies')
    def test_performance_benchmark(self, mock_load_policies):
        """성능 벤치마크 테스트"""
        mock_load_policies.return_value = {
            "legal": {"disclaimer": "본 답변은 참고용 정보입니다."},
            "answer": {"min_citations": 1, "min_context": 1},
            "intent_requirements": {
                "qa": {"min_context": 1, "min_citations": 1, "min_insurers": 1}
            },
            "quality": {"min_score": 0.3, "max_age_days": 365},
            "source_weights": {"공식약관": 1.0, "공지": 0.9, "안내": 0.8, "기타": 0.7}
        }
        
        # 대량의 문서로 성능 테스트
        large_refined = []
        for i in range(100):
            large_refined.append({
                "doc_id": f"doc{i}",
                "page": 1,
                "version": "v1",
                "insurer": f"보험사{i % 5}",
                "text": f"여행자보험 문서 {i}",
                "score": 0.8 - (i % 10) * 0.05,
                "version_date": "2025-01-01",
                "doc_type": "공식약관"
            })
        
        state = {
            "refined": large_refined,
            "intent": "qa",
            "warnings": []
        }
        
        start_time = time.time()
        result = verify_refine_node(state)
        end_time = time.time()
        
        # 성능 확인
        processing_time = end_time - start_time
        assert processing_time < 1.0  # 1초 이내 처리
        
        # 결과 확인
        assert result["verification_status"] == "pass"
        assert len(result["citations"]) == 100
        assert result["metrics"]["total_documents"] == 100
        assert result["metrics"]["unique_insurers"] == 5
    
    @patch('graph.nodes.verify_refine._load_policies')
    def test_complex_scenario_integration(self, mock_load_policies):
        """복잡한 시나리오 통합 테스트"""
        mock_load_policies.return_value = {
            "legal": {"disclaimer": "본 답변은 참고용 정보입니다."},
            "answer": {"min_citations": 1, "min_context": 1},
            "intent_requirements": {
                "compare": {"min_context": 3, "min_citations": 3, "min_insurers": 2}
            },
            "quality": {"min_score": 0.3, "max_age_days": 365},
            "source_weights": {"공식약관": 1.0, "공지": 0.9, "안내": 0.8, "기타": 0.7}
        }
        
        # 복잡한 시나리오: 상충, 중복, 낮은 스코어, 오래된 문서가 모두 포함
        old_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
        
        state = {
            "refined": [
                # 상충 문서 1
                {
                    "doc_id": "doc1",
                    "page": 1,
                    "version": "v1",
                    "insurer": "DB손해보험",
                    "text": "의료비 보장 한도는 1억원입니다",
                    "score": 0.8,
                    "version_date": "2025-01-01",
                    "doc_type": "공식약관"
                },
                # 상충 문서 2
                {
                    "doc_id": "doc2",
                    "page": 1,
                    "version": "v1",
                    "insurer": "삼성화재",
                    "text": "의료비 보장 한도는 5천만원입니다",
                    "score": 0.8,
                    "version_date": "2025-01-01",
                    "doc_type": "공식약관"
                },
                # 중복 문서
                {
                    "doc_id": "doc1",
                    "page": 1,
                    "version": "v1",
                    "insurer": "DB손해보험",
                    "text": "의료비 보장 한도는 1억원입니다",
                    "score": 0.8,
                    "version_date": "2025-01-01",
                    "doc_type": "공식약관"
                },
                # 낮은 스코어 문서
                {
                    "doc_id": "doc3",
                    "page": 1,
                    "version": "v1",
                    "insurer": "현대해상",
                    "text": "여행자보험 보험료",
                    "score": 0.1,
                    "version_date": "2025-01-01",
                    "doc_type": "공식약관"
                },
                # 오래된 문서
                {
                    "doc_id": "doc4",
                    "page": 1,
                    "version": "v1",
                    "insurer": "KB손해보험",
                    "text": "여행자보험 가입방법",
                    "score": 0.8,
                    "version_date": old_date,
                    "doc_type": "공식약관"
                }
            ],
            "intent": "compare",
            "warnings": []
        }
        
        result = verify_refine_node(state)
        
        # 복합적인 문제들 확인
        assert result["verification_status"] == "fail"
        assert result["next_action"] == "broaden_search"
        assert result["needs_more_search"] is True
        
        # 다양한 경고 확인
        warnings = result["warnings"]
        assert any("상충 탐지" in w for w in warnings)
        assert any("중복 문서 제거" in w for w in warnings)
        assert any("낮은 스코어" in w for w in warnings)
        assert any("오래된 문서" in w for w in warnings)
        
        # 메트릭 확인
        metrics = result["metrics"]
        assert metrics["total_documents"] == 4  # 중복 제거 후
        assert "coverage_conflict" in metrics["warning_counts"]
        assert "duplicate_document" in metrics["warning_counts"]
        assert "low_score" in metrics["warning_counts"]
        assert "outdated_document" in metrics["warning_counts"]
