"""
Rank Filter 노드 테스트
BGE 리랭커와 배치 정규화 기능을 테스트합니다.
"""

import sys
import os
import pytest
import math
from unittest.mock import Mock, patch, MagicMock

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from graph.nodes.rank_filter import (
    rank_filter_node,
    _dedup,
    _rerank_with_advanced_scoring,
    _apply_mmr,
    _calculate_similarity,
    _quality_filter,
    _sort_by_score
)


@pytest.mark.unit
class TestRankFilter:
    """Rank Filter 노드 테스트 클래스"""
    
    def test_dedup_function(self):
        """중복 제거 기능 테스트"""
        passages = [
            {"text": "여행자보험 보장내용", "score": 0.8},
            {"text": "여행자보험 보장내용", "score": 0.9},  # 중복
            {"text": "여행자보험 보험료", "score": 0.7},
            {"text": "", "score": 0.5},  # 빈 텍스트
        ]
        
        result = _dedup(passages)
        
        assert len(result) == 2
        assert result[0]["text"] == "여행자보험 보장내용"
        assert result[1]["text"] == "여행자보험 보험료"
    
    def test_basic_functionality(self):
        """기본 기능 테스트"""
        # 간단한 기능 테스트
        passages = [
            {"text": "여행자보험 보장내용", "score": 0.5},
            {"text": "여행자보험 보험료", "score": 0.3}
        ]
        
        result = _dedup(passages)
        assert len(result) == 2
    
    def test_rerank_with_advanced_scoring(self):
        """고급 점수 계산 리랭크 테스트"""
        passages = [
            {
                "text": "여행자보험 보장내용은 상해보장과 질병보장을 포함합니다",
                "title": "여행자보험 보장내용",
                "score": 0.5
            },
            {
                "text": "보험료는 연령과 여행지에 따라 달라집니다",
                "title": "보험료 정보",
                "score": 0.3
            }
        ]
        question = "여행자보험 보장내용"
        
        result = _rerank_with_advanced_scoring(passages, question)
        
        assert len(result) == 2
        assert all("rerank_score" in p for p in result)
        assert all("keyword_matches" in p for p in result)
        
        # 첫 번째 문서가 더 높은 점수를 받아야 함 (질문과 더 관련성 높음)
        assert result[0]["score"] >= result[1]["score"]
    
    def test_apply_mmr(self):
        """MMR 다양성 확보 테스트"""
        passages = [
            {"text": "여행자보험 보장내용 A", "score": 0.9},
            {"text": "여행자보험 보장내용 B", "score": 0.8},  # 유사한 내용
            {"text": "여행자보험 보험료 정보", "score": 0.7},  # 다른 주제
            {"text": "여행자보험 가입방법", "score": 0.6},   # 다른 주제
        ]
        question = "여행자보험 정보"
        
        result = _apply_mmr(passages, question, lambda_param=0.7)
        
        assert len(result) <= 5  # 최대 5개
        assert len(result) > 0
        
        # 다양성 확인: 유사한 내용이 연속으로 나오지 않아야 함
        texts = [p["text"] for p in result]
        assert "여행자보험 보장내용 A" in texts or "여행자보험 보장내용 B" in texts
        assert "여행자보험 보험료 정보" in texts or "여행자보험 가입방법" in texts
    
    def test_calculate_similarity(self):
        """문서 유사도 계산 테스트"""
        doc1 = {"text": "여행자보험 보장내용과 보험료"}
        doc2 = {"text": "여행자보험 보장내용과 가입방법"}
        doc3 = {"text": "자동차보험 보장내용"}
        
        # 유사한 문서들
        similarity_high = _calculate_similarity(doc1, doc2)
        assert 0 < similarity_high < 1
        
        # 다른 주제 문서들
        similarity_low = _calculate_similarity(doc1, doc3)
        assert similarity_low < similarity_high
        
        # 빈 텍스트 처리
        empty_doc = {"text": ""}
        similarity_empty = _calculate_similarity(doc1, empty_doc)
        assert similarity_empty == 0.0
    
    def test_quality_filter(self):
        """품질 필터링 테스트"""
        passages = [
            {"text": "적절한 길이의 문서입니다. " * 20, "score": 0.8},  # 좋은 품질
            {"text": "짧음", "score": 0.7},  # 너무 짧음
            {"text": "긴 문서입니다. " * 500, "score": 0.6},  # 너무 김
            {"text": "적절한 길이의 문서입니다. " * 20, "score": 0.05},  # 낮은 점수
            {"text": "적절한 길이의 문서입니다. " * 20, "score": 0.3},  # 통과
        ]
        
        result = _quality_filter(passages)
        
        # 품질 기준을 통과한 문서만 남아야 함
        assert len(result) == 2
        assert all(p["score"] >= 0.1 for p in result)
        assert all(50 <= len(p["text"]) <= 2000 for p in result)
    
    def test_sort_by_score(self):
        """점수 기준 정렬 테스트"""
        passages = [
            {"text": "문서1", "score": 0.3},
            {"text": "문서2", "score": 0.8},
            {"text": "문서3", "score": 0.5},
        ]
        
        result = _sort_by_score(passages)
        
        assert result[0]["score"] == 0.8
        assert result[1]["score"] == 0.5
        assert result[2]["score"] == 0.3
    
    def test_traditional_rerank_only(self):
        """전통적 리랭크만 사용하는 테스트"""
        passages = [
            {"text": "여행자보험 보장내용은 상해보장과 질병보장을 포함합니다", "score": 0.5},
            {"text": "여행자보험 보험료는 연령과 여행지에 따라 달라집니다", "score": 0.3},
            {"text": "여행자보험 가입방법은 온라인과 전화로 가능합니다", "score": 0.7},
        ]
        question = "여행자보험 보장내용"
        
        result = _rerank_with_advanced_scoring(passages, question)
        
        assert len(result) == 3
        assert all("rerank_score" in p for p in result)
        assert all("keyword_matches" in p for p in result)
    
    def test_rank_filter_node_integration(self):
        """Rank Filter 노드 통합 테스트"""
        state = {
            "passages": [
                {
                    "text": "여행자보험 보장내용은 상해보장과 질병보장을 포함합니다",
                    "title": "여행자보험 보장내용",
                    "score": 0.5
                },
                {
                    "text": "여행자보험 보장내용은 상해보장과 질병보장을 포함합니다",  # 중복
                    "title": "여행자보험 보장내용",
                    "score": 0.6
                },
                {
                    "text": "보험료는 연령과 여행지에 따라 달라집니다",
                    "title": "보험료 정보",
                    "score": 0.3
                }
            ],
            "question": "여행자보험 보장내용"
        }
        
        result = rank_filter_node(state)
        
        # 결과 검증
        assert "refined" in result
        assert "rank_meta" in result
        assert len(result["refined"]) <= 5
        
        # 메타데이터 검증
        meta = result["rank_meta"]
        assert "original_count" in meta
        assert "final_count" in meta
        assert "rerank_method" in meta
        assert "rerank_applied" in meta
        assert meta["rerank_applied"] == True


@pytest.mark.integration
class TestRankFilterIntegration:
    """Rank Filter 통합 테스트"""
    
    def test_rank_filter_with_real_data(self):
        """실제 데이터로 Rank Filter 테스트"""
        # 실제 보험 문서 데이터 시뮬레이션
        passages = [
            {
                "text": "여행자보험의 보장내용은 상해보장, 질병보장, 휴대품보장, 여행지연보장 등을 포함합니다. 상해보장은 해외여행 중 발생한 상해로 인한 의료비와 사망보험금을 지급합니다.",
                "title": "여행자보험 보장내용",
                "score": 0.7,
                "doc_id": "doc1",
                "page": 1
            },
            {
                "text": "여행자보험 보험료는 연령, 여행지, 보장기간, 보장금액에 따라 달라집니다. 20대의 경우 월 1만원 내외이며, 50대는 월 2-3만원 정도입니다.",
                "title": "여행자보험 보험료",
                "score": 0.5,
                "doc_id": "doc2", 
                "page": 1
            },
            {
                "text": "여행자보험 가입방법은 온라인, 전화, 대리점을 통해 가능합니다. 온라인 가입이 가장 간편하며 24시간 가능합니다.",
                "title": "여행자보험 가입방법",
                "score": 0.4,
                "doc_id": "doc3",
                "page": 1
            }
        ]
        
        state = {
            "passages": passages,
            "question": "여행자보험 보장내용과 보험료에 대해 알려주세요"
        }
        
        result = rank_filter_node(state)
        
        # 결과 검증
        assert "refined" in result
        assert len(result["refined"]) > 0
        assert len(result["refined"]) <= 5
        
        # 품질 검증
        for passage in result["refined"]:
            assert "text" in passage
            assert "score" in passage
            assert passage["score"] >= 0.0
            assert len(passage["text"]) > 0
        
        # 메타데이터 검증
        meta = result["rank_meta"]
        assert meta["original_count"] == 3
        assert meta["final_count"] <= 5
        assert meta["rerank_applied"] == True


@pytest.mark.performance
class TestRankFilterPerformance:
    """Rank Filter 성능 테스트"""
    
    def test_large_batch_processing(self):
        """대량 데이터 처리 성능 테스트"""
        # 대량의 테스트 데이터 생성
        passages = []
        for i in range(100):
            passages.append({
                "text": f"여행자보험 관련 문서 {i}. " * 10,
                "title": f"문서 {i}",
                "score": 0.1 + (i % 10) * 0.1
            })
        
        state = {
            "passages": passages,
            "question": "여행자보험 보장내용"
        }
        
        import time
        start_time = time.time()
        result = rank_filter_node(state)
        end_time = time.time()
        
        # 성능 검증 (5초 이내 완료)
        assert end_time - start_time < 5.0
        
        # 결과 검증
        assert "refined" in result
        assert len(result["refined"]) <= 5
        
        # 메타데이터에서 처리 통계 확인
        meta = result["rank_meta"]
        assert meta["original_count"] == 100
        assert meta["deduped_count"] <= 100  # 중복 제거 (같을 수도 있음)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
