"""
Rank Filter 통합 테스트
실제 환경에서 BGE 리랭커와 배치 정규화 기능을 테스트합니다.
"""

import sys
import os
import pytest
import time
from unittest.mock import patch, Mock

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from graph.nodes.rank_filter import rank_filter_node
from graph.nodes.search import search_node


@pytest.mark.integration
class TestRankFilterIntegration:
    """Rank Filter 통합 테스트 클래스"""
    
    def test_full_pipeline_with_mock_search(self):
        """전체 파이프라인 테스트 (검색 모킹)"""
        # 검색 결과 모킹
        mock_passages = [
            {
                "text": "여행자보험의 보장내용은 상해보장, 질병보장, 휴대품보장을 포함합니다. 상해보장은 해외여행 중 발생한 상해로 인한 의료비를 지급합니다.",
                "title": "여행자보험 보장내용",
                "score": 0.8,
                "doc_id": "doc1",
                "page": 1,
                "insurer": "DB손해보험"
            },
            {
                "text": "여행자보험 보험료는 연령과 여행지에 따라 달라집니다. 20대의 경우 월 1만원 내외입니다.",
                "title": "여행자보험 보험료",
                "score": 0.6,
                "doc_id": "doc2",
                "page": 1,
                "insurer": "삼성화재"
            },
            {
                "text": "여행자보험 가입방법은 온라인, 전화, 대리점을 통해 가능합니다.",
                "title": "여행자보험 가입방법",
                "score": 0.4,
                "doc_id": "doc3",
                "page": 1,
                "insurer": "현대해상"
            }
        ]
        
        state = {
            "passages": mock_passages,
            "question": "여행자보험 보장내용과 보험료에 대해 알려주세요"
        }
        
        # Rank Filter 실행
        result = rank_filter_node(state)
        
        # 결과 검증
        assert "refined" in result
        assert "rank_meta" in result
        assert len(result["refined"]) <= 5
        assert len(result["refined"]) > 0
        
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
        assert meta["mmr_applied"] == True
    
    def test_traditional_rerank_method(self):
        """전통적 리랭크 방법 테스트"""
        state = {
            "passages": [
                {"text": "여행자보험 보장내용", "score": 0.5},
                {"text": "여행자보험 보험료", "score": 0.3}
            ],
            "question": "여행자보험 정보"
        }
        
        result = rank_filter_node(state)
        
        # 메타데이터에서 리랭크 방법 확인
        meta = result["rank_meta"]
        assert "rerank_method" in meta
        assert meta["rerank_method"] == "traditional"
    
    def test_simplified_processing(self):
        """단순화된 처리 테스트"""
        state = {
            "passages": [
                {"text": "여행자보험 보장내용", "score": 0.5},
                {"text": "여행자보험 보험료", "score": 0.3}
            ],
            "question": "여행자보험 정보"
        }
        
        result = rank_filter_node(state)
        
        # 결과 검증
        assert "refined" in result
        assert "rank_meta" in result
        assert len(result["refined"]) <= 5
        
        # 메타데이터 검증
        meta = result["rank_meta"]
        assert meta["rerank_method"] == "traditional"
        assert meta["rerank_applied"] == True
        assert meta["mmr_applied"] == True
    
    def test_performance_with_large_dataset(self):
        """대량 데이터셋 성능 테스트"""
        # 대량의 테스트 데이터 생성
        passages = []
        for i in range(50):  # 50개 문서
            passages.append({
                "text": f"여행자보험 관련 문서 {i}. " * 20,  # 긴 텍스트
                "title": f"문서 {i}",
                "score": 0.1 + (i % 10) * 0.1,
                "doc_id": f"doc{i}",
                "page": 1
            })
        
        state = {
            "passages": passages,
            "question": "여행자보험 보장내용과 보험료에 대해 알려주세요"
        }
        
        # 성능 측정
        start_time = time.time()
        result = rank_filter_node(state)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # 성능 검증 (10초 이내 완료)
        assert processing_time < 10.0
        
        # 결과 검증
        assert "refined" in result
        assert len(result["refined"]) <= 5
        
        # 메타데이터 검증
        meta = result["rank_meta"]
        assert meta["original_count"] == 50
        assert meta["deduped_count"] <= 50
        assert meta["final_count"] <= 5
        
        print(f"처리 시간: {processing_time:.2f}초")
        print(f"처리된 문서 수: {meta['original_count']}")
        print(f"최종 선택된 문서 수: {meta['final_count']}")
    
    def test_basic_processing(self):
        """기본 처리 테스트 (다양성 테스트 대체)"""
        # 기본 문서들
        passages = [
            {
                "text": "여행자보험의 보장내용은 상해보장과 질병보장을 포함합니다. 상해보장은 해외여행 중 발생한 상해로 인한 의료비를 지급합니다.",
                "title": "여행자보험 보장내용",
                "score": 0.8
            },
            {
                "text": "여행자보험 보험료는 연령과 여행지에 따라 달라집니다. 20대의 경우 월 1만원 내외입니다.",
                "title": "여행자보험 보험료",
                "score": 0.6
            }
        ]
        
        state = {
            "passages": passages,
            "question": "여행자보험 정보"
        }
        
        result = rank_filter_node(state)
        
        # 기본 결과 검증
        refined = result["refined"]
        assert len(refined) <= 5
        assert len(refined) > 0, "최종 결과가 비어있습니다"
        
        # 메타데이터 검증
        meta = result["rank_meta"]
        assert meta["rerank_method"] == "traditional"
        assert meta["rerank_applied"] == True
    
    def test_error_handling(self):
        """에러 처리 테스트"""
        # 빈 상태 테스트
        empty_state = {"passages": [], "question": ""}
        result = rank_filter_node(empty_state)
        assert result["refined"] == []
        
        # 잘못된 데이터 테스트
        invalid_state = {
            "passages": [
                {"text": "", "score": 0.5},  # 빈 텍스트
                {"text": "a", "score": 0.3},  # 너무 짧은 텍스트
            ],
            "question": "테스트 질문"
        }
        
        result = rank_filter_node(invalid_state)
        assert "refined" in result
        assert "rank_meta" in result
        
        # 메타데이터에서 처리 과정 확인
        meta = result["rank_meta"]
        assert meta["original_count"] == 2
        assert meta["filtered_count"] <= 2  # 품질 필터링 적용


@pytest.mark.performance
class TestRankFilterPerformance:
    """Rank Filter 성능 테스트"""
    
    def test_benchmark_different_sizes(self):
        """다양한 크기의 데이터셋 성능 벤치마크"""
        sizes = [10, 25, 50, 100]
        results = {}
        
        for size in sizes:
            # 테스트 데이터 생성
            passages = []
            for i in range(size):
                passages.append({
                    "text": f"여행자보험 관련 문서 {i}. " * 10,
                    "title": f"문서 {i}",
                    "score": 0.1 + (i % 10) * 0.1,
                    "doc_id": f"doc{i}",
                    "page": 1
                })
            
            state = {
                "passages": passages,
                "question": "여행자보험 보장내용"
            }
            
            # 성능 측정
            start_time = time.time()
            result = rank_filter_node(state)
            end_time = time.time()
            
            processing_time = end_time - start_time
            results[size] = {
                "time": processing_time,
                "final_count": len(result["refined"]),
                "meta": result["rank_meta"]
            }
            
            print(f"크기 {size}: {processing_time:.2f}초, 최종 {len(result['refined'])}개")
        
        # 성능 검증
        for size, data in results.items():
            assert data["time"] < 15.0  # 15초 이내 완료
            assert data["final_count"] <= 5  # 최대 5개 선택
    
    def test_memory_usage(self):
        """메모리 사용량 테스트"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 대량 데이터 처리
        passages = []
        for i in range(100):
            passages.append({
                "text": f"여행자보험 관련 문서 {i}. " * 50,  # 긴 텍스트
                "title": f"문서 {i}",
                "score": 0.1 + (i % 10) * 0.1,
                "doc_id": f"doc{i}",
                "page": 1
            })
        
        state = {
            "passages": passages,
            "question": "여행자보험 보장내용"
        }
        
        result = rank_filter_node(state)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # 메모리 사용량 검증 (100MB 이내 증가)
        assert memory_increase < 100.0
        
        print(f"메모리 사용량 증가: {memory_increase:.2f}MB")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
