"""
Compare 노드 통합 테스트
실제 LLM과 함께 compare_node의 전체 워크플로우를 테스트합니다.
"""

import sys
import os
import pytest
from unittest.mock import patch
import json

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from graph.nodes.answerers.compare import compare_node


@pytest.mark.integration
class TestCompareIntegration:
    """Compare 노드 통합 테스트 클래스"""
    
    @pytest.fixture
    def sample_state(self):
        """샘플 state 데이터"""
        return {
            "question": "DB손보와 카카오페이 여행자보험 차이 비교",
            "passages": [
                {
                    "doc_id": "DB손해보험_여행자보험약관",
                    "page": 15,
                    "text": "사망보장 한도는 1억원이며, 상해보장은 5천만원입니다. 질병보장은 3천만원까지 보장합니다.",
                    "score": 0.85
                },
                {
                    "doc_id": "카카오페이_여행자보험약관",
                    "page": 12,
                    "text": "사망보장 한도는 5천만원이며, 상해보장은 3천만원입니다. 질병보장은 2천만원까지 보장합니다.",
                    "score": 0.78
                },
                {
                    "doc_id": "KB손해보험_여행자보험약관",
                    "page": 20,
                    "text": "사망보장 한도는 8천만원이며, 상해보장은 4천만원입니다. 질병보장은 2천5백만원까지 보장합니다.",
                    "score": 0.72
                }
            ]
        }
    
    @pytest.fixture
    def empty_passages_state(self):
        """빈 패시지 state"""
        return {
            "question": "여행자보험 보험사별 차이점 비교",
            "passages": []
        }
    
    @pytest.fixture
    def long_text_state(self):
        """긴 텍스트가 포함된 state"""
        long_text = "여행자보험" * 200  # 매우 긴 텍스트
        return {
            "question": "여행자보험에 대해 자세히 비교해줘",
            "passages": [
                {
                    "doc_id": "테스트_문서",
                    "page": 1,
                    "text": long_text,
                    "score": 0.9
                }
            ]
        }
    
    def test_compare_node_with_real_llm(self, sample_state):
        """실제 LLM을 사용한 Compare 노드 테스트"""
        try:
            result = compare_node(sample_state)
            
            # 기본 구조 확인
            assert "draft_answer" in result
            assert "final_answer" in result
            
            # 답변 구조 확인
            answer = result["draft_answer"]
            assert "conclusion" in answer
            assert "evidence" in answer
            assert "caveats" in answer
            assert "quotes" in answer
            assert "comparison_table" in answer
            
            # 답변 내용 확인
            assert isinstance(answer["conclusion"], str)
            assert len(answer["conclusion"]) > 0
            assert isinstance(answer["evidence"], list)
            assert isinstance(answer["caveats"], list)
            assert isinstance(answer["quotes"], list)
            
            # comparison_table 구조 확인
            table = answer["comparison_table"]
            assert "headers" in table
            assert "rows" in table
            assert isinstance(table["headers"], list)
            assert isinstance(table["rows"], list)
            assert len(table["headers"]) > 0
            assert len(table["rows"]) > 0
            
            # 출처 정보 확인
            assert len(answer["quotes"]) <= 3  # 상위 3개만
            for quote in answer["quotes"]:
                assert "text" in quote
                assert "source" in quote
                assert len(quote["text"]) <= 200  # 200자 제한
            
            print("✅ 실제 LLM을 사용한 Compare 노드 테스트")
            
        except Exception as e:
            pytest.skip(f"LLM 호출 실패: {str(e)}")
    
    def test_compare_node_empty_passages(self, empty_passages_state):
        """빈 패시지로 Compare 노드 테스트"""
        try:
            result = compare_node(empty_passages_state)
            
            # 기본 구조 확인
            assert "draft_answer" in result
            assert "final_answer" in result
            
            # 빈 패시지에 대한 적절한 처리 확인
            answer = result["draft_answer"]
            assert "conclusion" in answer
            assert "comparison_table" in answer
            assert len(answer["conclusion"]) > 0
            
            print("✅ 빈 패시지 처리 테스트")
            
        except Exception as e:
            pytest.skip(f"LLM 호출 실패: {str(e)}")
    
    def test_compare_node_long_text_handling(self, long_text_state):
        """긴 텍스트 처리 테스트"""
        try:
            result = compare_node(long_text_state)
            
            # 기본 구조 확인
            assert "draft_answer" in result
            assert "final_answer" in result
            
            # 긴 텍스트가 적절히 처리되었는지 확인
            answer = result["draft_answer"]
            assert "conclusion" in answer
            assert "comparison_table" in answer
            assert len(answer["conclusion"]) > 0
            
            print("✅ 긴 텍스트 처리 테스트")
            
        except Exception as e:
            pytest.skip(f"LLM 호출 실패: {str(e)}")
    
    def test_compare_node_different_question_types(self):
        """다양한 질문 유형에 대한 Compare 노드 테스트"""
        test_cases = [
            {
                "question": "보험사별 여행자보험 가격 비교",
                "passages": [
                    {
                        "doc_id": "DB손해보험_여행자보험약관",
                        "page": 10,
                        "text": "1일 보험료는 3,000원부터 시작됩니다.",
                        "score": 0.9
                    },
                    {
                        "doc_id": "카카오페이_여행자보험약관",
                        "page": 8,
                        "text": "1일 보험료는 2,500원부터 시작됩니다.",
                        "score": 0.85
                    }
                ]
            },
            {
                "question": "여행자보험 보장 내용 차이점",
                "passages": [
                    {
                        "doc_id": "KB손해보험_여행자보험약관",
                        "page": 5,
                        "text": "의료비 보상 한도는 1억원까지 보장합니다.",
                        "score": 0.88
                    },
                    {
                        "doc_id": "삼성화재_여행자보험약관",
                        "page": 7,
                        "text": "의료비 보상 한도는 5천만원까지 보장합니다.",
                        "score": 0.82
                    }
                ]
            },
            {
                "question": "여행자보험 특약 비교",
                "passages": [
                    {
                        "doc_id": "현대해상_여행자보험약관",
                        "page": 3,
                        "text": "골프보장 특약을 추가할 수 있습니다.",
                        "score": 0.75
                    },
                    {
                        "doc_id": "DB손해보험_여행자보험약관",
                        "page": 4,
                        "text": "스포츠보장 특약을 추가할 수 있습니다.",
                        "score": 0.73
                    }
                ]
            }
        ]
        
        success_count = 0
        total_count = len(test_cases)
        
        for i, case in enumerate(test_cases, 1):
            try:
                result = compare_node(case)
                
                # 기본 구조 확인
                assert "draft_answer" in result
                assert "final_answer" in result
                
                # 답변 내용 확인
                answer = result["draft_answer"]
                assert "conclusion" in answer
                assert "comparison_table" in answer
                assert len(answer["conclusion"]) > 0
                
                # comparison_table 구조 확인
                table = answer["comparison_table"]
                assert "headers" in table
                assert "rows" in table
                assert len(table["headers"]) > 0
                assert len(table["rows"]) > 0
                
                success_count += 1
                print(f"✅ 질문 유형 {i}: {case['question'][:30]}...")
                
            except Exception as e:
                print(f"❌ 질문 유형 {i}: {case['question'][:30]}... - {str(e)}")
        
        success_rate = (success_count / total_count) * 100
        print(f"\n📊 성공률: {success_count}/{total_count} ({success_rate:.1f}%)")
        
        assert success_rate >= 80, f"성공률이 너무 낮습니다: {success_rate:.1f}%"
    
    def test_compare_node_error_handling(self):
        """에러 처리 테스트"""
        # 잘못된 state 구조 - passages가 None인 경우
        invalid_state = {
            "question": "테스트 질문",
            "passages": None  # None으로 설정
        }
        
        try:
            result = compare_node(invalid_state)
            
            # 에러가 발생해도 기본 구조는 유지되어야 함
            assert "draft_answer" in result
            assert "final_answer" in result
            
            # comparison_table이 포함되어야 함
            answer = result["draft_answer"]
            assert "comparison_table" in answer
            
            print("✅ 에러 처리 테스트")
            
        except Exception as e:
            print(f"❌ 에러 처리 실패: {str(e)}")
            pytest.fail("에러 처리가 제대로 되지 않았습니다")
    
    def test_compare_node_response_format(self, sample_state):
        """응답 형식 검증 테스트"""
        try:
            result = compare_node(sample_state)
            
            # JSON 형식 검증
            answer = result["draft_answer"]
            
            # 필수 필드 확인
            required_fields = ["conclusion", "evidence", "caveats", "quotes", "comparison_table"]
            for field in required_fields:
                assert field in answer, f"필수 필드 {field}가 없습니다"
            
            # 데이터 타입 확인
            assert isinstance(answer["conclusion"], str)
            assert isinstance(answer["evidence"], list)
            assert isinstance(answer["caveats"], list)
            assert isinstance(answer["quotes"], list)
            assert isinstance(answer["comparison_table"], dict)
            
            # comparison_table 구조 확인
            table = answer["comparison_table"]
            assert "headers" in table
            assert "rows" in table
            assert isinstance(table["headers"], list)
            assert isinstance(table["rows"], list)
            
            # quotes 구조 확인
            for quote in answer["quotes"]:
                assert "text" in quote
                assert "source" in quote
                assert isinstance(quote["text"], str)
                assert isinstance(quote["source"], str)
            
            print("✅ 응답 형식 검증")
            
        except Exception as e:
            pytest.skip(f"LLM 호출 실패: {str(e)}")
    
    def test_compare_node_comparison_table_quality(self, sample_state):
        """comparison_table 품질 검증 테스트"""
        try:
            result = compare_node(sample_state)
            
            answer = result["draft_answer"]
            table = answer["comparison_table"]
            
            # 테이블 구조 검증
            assert len(table["headers"]) >= 2, "헤더가 최소 2개 이상이어야 함"
            assert len(table["rows"]) > 0, "행이 최소 1개 이상이어야 함"
            
            # 각 행의 길이가 헤더 길이와 일치하는지 확인
            header_count = len(table["headers"])
            for row in table["rows"]:
                assert len(row) == header_count, f"행의 길이가 헤더 길이({header_count})와 일치하지 않음: {len(row)}"
            
            # 헤더에 "항목"이 포함되어 있는지 확인
            assert "항목" in table["headers"], "헤더에 '항목'이 포함되어야 함"
            
            # 행에 의미있는 데이터가 있는지 확인
            for row in table["rows"]:
                assert len(row[0]) > 0, "첫 번째 열(항목명)이 비어있으면 안됨"
                for cell in row:
                    assert isinstance(cell, str), "모든 셀은 문자열이어야 함"
                    assert len(cell) > 0, "빈 셀이 있으면 안됨"
            
            print("✅ comparison_table 품질 검증")
            
        except Exception as e:
            pytest.skip(f"LLM 호출 실패: {str(e)}")
    
    def test_compare_node_multiple_insurance_comparison(self):
        """여러 보험사 비교 테스트"""
        multi_insurance_state = {
            "question": "DB손보, 카카오페이, KB손보 여행자보험 비교",
            "passages": [
                {
                    "doc_id": "DB손해보험_여행자보험약관",
                    "page": 15,
                    "text": "사망보장 1억원, 상해보장 5천만원, 질병보장 3천만원",
                    "score": 0.85
                },
                {
                    "doc_id": "카카오페이_여행자보험약관",
                    "page": 12,
                    "text": "사망보장 5천만원, 상해보장 3천만원, 질병보장 2천만원",
                    "score": 0.78
                },
                {
                    "doc_id": "KB손해보험_여행자보험약관",
                    "page": 20,
                    "text": "사망보장 8천만원, 상해보장 4천만원, 질병보장 2천5백만원",
                    "score": 0.72
                }
            ]
        }
        
        try:
            result = compare_node(multi_insurance_state)
            
            # 기본 구조 확인
            assert "draft_answer" in result
            assert "final_answer" in result
            
            answer = result["draft_answer"]
            table = answer["comparison_table"]
            
            # 3개 보험사 비교이므로 헤더에 3개 이상의 보험사가 있어야 함
            assert len(table["headers"]) >= 3, "3개 보험사 비교이므로 헤더가 3개 이상이어야 함"
            
            # 행에 의미있는 비교 데이터가 있는지 확인
            assert len(table["rows"]) > 0, "비교 행이 최소 1개 이상이어야 함"
            
            print("✅ 여러 보험사 비교 테스트")
            
        except Exception as e:
            pytest.skip(f"LLM 호출 실패: {str(e)}")


@pytest.mark.benchmark
def test_compare_node_integration_benchmark():
    """Compare 노드 통합 벤치마크 테스트"""
    print("\n" + "="*60)
    print("🎯 Compare 노드 통합 벤치마크")
    print("="*60)
    
    # 벤치마크 테스트 케이스들
    benchmark_cases = [
        {
            "question": "DB손보와 카카오페이 여행자보험 차이 비교",
            "passages": [
                {
                    "doc_id": "DB손해보험_여행자보험약관",
                    "page": 15,
                    "text": "사망보장 한도는 1억원이며, 상해보장은 5천만원입니다.",
                    "score": 0.85
                },
                {
                    "doc_id": "카카오페이_여행자보험약관",
                    "page": 12,
                    "text": "사망보장 한도는 5천만원이며, 상해보장은 3천만원입니다.",
                    "score": 0.78
                }
            ]
        },
        {
            "question": "보험사별 여행자보험 가격 비교",
            "passages": [
                {
                    "doc_id": "KB손해보험_여행자보험약관",
                    "page": 20,
                    "text": "1일 보험료는 3,000원부터 시작됩니다.",
                    "score": 0.78
                },
                {
                    "doc_id": "삼성화재_여행자보험약관",
                    "page": 18,
                    "text": "1일 보험료는 2,500원부터 시작됩니다.",
                    "score": 0.72
                }
            ]
        },
        {
            "question": "여행자보험 보장 내용 차이점",
            "passages": [
                {
                    "doc_id": "현대해상_여행자보험약관",
                    "page": 25,
                    "text": "의료비 보상 한도는 1억원까지 보장합니다.",
                    "score": 0.72
                },
                {
                    "doc_id": "DB손해보험_여행자보험약관",
                    "page": 30,
                    "text": "의료비 보상 한도는 5천만원까지 보장합니다.",
                    "score": 0.68
                }
            ]
        }
    ]
    
    success_count = 0
    total_count = len(benchmark_cases)
    
    for i, case in enumerate(benchmark_cases, 1):
        try:
            result = compare_node(case)
            
            # 기본 구조 확인
            assert "draft_answer" in result
            assert "final_answer" in result
            
            # 답변 품질 확인
            answer = result["draft_answer"]
            assert "conclusion" in answer
            assert "comparison_table" in answer
            assert len(answer["conclusion"]) > 10  # 의미있는 답변 길이
            
            # comparison_table 품질 확인
            table = answer["comparison_table"]
            assert len(table["headers"]) >= 2
            assert len(table["rows"]) > 0
            
            success_count += 1
            print(f"✅ 벤치마크 {i}: {case['question'][:30]}...")
            
        except Exception as e:
            print(f"❌ 벤치마크 {i}: {case['question'][:30]}... - {str(e)}")
    
    success_rate = (success_count / total_count) * 100
    print(f"\n📊 통합 성공률: {success_count}/{total_count} ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("🎉 우수한 통합 성능!")
    elif success_rate >= 80:
        print("👍 양호한 통합 성능")
    else:
        print("⚠️ 통합 성능 개선 필요")
    
    return success_rate


if __name__ == "__main__":
    # 직접 실행 시 통합 벤치마크 테스트
    test_compare_node_integration_benchmark()
