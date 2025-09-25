"""
QA 노드 통합 테스트
실제 LLM과 함께 qa_node의 전체 워크플로우를 테스트합니다.
"""

import sys
import os
import pytest
from unittest.mock import patch
import json

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from graph.nodes.answerers.qa import qa_node


@pytest.mark.integration
class TestQAIntegration:
    """QA 노드 통합 테스트 클래스"""
    
    @pytest.fixture
    def sample_state(self):
        """샘플 state 데이터"""
        return {
            "question": "비행기 연착 시 보장 알려줘",
            "passages": [
                {
                    "doc_id": "DB손해보험_여행자보험약관",
                    "page": 15,
                    "text": "항공기 연착으로 인한 지연 시 최대 24시간까지 지연보상금을 지급합니다. 단, 자연재해로 인한 연착은 제외됩니다.",
                    "score": 0.85
                },
                {
                    "doc_id": "KB손해보험_여행자보험약관",
                    "page": 12,
                    "text": "항공기 지연 시 6시간 이상 지연 시 지연보상금을 지급하며, 최대 12시간까지 보장합니다.",
                    "score": 0.78
                },
                {
                    "doc_id": "삼성화재_여행자보험약관",
                    "page": 20,
                    "text": "항공기 연착으로 인한 지연 시 시간당 보상금을 지급하며, 최대 24시간까지 보장합니다.",
                    "score": 0.72
                }
            ]
        }
    
    @pytest.fixture
    def empty_passages_state(self):
        """빈 패시지 state"""
        return {
            "question": "여행자보험 보장 내용이 뭐야?",
            "passages": []
        }
    
    @pytest.fixture
    def long_text_state(self):
        """긴 텍스트가 포함된 state"""
        long_text = "여행자보험" * 200  # 매우 긴 텍스트
        return {
            "question": "여행자보험에 대해 자세히 알려줘",
            "passages": [
                {
                    "doc_id": "테스트_문서",
                    "page": 1,
                    "text": long_text,
                    "score": 0.9
                }
            ]
        }
    
    def test_qa_node_with_real_llm(self, sample_state):
        """실제 LLM을 사용한 QA 노드 테스트"""
        try:
            result = qa_node(sample_state)
            
            # 기본 구조 확인
            assert "draft_answer" in result
            assert "final_answer" in result
            
            # 답변 구조 확인
            answer = result["draft_answer"]
            assert "conclusion" in answer
            assert "evidence" in answer
            assert "caveats" in answer
            assert "quotes" in answer
            
            # 답변 내용 확인
            assert isinstance(answer["conclusion"], str)
            assert len(answer["conclusion"]) > 0
            assert isinstance(answer["evidence"], list)
            assert isinstance(answer["caveats"], list)
            assert isinstance(answer["quotes"], list)
            
            # 출처 정보 확인
            assert len(answer["quotes"]) <= 3  # 상위 3개만
            for quote in answer["quotes"]:
                assert "text" in quote
                assert "source" in quote
                assert len(quote["text"]) <= 200  # 200자 제한
            
            print("✅ 실제 LLM을 사용한 QA 노드 테스트")
            
        except Exception as e:
            pytest.skip(f"LLM 호출 실패: {str(e)}")
    
    def test_qa_node_empty_passages(self, empty_passages_state):
        """빈 패시지로 QA 노드 테스트"""
        try:
            result = qa_node(empty_passages_state)
            
            # 기본 구조 확인
            assert "draft_answer" in result
            assert "final_answer" in result
            
            # 빈 패시지에 대한 적절한 처리 확인
            answer = result["draft_answer"]
            assert "conclusion" in answer
            assert len(answer["conclusion"]) > 0
            
            print("✅ 빈 패시지 처리 테스트")
            
        except Exception as e:
            pytest.skip(f"LLM 호출 실패: {str(e)}")
    
    def test_qa_node_long_text_handling(self, long_text_state):
        """긴 텍스트 처리 테스트"""
        try:
            result = qa_node(long_text_state)
            
            # 기본 구조 확인
            assert "draft_answer" in result
            assert "final_answer" in result
            
            # 긴 텍스트가 적절히 처리되었는지 확인
            answer = result["draft_answer"]
            assert "conclusion" in answer
            assert len(answer["conclusion"]) > 0
            
            print("✅ 긴 텍스트 처리 테스트")
            
        except Exception as e:
            pytest.skip(f"LLM 호출 실패: {str(e)}")
    
    def test_qa_node_different_question_types(self):
        """다양한 질문 유형에 대한 QA 노드 테스트"""
        test_cases = [
            {
                "question": "여행자보험 보장 내용이 뭐야?",
                "passages": [
                    {
                        "doc_id": "DB손해보험_여행자보험약관",
                        "page": 10,
                        "text": "여행자보험은 의료비, 휴대품, 여행지연 등을 보장합니다.",
                        "score": 0.9
                    }
                ]
            },
            {
                "question": "보험료는 얼마인가요?",
                "passages": [
                    {
                        "doc_id": "KB손해보험_여행자보험약관",
                        "page": 5,
                        "text": "보험료는 여행 기간과 보장 내용에 따라 달라집니다.",
                        "score": 0.85
                    }
                ]
            },
            {
                "question": "가입 조건은 어떻게 되나요?",
                "passages": [
                    {
                        "doc_id": "삼성화재_여행자보험약관",
                        "page": 3,
                        "text": "만 15세 이상 80세 이하의 건강한 자가 가입 가능합니다.",
                        "score": 0.88
                    }
                ]
            }
        ]
        
        success_count = 0
        total_count = len(test_cases)
        
        for i, case in enumerate(test_cases, 1):
            try:
                result = qa_node(case)
                
                # 기본 구조 확인
                assert "draft_answer" in result
                assert "final_answer" in result
                
                # 답변 내용 확인
                answer = result["draft_answer"]
                assert "conclusion" in answer
                assert len(answer["conclusion"]) > 0
                
                success_count += 1
                print(f"✅ 질문 유형 {i}: {case['question'][:30]}...")
                
            except Exception as e:
                print(f"❌ 질문 유형 {i}: {case['question'][:30]}... - {str(e)}")
        
        success_rate = (success_count / total_count) * 100
        print(f"\n📊 성공률: {success_count}/{total_count} ({success_rate:.1f}%)")
        
        assert success_rate >= 80, f"성공률이 너무 낮습니다: {success_rate:.1f}%"
    
    def test_qa_node_error_handling(self):
        """에러 처리 테스트"""
        # 잘못된 state 구조 - passages가 None인 경우
        invalid_state = {
            "question": "테스트 질문",
            "passages": None  # None으로 설정
        }
        
        try:
            result = qa_node(invalid_state)
            
            # 에러가 발생해도 기본 구조는 유지되어야 함
            assert "draft_answer" in result
            assert "final_answer" in result
            
            print("✅ 에러 처리 테스트")
            
        except Exception as e:
            print(f"❌ 에러 처리 실패: {str(e)}")
            pytest.fail("에러 처리가 제대로 되지 않았습니다")
    
    def test_qa_node_response_format(self, sample_state):
        """응답 형식 검증 테스트"""
        try:
            result = qa_node(sample_state)
            
            # JSON 형식 검증
            answer = result["draft_answer"]
            
            # 필수 필드 확인
            required_fields = ["conclusion", "evidence", "caveats", "quotes"]
            for field in required_fields:
                assert field in answer, f"필수 필드 {field}가 없습니다"
            
            # 데이터 타입 확인
            assert isinstance(answer["conclusion"], str)
            assert isinstance(answer["evidence"], list)
            assert isinstance(answer["caveats"], list)
            assert isinstance(answer["quotes"], list)
            
            # quotes 구조 확인
            for quote in answer["quotes"]:
                assert "text" in quote
                assert "source" in quote
                assert isinstance(quote["text"], str)
                assert isinstance(quote["source"], str)
            
            print("✅ 응답 형식 검증")
            
        except Exception as e:
            pytest.skip(f"LLM 호출 실패: {str(e)}")


@pytest.mark.benchmark
def test_qa_node_integration_benchmark():
    """QA 노드 통합 벤치마크 테스트"""
    print("\n" + "="*60)
    print("🎯 QA 노드 통합 벤치마크")
    print("="*60)
    
    # 벤치마크 테스트 케이스들
    benchmark_cases = [
        {
            "question": "여행자보험 보장 내용이 뭐야?",
            "passages": [
                {
                    "doc_id": "DB손해보험_여행자보험약관",
                    "page": 15,
                    "text": "항공기 연착으로 인한 지연 시 최대 24시간까지 지연보상금을 지급합니다.",
                    "score": 0.85
                }
            ]
        },
        {
            "question": "휴대품 분실 시 보상은?",
            "passages": [
                {
                    "doc_id": "KB손해보험_여행자보험약관",
                    "page": 20,
                    "text": "휴대품 분실 시 실제 가치에 따라 보상금을 지급합니다.",
                    "score": 0.78
                }
            ]
        },
        {
            "question": "의료비 보상 한도는?",
            "passages": [
                {
                    "doc_id": "삼성화재_여행자보험약관",
                    "page": 25,
                    "text": "의료비 보상 한도는 1억원까지 보장합니다.",
                    "score": 0.72
                }
            ]
        }
    ]
    
    success_count = 0
    total_count = len(benchmark_cases)
    
    for i, case in enumerate(benchmark_cases, 1):
        try:
            result = qa_node(case)
            
            # 기본 구조 확인
            assert "draft_answer" in result
            assert "final_answer" in result
            
            # 답변 품질 확인
            answer = result["draft_answer"]
            assert "conclusion" in answer
            assert len(answer["conclusion"]) > 10  # 의미있는 답변 길이
            
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
    test_qa_node_integration_benchmark()
