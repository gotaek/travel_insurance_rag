#!/usr/bin/env python3
"""
간단한 평가 테스트 스크립트
- 단일 질문으로 시스템 동작 확인
- 기본 지표 계산
"""

import json
from pathlib import Path
from typing import Dict, Any

from graph.builder import build_graph
from eval.recall_at_k import recall_at_k
from eval.faithfulness import simple_faithfulness

def test_single_question(question: str, gold_doc_ids: list = None) -> Dict[str, Any]:
    """단일 질문 테스트"""
    print(f"🔍 테스트 질문: {question}")
    
    # 그래프 빌드 및 실행
    g = build_graph()
    state = {"question": question}
    result = g.invoke(state)
    
    # 기본 지표 계산
    recall = recall_at_k(result, gold_doc_ids or [], k=5)
    faithfulness = simple_faithfulness(result)
    
    # 결과 출력
    print(f"✅ Intent: {result.get('intent')}")
    print(f"✅ Passages: {len(result.get('passages', []))}개")
    print(f"✅ Refined: {len(result.get('refined', []))}개")
    print(f"✅ Recall@5: {recall:.3f}")
    print(f"✅ Faithfulness: {faithfulness:.3f}")
    
    answer = result.get("draft_answer", {})
    print(f"✅ 답변: {answer.get('conclusion', '없음')}")
    
    return {
        "question": question,
        "intent": result.get("intent"),
        "recall_at_5": recall,
        "faithfulness": faithfulness,
        "answer": answer.get("conclusion", ""),
        "passages_count": len(result.get("passages", [])),
        "refined_count": len(result.get("refined", []))
    }

def main():
    """메인 함수"""
    print("🚀 간단한 평가 테스트 시작\n")
    
    # 테스트 질문들
    test_questions = [
        {
            "question": "비행기 연착 시 보장 알려줘",
            "gold_doc_ids": ["카카오페이_2025_여행자보험약관요약"]
        },
        {
            "question": "카카오페이 여행자보험 약관 요약",
            "gold_doc_ids": ["카카오페이_2025_여행자보험약관요약"]
        }
    ]
    
    results = []
    for i, test in enumerate(test_questions, 1):
        print(f"\n--- 테스트 {i}/{len(test_questions)} ---")
        result = test_single_question(test["question"], test["gold_doc_ids"])
        results.append(result)
        print("-" * 50)
    
    # 전체 결과 요약
    print(f"\n📊 전체 결과 요약:")
    print(f"  - 총 테스트: {len(results)}개")
    print(f"  - 평균 Recall@5: {sum(r['recall_at_5'] for r in results) / len(results):.3f}")
    print(f"  - 평균 Faithfulness: {sum(r['faithfulness'] for r in results) / len(results):.3f}")
    
    # 결과 저장
    output_file = Path("eval/out/simple_test_results.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 결과 저장: {output_file}")

if __name__ == "__main__":
    main()
