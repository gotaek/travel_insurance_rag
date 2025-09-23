#!/usr/bin/env python3
"""
평가 결과 분석 리포트 생성
"""

import json
import pandas as pd
from pathlib import Path

def analyze_results():
    """평가 결과 분석"""
    # 결과 로드
    metrics_file = Path("eval/out/metrics.csv")
    raw_file = Path("eval/out/raw.json")
    
    if not metrics_file.exists():
        print("❌ 평가 결과 파일이 없습니다.")
        return
    
    # 메트릭 로드
    df = pd.read_csv(metrics_file)
    
    print("📊 평가 결과 분석 리포트")
    print("=" * 50)
    
    # 전체 통계
    print(f"\n🎯 전체 성능:")
    print(f"  - 총 질문 수: {len(df)}")
    print(f"  - 평균 Recall@5: {df['recall_at_5'].mean():.3f}")
    print(f"  - 평균 Faithfulness: {df['faithfulness_simple'].mean():.3f}")
    
    # Intent별 성능
    print(f"\n📋 Intent별 성능:")
    intent_stats = df.groupby('intent').agg({
        'recall_at_5': 'mean',
        'faithfulness_simple': 'mean',
        'id': 'count'
    }).round(3)
    intent_stats.columns = ['Recall@5', 'Faithfulness', 'Count']
    print(intent_stats)
    
    # 문제점 분석
    print(f"\n⚠️ 문제점 분석:")
    low_faithfulness = df[df['faithfulness_simple'] < 0.1]
    if len(low_faithfulness) > 0:
        print(f"  - 낮은 Faithfulness ({len(low_faithfulness)}개):")
        for _, row in low_faithfulness.iterrows():
            print(f"    * {row['id']}: {row['faithfulness_simple']:.3f}")
    
    # 개선 제안
    print(f"\n💡 개선 제안:")
    if df['faithfulness_simple'].mean() < 0.1:
        print("  1. Faithfulness 개선 필요:")
        print("     - 검색된 문서와 답변 간 연관성 강화")
        print("     - 프롬프트 엔지니어링 개선")
        print("     - 컨텍스트 품질 향상")
    
    if df['recall_at_5'].mean() < 1.0:
        print("  2. Recall@5 개선 필요:")
        print("     - 검색 알고리즘 튜닝")
        print("     - 하이브리드 검색 가중치 조정")
        print("     - 문서 청킹 전략 개선")
    
    print("  3. 일반적인 개선사항:")
    print("     - 더 많은 평가 데이터 추가")
    print("     - 다양한 질문 유형 테스트")
    print("     - RAGAS 메트릭 추가 활용")

if __name__ == "__main__":
    analyze_results()
