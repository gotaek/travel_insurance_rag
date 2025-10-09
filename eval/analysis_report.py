#!/usr/bin/env python3
"""
평가 결과 분석 리포트 생성 및 내보내기
"""

import json
import logging
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings

import pandas as pd
import numpy as np

# 경고 메시지 억제
warnings.filterwarnings('ignore', category=FutureWarning)

# 로깅 설정
logger = logging.getLogger(__name__)

class EvaluationAnalyzer:
    """RAG 시스템 평가 결과 분석기"""
    
    def __init__(self, metrics_path: str = "eval/out/metrics.csv", raw_path: str = "eval/out/raw.json"):
        """
        분석기 초기화.
        
        Args:
            metrics_path: 메트릭 CSV 파일 경로
            raw_path: 원시 결과 JSON 파일 경로
        """
        self.metrics_path = Path(metrics_path)
        self.raw_path = Path(raw_path)
        self.metrics_df: Optional[pd.DataFrame] = None
        self.raw_data: Optional[List[Dict[str, Any]]] = None
        self.analysis_results: Dict[str, Any] = {}
        
    def load_data(self) -> bool:
        """
        평가 결과 데이터 로드.
        
        Returns:
            bool: 데이터 로드 성공 여부
        """
        try:
            # 메트릭 데이터 로드
            if not self.metrics_path.exists():
                logger.error(f"메트릭 파일을 찾을 수 없습니다: {self.metrics_path}")
                return False
                
            self.metrics_df = pd.read_csv(self.metrics_path)
            logger.info(f"메트릭 데이터 로드 완료: {len(self.metrics_df)}개 질문")
            
            # 원시 데이터 로드 (선택적)
            if self.raw_path.exists():
                with open(self.raw_path, 'r', encoding='utf-8') as f:
                    self.raw_data = json.load(f)
                logger.info(f"원시 데이터 로드 완료: {len(self.raw_data)}개 결과")
            else:
                logger.warning(f"원시 데이터 파일을 찾을 수 없습니다: {self.raw_path}")
                self.raw_data = []
            
            return True
            
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {e}")
            return False
    
    def validate_data(self) -> bool:
        """데이터 유효성 검증."""
        if self.metrics_df is None:
            return False
            
        required_columns = ['id', 'recall_at_5', 'faithfulness_simple']
        missing_columns = [col for col in required_columns if col not in self.metrics_df.columns]
        
        if missing_columns:
            logger.error(f"필수 컬럼이 누락되었습니다: {missing_columns}")
            return False
            
        # 데이터 타입 검증
        try:
            self.metrics_df['recall_at_5'] = pd.to_numeric(self.metrics_df['recall_at_5'], errors='coerce')
            self.metrics_df['faithfulness_simple'] = pd.to_numeric(self.metrics_df['faithfulness_simple'], errors='coerce')
        except Exception as e:
            logger.error(f"데이터 타입 변환 오류: {e}")
            return False
            
        return True
    
    def calculate_overall_statistics(self) -> Dict[str, Any]:
        """전체 통계 계산."""
        if self.metrics_df is None:
            return {}
            
        # 기본 통계
        total_questions = len(self.metrics_df)
        successful_questions = len(self.metrics_df.dropna(subset=['recall_at_5', 'faithfulness_simple']))
        
        # 성능 지표
        recall_scores = self.metrics_df['recall_at_5'].dropna()
        faith_scores = self.metrics_df['faithfulness_simple'].dropna()
        
        stats = {
            'total_questions': total_questions,
            'successful_questions': successful_questions,
            'success_rate': successful_questions / total_questions if total_questions > 0 else 0,
            'recall_at_5': {
                'mean': recall_scores.mean() if len(recall_scores) > 0 else 0,
                'median': recall_scores.median() if len(recall_scores) > 0 else 0,
                'std': recall_scores.std() if len(recall_scores) > 0 else 0,
                'min': recall_scores.min() if len(recall_scores) > 0 else 0,
                'max': recall_scores.max() if len(recall_scores) > 0 else 0,
            },
            'faithfulness': {
                'mean': faith_scores.mean() if len(faith_scores) > 0 else 0,
                'median': faith_scores.median() if len(faith_scores) > 0 else 0,
                'std': faith_scores.std() if len(faith_scores) > 0 else 0,
                'min': faith_scores.min() if len(faith_scores) > 0 else 0,
                'max': faith_scores.max() if len(faith_scores) > 0 else 0,
            }
        }
        
        # 실행 시간 통계 (있는 경우)
        if 'execution_time' in self.metrics_df.columns:
            exec_times = self.metrics_df['execution_time'].dropna()
            stats['execution_time'] = {
                'mean': exec_times.mean() if len(exec_times) > 0 else 0,
                'median': exec_times.median() if len(exec_times) > 0 else 0,
                'std': exec_times.std() if len(exec_times) > 0 else 0,
                'total': exec_times.sum() if len(exec_times) > 0 else 0,
            }
        
        return stats
    
    def analyze_by_intent(self) -> Dict[str, Any]:
        """Intent별 성능 분석."""
        if self.metrics_df is None or 'intent' not in self.metrics_df.columns:
            return {}
            
        intent_stats = self.metrics_df.groupby('intent').agg({
            'recall_at_5': ['mean', 'std', 'count'],
            'faithfulness_simple': ['mean', 'std'],
        }).round(3)
        
        # 컬럼명 정리
        intent_stats.columns = ['Recall@5_Mean', 'Recall@5_Std', 'Count', 'Faithfulness_Mean', 'Faithfulness_Std']
        
        # Intent별 상세 분석
        intent_analysis = {}
        for intent in self.metrics_df['intent'].dropna().unique():
            intent_data = self.metrics_df[self.metrics_df['intent'] == intent]
            
            intent_analysis[intent] = {
                'count': len(intent_data),
                'recall_at_5_mean': intent_data['recall_at_5'].mean(),
                'faithfulness_mean': intent_data['faithfulness_simple'].mean(),
                'performance_grade': self._calculate_performance_grade(
                    intent_data['recall_at_5'].mean(),
                    intent_data['faithfulness_simple'].mean()
                )
            }
        
        return {
            'summary_table': intent_stats,
            'detailed_analysis': intent_analysis
        }
    
    def analyze_by_difficulty(self) -> Dict[str, Any]:
        """난이도별 성능 분석."""
        if self.metrics_df is None or 'difficulty' not in self.metrics_df.columns:
            return {}
            
        difficulty_stats = self.metrics_df.groupby('difficulty').agg({
            'recall_at_5': ['mean', 'std', 'count'],
            'faithfulness_simple': ['mean', 'std'],
        }).round(3)
        
        difficulty_stats.columns = ['Recall@5_Mean', 'Recall@5_Std', 'Count', 'Faithfulness_Mean', 'Faithfulness_Std']
        
        difficulty_analysis = {}
        for difficulty in self.metrics_df['difficulty'].dropna().unique():
            difficulty_data = self.metrics_df[self.metrics_df['difficulty'] == difficulty]
            
            difficulty_analysis[difficulty] = {
                'count': len(difficulty_data),
                'recall_at_5_mean': difficulty_data['recall_at_5'].mean(),
                'faithfulness_mean': difficulty_data['faithfulness_simple'].mean(),
                'performance_grade': self._calculate_performance_grade(
                    difficulty_data['recall_at_5'].mean(),
                    difficulty_data['faithfulness_simple'].mean()
                )
            }
        
        return {
            'summary_table': difficulty_stats,
            'detailed_analysis': difficulty_analysis
        }
    
    def analyze_by_category(self) -> Dict[str, Any]:
        """카테고리별 성능 분석."""
        if self.metrics_df is None or 'category' not in self.metrics_df.columns:
            return {}
            
        category_stats = self.metrics_df.groupby('category').agg({
            'recall_at_5': ['mean', 'std', 'count'],
            'faithfulness_simple': ['mean', 'std'],
        }).round(3)
        
        category_stats.columns = ['Recall@5_Mean', 'Recall@5_Std', 'Count', 'Faithfulness_Mean', 'Faithfulness_Std']
        
        category_analysis = {}
        for category in self.metrics_df['category'].dropna().unique():
            category_data = self.metrics_df[self.metrics_df['category'] == category]
            
            category_analysis[category] = {
                'count': len(category_data),
                'recall_at_5_mean': category_data['recall_at_5'].mean(),
                'faithfulness_mean': category_data['faithfulness_simple'].mean(),
                'performance_grade': self._calculate_performance_grade(
                    category_data['recall_at_5'].mean(),
                    category_data['faithfulness_simple'].mean()
                )
            }
        
        return {
            'summary_table': category_stats,
            'detailed_analysis': category_analysis
        }
    
    def _calculate_performance_grade(self, recall: float, faithfulness: float) -> str:
        """성능 등급 계산."""
        if pd.isna(recall) or pd.isna(faithfulness):
            return "N/A"
            
        avg_score = (recall + faithfulness) / 2
        
        if avg_score >= 0.8:
            return "우수"
        elif avg_score >= 0.6:
            return "양호"
        elif avg_score >= 0.4:
            return "보통"
        else:
            return "개선필요"
    
    def identify_issues(self) -> Dict[str, Any]:
        """문제점 식별 및 분석."""
        if self.metrics_df is None:
            return {}
            
        issues = {
            'low_faithfulness': [],
            'low_recall': [],
            'failed_questions': [],
            'slow_execution': [],
            'poor_answers': [],
            'context_issues': []
        }
        
        # 낮은 Faithfulness 식별 (더 엄격한 기준)
        low_faith_threshold = 0.2  # 0.1에서 0.2로 상향 조정
        low_faith = self.metrics_df[self.metrics_df['faithfulness_simple'] < low_faith_threshold]
        for _, row in low_faith.iterrows():
            issues['low_faithfulness'].append({
                'id': row.get('id', 'unknown'),
                'question': row.get('question', ''),
                'faithfulness': row['faithfulness_simple'],
                'intent': row.get('intent', 'unknown')
            })
        
        # 낮은 Recall 식별
        low_recall_threshold = 0.8  # 0.5에서 0.8로 상향 조정 (현재는 모두 1.0이므로)
        low_recall = self.metrics_df[self.metrics_df['recall_at_5'] < low_recall_threshold]
        for _, row in low_recall.iterrows():
            issues['low_recall'].append({
                'id': row.get('id', 'unknown'),
                'question': row.get('question', ''),
                'recall': row['recall_at_5'],
                'intent': row.get('intent', 'unknown')
            })
        
        # 실패한 질문 식별
        if 'error' in self.metrics_df.columns:
            failed = self.metrics_df[self.metrics_df['error'].notna()]
            for _, row in failed.iterrows():
                issues['failed_questions'].append({
                    'id': row.get('id', 'unknown'),
                    'question': row.get('question', ''),
                    'error': row['error']
                })
        
        # 느린 실행 시간 식별
        if 'execution_time' in self.metrics_df.columns:
            exec_times = self.metrics_df['execution_time'].dropna()
            if len(exec_times) > 0:
                slow_threshold = exec_times.quantile(0.9)  # 상위 10%
                slow_exec = self.metrics_df[self.metrics_df['execution_time'] > slow_threshold]
                for _, row in slow_exec.iterrows():
                    issues['slow_execution'].append({
                        'id': row.get('id', 'unknown'),
                        'question': row.get('question', ''),
                        'execution_time': row['execution_time'],
                        'intent': row.get('intent', 'unknown')
                    })
        
        # 답변 품질 문제 식별 (raw 데이터에서)
        if self.raw_data:
            for item in self.raw_data:
                ragas_item = item.get('ragas_item', {})
                answer = ragas_item.get('answer', '')
                
                # 의미없는 답변 패턴 감지
                meaningless_patterns = [
                    '질문을 확인했습니다',
                    '질문을 이해했습니다',
                    '질문을 받았습니다',
                    '질문을 파악했습니다',
                    '질문을 분석했습니다',
                    '질문을 처리했습니다',
                    '질문을 검토했습니다',
                    '질문을 확인하겠습니다',
                    '질문을 살펴보겠습니다',
                    '질문을 검토하겠습니다'
                ]
                
                if any(pattern in answer for pattern in meaningless_patterns):
                    issues['poor_answers'].append({
                        'id': item.get('id', 'unknown'),
                        'question': ragas_item.get('question', ''),
                        'answer': answer[:100] + '...' if len(answer) > 100 else answer,
                        'intent': item.get('intent', 'unknown'),
                        'issue_type': 'meaningless_response'
                    })
                
                # 너무 짧은 답변
                if len(answer.strip()) < 20:
                    issues['poor_answers'].append({
                        'id': item.get('id', 'unknown'),
                        'question': ragas_item.get('question', ''),
                        'answer': answer,
                        'intent': item.get('intent', 'unknown'),
                        'issue_type': 'too_short'
                    })
                
                # 컨텍스트 문제 분석
                contexts = ragas_item.get('contexts', [])
                if contexts:
                    # 너무 짧은 컨텍스트
                    short_contexts = [ctx for ctx in contexts if len(ctx.strip()) < 10]
                    if len(short_contexts) > len(contexts) * 0.5:  # 50% 이상이 짧은 경우
                        issues['context_issues'].append({
                            'id': item.get('id', 'unknown'),
                            'question': ragas_item.get('question', ''),
                            'context_count': len(contexts),
                            'short_context_count': len(short_contexts),
                            'intent': item.get('intent', 'unknown'),
                            'issue_type': 'short_contexts'
                        })
                    
                    # 컨텍스트가 없는 경우
                    if not contexts or all(not ctx.strip() for ctx in contexts):
                        issues['context_issues'].append({
                            'id': item.get('id', 'unknown'),
                            'question': ragas_item.get('question', ''),
                            'context_count': len(contexts),
                            'intent': item.get('intent', 'unknown'),
                            'issue_type': 'no_context'
                        })
        
        return issues
    
    def generate_recommendations(self) -> List[str]:
        """개선 제안 생성."""
        if self.metrics_df is None:
            return []
            
        recommendations = []
        issues = self.identify_issues()
        overall_stats = self.calculate_overall_statistics()
        
        # Faithfulness 문제 분석 및 제안
        if overall_stats['faithfulness']['mean'] < 0.2:
            recommendations.extend([
                "🚨 심각한 Faithfulness 문제 발견:",
                "  - 현재 평균 Faithfulness가 매우 낮음 (0.2 미만)",
                "  - 답변이 검색된 컨텍스트와 거의 연관성이 없음",
                "  - 즉시 프롬프트 엔지니어링 개선 필요"
            ])
        
        # 답변 품질 문제 분석
        if issues['poor_answers']:
            meaningless_count = len([p for p in issues['poor_answers'] if p.get('issue_type') == 'meaningless_response'])
            short_count = len([p for p in issues['poor_answers'] if p.get('issue_type') == 'too_short'])
            
            if meaningless_count > 0:
                recommendations.extend([
                    f"💬 답변 품질 문제 ({meaningless_count}개 질문):",
                    "  - '질문을 확인했습니다' 같은 의미없는 답변 생성",
                    "  - 답변 생성 프롬프트 전면 개선 필요",
                    "  - 컨텍스트 활용하도록 프롬프트 수정"
                ])
            
            if short_count > 0:
                recommendations.extend([
                    f"📏 너무 짧은 답변 ({short_count}개 질문):",
                    "  - 20자 미만의 불충분한 답변",
                    "  - 최소 답변 길이 요구사항 추가",
                    "  - 답변 생성 모델 파라미터 조정"
                ])
        
        # 컨텍스트 문제 분석
        if issues['context_issues']:
            no_context_count = len([c for c in issues['context_issues'] if c.get('issue_type') == 'no_context'])
            short_context_count = len([c for c in issues['context_issues'] if c.get('issue_type') == 'short_contexts'])
            
            if no_context_count > 0:
                recommendations.extend([
                    f"📄 컨텍스트 부족 ({no_context_count}개 질문):",
                    "  - 검색된 컨텍스트가 없거나 비어있음",
                    "  - 검색 알고리즘 점검 및 문서 인덱싱 확인",
                    "  - 검색 쿼리 전처리 개선"
                ])
            
            if short_context_count > 0:
                recommendations.extend([
                    f"📝 짧은 컨텍스트 ({short_context_count}개 질문):",
                    "  - 검색된 컨텍스트가 너무 짧음 (10자 미만)",
                    "  - 문서 청킹 전략 개선 필요",
                    "  - 최소 컨텍스트 길이 보장"
                ])
        
        # Recall 성능 분석
        if overall_stats['recall_at_5']['mean'] < 1.0:
            recommendations.extend([
                "🔍 Recall@5 개선 필요:",
                "  - 일부 질문에서 정답 문서를 찾지 못함",
                "  - 검색 알고리즘 튜닝 및 하이브리드 검색 가중치 조정",
                "  - 문서 청킹 전략 개선 및 임베딩 모델 최적화"
            ])
        else:
            recommendations.append("✅ Recall@5는 양호함 (모든 질문에서 정답 문서 검색 성공)")
        
        # Intent별 제안
        intent_analysis = self.analyze_by_intent()
        for intent, analysis in intent_analysis.get('detailed_analysis', {}).items():
            if analysis['performance_grade'] in ['보통', '개선필요']:
                recommendations.append(f"🎯 {intent} Intent 개선: 현재 성능 등급 '{analysis['performance_grade']}'")
        
        # 우선순위 기반 제안
        recommendations.extend([
            "\n🎯 우선순위별 개선 방안:",
            "  1순위: 답변 생성 프롬프트 전면 개선 (Faithfulness 0.0 문제 해결)",
            "  2순위: 컨텍스트 품질 향상 (문서 청킹 및 검색 개선)",
            "  3순위: 답변 길이 및 품질 보장 (최소 요구사항 설정)",
            "  4순위: Intent별 맞춤형 프롬프트 개발"
        ])
        
        # 구체적인 기술적 제안
        recommendations.extend([
            "\n🔧 구체적인 기술적 개선사항:",
            "  - QA 프롬프트에 '반드시 검색된 문서 내용을 바탕으로 답변하라' 추가",
            "  - 답변 생성 시 컨텍스트 활용도 검증 로직 추가",
            "  - 의미없는 답변 패턴 감지 및 재생성 메커니즘 구현",
            "  - 최소 답변 길이 제한 (예: 50자 이상) 설정"
        ])
        
        return recommendations
    
    def print_analysis_report(self) -> None:
        """분석 리포트 콘솔 출력."""
        if not self.load_data() or not self.validate_data():
            print("❌ 데이터 로드 또는 검증 실패")
            return
        
        # 분석 실행
        overall_stats = self.calculate_overall_statistics()
        intent_analysis = self.analyze_by_intent()
        difficulty_analysis = self.analyze_by_difficulty()
        category_analysis = self.analyze_by_category()
        issues = self.identify_issues()
        recommendations = self.generate_recommendations()
        
        # 리포트 출력
        print("\n" + "="*80)
        print("📊 RAG 시스템 평가 결과 분석 리포트")
        print("="*80)
        print(f"📅 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 전체 성능
        print(f"\n🎯 전체 성능 요약:")
        print(f"  📊 총 질문 수: {overall_stats['total_questions']}")
        print(f"  ✅ 성공한 질문: {overall_stats['successful_questions']} ({overall_stats['success_rate']*100:.1f}%)")
        print(f"  🎯 평균 Recall@5: {overall_stats['recall_at_5']['mean']:.3f} ± {overall_stats['recall_at_5']['std']:.3f}")
        print(f"  🔍 평균 Faithfulness: {overall_stats['faithfulness']['mean']:.3f} ± {overall_stats['faithfulness']['std']:.3f}")
        
        if 'execution_time' in overall_stats:
            print(f"  ⏱️  평균 실행 시간: {overall_stats['execution_time']['mean']:.2f}초")
            print(f"  ⏱️  총 실행 시간: {overall_stats['execution_time']['total']:.2f}초")
        
        # Intent별 성능
        if intent_analysis:
            print(f"\n📋 Intent별 성능 분석:")
            for intent, analysis in intent_analysis['detailed_analysis'].items():
                grade_emoji = {"우수": "🟢", "양호": "🟡", "보통": "🟠", "개선필요": "🔴"}.get(analysis['performance_grade'], "⚪")
                print(f"  {grade_emoji} {intent}: {analysis['count']}개 질문, "
                      f"Recall@5 {analysis['recall_at_5_mean']:.3f}, "
                      f"Faithfulness {analysis['faithfulness_mean']:.3f} "
                      f"({analysis['performance_grade']})")
        
        # 난이도별 성능
        if difficulty_analysis:
            print(f"\n🎯 난이도별 성능 분석:")
            for difficulty, analysis in difficulty_analysis['detailed_analysis'].items():
                grade_emoji = {"우수": "🟢", "양호": "🟡", "보통": "🟠", "개선필요": "🔴"}.get(analysis['performance_grade'], "⚪")
                difficulty_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(difficulty, "⚪")
                print(f"  {difficulty_emoji} {difficulty}: {analysis['count']}개 질문, "
                      f"Recall@5 {analysis['recall_at_5_mean']:.3f}, "
                      f"Faithfulness {analysis['faithfulness_mean']:.3f} "
                      f"({analysis['performance_grade']})")
        
        # 카테고리별 성능
        if category_analysis:
            print(f"\n📂 카테고리별 성능 분석:")
            for category, analysis in category_analysis['detailed_analysis'].items():
                grade_emoji = {"우수": "🟢", "양호": "🟡", "보통": "🟠", "개선필요": "🔴"}.get(analysis['performance_grade'], "⚪")
                print(f"  {grade_emoji} {category}: {analysis['count']}개 질문, "
                      f"Recall@5 {analysis['recall_at_5_mean']:.3f}, "
                      f"Faithfulness {analysis['faithfulness_mean']:.3f} "
                      f"({analysis['performance_grade']})")
    
        # 문제점 분석
        print(f"\n⚠️ 문제점 분석:")
        if issues['low_faithfulness']:
            print(f"  🔍 낮은 Faithfulness ({len(issues['low_faithfulness'])}개):")
            for issue in issues['low_faithfulness'][:5]:  # 상위 5개만 표시
                print(f"    * {issue['id']}: {issue['faithfulness']:.3f} ({issue['intent']})")
        
        if issues['low_recall']:
            print(f"  🎯 낮은 Recall@5 ({len(issues['low_recall'])}개):")
            for issue in issues['low_recall'][:5]:  # 상위 5개만 표시
                print(f"    * {issue['id']}: {issue['recall']:.3f} ({issue['intent']})")
        
        if issues['failed_questions']:
            print(f"  ❌ 실패한 질문 ({len(issues['failed_questions'])}개):")
            for issue in issues['failed_questions'][:3]:  # 상위 3개만 표시
                print(f"    * {issue['id']}: {issue['error'][:50]}...")
        
        # 답변 품질 문제
        if issues['poor_answers']:
            print(f"  💬 답변 품질 문제 ({len(issues['poor_answers'])}개):")
            meaningless = [p for p in issues['poor_answers'] if p.get('issue_type') == 'meaningless_response']
            short_answers = [p for p in issues['poor_answers'] if p.get('issue_type') == 'too_short']
            
            if meaningless:
                print(f"    - 의미없는 답변: {len(meaningless)}개")
                for issue in meaningless[:3]:
                    print(f"      * {issue['id']}: '{issue['answer'][:30]}...' ({issue['intent']})")
            
            if short_answers:
                print(f"    - 너무 짧은 답변: {len(short_answers)}개")
                for issue in short_answers[:3]:
                    print(f"      * {issue['id']}: '{issue['answer']}' ({issue['intent']})")
        
        # 컨텍스트 문제
        if issues['context_issues']:
            print(f"  📄 컨텍스트 문제 ({len(issues['context_issues'])}개):")
            no_context = [c for c in issues['context_issues'] if c.get('issue_type') == 'no_context']
            short_context = [c for c in issues['context_issues'] if c.get('issue_type') == 'short_contexts']
            
            if no_context:
                print(f"    - 컨텍스트 없음: {len(no_context)}개")
                for issue in no_context[:3]:
                    print(f"      * {issue['id']}: {issue['question'][:30]}... ({issue['intent']})")
            
            if short_context:
                print(f"    - 짧은 컨텍스트: {len(short_context)}개")
                for issue in short_context[:3]:
                    print(f"      * {issue['id']}: {issue['short_context_count']}/{issue['context_count']}개 짧음 ({issue['intent']})")
    
        # 개선 제안
        print(f"\n💡 개선 제안:")
        for recommendation in recommendations:
            print(f"  {recommendation}")
        
        print("\n" + "="*80)
    
    def export_to_html(self, output_path: str = "eval/out/analysis_report.html") -> None:
        """HTML 형식으로 분석 리포트 내보내기."""
        # TODO: HTML 리포트 생성 기능 구현
        logger.info(f"HTML 리포트 내보내기: {output_path}")
        pass
    
    def export_to_json(self, output_path: str = "eval/out/analysis_results.json") -> None:
        """JSON 형식으로 분석 결과 내보내기."""
        if not self.load_data() or not self.validate_data():
            return
        
        # DataFrame을 JSON 직렬화 가능한 형태로 변환
        intent_analysis = self.analyze_by_intent()
        if 'summary_table' in intent_analysis:
            # DataFrame을 딕셔너리로 변환
            intent_analysis['summary_table'] = intent_analysis['summary_table'].to_dict('index')
        
        difficulty_analysis = self.analyze_by_difficulty()
        if 'summary_table' in difficulty_analysis:
            # DataFrame을 딕셔너리로 변환
            difficulty_analysis['summary_table'] = difficulty_analysis['summary_table'].to_dict('index')
        
        category_analysis = self.analyze_by_category()
        if 'summary_table' in category_analysis:
            # DataFrame을 딕셔너리로 변환
            category_analysis['summary_table'] = category_analysis['summary_table'].to_dict('index')
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_statistics': self.calculate_overall_statistics(),
            'intent_analysis': intent_analysis,
            'difficulty_analysis': difficulty_analysis,
            'category_analysis': category_analysis,
            'issues': self.identify_issues(),
            'recommendations': self.generate_recommendations()
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"분석 결과 JSON 내보내기 완료: {output_file}")

def main():
    """메인 분석 함수."""
    analyzer = EvaluationAnalyzer()
    analyzer.print_analysis_report()
    analyzer.export_to_json()

if __name__ == "__main__":
    main()
