#!/usr/bin/env python3
"""
기본적인 RAG 시스템 평가 도구
RAGAS 없이 간단하고 효과적인 평가 시스템
"""

import json
import logging
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from graph.builder import build_graph

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 경로 설정
QUESTIONS_PATH = Path("eval/questions.jsonl")
OUTPUT_DIR = Path("eval/out")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_questions() -> List[Dict[str, Any]]:
    """평가 질문들을 JSONL 파일에서 로드."""
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"평가 질문 파일을 찾을 수 없습니다: {QUESTIONS_PATH}")
    
    questions = []
    for line_num, line in enumerate(QUESTIONS_PATH.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        
        try:
            question_data = json.loads(line)
            if isinstance(question_data, dict):
                questions.append(question_data)
        except json.JSONDecodeError as e:
            logger.error(f"라인 {line_num} JSON 파싱 오류: {e}")
            continue
    
    logger.info(f"로드된 평가 질문: {len(questions)}개")
    return questions


def run_rag_system(graph, question: str) -> Dict[str, Any]:
    """
    RAG 시스템을 실행하여 답변과 컨텍스트를 생성.
    
    Args:
        graph: RAG 시스템 그래프
        question: 질문
        
    Returns:
        RAG 시스템 실행 결과
    """
    try:
        start_time = time.time()
        state = {"question": question}
        result = graph.invoke(state)
        end_time = time.time()
        
        # 답변 추출
        draft_answer = result.get("draft_answer", {})
        answer_text = ""
        if isinstance(draft_answer, dict):
            answer_text = draft_answer.get("conclusion", "")
        
        # 컨텍스트 추출
        refined_docs = result.get("refined", [])
        contexts = []
        for doc in refined_docs:
            if isinstance(doc, dict) and doc.get("text"):
                contexts.append(doc["text"])
        
        return {
            "answer": answer_text,
            "contexts": contexts,
            "response_time": end_time - start_time,
            "success": True,
            "raw_result": result
        }
        
    except Exception as e:
        logger.error(f"RAG 시스템 실행 중 오류: {e}")
        return {
            "answer": "",
            "contexts": [],
            "response_time": 0,
            "success": False,
            "error": str(e)
        }


def calculate_recall_at_k_direct(ground_truths: List[str], contexts: List[str], k: int) -> float:
    """
    특정 k 값에 대한 Recall@K 직접 계산 (재귀 없음).
    개선된 유연한 매칭 알고리즘 사용.
    
    Args:
        ground_truths: 정답 키워드 리스트
        contexts: 검색된 컨텍스트 리스트
        k: 상위 K개 컨텍스트만 고려
        
    Returns:
        Recall@K 점수
    """
    if not ground_truths or not contexts:
        return 0.0
    
    # 상위 k개 컨텍스트만 사용
    top_k_contexts = contexts[:k]
    
    # 모든 컨텍스트를 하나의 텍스트로 합치기
    combined_context = " ".join(top_k_contexts).lower()
    
    # 정답 키워드와 매칭 확인
    matched_ground_truths = []
    
    for ground_truth in ground_truths:
        # 키워드를 소문자로 변환하고 특수문자 제거
        clean_ground_truth = re.sub(r'[^\w\s]', '', ground_truth.lower())
        clean_ground_truth = re.sub(r'\s+', ' ', clean_ground_truth).strip()
        
        # 단어 단위로 분할
        ground_truth_words = clean_ground_truth.split()
        
        # 유연한 매칭을 위한 키워드 매핑
        keyword_mappings = {
            "항공기": ["항공편", "항공", "비행기"],
            "출발": ["출국", "이륙"],
            "지연": ["연착", "지연"],
            "보상": ["보장", "지급", "보험금"],
            "금액": ["비용", "한도", "금액"],
            "시간": ["시간", "분", "시"],
            "탑승권": ["항공권", "티켓", "승차권"],
            "확인서": ["증명서", "서류", "문서"],
            "제출": ["제출", "제시", "갖춤"]
        }
        
        # 각 단어가 컨텍스트에 포함되는지 확인 (유연한 매칭 포함)
        word_matches = 0
        for word in ground_truth_words:
            if len(word) > 1:  # 1글자 이상 단어 고려
                # 직접 매칭
                if word in combined_context:
                    word_matches += 1
                # 유연한 매칭 (동의어/유사어)
                elif word in keyword_mappings:
                    synonyms = keyword_mappings[word]
                    for synonym in synonyms:
                        if synonym in combined_context:
                            word_matches += 1
                            break
        
        # 30% 이상의 단어가 매칭되면 해당 ground_truth는 매칭된 것으로 간주 (임계값 낮춤)
        if word_matches >= len(ground_truth_words) * 0.3:
            matched_ground_truths.append(ground_truth)
    
    # Recall@K 계산
    total_ground_truths = len(ground_truths)
    matched_count = len(matched_ground_truths)
    
    return matched_count / total_ground_truths if total_ground_truths > 0 else 0.0


def calculate_recall_at_k(ground_truths: List[str], contexts: List[str], k: int = 5) -> Dict[str, Any]:
    """
    Recall@K 메트릭 계산.
    
    Args:
        ground_truths: 정답 키워드 리스트
        contexts: 검색된 컨텍스트 리스트
        k: 상위 K개 컨텍스트만 고려
        
    Returns:
        Recall@K 메트릭
    """
    if not ground_truths or not contexts:
        return {
            "recall_at_k": 0.0,
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "matched_ground_truths": [],
            "total_ground_truths": len(ground_truths)
        }
    
    # 상위 k개 컨텍스트만 사용
    top_k_contexts = contexts[:k]
    
    # 모든 컨텍스트를 하나의 텍스트로 합치기
    combined_context = " ".join(top_k_contexts).lower()
    
    # 정답 키워드와 매칭 확인
    matched_ground_truths = []
    
    for ground_truth in ground_truths:
        # 키워드를 소문자로 변환하고 특수문자 제거
        clean_ground_truth = re.sub(r'[^\w\s]', '', ground_truth.lower())
        clean_ground_truth = re.sub(r'\s+', ' ', clean_ground_truth).strip()
        
        # 단어 단위로 분할
        ground_truth_words = clean_ground_truth.split()
        
        # 각 단어가 컨텍스트에 포함되는지 확인
        word_matches = 0
        for word in ground_truth_words:
            if len(word) > 2 and word in combined_context:  # 2글자 이상 단어만 고려
                word_matches += 1
        
        # 50% 이상의 단어가 매칭되면 해당 ground_truth는 매칭된 것으로 간주
        if word_matches >= len(ground_truth_words) * 0.5:
            matched_ground_truths.append(ground_truth)
    
    # Recall@K 계산
    total_ground_truths = len(ground_truths)
    matched_count = len(matched_ground_truths)
    
    recall_at_k = matched_count / total_ground_truths if total_ground_truths > 0 else 0.0
    
    # 다양한 k 값에 대한 recall 계산 (재귀 호출 대신 직접 계산)
    recall_at_1 = calculate_recall_at_k_direct(ground_truths, contexts, 1)
    recall_at_3 = calculate_recall_at_k_direct(ground_truths, contexts, 3)
    recall_at_5 = recall_at_k  # 이미 k=5로 계산됨
    
    return {
        "recall_at_k": recall_at_k,
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "recall_at_5": recall_at_5,
        "matched_ground_truths": matched_ground_truths,
        "total_ground_truths": total_ground_truths,
        "matched_count": matched_count
    }


def calculate_basic_metrics(question_data: Dict[str, Any], rag_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    기본적인 평가 메트릭 계산 (Recall@K 포함).
    
    Args:
        question_data: 질문 데이터
        rag_result: RAG 시스템 실행 결과
        
    Returns:
        평가 메트릭
    """
    metrics = {
        "question_id": question_data.get("id", "unknown"),
        "question": question_data.get("question", ""),
        "category": question_data.get("category", "unknown"),
        "intent": question_data.get("intent", "unknown"),
        "response_time": rag_result.get("response_time", 0),
        "success": rag_result.get("success", False),
        "answer_length": len(rag_result.get("answer", "")),
        "context_count": len(rag_result.get("contexts", [])),
        "context_length": sum(len(ctx) for ctx in rag_result.get("contexts", [])),
    }
    
    # Recall@K 메트릭 계산
    ground_truths = question_data.get("ground_truths", [])
    contexts = rag_result.get("contexts", [])
    
    recall_metrics = calculate_recall_at_k(ground_truths, contexts, k=5)
    metrics.update(recall_metrics)
    
    # 기존 키워드 매칭 점수 (답변 기반)
    answer = rag_result.get("answer", "").lower()
    
    if ground_truths and answer:
        matched_keywords = 0
        total_keywords = len(ground_truths)
        
        for keyword in ground_truths:
            # 키워드를 소문자로 변환하고 공백 제거
            clean_keyword = re.sub(r'\s+', '', keyword.lower())
            clean_answer = re.sub(r'\s+', '', answer)
            
            if clean_keyword in clean_answer:
                matched_keywords += 1
        
        metrics["answer_keyword_match_score"] = matched_keywords / total_keywords if total_keywords > 0 else 0
        metrics["answer_matched_keywords"] = matched_keywords
        metrics["answer_total_keywords"] = total_keywords
    else:
        metrics["answer_keyword_match_score"] = 0
        metrics["answer_matched_keywords"] = 0
        metrics["answer_total_keywords"] = 0
    
    # 답변 품질 점수 (간단한 휴리스틱)
    quality_score = 0
    
    # 답변 길이 점수 (너무 짧거나 길면 감점)
    answer_length = metrics["answer_length"]
    if 50 <= answer_length <= 500:
        quality_score += 0.2
    elif 20 <= answer_length < 50 or 500 < answer_length <= 1000:
        quality_score += 0.15
    else:
        quality_score += 0.1
    
    # 컨텍스트 점수
    if metrics["context_count"] > 0:
        quality_score += 0.2
    
    # Recall@K 점수 (가장 중요)
    quality_score += metrics["recall_at_k"] * 0.5
    
    # 답변 키워드 매칭 점수
    quality_score += metrics["answer_keyword_match_score"] * 0.1
    
    metrics["quality_score"] = min(quality_score, 1.0)
    
    return metrics


def run_evaluation(questions: List[Dict[str, Any]], graph) -> List[Dict[str, Any]]:
    """
    전체 평가 실행.
    
    Args:
        questions: 평가 질문 리스트
        graph: RAG 시스템 그래프
        
    Returns:
        평가 결과 리스트
    """
    results = []
    
    for i, question_data in enumerate(questions, 1):
        question = question_data.get("question", "")
        if not question:
            continue
        
        logger.info(f"평가 진행 중: {i}/{len(questions)} - {question_data.get('id', 'unknown')}")
        
        # RAG 시스템 실행
        rag_result = run_rag_system(graph, question)
        
        # 메트릭 계산
        metrics = calculate_basic_metrics(question_data, rag_result)
        
        # 결과 저장
        result = {
            **metrics,
            "answer": rag_result.get("answer", ""),
            "contexts": rag_result.get("contexts", []),
            "ground_truths": question_data.get("ground_truths", []),
            "error": rag_result.get("error", None)
        }
        
        results.append(result)
        
        # 진행 상황 출력
        logger.info(f"  - 응답시간: {metrics['response_time']:.2f}초")
        logger.info(f"  - 답변길이: {metrics['answer_length']}자")
        logger.info(f"  - 컨텍스트: {metrics['context_count']}개")
        logger.info(f"  - Recall@1: {metrics['recall_at_1']:.3f}")
        logger.info(f"  - Recall@3: {metrics['recall_at_3']:.3f}")
        logger.info(f"  - Recall@5: {metrics['recall_at_5']:.3f}")
        logger.info(f"  - 매칭된정답: {metrics['matched_count']}/{metrics['total_ground_truths']}")
        logger.info(f"  - 답변키워드매칭: {metrics['answer_matched_keywords']}/{metrics['answer_total_keywords']}")
        logger.info(f"  - 품질점수: {metrics['quality_score']:.3f}")
    
    return results


def save_results(results: List[Dict[str, Any]]) -> None:
    """평가 결과를 파일로 저장."""
    if not results:
        logger.warning("저장할 결과가 없습니다.")
        return
    
    try:
        # CSV 형태로 저장
        csv_path = OUTPUT_DIR / "simple_eval_results.csv"
        with open(csv_path, 'w', encoding='utf-8') as f:
            # 헤더 작성 (Recall@K 메트릭 포함)
            headers = [
                "question_id", "question", "category", "intent",
                "response_time", "success", "answer_length", "context_count", "context_length",
                "recall_at_k", "recall_at_1", "recall_at_3", "recall_at_5",
                "matched_ground_truths", "total_ground_truths", "matched_count",
                "answer_keyword_match_score", "answer_matched_keywords", "answer_total_keywords",
                "quality_score"
            ]
            f.write(",".join(headers) + "\n")
            
            # 데이터 작성
            for result in results:
                row = [
                    str(result.get(header, "")) for header in headers
                ]
                f.write(",".join(row) + "\n")
        
        logger.info(f"CSV 결과 저장: {csv_path}")
        
        # JSON 형태로 저장
        json_path = OUTPUT_DIR / "simple_eval_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON 결과 저장: {json_path}")
        
        # 요약 통계 저장
        summary_path = OUTPUT_DIR / "simple_eval_summary.json"
        summary = calculate_summary_stats(results)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"요약 통계 저장: {summary_path}")
        
    except Exception as e:
        logger.error(f"결과 저장 중 오류: {e}")


def calculate_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """요약 통계 계산."""
    if not results:
        return {}
    
    # 기본 통계
    total_questions = len(results)
    successful_questions = sum(1 for r in results if r.get("success", False))
    
    # 평균값 계산
    avg_response_time = sum(r.get("response_time", 0) for r in results) / total_questions
    avg_answer_length = sum(r.get("answer_length", 0) for r in results) / total_questions
    avg_context_count = sum(r.get("context_count", 0) for r in results) / total_questions
    avg_recall_at_1 = sum(r.get("recall_at_1", 0) for r in results) / total_questions
    avg_recall_at_3 = sum(r.get("recall_at_3", 0) for r in results) / total_questions
    avg_recall_at_5 = sum(r.get("recall_at_5", 0) for r in results) / total_questions
    avg_answer_keyword_match = sum(r.get("answer_keyword_match_score", 0) for r in results) / total_questions
    avg_quality_score = sum(r.get("quality_score", 0) for r in results) / total_questions
    
    # 카테고리별 통계
    category_stats = {}
    for result in results:
        category = result.get("category", "unknown")
        if category not in category_stats:
            category_stats[category] = {
                "count": 0,
                "avg_quality_score": 0,
                "avg_recall_at_5": 0,
                "avg_answer_keyword_match": 0,
                "avg_response_time": 0
            }
        
        category_stats[category]["count"] += 1
        category_stats[category]["avg_quality_score"] += result.get("quality_score", 0)
        category_stats[category]["avg_recall_at_5"] += result.get("recall_at_5", 0)
        category_stats[category]["avg_answer_keyword_match"] += result.get("answer_keyword_match_score", 0)
        category_stats[category]["avg_response_time"] += result.get("response_time", 0)
    
    # 카테고리별 평균 계산
    for category in category_stats:
        count = category_stats[category]["count"]
        category_stats[category]["avg_quality_score"] /= count
        category_stats[category]["avg_recall_at_5"] /= count
        category_stats[category]["avg_answer_keyword_match"] /= count
        category_stats[category]["avg_response_time"] /= count
    
    return {
        "evaluation_date": datetime.now().isoformat(),
        "total_questions": total_questions,
        "successful_questions": successful_questions,
        "success_rate": successful_questions / total_questions,
        "average_metrics": {
            "response_time": avg_response_time,
            "answer_length": avg_answer_length,
            "context_count": avg_context_count,
            "recall_at_1": avg_recall_at_1,
            "recall_at_3": avg_recall_at_3,
            "recall_at_5": avg_recall_at_5,
            "answer_keyword_match_score": avg_answer_keyword_match,
            "quality_score": avg_quality_score
        },
        "category_stats": category_stats
    }


def print_summary(results: List[Dict[str, Any]]) -> None:
    """평가 결과 요약 출력."""
    if not results:
        print("❌ 출력할 결과가 없습니다.")
        return
    
    summary = calculate_summary_stats(results)
    
    print("\n" + "="*60)
    print("📊 기본 평가 결과 요약")
    print("="*60)
    
    print(f"📋 총 평가 질문: {summary['total_questions']}개")
    print(f"✅ 성공한 질문: {summary['successful_questions']}개")
    print(f"📈 성공률: {summary['success_rate']:.1%}")
    
    print(f"\n📊 평균 메트릭:")
    avg_metrics = summary['average_metrics']
    print(f"  ⏱️ 평균 응답시간: {avg_metrics['response_time']:.2f}초")
    print(f"  📝 평균 답변길이: {avg_metrics['answer_length']:.0f}자")
    print(f"  📚 평균 컨텍스트 수: {avg_metrics['context_count']:.1f}개")
    print(f"  🎯 Recall@1: {avg_metrics['recall_at_1']:.3f}")
    print(f"  🎯 Recall@3: {avg_metrics['recall_at_3']:.3f}")
    print(f"  🎯 Recall@5: {avg_metrics['recall_at_5']:.3f}")
    print(f"  🔍 평균 답변키워드매칭: {avg_metrics['answer_keyword_match_score']:.3f}")
    print(f"  ⭐ 평균 품질점수: {avg_metrics['quality_score']:.3f}")
    
    print(f"\n📂 카테고리별 성능:")
    for category, stats in summary['category_stats'].items():
        print(f"  {category}:")
        print(f"    - 질문 수: {stats['count']}개")
        print(f"    - 품질점수: {stats['avg_quality_score']:.3f}")
        print(f"    - Recall@5: {stats['avg_recall_at_5']:.3f}")
        print(f"    - 답변키워드매칭: {stats['avg_answer_keyword_match']:.3f}")
        print(f"    - 응답시간: {stats['avg_response_time']:.2f}초")
    
    print(f"\n📁 결과 파일:")
    print(f"  - CSV: eval/out/simple_eval_results.csv")
    print(f"  - JSON: eval/out/simple_eval_results.json")
    print(f"  - 요약: eval/out/simple_eval_summary.json")


def main():
    """메인 실행 함수."""
    try:
        print("🚀 기본 평가 시스템 시작")
        
        # 질문 로드
        questions = load_questions()
        if not questions:
            logger.error("평가할 질문이 없습니다.")
            return
        
        # RAG 그래프 빌드
        logger.info("RAG 그래프 빌드 중...")
        graph = build_graph()
        
        # 평가 실행
        results = run_evaluation(questions, graph)
        if not results:
            logger.error("평가 실행에 실패했습니다.")
            return
        
        # 결과 저장
        save_results(results)
        
        # 요약 출력
        print_summary(results)
        
        print("\n✅ 기본 평가 완료!")
        
    except Exception as e:
        logger.error(f"평가 실행 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
