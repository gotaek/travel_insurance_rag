import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import pandas as pd

from graph.builder import build_graph
from eval.recall_at_k import recall_at_k
from eval.faithfulness import simple_faithfulness

# 로깅 설정
logger = logging.getLogger(__name__)

# RAGAS는 선택적으로 실행 (설치 실패/환경 문제 대비)
_USE_RAGAS = True
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness as ragas_faithfulness, context_precision
    logger.info("RAGAS 라이브러리가 성공적으로 로드되었습니다.")
except ImportError as e:
    _USE_RAGAS = False
    logger.warning(f"RAGAS 라이브러리 로드 실패: {e}. 기본 지표만 사용합니다.")
except Exception as e:
    _USE_RAGAS = False
    logger.error(f"RAGAS 라이브러리 초기화 오류: {e}. 기본 지표만 사용합니다.")

DATA_PATH = Path("eval/questions.jsonl")
OUT_DIR = Path("eval/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run_one(graph, row: Dict[str, Any]) -> Dict[str, Any]:
    """
    단일 질문에 대해 RAG 시스템을 실행하고 평가 지표를 계산.
    
    Args:
        graph: RAG 시스템 그래프
        row: 평가 질문 정보 딕셔너리
        
    Returns:
        평가 결과 딕셔너리 (지표, 메타데이터, RAGAS 입력 포함)
    """
    try:
        # 질문 정보 추출
        question = row.get("question", "")
        if not question:
            logger.warning(f"빈 질문 발견: {row.get('id', 'unknown')}")
            return _create_empty_result(row)
        
        gold_doc_ids = row.get("gold_doc_ids", [])
        state = {"question": question}
        
        # RAG 시스템 실행
        start_time = time.time()
        out = graph.invoke(state)
        execution_time = time.time() - start_time
        
        # 기본 지표 계산
        r_at5 = recall_at_k(out, gold_doc_ids, k=5)
        faith = simple_faithfulness(out)

        # RAGAS 입력 준비
        ragas_item = _prepare_ragas_item(question, out)

        return {
            "id": row.get("id"),
            "question": question,
            "intent": out.get("intent"),
            "needs_web": out.get("needs_web"),
            "recall_at_5": r_at5,
            "faithfulness_simple": faith,
            "execution_time": execution_time,
            "ragas_item": ragas_item,
            "trace": out.get("trace"),
            "gold_doc_ids": gold_doc_ids,
            # 새로운 메타데이터 추가
            "expected_answer_type": row.get("expected_answer_type"),
            "difficulty": row.get("difficulty"),
            "category": row.get("category"),
            "tags": row.get("tags", []),
        }
        
    except Exception as e:
        logger.error(f"질문 처리 중 오류 발생 (ID: {row.get('id', 'unknown')}): {e}")
        return _create_empty_result(row, error=str(e))

def _create_empty_result(row: Dict[str, Any], error: Optional[str] = None) -> Dict[str, Any]:
    """오류 발생 시 빈 결과 딕셔너리 생성."""
    return {
        "id": row.get("id"),
        "question": row.get("question", ""),
        "intent": None,
        "needs_web": None,
        "recall_at_5": 0.0,
        "faithfulness_simple": 0.0,
        "execution_time": 0.0,
        "ragas_item": {
            "question": row.get("question", ""),
            "answer": "",
            "contexts": [],
            "ground_truths": []
        },
        "trace": None,
        "gold_doc_ids": row.get("gold_doc_ids", []),
        "error": error,
        # 새로운 메타데이터 추가
        "expected_answer_type": row.get("expected_answer_type"),
        "difficulty": row.get("difficulty"),
        "category": row.get("category"),
        "tags": row.get("tags", []),
    }

def _prepare_ragas_item(question: str, out: Dict[str, Any]) -> Dict[str, Any]:
    """RAGAS 평가를 위한 입력 데이터 준비."""
    draft_answer = out.get("draft_answer", {})
    refined_docs = out.get("refined", [])
    
    # 답변 텍스트 추출
    answer_text = ""
    if isinstance(draft_answer, dict):
        answer_text = draft_answer.get("conclusion", "")
    
    # 컨텍스트 텍스트 추출
    contexts = []
    for doc in refined_docs:
        if isinstance(doc, dict) and doc.get("text"):
            contexts.append(doc["text"])
    
    return {
        "question": question,
        "answer": answer_text,
        "contexts": contexts,
        "ground_truths": []  # 필요시 정답 문장 추가
    }

def load_questions() -> List[Dict[str, Any]]:
    """
    평가 질문들을 JSONL 파일에서 로드.
    
    Returns:
        질문 정보 딕셔너리 리스트
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"평가 질문 파일을 찾을 수 없습니다: {DATA_PATH}")
    
    rows = []
    for line_num, line in enumerate(DATA_PATH.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('//'):  # 빈 줄이나 주석 제외
            continue
            
        try:
            question_data = json.loads(line)
            if not isinstance(question_data, dict):
                logger.warning(f"라인 {line_num}: 딕셔너리가 아닌 데이터 무시")
                continue
            rows.append(question_data)
        except json.JSONDecodeError as e:
            logger.error(f"라인 {line_num} JSON 파싱 오류: {line} - {e}")
            continue
    
    logger.info(f"로드된 평가 질문: {len(rows)}개")
    return rows

def run_evaluation_parallel(graph, questions: List[Dict[str, Any]], max_workers: int = 2) -> List[Dict[str, Any]]:
    """
    병렬 처리를 통한 평가 실행.
    
    Args:
        graph: RAG 시스템 그래프
        questions: 평가 질문 리스트
        max_workers: 최대 워커 수
        
    Returns:
        평가 결과 리스트
    """
    results = []
    total_questions = len(questions)
    
    logger.info(f"병렬 평가 시작 (워커 수: {max_workers})")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 각 질문에 대해 Future 객체 생성
        future_to_question = {
            executor.submit(run_one, graph, question): question 
            for question in questions
        }
        
        # 완료된 작업들을 순서대로 처리
        completed = 0
        for future in as_completed(future_to_question):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                # 진행 상황 로깅
                if completed % max(1, total_questions // 10) == 0 or completed == total_questions:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed
                    remaining = (total_questions - completed) * avg_time
                    logger.info(f"진행률: {completed}/{total_questions} ({completed/total_questions*100:.1f}%) - "
                              f"예상 남은 시간: {remaining:.1f}초")
                              
            except Exception as e:
                question = future_to_question[future]
                logger.error(f"질문 처리 실패 (ID: {question.get('id', 'unknown')}): {e}")
                results.append(_create_empty_result(question, error=str(e)))
    
    total_time = time.time() - start_time
    logger.info(f"평가 완료 - 총 소요 시간: {total_time:.2f}초, 평균: {total_time/total_questions:.2f}초/질문")
    
    return results

def save_results(results: List[Dict[str, Any]]) -> None:
    """평가 결과를 다양한 형식으로 저장."""
    if not results:
        logger.warning("저장할 결과가 없습니다.")
        return
    
    # 기본 메트릭 CSV 저장 (새로운 메타데이터 포함)
    df_data = {
        "id": [x.get("id") for x in results],
        "question": [x.get("question", "") for x in results],
        "intent": [x.get("intent") for x in results],
        "needs_web": [x.get("needs_web") for x in results],
        "recall_at_5": [x.get("recall_at_5", 0.0) for x in results],
        "faithfulness_simple": [x.get("faithfulness_simple", 0.0) for x in results],
        "execution_time": [x.get("execution_time", 0.0) for x in results],
        "expected_answer_type": [x.get("expected_answer_type") for x in results],
        "difficulty": [x.get("difficulty") for x in results],
        "category": [x.get("category") for x in results],
    }
    
    # 오류가 있는 경우 오류 정보 추가
    if any("error" in x for x in results):
        df_data["error"] = [x.get("error", "") for x in results]
    
    df = pd.DataFrame(df_data)
    metrics_path = OUT_DIR / "metrics.csv"
    df.to_csv(metrics_path, index=False, encoding="utf-8")
    logger.info(f"기본 메트릭 저장: {metrics_path}")

    # RAGAS 평가 실행 (선택적)
    if _USE_RAGAS:
        try:
            ragas_items = [x.get("ragas_item", {}) for x in results]
            # 빈 항목 필터링
            valid_ragas_items = [item for item in ragas_items if item.get("question") and item.get("answer")]
            
            if valid_ragas_items:
                ds = Dataset.from_list(valid_ragas_items)
                ragas_res = evaluate(
                    ds,
                    metrics=[answer_relevancy, ragas_faithfulness, context_precision],
                )
                ragas_df = ragas_res.to_pandas()
                ragas_path = OUT_DIR / "ragas.csv"
                ragas_df.to_csv(ragas_path, index=False, encoding="utf-8")
                logger.info(f"RAGAS 메트릭 저장: {ragas_path}")
            else:
                logger.warning("유효한 RAGAS 평가 데이터가 없습니다.")
                
        except Exception as e:
            logger.error(f"RAGAS 평가 실행 실패: {e}")

    # 전체 결과 JSON 저장
    raw_path = OUT_DIR / "raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"전체 결과 저장: {raw_path}")

def print_summary(results: List[Dict[str, Any]]) -> None:
    """평가 결과 요약 출력."""
    if not results:
        logger.warning("출력할 결과가 없습니다.")
        return
    
    # 기본 통계 계산
    total_questions = len(results)
    successful_results = [r for r in results if "error" not in r]
    failed_count = total_questions - len(successful_results)
    
    if successful_results:
        recall_scores = [r.get("recall_at_5", 0.0) for r in successful_results]
        faith_scores = [r.get("faithfulness_simple", 0.0) for r in successful_results]
        exec_times = [r.get("execution_time", 0.0) for r in successful_results]
        
        avg_recall = sum(recall_scores) / len(recall_scores)
        avg_faith = sum(faith_scores) / len(faith_scores)
        avg_exec_time = sum(exec_times) / len(exec_times)
        
        # Intent 분포
        intents = [r.get("intent") for r in successful_results if r.get("intent")]
        intent_counts = {}
        for intent in intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
    else:
        avg_recall = avg_faith = avg_exec_time = 0.0
        intent_counts = {}
    
    print("\n" + "="*60)
    print("📈 평가 결과 요약")
    print("="*60)
    print(f"  📊 총 질문 수: {total_questions}")
    print(f"  ✅ 성공: {len(successful_results)}")
    print(f"  ❌ 실패: {failed_count}")
    
    if successful_results:
        print(f"  🎯 평균 Recall@5: {avg_recall:.3f}")
        print(f"  🔍 평균 Faithfulness: {avg_faith:.3f}")
        print(f"  ⏱️  평균 실행 시간: {avg_exec_time:.2f}초")
        
        if intent_counts:
            print(f"  📋 Intent 분포:")
            for intent, count in sorted(intent_counts.items()):
                print(f"    - {intent}: {count}개")
    
    print(f"\n✅ 저장된 파일:")
    print(f"  - {OUT_DIR / 'metrics.csv'}")
    if _USE_RAGAS:
        print(f"  - {OUT_DIR / 'ragas.csv'}")
    print(f"  - {OUT_DIR / 'raw.json'}")

def main():
    """메인 평가 파이프라인 실행."""
    try:
        # 질문 로드
        questions = load_questions()
        if not questions:
            logger.error("평가할 질문이 없습니다.")
            return
        
        # RAG 그래프 빌드
        logger.info("RAG 그래프 빌드 중...")
        graph = build_graph()
        
        # 평가 실행 (병렬 처리)
        results = run_evaluation_parallel(graph, questions, max_workers=2)
        
        # 결과 저장
        save_results(results)
        
        # 요약 출력
        print_summary(results)
        
    except Exception as e:
        logger.error(f"평가 파이프라인 실행 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()