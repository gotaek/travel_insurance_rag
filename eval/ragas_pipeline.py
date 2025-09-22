import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from graph.builder import build_graph
from eval.recall_at_k import recall_at_k
from eval.faithfulness import simple_faithfulness

# RAGAS는 선택적으로 실행 (설치 실패/환경 문제 대비)
_USE_RAGAS = True
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness as ragas_faithfulness, context_precision
except Exception:
    _USE_RAGAS = False

DATA_PATH = Path("eval/questions.jsonl")
OUT_DIR = Path("eval/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run_one(graph, row: Dict[str, Any]) -> Dict[str, Any]:
    q = row["question"]
    gold = row.get("gold_doc_ids", [])
    state = {"question": q}
    out = graph.invoke(state)

    # 기본 지표
    r_at5 = recall_at_k(out, gold, k=5)
    faith = simple_faithfulness(out)

    # RAGAS 입력 준비
    ragas_item = {
        "question": q,
        "answer": (out.get("draft_answer") or {}).get("conclusion",""),
        "contexts": [p.get("text","") for p in (out.get("refined") or [])],
        "ground_truths": []  # 필요시 채우기 (정답 문장)
    }

    return {
        "id": row.get("id"),
        "intent": out.get("intent"),
        "needs_web": out.get("needs_web"),
        "recall_at_5": r_at5,
        "faithfulness_simple": faith,
        "ragas_item": ragas_item,
        "trace": out.get("trace"),
    }

def main():
    # 질문 로드
    rows = [json.loads(l) for l in DATA_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    g = build_graph()
    results: List[Dict[str, Any]] = []

    for r in rows:
        results.append(run_one(g, r))

    # 테이블 저장
    df = pd.DataFrame(
        {
            "id": [x["id"] for x in results],
            "intent": [x["intent"] for x in results],
            "needs_web": [x["needs_web"] for x in results],
            "recall_at_5": [x["recall_at_5"] for x in results],
            "faithfulness_simple": [x["faithfulness_simple"] for x in results],
        }
    )
    df.to_csv(OUT_DIR / "metrics.csv", index=False)

    # 선택적 RAGAS 실행
    if _USE_RAGAS:
        ds = Dataset.from_list([x["ragas_item"] for x in results])
        ragas_res = evaluate(
            ds,
            metrics=[answer_relevancy, ragas_faithfulness, context_precision],
        )
        ragas_df = ragas_res.to_pandas()
        ragas_df.to_csv(OUT_DIR / "ragas.csv", index=False)

    # raw 결과 저장(트레이스 포함)
    with open(OUT_DIR / "raw.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("✅ Saved:", OUT_DIR / "metrics.csv")
    if _USE_RAGAS:
        print("✅ Saved:", OUT_DIR / "ragas.csv")
    print("✅ Saved:", OUT_DIR / "raw.json")

if __name__ == "__main__":
    main()