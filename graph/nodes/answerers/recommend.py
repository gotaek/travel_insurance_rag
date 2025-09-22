from typing import Dict, Any
from app.deps import get_llm
from graph.prompts.utils import load_prompt

def recommend_node(state: Dict[str, Any]) -> Dict[str, Any]:
    refined = state.get("refined", [])
    citations = state.get("citations", [])
    warnings = state.get("warnings", []) or []
    policy_disclaimer = state.get("policy_disclaimer")
    web_results = state.get("web_results", [])

    passages_text = "\n".join([p.get("text","") for p in refined])
    news_text = "\n".join([r["snippet"] for r in web_results]) if web_results else ""
    prompt = (
        load_prompt("recommend")
        + f"\n\n질문: {state['question']}\n\n참고 문서:\n{passages_text}\n\n실시간 뉴스:\n{news_text}"
    )

    try:
        llm = get_llm()
        resp = llm.generate_content(prompt)
        rec = (resp.text or "").strip()
    except Exception as e:
        rec = f"(LLM 호출 실패: {e})"

    caveats = warnings + ([policy_disclaimer] if policy_disclaimer else [])
    answer = {
        "conclusion": rec[:150],
        "evidence": [p.get("text") for p in refined[:2]] + ([news_text] if news_text else []),
        "caveats": caveats,
        "quotes": citations,
    }
    return {**state, "draft_answer": answer}