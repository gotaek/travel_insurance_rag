from typing import Dict, Any
from app.deps import get_llm
from graph.prompts.utils import load_prompt

def summarize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    refined = state.get("refined", [])
    citations = state.get("citations", [])
    warnings = state.get("warnings", []) or []
    policy_disclaimer = state.get("policy_disclaimer")

    passages_text = "\n".join([p.get("text","") for p in refined])
    prompt = (
        load_prompt("summarize")
        + f"\n\n질문: {state['question']}\n\n참고 문서:\n{passages_text}"
    )

    try:
        llm = get_llm()
        resp = llm.generate_content(prompt)
        summary = (resp.text or "").strip()
    except Exception as e:
        summary = f"(LLM 호출 실패: {e})"

    caveats = warnings + ([policy_disclaimer] if policy_disclaimer else [])
    answer = {
        "conclusion": summary[:150],
        "evidence": [p.get("text") for p in refined[:2]],
        "caveats": caveats,
        "quotes": citations,
    }
    return {**state, "draft_answer": answer}