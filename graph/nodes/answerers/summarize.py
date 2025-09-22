from typing import Dict, Any
from app.deps import get_llm
from graph.prompts.utils import load_prompt
from graph.prompts.parser import parse_llm_output

def summarize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    refined = state.get("refined", [])
    warnings = state.get("warnings", []) or []
    policy_disclaimer = state.get("policy_disclaimer")

    passages_text = "\n".join([p.get("text","") for p in refined])
    prompt = load_prompt("summarize") + f"\n\n질문: {state['question']}\n\n참고 문서:\n{passages_text}"

    try:
        llm = get_llm()
        resp = llm.generate_content(prompt)
        parsed = parse_llm_output(resp.text)
    except Exception as e:
        parsed = {"conclusion": f"(LLM 호출 실패: {e})", "evidence": [], "caveats": [], "quotes": []}

    caveats = parsed.get("caveats", []) + warnings
    if policy_disclaimer:
        caveats.append(policy_disclaimer)

    answer = {
        "conclusion": parsed.get("conclusion", ""),
        "evidence": parsed.get("evidence", []),
        "caveats": caveats,
        "quotes": parsed.get("quotes", []),
    }
    return {**state, "draft_answer": answer}