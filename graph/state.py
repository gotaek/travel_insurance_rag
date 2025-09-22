from typing import TypedDict, List, Dict, Any

class RAGState(TypedDict, total=False):
    question: str
    intent: str            # "summary" | "compare" | "qa" | "recommend"
    needs_web: bool
    plan: List[str]
    passages: List[Dict[str, Any]]
    refined: List[Dict[str, Any]]
    draft_answer: dict
    citations: List[Dict[str, Any]]
    warnings: List[str]
    trace: List[Dict[str, Any]]