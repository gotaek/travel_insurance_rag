from langgraph.graph import StateGraph, END
from graph.state import RAGState
from graph.nodes.planner import planner_node
from graph.nodes.websearch import websearch_node
from graph.nodes.search import search_node
from graph.nodes.rank_filter import rank_filter_node
from graph.nodes.verify_refine import verify_refine_node
from graph.nodes.answerers.qa import qa_node
from graph.nodes.answerers.summarize import summarize_node
from graph.nodes.answerers.compare import compare_node
from graph.nodes.answerers.recommend import recommend_node
from graph.nodes.trace import wrap_with_trace

def _decide_answer_node(state: RAGState) -> str:
    intent = state.get("intent", "qa")
    if intent == "summary":
        return "answer_summary"
    if intent == "compare":
        return "answer_compare"
    if intent == "recommend":
        return "answer_recommend"
    return "answer_qa"

def _needs_web_edge(state: RAGState) -> str:
    return "websearch" if state.get("needs_web") else "search"

def build_graph():
    g = StateGraph(RAGState)

    # 모든 노드 trace 래핑
    g.add_node("planner", wrap_with_trace(planner_node, "planner"))
    g.add_node("websearch", wrap_with_trace(websearch_node, "websearch"))
    g.add_node("search", wrap_with_trace(search_node, "search"))
    g.add_node("rank_filter", wrap_with_trace(rank_filter_node, "rank_filter"))
    g.add_node("verify_refine", wrap_with_trace(verify_refine_node, "verify_refine"))
    g.add_node("answer_summary", wrap_with_trace(summarize_node, "answer_summary"))
    g.add_node("answer_compare", wrap_with_trace(compare_node, "answer_compare"))
    g.add_node("answer_recommend", wrap_with_trace(recommend_node, "answer_recommend"))
    g.add_node("answer_qa", wrap_with_trace(qa_node, "answer_qa"))

    g.set_entry_point("planner")
    g.add_conditional_edges("planner", _needs_web_edge, {"websearch": "websearch", "search": "search"})
    g.add_edge("websearch", "search")
    g.add_edge("search", "rank_filter")
    g.add_edge("rank_filter", "verify_refine")
    g.add_conditional_edges(
        "verify_refine",
        _decide_answer_node,
        {
            "answer_summary": "answer_summary",
            "answer_compare": "answer_compare",
            "answer_recommend": "answer_recommend",
            "answer_qa": "answer_qa",
        },
    )
    g.add_edge("answer_summary", END)
    g.add_edge("answer_compare", END)
    g.add_edge("answer_recommend", END)
    g.add_edge("answer_qa", END)
    return g.compile()