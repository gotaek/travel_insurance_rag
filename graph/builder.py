from langgraph.graph import StateGraph, END
from graph.state import RAGState
from graph.nodes.planner import planner_node
from graph.nodes.search import search_node
from graph.nodes.rank_filter import rank_filter_node
from graph.nodes.verify_refine import verify_refine_node
from graph.nodes.answerers.qa import qa_node
from graph.nodes.answerers.summarize import summarize_node
from graph.nodes.answerers.compare import compare_node
from graph.nodes.answerers.recommend import recommend_node

def _decide_answer_node(state: RAGState) -> str:
    intent = state.get("intent", "qa")
    if intent == "summary":
        return "answer_summary"
    if intent == "compare":
        return "answer_compare"
    if intent == "recommend":
        return "answer_recommend"
    return "answer_qa"

def build_graph():
    g = StateGraph(RAGState)

    # Nodes
    g.add_node("planner", planner_node)
    g.add_node("search", search_node)
    g.add_node("rank_filter", rank_filter_node)
    g.add_node("verify_refine", verify_refine_node)
    g.add_node("answer_summary", summarize_node)
    g.add_node("answer_compare", compare_node)
    g.add_node("answer_recommend", recommend_node)
    g.add_node("answer_qa", qa_node)

    # Edges
    g.set_entry_point("planner")
    g.add_edge("planner", "search")
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