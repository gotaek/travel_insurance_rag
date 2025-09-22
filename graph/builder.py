from langgraph.graph import StateGraph, END
from graph.state import RAGState
from graph.nodes.planner import planner_node
from graph.nodes.search import search_node
from graph.nodes.rank_filter import rank_filter_node
from graph.nodes.verify_refine import verify_refine_node
from graph.nodes.answerers.qa import qa_node

def build_graph():
    g = StateGraph(RAGState)
    g.add_node("planner", planner_node)
    g.add_node("search", search_node)
    g.add_node("rank_filter", rank_filter_node)
    g.add_node("verify_refine", verify_refine_node)
    g.add_node("answer", qa_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "search")
    g.add_edge("search", "rank_filter")
    g.add_edge("rank_filter", "verify_refine")
    g.add_edge("verify_refine", "answer")
    g.add_edge("answer", END)

    return g.compile()