from langgraph.graph import StateGraph, END
from graph.state import RAGState
from graph.nodes.planner import planner_node
from graph.nodes.answerers.qa import qa_node

def build_graph():
    g = StateGraph(RAGState)
    g.add_node("planner", planner_node)
    g.add_node("answer", qa_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "answer")
    g.add_edge("answer", END)
    return g.compile()