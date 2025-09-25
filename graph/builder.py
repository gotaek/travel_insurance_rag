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
from graph.nodes.reevaluate import reevaluate_node
from graph.nodes.replan import replan_node
from graph.nodes.trace import wrap_with_trace
from graph.langsmith_integration import get_langsmith_callbacks, is_langsmith_enabled

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

def _quality_check_edge(state: RAGState) -> str:
    """품질 평가 후 분기 결정 (무한루프 방지 포함)"""
    needs_replan = state.get("needs_replan", False)
    replan_count = state.get("replan_count", 0)
    max_attempts = state.get("max_replan_attempts", 3)
    
    print(f"🔍 _quality_check_edge 호출 - needs_replan: {needs_replan}, replan_count: {replan_count}, max_attempts: {max_attempts}")
    
    # 재검색 횟수가 최대 시도 횟수를 초과하면 강제 종료
    if replan_count >= max_attempts:
        print(f"🚨 최대 재검색 횟수({max_attempts}) 초과 - 강제 종료")
        return "final_answer"
    
    # needs_replan이 False이면 즉시 종료
    if not needs_replan:
        print(f"✅ needs_replan이 False - 답변 완료")
        return "final_answer"
    
    # 재검색이 필요한 경우
    print(f"🔄 재검색 필요 - 횟수: {replan_count}/{max_attempts}")
    return "replan"

def build_graph():
    """LangGraph 빌드 함수 - LangSmith 추적 통합"""
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
    g.add_node("reevaluate", wrap_with_trace(reevaluate_node, "reevaluate"))
    g.add_node("replan", wrap_with_trace(replan_node, "replan"))

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
    
    # 모든 답변 노드에서 reevaluate로 연결
    g.add_edge("answer_summary", "reevaluate")
    g.add_edge("answer_compare", "reevaluate")
    g.add_edge("answer_recommend", "reevaluate")
    g.add_edge("answer_qa", "reevaluate")
    
    # 품질 평가 후 분기
    g.add_conditional_edges("reevaluate", _quality_check_edge, {"replan": "replan", "final_answer": END})
    
    # 재검색 루프 - replan에서 planner로 다시 돌아가서 새로운 질문 처리
    g.add_edge("replan", "planner")
    
    # LangSmith 콜백 통합 (호환성 고려)
    callbacks = get_langsmith_callbacks()
    
    if callbacks:
        print(f"🔍 LangSmith 추적 활성화 - 콜백 수: {len(callbacks)}")
        # LangGraph 버전에 따라 다른 방식으로 콜백 전달
        try:
            # 최신 버전 시도
            return g.compile(callbacks=callbacks)
        except TypeError:
            # 구버전 호환성
            print("⚠️ LangGraph 구버전 감지 - 콜백 없이 컴파일")
            return g.compile()
    else:
        print("⚠️ LangSmith 추적 비활성화 - 콜백 없음")
        return g.compile()