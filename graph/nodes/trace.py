import time
from typing import Any, Dict, Callable, List

def _count_tokens_from_state(state: Dict[str, Any]) -> int:
    """
    매우 단순한 토큰 추정(글자수 기반). 실제 토크나이저로 교체 예정.
    question + passages/refined의 text 길이 합산.
    """
    total = 0
    q = state.get("question", "")
    total += len(str(q))
    for key in ("passages", "refined"):
        items: List[Dict[str, Any]] = state.get(key) or []
        for p in items:
            total += len(str(p.get("text", "")))
    return total

def wrap_with_trace(fn: Callable[[Dict[str, Any]], Dict[str, Any]], node_name: str):
    """
    LangGraph 노드 함수를 감싸서:
      - 실행 시간(ms) 측정
      - in/out token 추정
      - state['trace']에 append
    """
    import logging
    logger = logging.getLogger(__name__)
    
    def _wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
        trace_list = state.get("trace") or []
        in_tokens = _count_tokens_from_state(state)
        t0 = time.perf_counter()
        
        logger.info(f"⏱️ [Trace] {node_name} 시작 - 입력 토큰: {in_tokens}개")

        out = fn(state)  # 실제 노드 호출

        dur_ms = int((time.perf_counter() - t0) * 1000)
        result_state = out if isinstance(out, dict) else state
        out_tokens = _count_tokens_from_state(result_state)

        trace_item = {
            "node": node_name,
            "latency_ms": dur_ms,
            "in_tokens_approx": in_tokens,
            "out_tokens_approx": out_tokens,
        }
        trace_list.append(trace_item)
        result_state["trace"] = trace_list
        
        logger.info(f"⏱️ [Trace] {node_name} 완료 - 실행시간: {dur_ms}ms, 출력 토큰: {out_tokens}개")
        return result_state
    return _wrapped