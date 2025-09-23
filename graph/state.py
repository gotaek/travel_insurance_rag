from typing import TypedDict, List, Dict, Any, Annotated, Optional
from graph.context import ConversationContext, ConversationTurn

class RAGState(TypedDict, total=False):
    # 기본 질문 정보
    question: Annotated[str, "현재 사용자 질문"]
    intent: Annotated[str, "질문 의도 (qa/summarize/compare/recommend)"]
    needs_web: Annotated[bool, "웹 검색 필요 여부"]
    plan: Annotated[List[str], "실행 계획"]
    
    # 검색 및 문서 정보
    passages: Annotated[List[Dict[str, Any]], "검색된 문서 패시지"]
    refined: Annotated[List[Dict[str, Any]], "정제된 문서 패시지"]
    web_results: Annotated[List[Dict[str, Any]], "웹 검색 결과"]
    
    # 답변 정보
    draft_answer: Annotated[Dict[str, Any], "초안 답변"]
    citations: Annotated[List[Dict[str, Any]], "인용 정보"]
    
    # 시스템 정보
    warnings: Annotated[List[str], "경고 메시지"]
    trace: Annotated[List[Dict[str, Any]], "실행 추적 정보"]
    
    # 멀티턴 대화 컨텍스트 (새로 추가)
    conversation_context: Annotated[Optional[ConversationContext], "대화 컨텍스트"]
    current_turn: Annotated[Optional[ConversationTurn], "현재 턴 정보"]
    session_id: Annotated[Optional[str], "세션 ID"]   