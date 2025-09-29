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
    
    # 보험사 필터링 정보
    insurer_filter: Annotated[Optional[List[str]], "보험사 필터 리스트"]
    extracted_insurers: Annotated[List[str], "추출된 보험사 목록"]
    owned_insurers: Annotated[List[str], "보유 보험사 목록"]
    non_owned_insurers: Annotated[List[str], "비보유 보험사 목록"]
    
    # 답변 정보
    draft_answer: Annotated[Dict[str, Any], "초안 답변"]
    citations: Annotated[List[Dict[str, Any]], "인용 정보"]
    final_answer: Annotated[Dict[str, Any], "최종 답변"]
    
    # 답변 품질 평가
    quality_score: Annotated[float, "답변 품질 점수 (0-1)"]
    quality_feedback: Annotated[str, "품질 평가 피드백"]
    needs_replan: Annotated[bool, "재검색 필요 여부"]
    replan_query: Annotated[str, "재검색을 위한 새로운 질문"]
    
    # 무한루프 방지
    replan_count: Annotated[int, "재검색 횟수 (최대 3회)"]
    max_replan_attempts: Annotated[int, "최대 재검색 시도 횟수"]
    
    # 구조화 실패 감지 (간소화됨)
    # structured_failure_count, max_structured_failures, emergency_fallback_used 제거됨
    
    # 시스템 정보
    warnings: Annotated[List[str], "경고 메시지"]
    trace: Annotated[List[Dict[str, Any]], "실행 추적 정보"]
    
    # 멀티턴 대화 컨텍스트 
    conversation_context: Annotated[Optional[ConversationContext], "대화 컨텍스트"]
    current_turn: Annotated[Optional[ConversationTurn], "현재 턴 정보"]
    session_id: Annotated[Optional[str], "세션 ID"]   