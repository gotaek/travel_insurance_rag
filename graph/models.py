"""
LLM 응답을 위한 Pydantic 모델들
with_structured_output을 사용하여 안정적인 JSON 파싱을 보장합니다.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class PlannerResponse(BaseModel):
    """Planner 노드 응답 모델"""
    intent: str = Field(description="질문 의도 (qa/summarize/compare/recommend)")
    needs_web: bool = Field(description="웹 검색 필요 여부")
    reasoning: str = Field(description="분류 근거")


class AnswerResponse(BaseModel):
    """기본 답변 응답 모델 (QA, Summarize 공통)"""
    conclusion: str = Field(description="답변의 핵심 결론")
    evidence: List[str] = Field(description="근거 정보 목록")
    caveats: List[str] = Field(description="주의사항 목록")
    quotes: List[Dict[str, str]] = Field(description="인용 정보 목록", default_factory=list)


class CompareResponse(BaseModel):
    """비교 답변 응답 모델"""
    conclusion: str = Field(description="비교 결과 핵심 결론")
    evidence: List[str] = Field(description="근거 정보 목록")
    caveats: List[str] = Field(description="주의사항 목록")
    quotes: List[Dict[str, str]] = Field(description="인용 정보 목록", default_factory=list)
    comparison_table: Dict[str, Any] = Field(description="비교 표 데이터")


class RecommendResponse(BaseModel):
    """추천 답변 응답 모델"""
    conclusion: str = Field(description="추천 결과 핵심 결론")
    evidence: List[str] = Field(description="근거 정보 목록")
    caveats: List[str] = Field(description="주의사항 목록")
    quotes: List[Dict[str, str]] = Field(description="인용 정보 목록", default_factory=list)
    recommendations: List[Dict[str, Any]] = Field(description="추천 항목 목록", default_factory=list)
    web_info: Dict[str, Any] = Field(description="웹 검색 정보", default_factory=dict)


class QualityEvaluationResponse(BaseModel):
    """품질 평가 응답 모델"""
    score: float = Field(description="품질 점수 (0.0-1.0)", ge=0.0, le=1.0)
    feedback: str = Field(description="품질 평가 상세 설명")
    needs_replan: bool = Field(description="재검색 필요 여부")
    replan_query: Optional[str] = Field(description="재검색을 위한 새로운 질문", default=None)


class ReplanResponse(BaseModel):
    """재검색 질문 생성 응답 모델"""
    new_question: str = Field(description="개선된 검색 질문")
    needs_web: bool = Field(description="웹 검색 필요 여부")
    reasoning: str = Field(description="재검색 질문 개선 근거")
