"""
LLM 응답을 위한 Pydantic 모델들
with_structured_output을 사용하여 안정적인 JSON 파싱을 보장합니다.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional


class QuoteInfo(BaseModel):
    """인용 정보 서브모델 - additionalProperties 방지"""
    model_config = ConfigDict(extra='ignore')
    
    text: str = Field(description="인용 텍스트", default="")
    source: str = Field(description="인용 출처", default="")


class PlannerResponse(BaseModel):
    """Planner 노드 응답 모델"""
    intent: str = Field(description="질문 의도 (qa/summarize/compare/recommend)", default="qa")
    needs_web: bool = Field(description="웹 검색 필요 여부", default=False)
    reasoning: str = Field(description="분류 근거", default="기본 분류")


class AnswerResponse(BaseModel):
    """기본 답변 응답 모델 (QA, Summarize 공통)"""
    conclusion: str = Field(description="답변의 핵심 결론", default="답변을 생성할 수 없습니다.")
    evidence: List[str] = Field(description="근거 정보 목록", default_factory=list)
    caveats: List[str] = Field(description="주의사항 목록", default_factory=list)
    quotes: List[QuoteInfo] = Field(description="인용 정보 목록", default_factory=list)


class ComparisonTable(BaseModel):
    """비교 표 데이터 모델 - additionalProperties 방지"""
    model_config = ConfigDict(extra='ignore')
    
    headers: List[str] = Field(description="표 헤더 목록", default_factory=list)
    rows: List[List[str]] = Field(description="표 행 데이터", default_factory=list)


class CompareResponse(BaseModel):
    """비교 답변 응답 모델"""
    conclusion: str = Field(description="비교 결과 핵심 결론", default="비교 분석을 완료할 수 없습니다.")
    evidence: List[str] = Field(description="근거 정보 목록", default_factory=list)
    caveats: List[str] = Field(description="주의사항 목록", default_factory=list)
    quotes: List[QuoteInfo] = Field(description="인용 정보 목록", default_factory=list)
    comparison_table: ComparisonTable = Field(description="비교 표 데이터", default_factory=ComparisonTable)


class RecommendationItem(BaseModel):
    """추천 항목 모델 - additionalProperties 방지"""
    model_config = ConfigDict(extra='ignore')
    
    type: str = Field(description="추천 유형", default="")
    name: str = Field(description="추천 대상명", default="")
    reason: str = Field(description="추천 이유", default="")
    coverage: str = Field(description="보장 내용", default="")
    priority: str = Field(description="우선순위", default="보통")
    category: str = Field(description="카테고리", default="")


class WebInfo(BaseModel):
    """웹 검색 정보 모델 - additionalProperties 방지"""
    model_config = ConfigDict(extra='ignore')
    
    latest_news: str = Field(description="최신 뉴스", default="")
    travel_alerts: str = Field(description="여행 경보", default="")


class RecommendResponse(BaseModel):
    """추천 답변 응답 모델"""
    model_config = ConfigDict(extra='ignore')
    
    conclusion: str = Field(description="추천 결과 핵심 결론", default="추천 정보를 제공할 수 없습니다.")
    evidence: List[str] = Field(description="근거 정보 목록", default_factory=list)
    caveats: List[str] = Field(description="주의사항 목록", default_factory=list)
    quotes: List[QuoteInfo] = Field(description="인용 정보 목록", default_factory=list)
    web_quotes: List[QuoteInfo] = Field(description="웹 검색 결과 인용 목록", default_factory=list)
    recommendations: List[RecommendationItem] = Field(description="추천 항목 목록", default_factory=list)
    web_info: WebInfo = Field(description="웹 검색 정보", default_factory=WebInfo)


class QualityEvaluationResponse(BaseModel):
    """품질 평가 응답 모델"""
    score: float = Field(description="품질 점수 (0.0-1.0)", ge=0.0, le=1.0, default=0.5)
    feedback: str = Field(description="품질 평가 상세 설명", default="기본 평가")
    needs_replan: bool = Field(description="재검색 필요 여부", default=False)
    replan_query: Optional[str] = Field(description="재검색을 위한 새로운 질문", default=None)


class ReplanResponse(BaseModel):
    """재검색 질문 생성 응답 모델"""
    new_question: str = Field(description="개선된 검색 질문", default="원본 질문을 사용하세요.")
    needs_web: bool = Field(description="웹 검색 필요 여부", default=False)
    reasoning: str = Field(description="재검색 질문 개선 근거", default="기본 재검색")
