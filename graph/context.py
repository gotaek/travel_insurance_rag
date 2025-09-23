"""
멀티턴 대화를 위한 컨텍스트 관리 모듈
- Annotated 타입을 사용한 타입 안전성 보장
- 컨텍스트 윈도우 관리 및 압축
- 세션 기반 대화 히스토리 관리
"""

from typing import Annotated, List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
import json
import hashlib


class ConversationTurn(BaseModel):
    """대화 턴 정보"""
    turn_id: str = Field(..., description="턴 고유 ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="생성 시간")
    question: str = Field(..., description="사용자 질문")
    answer: Dict[str, Any] = Field(..., description="시스템 답변")
    intent: str = Field(..., description="질문 의도")
    passages_used: List[Dict[str, Any]] = Field(default_factory=list, description="사용된 문서")
    tokens_used: int = Field(default=0, description="사용된 토큰 수")
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "turn_id": self.turn_id,
            "timestamp": self.timestamp.isoformat(),
            "question": self.question,
            "answer": self.answer,
            "intent": self.intent,
            "passages_used": self.passages_used,
            "tokens_used": self.tokens_used
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        """딕셔너리에서 생성"""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class UserPreferences(BaseModel):
    """사용자 선호도 설정"""
    preferred_language: str = Field(default="ko", description="선호 언어")
    detail_level: str = Field(default="medium", description="답변 상세도 (low/medium/high)")
    include_citations: bool = Field(default=True, description="인용 포함 여부")
    max_response_length: int = Field(default=1000, description="최대 응답 길이")
    
    def to_dict(self) -> Dict[str, Any]:
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreferences":
        return cls(**data)


class ConversationContext(BaseModel):
    """대화 컨텍스트 정보"""
    session_id: str = Field(..., description="세션 고유 ID")
    user_id: Optional[str] = Field(default=None, description="사용자 ID")
    created_at: datetime = Field(default_factory=datetime.now, description="세션 생성 시간")
    last_activity: datetime = Field(default_factory=datetime.now, description="마지막 활동 시간")
    
    # 대화 히스토리
    conversation_history: List[ConversationTurn] = Field(default_factory=list, description="대화 히스토리")
    
    # 컨텍스트 정보
    current_topic: Optional[str] = Field(default=None, description="현재 주제")
    related_documents: List[str] = Field(default_factory=list, description="관련 문서 ID 목록")
    
    # 사용자 설정
    user_preferences: UserPreferences = Field(default_factory=UserPreferences, description="사용자 선호도")
    
    # 메타데이터
    total_tokens: int = Field(default=0, description="총 사용 토큰 수")
    turn_count: int = Field(default=0, description="총 턴 수")
    
    def add_turn(self, turn: ConversationTurn) -> None:
        """새로운 턴 추가"""
        self.conversation_history.append(turn)
        self.last_activity = datetime.now()
        self.total_tokens += turn.tokens_used
        self.turn_count += 1
    
    def get_recent_turns(self, n: int = 5) -> List[ConversationTurn]:
        """최근 N개 턴 반환"""
        return self.conversation_history[-n:] if self.conversation_history else []
    
    def get_context_summary(self) -> Dict[str, Any]:
        """컨텍스트 요약 정보"""
        recent_turns = self.get_recent_turns(3)
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "current_topic": self.current_topic,
            "recent_questions": [turn.question for turn in recent_turns],
            "total_tokens": self.total_tokens,
            "last_activity": self.last_activity.isoformat()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "conversation_history": [turn.to_dict() for turn in self.conversation_history],
            "current_topic": self.current_topic,
            "related_documents": self.related_documents,
            "user_preferences": self.user_preferences.to_dict(),
            "total_tokens": self.total_tokens,
            "turn_count": self.turn_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationContext":
        """딕셔너리에서 생성"""
        # 날짜 필드 변환
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["last_activity"] = datetime.fromisoformat(data["last_activity"])
        
        # 대화 히스토리 변환
        data["conversation_history"] = [
            ConversationTurn.from_dict(turn_data) 
            for turn_data in data["conversation_history"]
        ]
        
        # 사용자 선호도 변환
        data["user_preferences"] = UserPreferences.from_dict(data["user_preferences"])
        
        return cls(**data)


# Annotated 타입 정의
ConversationTurnType = Annotated[ConversationTurn, "대화 턴 정보"]
ConversationContextType = Annotated[ConversationContext, "대화 컨텍스트"]
UserPreferencesType = Annotated[UserPreferences, "사용자 선호도 설정"]

# 컨텍스트 윈도우 설정
MAX_CONTEXT_TOKENS = 4000  # 최대 컨텍스트 토큰 수
MAX_HISTORY_TURNS = 10     # 최대 히스토리 턴 수
COMPRESSION_THRESHOLD = 0.8  # 압축 임계값 (80% 도달 시 압축)


def generate_turn_id(question: str, session_id: str) -> str:
    """턴 ID 생성"""
    content = f"{session_id}:{question}:{datetime.now().isoformat()}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def generate_session_id(user_id: Optional[str] = None) -> str:
    """세션 ID 생성"""
    if user_id:
        content = f"{user_id}:{datetime.now().isoformat()}"
    else:
        content = f"anonymous:{datetime.now().isoformat()}"
    return hashlib.md5(content.encode()).hexdigest()[:16]
