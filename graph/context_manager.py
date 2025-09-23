"""
대화 컨텍스트 관리자
- 컨텍스트 윈도우 관리 및 압축
- 토큰 수 기반 히스토리 관리
- 지능적 컨텍스트 압축
"""

import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import tiktoken

from graph.context import (
    ConversationContext, ConversationTurn, UserPreferences,
    MAX_CONTEXT_TOKENS, MAX_HISTORY_TURNS, COMPRESSION_THRESHOLD
)
from app.deps import get_redis_client, get_settings


class ContextWindowManager:
    """컨텍스트 윈도우 관리자"""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = get_redis_client()
        # tiktoken 인코더 (GPT-4용)
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산"""
        try:
            return len(self.encoder.encode(text))
        except Exception:
            # tiktoken 실패 시 대략적 계산 (한글 기준)
            return len(text) // 2
    
    def compress_conversation_history(
        self, 
        history: List[ConversationTurn], 
        max_tokens: int = MAX_CONTEXT_TOKENS
    ) -> List[ConversationTurn]:
        """
        대화 히스토리 압축
        - 중요도 기반 턴 선택
        - 최근 턴 우선 보존
        - 토큰 수 제한 내에서 압축
        """
        if not history:
            return []
        
        # 토큰 수 계산
        total_tokens = sum(self.count_tokens(turn.question + str(turn.answer)) for turn in history)
        
        if total_tokens <= max_tokens:
            return history
        
        # 압축 필요
        compressed = []
        current_tokens = 0
        
        # 1. 최근 턴들 우선 보존 (최대 3개)
        recent_turns = history[-3:]
        for turn in reversed(recent_turns):
            turn_tokens = self.count_tokens(turn.question + str(turn.answer))
            if current_tokens + turn_tokens <= max_tokens:
                compressed.insert(0, turn)
                current_tokens += turn_tokens
            else:
                break
        
        # 2. 나머지 공간이 있으면 중요도 기반으로 추가
        remaining_turns = history[:-3]
        if remaining_turns and current_tokens < max_tokens * 0.8:
            # 중요도 점수 계산 (의도, 토큰 수, 시간 고려)
            scored_turns = []
            for turn in remaining_turns:
                score = self._calculate_turn_importance(turn, history)
                scored_turns.append((score, turn))
            
            # 점수 순으로 정렬
            scored_turns.sort(key=lambda x: x[0], reverse=True)
            
            for score, turn in scored_turns:
                turn_tokens = self.count_tokens(turn.question + str(turn.answer))
                if current_tokens + turn_tokens <= max_tokens:
                    compressed.insert(0, turn)
                    current_tokens += turn_tokens
                else:
                    break
        
        return compressed
    
    def _calculate_turn_importance(self, turn: ConversationTurn, full_history: List[ConversationTurn]) -> float:
        """턴의 중요도 점수 계산"""
        score = 0.0
        
        # 1. 의도별 가중치
        intent_weights = {
            "compare": 1.5,    # 비교는 중요
            "recommend": 1.3,  # 추천도 중요
            "summarize": 1.2,  # 요약도 중요
            "qa": 1.0          # 기본 Q&A
        }
        score += intent_weights.get(turn.intent, 1.0)
        
        # 2. 토큰 수 (많을수록 중요)
        tokens = turn.tokens_used
        score += min(tokens / 1000, 2.0)  # 최대 2점
        
        # 3. 시간 가중치 (최근일수록 중요)
        if full_history:
            latest_time = full_history[-1].timestamp
            time_diff = (latest_time - turn.timestamp).total_seconds()
            time_score = max(0, 1.0 - (time_diff / 3600))  # 1시간 기준
            score += time_score
        
        # 4. 문서 사용량 (많을수록 중요)
        doc_count = len(turn.passages_used)
        score += min(doc_count / 5, 1.0)  # 최대 1점
        
        return score
    
    def create_context_summary(self, history: List[ConversationTurn]) -> str:
        """압축된 히스토리의 요약 생성"""
        if not history:
            return "이전 대화 내용이 없습니다."
        
        # 주요 주제 추출
        topics = []
        for turn in history:
            if turn.intent in ["compare", "recommend"]:
                topics.append(f"{turn.intent}: {turn.question[:50]}...")
        
        # 요약 생성
        summary_parts = []
        if topics:
            summary_parts.append(f"주요 주제: {', '.join(topics[:3])}")
        
        summary_parts.append(f"총 {len(history)}개 대화 턴")
        
        return " | ".join(summary_parts)


class SessionManager:
    """세션 관리자"""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = get_redis_client()
        self.context_manager = ContextWindowManager()
    
    def get_session_key(self, session_id: str) -> str:
        """세션 Redis 키 생성"""
        return f"session:{session_id}"
    
    def save_context(self, context: ConversationContext) -> bool:
        """컨텍스트를 Redis에 저장"""
        if not self.redis_client:
            return False
        
        try:
            key = self.get_session_key(context.session_id)
            data = json.dumps(context.to_dict(), ensure_ascii=False)
            
            # TTL 설정
            ttl = self.settings.REDIS_SESSION_TTL
            self.redis_client.setex(key, ttl, data)
            return True
        except Exception as e:
            print(f"⚠️ 세션 저장 실패: {e}")
            return False
    
    def load_context(self, session_id: str) -> Optional[ConversationContext]:
        """Redis에서 컨텍스트 로드"""
        if not self.redis_client:
            return None
        
        try:
            key = self.get_session_key(session_id)
            data = self.redis_client.get(key)
            
            if data:
                context_data = json.loads(data)
                return ConversationContext.from_dict(context_data)
            return None
        except Exception as e:
            print(f"⚠️ 세션 로드 실패: {e}")
            return None
    
    def create_new_context(
        self, 
        session_id: str, 
        user_id: Optional[str] = None,
        preferences: Optional[UserPreferences] = None
    ) -> ConversationContext:
        """새로운 컨텍스트 생성"""
        return ConversationContext(
            session_id=session_id,
            user_id=user_id,
            user_preferences=preferences or UserPreferences()
        )
    
    def update_context_with_turn(
        self, 
        context: ConversationContext, 
        turn: ConversationTurn
    ) -> ConversationContext:
        """턴으로 컨텍스트 업데이트"""
        # 턴 추가
        context.add_turn(turn)
        
        # 컨텍스트 압축 필요성 확인
        if len(context.conversation_history) > MAX_HISTORY_TURNS:
            # 히스토리 압축
            compressed_history = self.context_manager.compress_conversation_history(
                context.conversation_history
            )
            context.conversation_history = compressed_history
        
        # 토큰 수 기반 압축
        total_tokens = sum(turn.tokens_used for turn in context.conversation_history)
        if total_tokens > MAX_CONTEXT_TOKENS * COMPRESSION_THRESHOLD:
            compressed_history = self.context_manager.compress_conversation_history(
                context.conversation_history,
                max_tokens=int(MAX_CONTEXT_TOKENS * 0.7)  # 70%로 압축
            )
            context.conversation_history = compressed_history
        
        return context
    
    def get_context_for_llm(self, context: ConversationContext) -> str:
        """LLM용 컨텍스트 문자열 생성"""
        if not context.conversation_history:
            return "이전 대화 내용이 없습니다."
        
        # 최근 턴들만 사용 (토큰 제한 고려)
        recent_turns = context.get_recent_turns(5)
        
        context_parts = []
        for turn in recent_turns:
            context_parts.append(f"Q: {turn.question}")
            context_parts.append(f"A: {turn.answer.get('conclusion', '')}")
        
        return "\n".join(context_parts)
    
    def cleanup_expired_sessions(self) -> int:
        """만료된 세션 정리 (Redis TTL이 자동 처리하지만 수동 정리도 가능)"""
        if not self.redis_client:
            return 0
        
        try:
            # Redis TTL이 자동으로 만료된 키를 정리하므로
            # 여기서는 통계만 반환
            keys = self.redis_client.keys("session:*")
            return len(keys)
        except Exception:
            return 0


# 전역 인스턴스
session_manager = SessionManager()
context_manager = ContextWindowManager()
