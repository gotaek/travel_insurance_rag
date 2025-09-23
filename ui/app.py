import streamlit as st
import requests
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

# 페이지 설정
st.set_page_config(
    page_title="여행자보험 RAG 시스템 - 멀티턴 대화",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-block {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .conclusion {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .evidence {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    .caveats {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    .quotes {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
    .trace-step {
        background-color: #f5f5f5;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
        font-size: 0.9rem;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
        margin-left: 2rem;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
        margin-right: 2rem;
    }
    .session-info {
        background-color: #e8f5e8;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# API 설정
API_BASE_URL = "http://api:8000"

# 세션 상태 초기화
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{int(time.time())}"
if "last_response_time" not in st.session_state:
    st.session_state.last_response_time = None

def create_session() -> Optional[str]:
    """새로운 세션 생성"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/rag/session/create",
            params={"user_id": st.session_state.user_id},
            timeout=10
        )
        if response.status_code == 200:
            session_data = response.json()
            return session_data["session_id"]
        else:
            st.error(f"세션 생성 실패: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"세션 생성 오류: {str(e)}")
        return None

def get_session_info(session_id: str) -> Optional[Dict[str, Any]]:
    """세션 정보 조회"""
    try:
        response = requests.get(f"{API_BASE_URL}/rag/session/{session_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"세션 정보 조회 오류: {str(e)}")
        return None

def call_multiturn_api(question: str, session_id: str) -> Dict[str, Any]:
    """멀티턴 RAG API 호출"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/rag/multiturn/ask",
            json={
                "question": question,
                "session_id": session_id,
                "user_id": st.session_state.user_id,
                "include_context": True
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API 호출 오류: {str(e)}")
        return None

def call_legacy_api(question: str) -> Dict[str, Any]:
    """기존 RAG API 호출 (하위 호환성)"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/rag/ask",
            json={
                "question": question,
                "session_id": st.session_state.session_id,
                "user_id": st.session_state.user_id,
                "include_context": True
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API 호출 오류: {str(e)}")
        return None

def render_answer_block(title: str, content: Any, block_type: str = ""):
    """답변 블록 렌더링"""
    if not content:
        return
    
    if isinstance(content, list):
        if not content:
            return
        content_text = "\n".join([f"• {item}" for item in content])
    else:
        content_text = str(content)
    
    st.markdown(f"""
    <div class="answer-block {block_type}">
        <h4>{title}</h4>
        <p>{content_text}</p>
    </div>
    """, unsafe_allow_html=True)

def render_chat_message(role: str, content: str, timestamp: str = None):
    """채팅 메시지 렌더링"""
    role_emoji = "👤" if role == "user" else "🤖"
    role_class = "user-message" if role == "user" else "assistant-message"
    
    time_str = f"<small>{timestamp}</small>" if timestamp else ""
    
    st.markdown(f"""
    <div class="chat-message {role_class}">
        <strong>{role_emoji} {role.title()}:</strong><br>
        {content}<br>
        {time_str}
    </div>
    """, unsafe_allow_html=True)

def main():
    # 헤더
    st.markdown('<h1 class="main-header">✈️ 여행자보험 RAG 시스템 - 멀티턴 대화</h1>', unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.markdown("## 🔧 세션 관리")
        
        # 세션 ID 표시
        if st.session_state.session_id:
            st.markdown(f"""
            <div class="session-info">
                <strong>현재 세션:</strong><br>
                <code>{st.session_state.session_id[:16]}...</code><br>
                <strong>사용자:</strong> {st.session_state.user_id}
            </div>
            """, unsafe_allow_html=True)
            
            # 세션 정보 조회
            if st.button("📊 세션 정보 조회"):
                with st.spinner("세션 정보를 조회하는 중..."):
                    session_info = get_session_info(st.session_state.session_id)
                    if session_info:
                        st.json(session_info)
                    else:
                        st.error("세션 정보를 가져올 수 없습니다.")
            
            # 새 세션 시작
            if st.button("🆕 새 대화 시작"):
                st.session_state.session_id = None
                st.session_state.conversation_history = []
                st.session_state.last_response_time = None
                st.rerun()
        else:
            st.info("새 대화를 시작하려면 '새 대화 시작' 버튼을 클릭하세요.")
            if st.button("🆕 새 대화 시작"):
                with st.spinner("새 세션을 생성하는 중..."):
                    session_id = create_session()
                    if session_id:
                        st.session_state.session_id = session_id
                        st.session_state.conversation_history = []
                        st.session_state.last_response_time = None
                        st.success("새 대화가 시작되었습니다!")
                        st.rerun()
        
        st.markdown("## 📋 사용 가이드")
        st.markdown("""
        **질문 유형:**
        - **QA**: "비행기 연착 시 보장 알려줘"
        - **요약**: "카카오페이 여행자보험 약관 요약"
        - **비교**: "보험사별 여행자보험 차이 비교"
        - **추천**: "LA 여행 특약 추천"
        
        **멀티턴 대화:**
        - 첫 질문: "여행자보험에 대해 알려주세요"
        - 후속 질문: "그 보험의 보상 한도는?"
        - 시스템이 이전 대화를 기억합니다!
        """)
        
        st.markdown("## 🔧 시스템 상태")
        try:
            response = requests.get(f"{API_BASE_URL}/healthz", timeout=5)
            if response.status_code == 200:
                st.success("✅ API 서버 정상")
            else:
                st.error("❌ API 서버 오류")
        except:
            st.error("❌ API 서버 연결 실패")
        
        # 캐시 통계
        try:
            response = requests.get(f"{API_BASE_URL}/rag/cache/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                st.markdown("## 📊 캐시 통계")
                cache_stats = stats.get("cache_stats", {})
                st.metric("임베딩 캐시", cache_stats.get("embeddings_count", 0))
                st.metric("검색 캐시", cache_stats.get("search_count", 0))
                st.metric("세션 수", cache_stats.get("session_count", 0))
        except:
            pass
    
    # 메인 컨텐츠
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 💬 대화하기")
        
        # 대화 히스토리 표시
        if st.session_state.conversation_history:
            st.markdown("### 📜 대화 히스토리")
            for msg in st.session_state.conversation_history:
                render_chat_message(
                    msg["role"], 
                    msg["content"], 
                    msg.get("timestamp")
                )
        
        # 질문 입력
        question = st.text_area(
            "여행자보험에 대해 궁금한 것을 물어보세요:",
            placeholder="예: 비행기 연착 시 보장 알려줘",
            height=100,
            key="question_input"
        )
        
        # 답변 요청 버튼
        col_btn1, col_btn2 = st.columns([1, 1])
        
        with col_btn1:
            if st.button("🚀 답변 생성", type="primary", disabled=not question):
                if question:
                    # 세션이 없으면 생성
                    if not st.session_state.session_id:
                        with st.spinner("새 세션을 생성하는 중..."):
                            session_id = create_session()
                            if session_id:
                                st.session_state.session_id = session_id
                            else:
                                st.error("세션 생성에 실패했습니다.")
                                return
                    
                    with st.spinner("답변을 생성하는 중..."):
                        start_time = time.time()
                        result = call_multiturn_api(question, st.session_state.session_id)
                        end_time = time.time()
                        
                        # 실행 시간을 세션 상태에 저장
                        st.session_state.last_response_time = end_time - start_time 
                        if result:
                            # 사용자 메시지 추가
                            st.session_state.conversation_history.append({
                                "role": "user",
                                "content": question,
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            
                            # 답변 추출
                            draft_answer = result.get("draft_answer", {})
                            conclusion = draft_answer.get("conclusion", "")
                            
                            # 어시스턴트 메시지 추가
                            st.session_state.conversation_history.append({
                                "role": "assistant",
                                "content": conclusion,
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                "full_response": result
                            })
                            
                            st.success("답변이 생성되었습니다!")
                            st.rerun()
                        else:
                            st.error("답변 생성에 실패했습니다.")
                else:
                    st.warning("질문을 입력해주세요.")
        
        # 수정된 코드
        with col_btn2:
            if st.button("🗑️ 대화 초기화"):
                st.session_state.conversation_history = []
                st.session_state.last_response_time = None
                st.rerun()
    
    with col2:
        st.markdown("## 📝 최신 답변")
        
        # 최신 답변 표시
        if st.session_state.conversation_history:
            latest_assistant = None
            for msg in reversed(st.session_state.conversation_history):
                if msg["role"] == "assistant":
                    latest_assistant = msg
                    break
            
            if latest_assistant:
                full_response = latest_assistant.get("full_response", {})
                draft_answer = full_response.get("draft_answer", {})
                
                # 결론
                conclusion = draft_answer.get("conclusion", "")
                if conclusion:
                    render_answer_block("🎯 결론", conclusion, "conclusion")
                
                # 근거
                evidence = draft_answer.get("evidence", [])
                if evidence:
                    render_answer_block("📚 근거", evidence, "evidence")
                
                # 예외/주의
                caveats = draft_answer.get("caveats", [])
                if caveats:
                    render_answer_block("⚠️ 예외/주의", caveats, "caveats")
                
                # 원문 인용
                quotes = draft_answer.get("quotes", [])
                if quotes:
                    st.markdown("### 📖 원문 인용")
                    for i, quote in enumerate(quotes, 1):
                        text = quote.get("text", "")
                        source = quote.get("source", "알 수 없음")
                        st.markdown(f"""
                        <div class="answer-block quotes">
                            <p><strong>인용 {i}:</strong> {text}</p>
                            <p><small>출처: {source}</small></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # 컨텍스트 정보
                conversation_context = full_response.get("conversation_context", {})
                if conversation_context:
                    st.markdown("### 🔄 대화 컨텍스트")
                    st.info(f"""
                    **턴 수:** {conversation_context.get('turn_count', 0)}  
                    **토큰 수:** {conversation_context.get('total_tokens', 0)}  
                    **마지막 활동:** {conversation_context.get('last_activity', 'N/A')}
                    """)
                
                # 실행 시간
                if st.session_state.last_response_time is not None:
                    st.info(f"⏱️ 총 실행 시간: {st.session_state.last_response_time:.2f}초")
                
                # 트레이스 로그
                trace = full_response.get("trace", [])
                if trace:
                    st.markdown("### 🔍 실행 과정")
                    for i, step in enumerate(trace, 1):
                        node_name = step.get("node", "알 수 없음")
                        duration = step.get("duration", 0)
                        status = step.get("status", "완료")
                        st.markdown(f"""
                        <div class="trace-step">
                            <strong>단계 {i}:</strong> {node_name} ({status}) - {duration:.2f}초
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("아직 대화가 없습니다. 질문을 입력해보세요!")

if __name__ == "__main__":
    main()