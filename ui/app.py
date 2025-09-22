import streamlit as st
import requests
import json
import time
from typing import Dict, Any, List

# 페이지 설정
st.set_page_config(
    page_title="여행자보험 RAG 시스템",
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
</style>
""", unsafe_allow_html=True)

# API 설정
API_BASE_URL = "http://api:8000"

def call_rag_api(question: str) -> Dict[str, Any]:
    """RAG API 호출"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/rag/ask",
            json={"question": question},
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

def main():
    # 헤더
    st.markdown('<h1 class="main-header">✈️ 여행자보험 RAG 시스템</h1>', unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.markdown("## 📋 사용 가이드")
        st.markdown("""
        **질문 유형:**
        - **QA**: "비행기 연착 시 보장 알려줘"
        - **요약**: "카카오페이 여행자보험 약관 요약"
        - **비교**: "보험사별 여행자보험 차이 비교"
        - **추천**: "LA 여행 특약 추천"
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
    
    # 메인 컨텐츠
    st.markdown("## 질문하기")
    
    # 질문 입력
    question = st.text_area(
        "여행자보험에 대해 궁금한 것을 물어보세요:",
        placeholder="예: 비행기 연착 시 보장 알려줘",
        height=100
    )
    
    # 답변 요청 버튼
    if st.button("🚀 답변 생성", type="primary"):
        if question:
            with st.spinner("답변을 생성하는 중..."):
                start_time = time.time()
                result = call_rag_api(question)
                end_time = time.time()
                
                if result:
                    st.markdown("## 📝 답변")
                    
                    # 답변 블록들 렌더링
                    draft_answer = result.get("draft_answer", {})
                    
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
                        st.markdown("### �� 원문 인용")
                        for i, quote in enumerate(quotes, 1):
                            text = quote.get("text", "")
                            source = quote.get("source", "알 수 없음")
                            st.markdown(f"""
                            <div class="answer-block quotes">
                                <p><strong>인용 {i}:</strong> {text}</p>
                                <p><small>출처: {source}</small></p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # 실행 시간
                    st.info(f"⏱️ 총 실행 시간: {end_time - start_time:.2f}초")
                    
                    # 트레이스 로그
                    trace = result.get("trace", [])
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
                    st.error("답변 생성에 실패했습니다.")
        else:
            st.warning("질문을 입력해주세요.")

if __name__ == "__main__":
    main()