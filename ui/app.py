"""
여행자보험 RAG 시스템 모니터링 UI
Streamlit을 사용한 실시간 파이프라인 모니터링 및 추적
"""

import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid
import os
import base64

# 페이지 설정
st.set_page_config(
    page_title="여행자보험 RAG 모니터링",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API 기본 설정 - Docker 환경 감지
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# CSS 제거 - Streamlit 기본 컴포넌트 사용

def render_chat_message(message_type: str, content: str, 
                       evidence: List[str] = None, caveats: List[str] = None, 
                       quality_score: float = None, error: bool = False,
                       comparison_table: Dict[str, Any] = None) -> None:
    """기본 채팅 메시지 렌더링 (마크다운 지원 포함)"""
    
    # 메시지 타입에 따른 헤더 결정
    if message_type == "user":
        header = "👤 사용자"
    else:  # assistant
        header = "🤖 AI"
    
    # 컨테이너 생성
    if message_type == "user":
        with st.container():
            st.markdown(f"**{header}**")
            st.write(content)
    else:  # assistant
        if error:
            st.error(f"**{header}**\n\n{content}")
        else:
            # AI 답변을 마크다운으로 렌더링
            st.markdown(f"**{header}**")
            st.markdown(content)  # 마크다운 렌더링
    
    # 비교 표 데이터가 있으면 표로 렌더링
    if comparison_table and isinstance(comparison_table, dict):
        headers = comparison_table.get("headers", [])
        rows = comparison_table.get("rows", [])
        
        if headers and rows:
            st.markdown("### 📊 비교 표")
            
            # DataFrame으로 변환하여 표시
            import pandas as pd
            df = pd.DataFrame(rows, columns=headers)
            st.dataframe(df, use_container_width=True)
    
    # 증거 정보 표시
    if evidence:
        with st.expander("📋 증거"):
            for i, ev in enumerate(evidence, 1):
                if isinstance(ev, dict):
                    # evidence가 객체인 경우
                    text = ev.get('text', '')
                    source = ev.get('source', '')
                    if source:
                        st.write(f"{i}. **{text}**")
                        st.caption(f"   출처: {source}")
                    else:
                        st.write(f"{i}. {text}")
                else:
                    # evidence가 문자열인 경우 (기존 호환성)
                    st.write(f"{i}. {ev}")
    
    # 주의사항 표시
    if caveats:
        with st.expander("⚠️ 주의사항"):
            for i, caveat in enumerate(caveats, 1):
                if isinstance(caveat, dict):
                    # caveat이 객체인 경우
                    text = caveat.get('text', '')
                    source = caveat.get('source', '')
                    if source:
                        st.write(f"{i}. **{text}**")
                        st.caption(f"   출처: {source}")
                    else:
                        st.write(f"{i}. {text}")
                else:
                    # caveat이 문자열인 경우 (기존 호환성)
                    st.write(f"{i}. {caveat}")
    
    # 품질 점수 표시
    if quality_score is not None:
        if quality_score >= 0.7:
            st.success(f"품질 점수: {quality_score:.2f}")
        elif quality_score >= 0.4:
            st.warning(f"품질 점수: {quality_score:.2f}")
        else:
            st.error(f"품질 점수: {quality_score:.2f}")
    
    st.markdown("---")

class RAGMonitor:
    """RAG 시스템 모니터링 클래스"""
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []
        
    def send_question(self, question: str, include_context: bool = True) -> Dict[str, Any]:
        """질문을 RAG API로 전송하고 결과 반환"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/rag/ask",
                json={
                    "question": question,
                    "session_id": self.session_id,
                    "include_context": include_context
                },
                timeout=120  # 타임아웃을 2분으로 증가
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API 호출 실패: {str(e)}")
            return {}
    
    
    def get_trace(self) -> List[Dict[str, Any]]:
        """최근 실행 trace 정보 조회"""
        try:
            response = requests.get(f"{API_BASE_URL}/rag/trace")
            response.raise_for_status()
            data = response.json()
            return data.get("trace", [])
        except requests.exceptions.RequestException as e:
            st.error(f"Trace 조회 실패: {str(e)}")
            return []
    
    def get_session_info(self) -> Dict[str, Any]:
        """세션 정보 조회"""
        try:
            response = requests.get(f"{API_BASE_URL}/rag/session/{self.session_id}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return {}

def render_pipeline_flow(trace_data: List[Dict[str, Any]]) -> None:
    """파이프라인 플로우 시각화"""
    if not trace_data:
        st.info("실행된 파이프라인이 없습니다.")
        return
    
    # 노드 순서 정의
    node_order = [
        "planner", "websearch", "search", "rank_filter", 
        "verify_refine", "answer_qa", "answer_summary", 
        "answer_compare", "answer_recommend", "reevaluate", "replan"
    ]
    
    # 실행된 노드들만 필터링
    executed_nodes = [node for node in trace_data if node.get("node") in node_order]
    
    if not executed_nodes:
        st.warning("실행된 노드 정보가 없습니다.")
        return
    
    # 플로우 차트 생성
    fig = go.Figure()
    
    # 노드 위치 계산
    node_positions = {}
    for i, node_name in enumerate(node_order):
        if any(node["node"] == node_name for node in executed_nodes):
            node_positions[node_name] = (i, 0)
    
    # 노드 그리기
    for node_name, (x, y) in node_positions.items():
        # 노드 정보 찾기
        node_info = next((node for node in executed_nodes if node["node"] == node_name), None)
        
        if node_info:
            # 실행 시간에 따른 색상 결정
            latency = node_info.get("latency_ms", 0)
            if latency > 5000:
                color = "red"
            elif latency > 2000:
                color = "orange"
            else:
                color = "green"
            
            # 노드 추가
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=50, color=color, line=dict(width=2, color='black')),
                text=[node_name],
                textposition="middle center",
                name=node_name,
                hovertemplate=f"<b>{node_name}</b><br>" +
                             f"실행시간: {latency}ms<br>" +
                             f"입력토큰: {node_info.get('in_tokens_approx', 0)}<br>" +
                             f"출력토큰: {node_info.get('out_tokens_approx', 0)}<extra></extra>"
            ))
    
    # 연결선 그리기
    for i in range(len(executed_nodes) - 1):
        current_node = executed_nodes[i]["node"]
        next_node = executed_nodes[i + 1]["node"]
        
        if current_node in node_positions and next_node in node_positions:
            x1, y1 = node_positions[current_node]
            x2, y2 = node_positions[next_node]
            
            fig.add_trace(go.Scatter(
                x=[x1, x2], y=[y1, y2],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # 레이아웃 설정
    fig.update_layout(
        title="RAG 파이프라인 실행 플로우",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_performance_metrics(trace_data: List[Dict[str, Any]]) -> None:
    """성능 메트릭 시각화"""
    if not trace_data:
        return
    
    # 데이터프레임 생성
    df = pd.DataFrame(trace_data)
    
    # 실행 시간 차트
    fig_time = px.bar(
        df, 
        x='node', 
        y='latency_ms',
        title="노드별 실행 시간 (ms)",
        color='latency_ms',
        color_continuous_scale='RdYlGn_r'
    )
    fig_time.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_time, use_container_width=True)
    
    # 토큰 사용량 차트
    col1, col2 = st.columns(2)
    
    with col1:
        fig_tokens_in = px.bar(
            df, 
            x='node', 
            y='in_tokens_approx',
            title="입력 토큰 수",
            color='in_tokens_approx',
            color_continuous_scale='Blues'
        )
        fig_tokens_in.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_tokens_in, use_container_width=True)
    
    with col2:
        fig_tokens_out = px.bar(
            df, 
            x='node', 
            y='out_tokens_approx',
            title="출력 토큰 수",
            color='out_tokens_approx',
            color_continuous_scale='Greens'
        )
        fig_tokens_out.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_tokens_out, use_container_width=True)

def render_document_analysis(passages: List[Dict[str, Any]], search_meta: Dict[str, Any] = None) -> None:
    """검색된 문서 분석"""
    if not passages:
        st.info("검색된 문서가 없습니다.")
        return
    
    st.subheader("📄 검색된 문서 분석")
    
    # 보험사 필터링 정보 표시
    if search_meta:
        insurer_filtered = search_meta.get('insurer_filtered', False)
        insurer_filter = search_meta.get('insurer_filter', [])
        filter_method = search_meta.get('filter_method', 'unknown')
        
        if insurer_filtered and insurer_filter:
            st.success(f"🎯 보험사 필터링 적용: {', '.join(insurer_filter)}")
            st.info(f"필터링 방법: {filter_method}")
            
            # 필터링된 보험사 문서 수 표시
            filtered_insurer_counts = {}
            for passage in passages:
                insurer = passage.get('insurer', 'Unknown')
                if insurer in insurer_filter:
                    filtered_insurer_counts[insurer] = filtered_insurer_counts.get(insurer, 0) + 1
            
            if filtered_insurer_counts:
                st.write("**필터링된 보험사별 문서 수:**")
                for insurer, count in filtered_insurer_counts.items():
                    st.write(f"  - {insurer}: {count}개")
        else:
            st.info("ℹ️ 보험사 필터링 없음 - 전체 문서 검색")
    
    # 문서 소스별 분류
    sources = {}
    for passage in passages:
        source = passage.get("source", "unknown")
        if source not in sources:
            sources[source] = []
        sources[source].append(passage)
    
    # 소스별 문서 수 표시
    source_counts = {source: len(docs) for source, docs in sources.items()}
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 소스별 문서 수 차트
        fig_sources = px.pie(
            values=list(source_counts.values()),
            names=list(source_counts.keys()),
            title="문서 소스별 분포"
        )
        st.plotly_chart(fig_sources, use_container_width=True)
    
    with col2:
        # 문서 점수 분포
        scores = [p.get("score", 0) for p in passages]
        fig_scores = px.histogram(
            x=scores,
            title="문서 관련성 점수 분포",
            nbins=20
        )
        st.plotly_chart(fig_scores, use_container_width=True)
    
    # 보험사별 문서 분석
    insurer_docs = {}
    for passage in passages:
        insurer = passage.get('insurer', 'Unknown')
        if insurer not in insurer_docs:
            insurer_docs[insurer] = []
        insurer_docs[insurer].append(passage)
    
    if insurer_docs:
        st.subheader("🏢 보험사별 문서 분포")
        
        # 보험사별 문서 수
        insurer_counts = {insurer: len(docs) for insurer, docs in insurer_docs.items()}
        
        # 보험사 필터링이 적용된 경우 강조 표시
        if search_meta and search_meta.get('insurer_filtered', False):
            target_insurers = search_meta.get('insurer_filter', [])
            st.info(f"🎯 필터링된 보험사: {', '.join(target_insurers)}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 보험사별 문서 수 차트
            fig_insurers = px.bar(
                x=list(insurer_counts.keys()),
                y=list(insurer_counts.values()),
                title="보험사별 문서 수",
                labels={'x': '보험사', 'y': '문서 수'}
            )
            st.plotly_chart(fig_insurers, use_container_width=True)
        
        with col2:
            # 보험사별 평균 점수
            insurer_avg_scores = {}
            for insurer, docs in insurer_docs.items():
                avg_score = sum(d.get('score', 0) for d in docs) / len(docs)
                insurer_avg_scores[insurer] = avg_score
            
            fig_scores = px.bar(
                x=list(insurer_avg_scores.keys()),
                y=list(insurer_avg_scores.values()),
                title="보험사별 평균 관련성 점수",
                labels={'x': '보험사', 'y': '평균 점수'}
            )
            st.plotly_chart(fig_scores, use_container_width=True)
    
    # 상세 문서 정보
    st.subheader("📋 문서 상세 정보")
    
    # 점수 높은 순으로 정렬
    sorted_passages = sorted(passages, key=lambda x: x.get('score', 0), reverse=True)
    
    for i, passage in enumerate(sorted_passages[:10]):  # 상위 10개만 표시
        # 보험사 부스트 여부 표시
        title_suffix = ""
        if passage.get('insurer_boost', False):
            title_suffix += " 🎯"
        if passage.get('target_insurer', False):
            title_suffix += " ⭐"
        
        # 보험사 필터링이 적용된 경우 해당 보험사 문서 강조
        insurer = passage.get('insurer', 'N/A')
        if search_meta and search_meta.get('insurer_filtered', False):
            target_insurers = search_meta.get('insurer_filter', [])
            if insurer in target_insurers:
                title_suffix += " 🎯"
        
        with st.expander(f"문서 {i+1}: {passage.get('title', '제목 없음')} (점수: {passage.get('score', 0):.3f}){title_suffix}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**소스**: {passage.get('source', 'unknown')}")
                st.write(f"**점수**: {passage.get('score', 0):.3f}")
            
            with col2:
                st.write(f"**페이지**: {passage.get('page', 'N/A')}")
                st.write(f"**문서ID**: {passage.get('doc_id', 'N/A')}")
            
            with col3:
                insurer_display = insurer
                if passage.get('target_insurer', False):
                    insurer_display += " ⭐"
                
                # 보험사 필터링이 적용된 경우 해당 보험사 강조
                if search_meta and search_meta.get('insurer_filtered', False):
                    target_insurers = search_meta.get('insurer_filter', [])
                    if insurer in target_insurers:
                        insurer_display += " 🎯"
                
                st.write(f"**보험사**: {insurer_display}")
                if passage.get('url'):
                    st.write(f"**URL**: {passage.get('url')}")
            
            # 보험사 부스트 정보
            if passage.get('insurer_boost', False):
                st.info("🎯 이 문서는 질문에서 언급된 보험사의 문서로 우선순위가 부여되었습니다.")
            
            # 보험사 필터링 정보
            if search_meta and search_meta.get('insurer_filtered', False):
                target_insurers = search_meta.get('insurer_filter', [])
                if insurer in target_insurers:
                    st.success(f"🎯 이 문서는 필터링된 보험사({insurer})의 문서입니다.")
            
            # 문서 내용 미리보기
            text = passage.get('text', '')
            if text:
                st.text_area(
                    "문서 내용",
                    text[:500] + "..." if len(text) > 500 else text,
                    height=100,
                    disabled=True
                )

    

def main():
    """메인 애플리케이션"""
    st.title("🛡️ 여행자보험 RAG 시스템 모니터링")
    st.markdown("---")
    
    # 세션 상태 초기화
    if 'monitor' not in st.session_state:
        st.session_state.monitor = RAGMonitor()
    
    monitor = st.session_state.monitor
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # API 연결 상태 확인
        try:
            response = requests.get(f"{API_BASE_URL}/", timeout=5)
            if response.status_code == 200:
                st.success("✅ API 연결됨")
            else:
                st.error("❌ API 연결 실패")
        except:
            st.error("❌ API 서버에 연결할 수 없습니다")
        
        # 상세 진단 정보
        if st.button("🔍 상세 진단", help="API 상태 및 설정을 자세히 확인합니다"):
            with st.spinner("진단 중..."):
                try:
                    api_status = requests.get(f"{API_BASE_URL}/api-status", timeout=10).json()
                    
                    st.subheader("📊 API 상태 진단")
                    
                    # Gemini 상태
                    gemini = api_status.get("gemini", {})
                    st.write("**🤖 Gemini API**")
                    st.write(f"상태: {gemini.get('status', '❓')}")
                    st.write(f"모델: {gemini.get('model', '❓')}")
                    st.write(f"API 키: {gemini.get('api_key', '❓')}")
                    
                    if gemini.get('error'):
                        st.error(f"오류: {gemini.get('error')}")
                        st.info("💡 해결 방법:")
                        if "API 키" in gemini.get('error', ''):
                            st.write("1. .env 파일에 GEMINI_API_KEY가 올바르게 설정되었는지 확인")
                            st.write("2. API 키가 유효한지 Google AI Studio에서 확인")
                        elif "할당량" in gemini.get('error', ''):
                            st.write("1. Google AI Studio에서 할당량 확인")
                            st.write("2. 잠시 후 다시 시도")
                        elif "권한" in gemini.get('error', ''):
                            st.write("1. API 키에 해당 모델 접근 권한이 있는지 확인")
                            st.write("2. 모델 이름이 올바른지 확인")
                        elif "네트워크" in gemini.get('error', ''):
                            st.write("1. 인터넷 연결 확인")
                            st.write("2. 방화벽 설정 확인")
                    
                    # 사용 가능한 모델
                    available_models = gemini.get('available_models', [])
                    if available_models:
                        st.write(f"**사용 가능한 모델**: {', '.join(available_models[:5])}")
                        if len(available_models) > 5:
                            st.write(f"... 외 {len(available_models) - 5}개")
                    
                    # Tavily 상태
                    tavily = api_status.get("tavily", {})
                    st.write("**🌐 Tavily API**")
                    st.write(f"상태: {tavily.get('status', '❓')}")
                    
                    # 전체 상태
                    overall = api_status.get("overall", "❓")
                    if "✅" in overall:
                        st.success(f"전체 상태: {overall}")
                    else:
                        st.error(f"전체 상태: {overall}")
                        
                except Exception as e:
                    st.error(f"진단 실패: {str(e)}")
                    st.info("API 서버가 실행 중인지 확인해주세요.")
        
        st.markdown("---")
        
        # 세션 정보
        st.subheader("📊 세션 정보")
        st.write(f"**세션 ID**: {monitor.session_id[:8]}...")
        
        session_info = monitor.get_session_info()
        if session_info:
            context = session_info.get('context', {})
            st.write(f"**대화 수**: {context.get('turn_count', 0)}")
            st.write(f"**생성 시간**: {context.get('created_at', 'N/A')}")
        
        st.markdown("---")
        
        # 캐시 통계
        try:
            cache_response = requests.get(f"{API_BASE_URL}/rag/cache/stats")
            if cache_response.status_code == 200:
                cache_data = cache_response.json()
                st.subheader("💾 캐시 통계")
                cache_stats = cache_data.get('cache_stats', {})
                st.write(f"**임베딩 캐시**: {cache_stats.get('embeddings', 0)}")
                st.write(f"**검색 캐시**: {cache_stats.get('search', 0)}")
                st.write(f"**LLM 캐시**: {cache_stats.get('llm_response', 0)}")
        except:
            pass
    
    # 메인 컨텐츠
    tab1, tab2, tab3 = st.tabs(["🔍 질문하기", "📊 파이프라인 모니터링", "📄 문서 분석"])
    
    with tab1:
        
        # 기존 대화 히스토리 표시
        if monitor.conversation_history:
            for chat in monitor.conversation_history:
                # 사용자 질문 표시
                render_chat_message(
                    message_type="user",
                    content=chat['question']
                )
                
                # AI 답변 표시
                result = chat['result']
                answer = None
                if 'final_answer' in result and result['final_answer']:
                    answer = result['final_answer']
                elif 'draft_answer' in result and result['draft_answer']:
                    answer = result['draft_answer']
                
                if answer:
                    conclusion = answer.get('conclusion', answer.get('content', '답변을 생성할 수 없습니다.'))
                    evidence = answer.get('evidence', [])
                    caveats = answer.get('caveats', [])
                    quality_score = result.get('quality_score', 0)
                    comparison_table = answer.get('comparison_table', None)
                    
                    render_chat_message(
                        message_type="assistant",
                        content=conclusion,
                        evidence=evidence,
                        caveats=caveats,
                        quality_score=quality_score,
                        comparison_table=comparison_table
                    )
                else:
                    render_chat_message(
                        message_type="assistant",
                        content="답변을 생성할 수 없습니다.",
                        error=True
                    )
        
        # 질문 입력 폼
        st.markdown("---")
        
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                question = st.text_area(
                    "여행자보험에 대해 질문해주세요:",
                    placeholder="예: 해외여행 보험료는 얼마인가요?",
                    height=80,
                    key="question_input"
                )
            
            with col2:
                include_context = st.checkbox("대화 컨텍스트 포함", value=True, key="context_checkbox")
                submit_button = st.form_submit_button("전송", type="primary")
        
        if submit_button and question.strip():
            # 사용자 질문을 즉시 표시
            render_chat_message(
                message_type="user",
                content=question
            )
            
            # 진행 상황 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                start_time = time.time()
                
                # 진행 상황 표시
                status_text.text("🔍 질문을 분석하고 있습니다...")
                progress_bar.progress(20)
                
                result = monitor.send_question(question, include_context)
                
                # 진행 상황 업데이트
                status_text.text("📊 파이프라인을 실행하고 있습니다...")
                progress_bar.progress(60)
                end_time = time.time()
                
                progress_bar.progress(100)
                status_text.text("✅ 답변 생성 완료!")
                
                if result:
                    # 답변 내용
                    answer = None
                    if 'final_answer' in result and result['final_answer']:
                        answer = result['final_answer']
                    elif 'draft_answer' in result and result['draft_answer']:
                        answer = result['draft_answer']
                    
                    if answer:
                        conclusion = answer.get('conclusion', answer.get('content', '답변을 생성할 수 없습니다.'))
                        evidence = answer.get('evidence', [])
                        caveats = answer.get('caveats', [])
                        quality_score = result.get('quality_score', 0)
                        comparison_table = answer.get('comparison_table', None)
                        
                        # AI 답변 표시
                        render_chat_message(
                            message_type="assistant",
                            content=conclusion,
                            evidence=evidence,
                            caveats=caveats,
                            quality_score=quality_score,
                            comparison_table=comparison_table
                        )
                    else:
                        render_chat_message(
                            message_type="assistant",
                            content="답변을 생성할 수 없습니다.",
                            error=True
                        )
                    
                    # 대화 히스토리에 추가
                    monitor.conversation_history.append({
                        'question': question,
                        'result': result,
                        'timestamp': datetime.now()
                    })
                    
                    # 진행 상황 초기화
                    progress_bar.progress(0)
                    status_text.text("")
                    
                    # 페이지 새로고침하여 모니터링 탭 업데이트
                    st.rerun()
                else:
                    render_chat_message(
                        message_type="assistant",
                        content="답변 생성에 실패했습니다. API 서버 상태를 확인해주세요.",
                        error=True
                    )
                    
            except Exception as e:
                render_chat_message(
                    message_type="assistant",
                    content=f"오류가 발생했습니다: {str(e)}",
                    error=True
                )
                
                # 진행 상황 초기화
                progress_bar.progress(0)
                status_text.text("")
        
        elif submit_button and not question.strip():
            st.warning("질문을 입력해주세요.")
        
    
    with tab2:
        st.header("파이프라인 모니터링")
        
        # 최근 trace 정보 조회
        trace_data = monitor.get_trace()
        
        if trace_data:
            # 파이프라인 플로우
            st.subheader("🔄 파이프라인 실행 플로우")
            render_pipeline_flow(trace_data)
            
            # 성능 메트릭
            st.subheader("📈 성능 메트릭")
            render_performance_metrics(trace_data)
            
            # 상세 trace 정보
            st.subheader("🔍 상세 실행 정보")
            
            # Trace 데이터프레임
            df = pd.DataFrame(trace_data)
            st.dataframe(df, use_container_width=True)
            
            # 총 실행 시간
            total_time = sum(node.get('latency_ms', 0) for node in trace_data)
            total_tokens = sum(node.get('out_tokens_approx', 0) for node in trace_data)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 실행 시간", f"{total_time}ms")
            with col2:
                st.metric("총 토큰 수", f"{total_tokens:,}")
            with col3:
                st.metric("실행된 노드 수", len(trace_data))
            
            # 보험사 필터링 정보
            planner_nodes = [node for node in trace_data if node.get('node_name') == 'planner']
            if planner_nodes:
                st.subheader("🎯 보험사 필터링 정보")
                planner_meta = planner_nodes[0]
                insurer_filter = planner_meta.get('insurer_filter', [])
                extracted_insurers = planner_meta.get('extracted_insurers', [])
                owned_insurers = planner_meta.get('owned_insurers', [])
                non_owned_insurers = planner_meta.get('non_owned_insurers', [])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("추출된 보험사", f"{len(extracted_insurers)}개")
                with col2:
                    st.metric("보유 보험사", f"{len(owned_insurers)}개")
                with col3:
                    st.metric("비보유 보험사", f"{len(non_owned_insurers)}개")
                
                if extracted_insurers:
                    st.info(f"🔍 질문에서 추출된 보험사: {', '.join(extracted_insurers)}")
                
                if insurer_filter:
                    st.success(f"🎯 적용된 보험사 필터: {', '.join(insurer_filter)}")
                else:
                    st.info("ℹ️ 보험사 필터링 없음 - 전체 문서 검색")
                
                if non_owned_insurers:
                    st.warning(f"⚠️ 비보유 보험사로 인한 웹검색 필요: {', '.join(non_owned_insurers)}")
            
            # re-evaluate 노드 정보
            reevaluate_nodes = [node for node in trace_data if node.get('node_name') == 'reevaluate']
            if reevaluate_nodes:
                st.subheader("🔍 답변 품질 평가")
                reevaluate_meta = reevaluate_nodes[0].get('reevaluate_meta', {})
                quality_score = reevaluate_meta.get('quality_score', 0)
                needs_replan = reevaluate_meta.get('needs_replan', False)
                replan_count = reevaluate_meta.get('replan_count', 0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("품질 점수", f"{quality_score:.2f}")
                with col2:
                    st.metric("재검색 필요", "✅" if needs_replan else "❌")
                with col3:
                    st.metric("재검색 횟수", f"{replan_count}/3")
                
                if needs_replan:
                    st.warning("⚠️ 답변 품질이 기준(0.7) 미만으로 재검색이 필요합니다.")
                else:
                    st.success("✅ 답변 품질이 기준을 만족합니다.")
        else:
            st.info("실행된 파이프라인이 없습니다. 먼저 질문을 해보세요.")
    
    with tab3:
        st.header("문서 분석")
        
        # 최근 질문의 검색 결과 분석
        if monitor.conversation_history:
            latest_result = monitor.conversation_history[-1]['result']
            passages = latest_result.get('passages', [])
            search_meta = latest_result.get('search_meta', {})
            
            if passages:
                render_document_analysis(passages, search_meta)
            else:
                st.info("검색된 문서가 없습니다.")
        else:
            st.info("질문을 먼저 해보세요.")
    

if __name__ == "__main__":
    main()
