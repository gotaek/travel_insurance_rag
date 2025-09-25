INSURANCE_RAG/
├─ app/                         # FastAPI 서비스 계층: 엔드포인트, 의존성 주입, 예외/로그 표준화
│  ├─ main.py                   # FastAPI 인스턴스 생성, 미들웨어(CORS, OTEL), 라우터 include, /healthz 등록
│  ├─ deps.py                   # DI 컨테이너: LLM 클라이언트, Redis, VectorStore 핸들러, Graph 인스턴스 팩토리
│  └─ routers/
│     ├─ rag.py                 # /rag/ask, /rag/plan, /rag/trace
│     │                         #  - 입력: {question, trip_info?, k?, temperature?}
│     │                         #  - 내부: LangGraph.invoke(state) 호출
│     │                         #  - 출력: (결론/근거/예외/원문인용 + trace) JSON
│     │                         #  - 실패: GraphError → 422/500 매핑, 표준 에러 포맷 반환
│     └─ health.py              # /healthz, /readyz, /metrics(선택) — 벡터 인덱스/Redis/LLM 키 점검
│
├─ graph/                       # LangGraph 오케스트레이션: 상태 정의, 노드/엣지, 프롬프트
│  ├─ __init__.py               # 그래프 버전/이름 상수, 빌더 export
│  ├─ state.py                  # RAGState TypedDict: question, intent, needs_web, passages, citations, trace 등
│  ├─ nodes/
│  │  ├─ planner.py             # 쿼리 분석/플래너: intent 판별(summary/compare/qa/recommend), needs_web 결정, plan 작성
│  │  ├─ search.py              # 검색 단계: retriever.hybrid 호출, k/스코어 정규화, 문서/페이지 메타 부착
│  │  ├─ rank_filter.py         # 컨텍스트 정제: LLM/규칙 기반 필터링, 중복 제거, 문단 가위질, 토큰 컷
│  │  ├─ answerers/             # 답변 생성 계층: 유형별 프롬프트와 출력 스키마 고정
│  │  │  ├─ summarize.py        # 약관 요약: “초등학생 설명” 톤, 불릿/용어풀이, 페이지 표기 의무
│  │  │  ├─ compare.py          # 보험사 비교: 표 형태 기본, 차이점 위주, 동일항목 묶음/하이라이트
│  │  │  ├─ recommend.py        # 특약 추천: 일정/지역 + websearch 결과 반영, 근거·가정 명시
│  │  │  └─ qa.py               # 청구서류/절차 Q&A: 단계별 체크리스트 + 필수 서류/예외
│  │  ├─ verify_refine.py       # 사실성/인용 검증: 누락 시 재검색 루프(최대 N회), 법적 문구 삽입
│  │  ├─ websearch.py           # 뉴스/웹 검색: 필요시만 호출, 최신성 타임박스, 출처 도메인 화이트리스트
│  │  └─ trace.py               # 노드별 latency/tokens/score 기록, OpenTelemetry span 생성
│  ├─ edges.py                  # 조건부 전이 정의: needs_web True→websearch, intent별 answerer 라우팅, 실패 시 백트래킹
│  ├─ builder.py                # StateGraph 조립: 노드 등록/엣지 연결/메모리/체크포인트 구성, graph.compile() export
│  └─ prompts/                  # 에이전트별 시스템 프롬프트(버전 관리 대상)
│     ├─ system_core.md         # 공통 가드레일: 답변 포맷(결론/근거/예외/원문), 금칙어, 톤, 인용 규칙
│     ├─ summarize.md           # 요약 전용 지침: 용어 평이화, 숫자/연령/기간 강조 방식
│     ├─ compare.md             # 비교 전용 지침: 표 스펙, 정렬 기준, 동률 처리
│     ├─ recommend.md           # 추천 전용 지침: 위험도, 가정, 최신성 주석, 불확실성 표기
│     └─ qa.md                  # QA 전용 지침: 단계/체크리스트/필수 서류 템플릿
│
├─ retriever/                   # 검색 계층: Chroma DB 백엔드, 하이브리드 전략
│  ├─ __init__.py               # 팩토리/인터페이스 export
│  ├─ vector.py                 # Chroma DB 벡터 스토어, top_k 검색, 컬렉션 관리
│  ├─ keyword.py                # BM25/TF-IDF 래퍼, 토큰화/불용어, 정규화 스코어 반환
│  └─ hybrid.py                 # 벡터+키워드 머지: 스코어 min-max 정규화, 중복 문서 병합, 소스메타 유지
│
├─ data/
│  ├─ documents/                # PDF 원천: 파일명 규칙 insurer_product_version(pages).pdf 유지
│  └─ vector_db/                # Chroma DB (자동 생성, git 제외)
│     └─ insurance_docs/        # Chroma DB 컬렉션 (make ingest로 재생성)
│
├─ ui/                          # Streamlit 모니터링 UI
│  ├─ app.py                    # 메인 모니터링 UI: 파이프라인 추적, 문서 분석, 대화 히스토리
│  ├─ demo.py                   # 데모 UI: API 서버 없이 UI 기능 테스트
│  ├─ run_ui.py                 # UI 실행 스크립트
│  └─ README.md                  # UI 사용법 및 기능 설명
│
├─ eval/                        # 오프라인 평가: 회귀 테스트와 지표 일괄 산출
│  ├─ ragas_pipeline.py         # RAGAS 실행 파이프라인(questions.jsonl→scores.json)
│  ├─ faithfulness.py           # entailment 기반 사실성 점수(LLM/모델 선택 가능)
│  └─ recall_at_k.py            # 검색 단계 정답 포함률 측정(골든 인용 대비)
│
├─ scripts/
│  ├─ ingest.py                 # 파서→청크→임베딩→Chroma DB 빌드; 증분 업데이트, 실패 재시도, 중복 방지
│  ├─ rebuild_vector.sh         # 인덱스 재생성 셸: 환경변수 로드, 병렬처리 옵션, 산출물 검증 해시 출력
│  └─ smoke_test.py             # 최소 동작 확인: 대표 질의 세트로 planner/answerer 라우팅 검증
│
├─ config/
│  ├─ settings.py               # pydantic-settings: API 키, 경로, 검색 파라미터, 추천 기본값, 타임아웃/리트라이
│  └─ policies.yaml             # 컴플라이언스 규칙/법적 문구/민감어 리스트/출처 화이트리스트(웹검색)
│
├─ Makefile                     # make dev/run/test/eval/ingest/docker 빌드 타깃 정의
├─ docker-compose.yml           # fastapi + (선택)redis + (선택)otel-collector; 헬스체크/리소스 제한
├─ Dockerfile                   # 멀티스테이지 빌드, 캐시 최적화, 비루트 사용자, 건강검진(CMD curl /healthz)
├─ README.md                    # 로컬 실행, API 스펙, 평가 방법, 아키텍처 다이어그램, 운영 가이드
└─ requirements.txt             # 정확한 버전 핀 고정; prod/dev extras 분리 가능(requirements-dev.txt)

## 🖥️ UI 모니터링 시스템

### Streamlit 기반 실시간 모니터링

이 프로젝트는 RAG 파이프라인의 실시간 모니터링을 위한 Streamlit UI를 제공합니다.

#### 주요 기능

1. **🔍 질문하기**: 여행자보험 관련 질문 입력 및 실시간 답변 생성
2. **📊 파이프라인 모니터링**: 
   - 노드별 실행 플로우 시각화
   - 성능 메트릭 (실행 시간, 토큰 사용량)
   - 실시간 추적 및 로그 모니터링
3. **📄 문서 분석**: 
   - 검색된 문서 분석 및 시각화
   - 문서 소스별 분포 (로컬/웹)
   - 관련성 점수 분포
4. **💬 대화 히스토리**: 세션별 대화 기록 및 통계

#### 실행 방법

```bash
# 1. API 서버 실행
make run

# 2. UI 실행 (새 터미널에서)
make ui
# 또는
streamlit run ui/app.py --server.port 8501

# 3. 데모 UI 실행 (API 서버 없이)
make ui-demo
# 또는
streamlit run ui/demo.py --server.port 8502
```

#### 접속 URL
- **메인 UI**: http://localhost:8501
- **데모 UI**: http://localhost:8502

#### 파이프라인 노드 모니터링

각 RAG 파이프라인 노드의 input/output을 실시간으로 모니터링할 수 있습니다:

- **Planner**: 질문 의도 분석 및 실행 계획 수립
- **Websearch**: 실시간 웹 정보 수집
- **Search**: 벡터/키워드/하이브리드 검색
- **Rank Filter**: 문서 관련성 기반 필터링
- **Verify Refine**: 문서 품질 검증
- **Answer Nodes**: 의도별 답변 생성 (QA/Summary/Compare/Recommend)
- **Reevaluate**: 답변 품질 평가
- **Replan**: 품질 개선을 위한 재검색 계획

자세한 사용법은 `ui/README.md`를 참조하세요.