# LangSmith 통합 설정 가이드

## 🎉 LangSmith 통합 완료!

프로젝트에 LangSmith가 성공적으로 통합되었습니다. 이제 고급 추적 및 모니터링 기능을 사용할 수 있습니다.

## 📋 설정 단계

### 1. LangSmith 계정 생성 및 API 키 발급

1. [LangSmith 웹사이트](https://smith.langchain.com/)에 접속
2. 계정 생성 또는 로그인
3. API 키 발급 (Settings > API Keys)

### 2. 환경 변수 설정

`.env` 파일에 다음 환경 변수들을 추가하세요:

```bash
# LangSmith 설정
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_actual_langsmith_api_key_here
LANGCHAIN_PROJECT=travel-insurance-rag
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### 3. 서버 재시작

환경 변수 설정 후 서버를 재시작하세요:

```bash
# FastAPI 서버 재시작
uvicorn app.main:app --reload

# 또는 Streamlit UI 재시작
streamlit run ui/app.py
```

## 🔍 통합된 기능들

### 1. **LangGraph 추적**
- 모든 노드의 실행 과정 추적
- 실행 시간 및 토큰 사용량 모니터링
- 에러 발생 시점 정확한 파악

### 2. **LLM 호출 추적**
- Google Gemini 호출 전체 기록
- 프롬프트/응답 쌍 저장
- 토큰 사용량 정확한 측정

### 3. **자동 품질 모니터링**
- 답변 품질 자동 평가
- 사용자 피드백 연동
- A/B 테스트 지원

## 📊 LangSmith 대시보드에서 확인할 수 있는 정보

### 실행 추적 (Traces)
- 각 질문에 대한 전체 실행 흐름
- 노드별 실행 시간 및 성공/실패 상태
- LLM 호출의 입력/출력 전체 기록

### 성능 메트릭
- 평균 응답 시간
- 토큰 사용량 통계
- 에러 발생률

### 품질 분석
- 답변 정확도 추적
- 사용자 만족도 지표
- 개선이 필요한 영역 식별

## 🛠️ 고급 기능

### 수동 실행 추적
```python
from graph.langsmith_integration import create_langsmith_run, update_langsmith_run

# 실행 시작
run_id = create_langsmith_run(
    name="custom_operation",
    inputs={"question": "사용자 질문"},
    extra={"metadata": {"user_id": "123"}}
)

# 실행 완료
update_langsmith_run(
    run_id=run_id,
    outputs={"result": "처리 결과"},
    extra={"metadata": {"success": True}}
)
```

### 프로젝트별 분리
```bash
# 개발 환경
LANGCHAIN_PROJECT=travel-insurance-rag-dev

# 프로덕션 환경
LANGCHAIN_PROJECT=travel-insurance-rag-prod
```

## 🔧 문제 해결

### 1. API 키 오류
```
⚠️ LangSmith API 키가 설정되지 않았습니다.
```
**해결방법**: `.env` 파일에 올바른 `LANGCHAIN_API_KEY` 설정

### 2. 추적이 작동하지 않는 경우
```bash
# 환경 변수 확인
echo $LANGCHAIN_TRACING_V2
echo $LANGCHAIN_API_KEY
echo $LANGCHAIN_PROJECT
```

### 3. LangChain 버전 호환성 문제
```bash
# 필요한 패키지 재설치
pip install --upgrade langsmith langchain-core
```

## 📈 예상 효과

### 개발 효율성
- **디버깅 시간 50% 단축**: 문제 발생 시점 정확한 파악
- **성능 최적화**: 병목 지점 식별로 응답 속도 개선
- **비용 절감**: 토큰 사용량 최적화로 LLM 비용 절약

### 품질 향상
- **자동화된 품질 모니터링**: 답변 정확도 실시간 추적
- **데이터 기반 개선**: 실행 데이터 기반으로 시스템 개선
- **팀 협업**: LangSmith 대시보드로 팀 전체의 가시성 확보

## 🎯 다음 단계

1. **LangSmith API 키 설정**
2. **서버 재시작 후 테스트**
3. **LangSmith 대시보드에서 추적 데이터 확인**
4. **품질 메트릭 설정 및 알림 구성**

---

**문의사항이 있으시면 언제든지 연락주세요!** 🚀
