# 멀티턴 대화 및 컨텍스트 윈도우 관리 가이드

## 🚀 새로운 기능 개요

이번 업데이트에서는 Annotated 타입 기반의 멀티턴 대화 지원과 Redis 기반 캐싱 시스템을 도입했습니다.

### 주요 개선사항

1. **멀티턴 대화 지원**: 이전 대화 내용을 기억하고 연속적인 대화 가능
2. **컨텍스트 윈도우 관리**: 토큰 제한 내에서 지능적 히스토리 압축
3. **Redis 기반 캐싱**: 임베딩 및 검색 결과 캐싱으로 성능 향상
4. **임베딩 모델 사전 로딩**: 서버 시작 시 모델 로딩으로 첫 요청 지연 해결
5. **Annotated 타입**: 타입 안전성 및 코드 가독성 향상
6. **하위 호환성**: 기존 API 클라이언트와 완전 호환

## 🏗️ 아키텍처 변경사항

### 새로운 컴포넌트

```
graph/
├── context.py              # 대화 컨텍스트 모델
├── context_manager.py      # 컨텍스트 윈도우 관리
└── cache_manager.py        # Redis 캐싱 관리

app/
├── compatibility.py        # 하위 호환성 보장
└── deps.py                 # Redis 클라이언트 추가
```

### 데이터 흐름

```
사용자 질문 → 세션 관리 → 컨텍스트 로드 → 캐시 확인 → RAG 처리 → 컨텍스트 업데이트 → 응답
```

## 📡 API 엔드포인트

### 기존 API (하위 호환)

```http
POST /rag/ask
Content-Type: application/json

{
  "question": "여행자보험에 대해 알려주세요",
  "session_id": "optional_session_id",
  "user_id": "optional_user_id",
  "include_context": true
}
```

### 새로운 멀티턴 API

```http
POST /rag/multiturn/ask
Content-Type: application/json

{
  "question": "그 보험의 보상 한도는 얼마인가요?",
  "session_id": "required_session_id",
  "user_id": "optional_user_id",
  "include_context": true
}
```

### 세션 관리 API

```http
# 세션 생성
POST /rag/session/create?user_id=user123

# 세션 정보 조회
GET /rag/session/{session_id}

# 세션 삭제
DELETE /rag/session/{session_id}
```

### 캐시 관리 API

```http
# 캐시 통계 조회
GET /rag/cache/stats

# 캐시 초기화
POST /rag/cache/clear
```

## 🔧 설정 및 배포

### 1. Redis 설정

```bash
# Docker Compose로 Redis 시작
docker-compose up redis -d

# 또는 로컬 Redis 설치
brew install redis  # macOS
sudo apt install redis-server  # Ubuntu
```

### 2. 환경 변수 설정

```bash
# .env 파일에 추가
REDIS_URL=redis://localhost:6379
REDIS_SESSION_TTL=3600
REDIS_CACHE_TTL=1800
```

### 3. 의존성 설치

```bash
pip install redis>=4.5.0 hiredis>=2.0.0
```

### 4. 서버 시작

```bash
# Docker Compose로 전체 스택 시작
docker-compose up

# 또는 개별 서비스 시작
docker-compose up redis api ui
```

## 🧪 테스트

### 자동 테스트 실행

```bash
python test_multiturn.py
```

### 수동 테스트 예시

```bash
# 1. 세션 생성
curl -X POST "http://localhost:8000/rag/session/create?user_id=test_user"

# 2. 첫 번째 질문
curl -X POST "http://localhost:8000/rag/multiturn/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "여행자보험에 대해 알려주세요",
    "session_id": "your_session_id"
  }'

# 3. 후속 질문 (컨텍스트 포함)
curl -X POST "http://localhost:8000/rag/multiturn/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "그 보험의 보상 한도는 얼마인가요?",
    "session_id": "your_session_id"
  }'
```

## 📊 성능 개선 효과

### 임베딩 모델 로딩

- **이전**: 첫 요청 시 5-10초 지연
- **개선 후**: 서버 시작 시 로딩, 첫 요청 즉시 응답

### 캐싱 효과

- **임베딩 캐싱**: 동일 쿼리 재요청 시 90% 성능 향상
- **검색 결과 캐싱**: 자주 묻는 질문 즉시 응답
- **LLM 응답 캐싱**: 동일 프롬프트 재사용 시 빠른 응답

### 컨텍스트 관리

- **토큰 제한**: 최대 4000 토큰 내에서 지능적 압축
- **히스토리 압축**: 중요도 기반 턴 선택 및 압축
- **세션 관리**: Redis 기반 영구 저장

## 🔍 모니터링 및 디버깅

### 캐시 통계 확인

```bash
curl http://localhost:8000/rag/cache/stats
```

### 세션 정보 확인

```bash
curl http://localhost:8000/rag/session/{session_id}
```

### 로그 확인

```bash
# Docker 로그 확인
docker-compose logs api

# Redis 로그 확인
docker-compose logs redis
```

## 🚨 주의사항

### Redis 의존성

- Redis가 없어도 기본 기능은 작동 (폴백 모드)
- 캐싱 및 세션 관리 기능은 제한됨
- 프로덕션 환경에서는 Redis 필수

### 메모리 사용량

- 임베딩 모델: ~500MB 메모리 사용
- Redis 캐시: 설정에 따라 가변적
- 컨텍스트 히스토리: 세션당 최대 10개 턴

### 보안 고려사항

- 세션 데이터는 Redis에 평문 저장
- 민감한 정보는 암호화 고려 필요
- Redis 접근 권한 설정 권장

## 🔄 마이그레이션 가이드

### 기존 클라이언트

기존 API 클라이언트는 수정 없이 계속 작동합니다:

```python
# 기존 코드 그대로 사용 가능
response = requests.post("http://localhost:8000/rag/ask", 
                        json={"question": "질문"})
```

### 새로운 기능 활용

```python
# 멀티턴 대화 활용
session_response = requests.post("http://localhost:8000/rag/session/create")
session_id = session_response.json()["session_id"]

# 첫 번째 질문
response1 = requests.post("http://localhost:8000/rag/multiturn/ask", 
                         json={"question": "첫 질문", "session_id": session_id})

# 후속 질문 (컨텍스트 포함)
response2 = requests.post("http://localhost:8000/rag/multiturn/ask", 
                         json={"question": "후속 질문", "session_id": session_id})
```


## 🆘 문제 해결

### 일반적인 문제

1. **Redis 연결 실패**: Redis 서버 상태 확인
2. **임베딩 모델 로딩 실패**: Hugging Face 토큰 확인
3. **캐시 미스**: 캐시 TTL 설정 확인
4. **세션 만료**: 세션 TTL 설정 조정

### 로그 레벨 조정

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

이 가이드를 통해 새로운 멀티턴 대화 기능을 효과적으로 활용하실 수 있습니다.
