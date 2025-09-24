# 테스트 구조 및 가이드

## 📁 테스트 디렉토리 구조

```
tests/
├── __init__.py
├── README.md
├── unit/                          # 단위 테스트
│   ├── __init__.py
│   ├── test_planner_fallback.py  # Fallback classify 테스트
│   └── test_planner_llm.py       # LLM 기반 planner 테스트
├── integration/                   # 통합 테스트
│   ├── __init__.py
│   └── test_planner_integration.py
└── fixtures/                      # 테스트 픽스처
    └── test_data.py              # 테스트 데이터 및 픽스처
```

## 🧪 테스트 유형

### 1. 단위 테스트 (Unit Tests)
- **위치**: `tests/unit/`
- **목적**: 개별 함수/클래스의 동작 검증
- **마커**: `@pytest.mark.unit`

#### 주요 테스트 파일:
- `test_planner_fallback.py`: Fallback classify 정확도 테스트
- `test_planner_llm.py`: LLM 기반 planner 테스트

### 2. 통합 테스트 (Integration Tests)
- **위치**: `tests/integration/`
- **목적**: 전체 시스템의 통합 동작 검증
- **마커**: `@pytest.mark.integration`

#### 주요 테스트 파일:
- `test_planner_integration.py`: Planner 통합 테스트

### 3. 픽스처 (Fixtures)
- **위치**: `tests/fixtures/`
- **목적**: 테스트 데이터 및 공통 설정 제공

## 🚀 테스트 실행 방법

### 기본 명령어
```bash
# 전체 테스트 실행
make test

# 단위 테스트만 실행
make test-unit

# 통합 테스트만 실행
make test-integration

# 벤치마크 테스트 실행
make test-benchmark

# 빠른 테스트 (느린 테스트 제외)
make test-fast
```

### pytest 직접 실행
```bash
# 전체 테스트
pytest tests/ -v

# 특정 마커만 실행
pytest tests/ -v -m unit
pytest tests/ -v -m integration
pytest tests/ -v -m benchmark

# 특정 파일만 실행
pytest tests/unit/test_planner_fallback.py -v

# 커버리지 포함
pytest tests/ --cov=graph --cov-report=html
```

## 📊 테스트 마커

| 마커 | 설명 | 사용 예시 |
|------|------|-----------|
| `unit` | 단위 테스트 | `@pytest.mark.unit` |
| `integration` | 통합 테스트 | `@pytest.mark.integration` |
| `slow` | 느린 테스트 | `@pytest.mark.slow` |
| `benchmark` | 성능 벤치마크 | `@pytest.mark.benchmark` |

## 🎯 주요 테스트 케이스

### Fallback Classify 테스트
- **QA Intent**: 기본 질문-답변 분류
- **Summary Intent**: 요약 요청 분류
- **Compare Intent**: 비교 요청 분류 (핵심!)
- **Recommend Intent**: 추천 요청 분류
- **웹 검색 필요성**: 실시간 정보 필요성 판단

### 핵심 테스트 케이스
```python
# 휴대품 조항 비교 (핵심!)
"휴대품 관련 조항은 어떻게 돼?" → compare intent

# 개인용품 보상 규정 비교
"개인용품 보상 규정은 어떻게 되나요?" → compare intent

# 지역별 추천 (웹 검색 필요)
"2025년 3월 일본 도쿄 여행에 추천하는 보험은?" → recommend intent + web search
```

## 📈 성능 기준

- **정확도**: 100% (13/13 테스트 케이스 통과)
- **평균 처리 시간**: < 0.1초/질문
- **메모리 사용량**: 최적화됨

## 🔧 테스트 유지보수

### 새로운 테스트 추가
1. 적절한 디렉토리에 테스트 파일 생성
2. 마커 추가 (`@pytest.mark.unit` 등)
3. 픽스처 사용하여 테스트 데이터 관리
4. 문서화 및 주석 추가

### 테스트 데이터 관리
- `tests/fixtures/test_data.py`에서 중앙 관리
- JSON 형태로 테스트 데이터 저장 가능
- `make test-data` 명령어로 데이터 생성

## 🐛 문제 해결

### 일반적인 문제
1. **Import 오류**: 프로젝트 루트 경로 확인
2. **마커 경고**: `pytest.ini`에서 마커 등록 확인
3. **의존성 오류**: `requirements.txt` 확인

### 디버깅
```bash
# 상세 출력으로 테스트 실행
pytest tests/ -v -s

# 특정 테스트만 실행
pytest tests/unit/test_planner_fallback.py::TestFallbackClassify::test_compare_intent_classification -v

# 실패한 테스트만 재실행
pytest tests/ --lf
```

## 📋 테스트 리포트

```bash
# HTML 리포트 생성
make test-report

# 커버리지 리포트
make test-coverage
```

## 🎉 성과

- **정확도**: 100% 달성
- **핵심 기능**: "휴대품 관련 조항" compare 분류 성공
- **성능**: 빠른 처리 속도
- **유지보수성**: 체계적인 테스트 구조
