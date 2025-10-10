# Travel Insurance RAG System - Makefile
# 도커 기반 개발 환경을 위한 간소화된 명령어들

.PHONY: help dev build up down logs clean restart

# 기본 도움말
help:
	@echo "🛡️ Travel Insurance RAG System"
	@echo ""
	@echo "🚀 개발 명령어:"
	@echo "  make dev          - 개발 환경 전체 실행 (API + Web + Redis)"
	@echo "  make up           - 백그라운드에서 전체 서비스 실행"
	@echo "  make down         - 모든 서비스 중지"
	@echo "  make restart      - 서비스 재시작"
	@echo "  make logs         - 실시간 로그 확인"
	@echo ""
	@echo "🏗️ 빌드 명령어:"
	@echo "  make build        - 모든 서비스 빌드"
	@echo "  make build.web    - 웹 서비스만 빌드"
	@echo "  make build.api    - API 서비스만 빌드"
	@echo ""
	@echo "🧪 테스트 명령어:"
	@echo "  make test         - 전체 테스트 실행"
	@echo "  make test.unit    - 단위 테스트만 실행"
	@echo "  make test.integration - 통합 테스트만 실행"
	@echo ""
	@echo "📊 평가 명령어:"
	@echo "  make eval         - 기본 평가 실행"
	@echo "  make ingest       - 벡터 DB 재구성"
	@echo ""
	@echo "🧹 정리 명령어:"
	@echo "  make clean        - 컨테이너 및 볼륨 정리"
	@echo "  make clean.all    - 모든 데이터 정리 (주의!)"

# =============================================================================
# 개발 환경 명령어
# =============================================================================

dev:
	@echo "🚀 개발 환경 실행 중..."
	docker compose up --build

up:
	@echo "🚀 백그라운드에서 서비스 실행 중..."
	docker compose up -d

down:
	@echo "🛑 서비스 중지 중..."
	docker compose down

restart:
	@echo "🔄 서비스 재시작 중..."
	docker compose restart

logs:
	@echo "📋 실시간 로그 확인 중..."
	docker compose logs -f --tail=100

# =============================================================================
# 빌드 명령어
# =============================================================================

build:
	@echo "🏗️ 전체 서비스 빌드 중..."
	docker compose build

build.web:
	@echo "🏗️ 웹 서비스 빌드 중..."
	docker compose build web

build.api:
	@echo "🏗️ API 서비스 빌드 중..."
	docker compose build api

# =============================================================================
# 테스트 명령어
# =============================================================================

test:
	@echo "🧪 전체 테스트 실행 중..."
	docker compose exec api pytest tests/ -v

test.unit:
	@echo "🔬 단위 테스트 실행 중..."
	docker compose exec api pytest tests/unit/ -v

test.integration:
	@echo "🔗 통합 테스트 실행 중..."
	docker compose exec api pytest tests/integration/ -v

test.coverage:
	@echo "📈 커버리지 포함 테스트 실행 중..."
	docker compose exec api pytest tests/ --cov=graph --cov-report=term

# =============================================================================
# 평가 및 데이터 명령어
# =============================================================================

eval:
	@echo "📊 기본 평가 실행 중..."
	docker compose exec api python eval/simple_eval.py

ingest:
	@echo "📚 벡터 DB 재구성 중..."
	docker compose exec api bash scripts/rebuild_vector.sh

# =============================================================================
# 정리 명령어
# =============================================================================

clean:
	@echo "🧹 컨테이너 및 볼륨 정리 중..."
	docker compose down -v
	docker system prune -f

clean.all:
	@echo "⚠️ 모든 데이터 정리 중... (주의: 되돌릴 수 없습니다!)"
	docker compose down -v --remove-orphans
	docker system prune -af
	docker volume prune -f

# =============================================================================
# 특별 명령어 (필요시 사용)
# =============================================================================

# API만 실행 (웹 없이)
api.only:
	@echo "🔧 API 서비스만 실행 중..."
	docker compose up api redis --build

# 웹만 실행 (API 없이, 로컬 API 사용)
web.only:
	@echo "🌐 웹 서비스만 실행 중..."
	docker compose up web --build

# Streamlit UI 실행
ui:
	@echo "📊 Streamlit UI 실행 중..."
	docker compose up ui --build

# 프로덕션 빌드 (필요시)
prod:
	@echo "🚀 프로덕션 빌드 실행 중..."
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d