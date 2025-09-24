# Docker helpers
.PHONY: d.build d.up d.upd d.logs d.down ingest eval ui

d.build:
	docker compose build

d.up:
	docker compose up -d

d.upd:
	docker compose up

d.logs:
	docker compose logs -f --tail=200

d.down:
	docker compose down

ingest:
	docker compose exec -T api bash scripts/rebuild_vector.sh

eval:
	docker compose exec -T api python eval/ragas_pipeline.py

eval.analysis:
	docker compose exec -T api python eval/analysis_report.py

eval.simple:
	docker compose exec -T api python eval/simple_test.py

ui:
	docker compose exec api streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0

# Docker 환경에서 테스트 실행
.PHONY: test-docker test-unit-docker test-integration-docker test-benchmark-docker

test-docker:
	@echo "🧪 도커 환경에서 전체 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/ -v'

test-unit-docker:
	@echo "🔬 도커 환경에서 단위 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/ -v'

test-integration-docker:
	@echo "🔗 도커 환경에서 통합 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/integration/ -v'

test-benchmark-docker:
	@echo "📊 도커 환경에서 벤치마크 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/ -v -m benchmark'

test-coverage-docker:
	@echo "📈 도커 환경에서 커버리지 포함 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/ --cov=graph --cov-report=term'

