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

# Planner 노드 전용 테스트
.PHONY: test-planner test-planner-unit test-planner-integration

test-planner:
	@echo "📋 Planner 노드 전체 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/test_planner_fallback.py tests/integration/test_planner_integration.py -v'

test-planner-unit:
	@echo "🔬 Planner 노드 단위 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/test_planner_fallback.py -v'

test-planner-integration:
	@echo "🔗 Planner 노드 통합 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/integration/test_planner_integration.py -v'

# Websearch 노드 전용 테스트
.PHONY: test-websearch test-websearch-unit test-websearch-integration

test-websearch:
	@echo "🔍 Websearch 노드 전체 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/test_websearch.py tests/integration/test_websearch_integration.py -v'

test-websearch-unit:
	@echo "🔬 Websearch 노드 단위 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/test_websearch.py -v'

test-websearch-integration:
	@echo "🔗 Websearch 노드 통합 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/integration/test_websearch_integration.py -v'


# Search 노드 전용 테스트
.PHONY: test-search test-search-unit test-search-integration

test-search:
	@echo "🔍 Search 노드 전체 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/test_search.py tests/integration/test_search_integration.py -v'

test-search-unit:
	@echo "🔬 Search 노드 단위 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/test_search.py -v'

test-search-integration:
	@echo "🔗 Search 노드 통합 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/integration/test_search_integration.py -v'


# Rank Filter 노드 전용 테스트
.PHONY: test-rank-filter test-rank-filter-unit test-rank-filter-integration

test-rank-filter:
	@echo "🔍 Rank Filter 노드 전체 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/test_rank_filter.py tests/integration/test_rank_filter_integration.py -v'

test-rank-filter-unit:
	@echo "🔬 Rank Filter 노드 단위 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/test_rank_filter.py -v'

test-rank-filter-integration:
	@echo "🔗 Rank Filter 노드 통합 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/integration/test_rank_filter_integration.py -v'


# Verify Refine 노드 전용 테스트
.PHONY: test-verify-refine test-verify-refine-unit test-verify-refine-integration

test-verify-refine:
	@echo "🔍 Verify Refine 노드 전체 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/test_verify_refine.py tests/integration/test_verify_refine_integration.py -v'

test-verify-refine-unit:
	@echo "🔬 Verify Refine 노드 단위 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/test_verify_refine.py -v'
test-verify-refine-integration:
	@echo "🔗 Verify Refine 노드 통합 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/integration/test_verify_refine_integration.py -v'

# QA 노드 전용 테스트
.PHONY: test-qa test-qa-unit test-qa-integration

test-qa:
	@echo "🔍 QA 노드 전체 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/test_qa.py tests/integration/test_qa_integration.py -v'

test-qa-unit:
	@echo "🔬 QA 노드 단위 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/test_qa.py -v'

test-qa-integration:
	@echo "🔗 QA 노드 통합 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/integration/test_qa_integration.py -v'


# Recommend 노드 전용 테스트
.PHONY: test-recommend test-recommend-unit test-recommend-integration

test-recommend:
	@echo "🔍 Recommend 노드 전체 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/test_recommend.py tests/integration/test_recommend_integration.py -v'

test-recommend-unit:
	@echo "🔬 Recommend 노드 단위 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/test_recommend.py -v'

test-recommend-integration:
	@echo "🔗 Recommend 노드 통합 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/integration/test_recommend_integration.py -v'


# Summarize 노드 전용 테스트
.PHONY: test-summarize test-summarize-unit test-summarize-integration

test-summarize:
	@echo "🔍 Summarize 노드 전체 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/test_summarize.py tests/integration/test_summarize_integration.py -v'

test-summarize-unit:
	@echo "🔬 Summarize 노드 단위 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/test_summarize.py -v'

test-summarize-integration:
	@echo "🔗 Summarize 노드 통합 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/integration/test_summarize_integration.py -v'

# Compare 노드 전용 테스트
.PHONY: test-compare test-compare-unit test-compare-integration

test-compare:
	@echo "🔍 Compare 노드 전체 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/test_compare.py tests/integration/test_compare_integration.py -v'

test-compare-unit:
	@echo "🔬 Compare 노드 단위 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/test_compare.py -v'

test-compare-integration:
	@echo "🔗 Compare 노드 통합 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/integration/test_compare_integration.py -v'

# Reevaluate 노드 전용 테스트
.PHONY: test-reevaluate test-reevaluate-unit test-reevaluate-integration

test-reevaluate:
	@echo "🔍 Reevaluate 노드 전체 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/test_reevaluate.py tests/integration/test_reevaluate_integration.py -v'

test-reevaluate-unit:
	@echo "🔬 Reevaluate 노드 단위 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/test_reevaluate.py -v'

test-reevaluate-integration:
	@echo "🔗 Reevaluate 노드 통합 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/integration/test_reevaluate_integration.py -v'

# Replan 노드 전용 테스트
.PHONY: test-replan test-replan-unit test-replan-integration

test-replan:
	@echo "🔍 Replan 노드 전체 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/test_replan.py tests/integration/test_replan_integration.py -v'

test-replan-unit:
	@echo "🔬 Replan 노드 단위 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/unit/test_replan.py -v'

test-replan-integration:
	@echo "🔗 Replan 노드 통합 테스트 실행..."
	docker compose exec api bash -c 'export PATH=$$PATH:/home/appuser/.local/bin && pytest tests/integration/test_replan_integration.py -v'
