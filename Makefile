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