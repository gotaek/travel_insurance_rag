# Docker helpers
.PHONY: d.build d.up d.upd d.logs d.down ingest eval

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