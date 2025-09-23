#!/bin/bash
set -e
export DOCUMENT_DIR=${DOCUMENT_DIR:-data/documents}
export VECTOR_DIR=${VECTOR_DIR:-data/vector_db}
export EMB_MODEL_NAME=${EMB_MODEL_NAME:-dragonkue/multilingual-e5-small-ko}
export EMB_BATCH=${EMB_BATCH:-32}
export CHUNK_SIZE=${CHUNK_SIZE:-800}
export CHUNK_OVERLAP=${CHUNK_OVERLAP:-120}

echo "[ingest] DOCS=$DOCUMENT_DIR → VEC=$VECTOR_DIR"
python scripts/ingest.py