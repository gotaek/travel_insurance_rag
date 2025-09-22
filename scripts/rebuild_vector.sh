#!/bin/bash
set -e
echo "Rebuilding FAISS index..."
export DOC_DIR=data/documents
export VECTOR_DIR=data/vector_db
python scripts/ingest.py