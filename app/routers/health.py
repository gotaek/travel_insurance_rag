from fastapi import APIRouter
from app.deps import get_settings
import os

router = APIRouter(tags=["health"])

@router.get("/healthz")
def healthz():
    s = get_settings()
    vector_exists = os.path.isdir(s.VECTOR_DIR)
    docs_exists = os.path.isdir(s.DOCUMENT_DIR)
    return {
        "ok": True,
        "env": s.ENV,
        "vector_dir": s.VECTOR_DIR,
        "documents_dir": s.DOCUMENT_DIR,
        "exists": {"vector": vector_exists, "documents": docs_exists},
    }

@router.get("/readyz")
def readyz():
    return {"ready": True}