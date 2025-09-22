from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import health, rag

app = FastAPI(title="Insurance RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(health.router, prefix="")

@app.get("/")
def root():
    return {"service": "insurance-rag", "status": "ready"}


app.include_router(health.router, prefix="")
app.include_router(rag.router, prefix="")