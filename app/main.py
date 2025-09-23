from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import health, rag
from retriever.embeddings import preload_embedding_model

app = FastAPI(title="Insurance RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(health.router, prefix="")
app.include_router(rag.router, prefix="")

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화 작업"""
    print("🚀 서버 시작 중...")
    
    # 임베딩 모델 사전 로딩
    success = preload_embedding_model()
    if success:
        print("✅ 임베딩 모델 로딩 완료")
    else:
        print("⚠️ 임베딩 모델 로딩 실패 - 첫 요청 시 로딩됩니다")
    
    print("🎉 서버 시작 완료")

@app.get("/")
def root():
    return {"service": "insurance-rag", "status": "ready"}