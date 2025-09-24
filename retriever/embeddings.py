import os
import logging
from functools import lru_cache
from typing import List, Optional, Dict, Any
import numpy as np

# FlagEmbedding (BGE 계열)
from FlagEmbedding import BGEM3FlagModel # m3용
from FlagEmbedding import FlagModel       # generic wrapper

# 캐싱 관리자
from graph.cache_manager import cache_manager

# 로깅 설정
logger = logging.getLogger(__name__)

EMB_NAME = os.getenv("EMB_MODEL_NAME", "dragonkue/multilingual-e5-small-ko")
EMB_BATCH = int(os.getenv("EMB_BATCH", "32"))

@lru_cache()
def _load_model():
    name = EMB_NAME.strip().lower()
    
    # Hugging Face 토큰 설정
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        logger.info("Hugging Face 토큰이 설정되었습니다.")
    
    logger.info(f"{EMB_NAME} 모델 로딩 중...")
    
    # multilingual-e5-small-ko 모델 사용
    if "e5" in name or "multilingual" in name:
        model = FlagModel(EMB_NAME, use_fp16=True)
        return ("generic", model)
    # BGE-m3-ko 모델 사용 (한국어 특화)
    elif "m3-ko" in name or "dragonkue" in name:
        model = BGEM3FlagModel(EMB_NAME, use_fp16=True)
        return ("m3", model)
    # bge-m3 (멀티링구얼) 지원
    elif "m3" in name:
        model = BGEM3FlagModel(EMB_NAME, use_fp16=True)
        return ("m3", model)
    # 기타 모델: generic FlagModel 사용
    else:
        model = FlagModel(EMB_NAME, use_fp16=True)
        return ("generic", model)

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    텍스트 임베딩 생성 (캐싱 지원)
    """
    if not texts:
        return np.array([])
    
    # 캐시에서 먼저 확인
    cached_embeddings = cache_manager.get_cached_embeddings(texts)
    if cached_embeddings is not None:
        logger.debug(f"임베딩 캐시 히트: {len(texts)}개 텍스트")
        return cached_embeddings
    
    # 캐시 미스 - 모델 로딩 및 임베딩 생성
    logger.debug(f"임베딩 생성 중: {len(texts)}개 텍스트")
    kind, model = _load_model()
    
    if kind == "m3":
        # returns {"dense_vecs": np.ndarray, ...}
        out = model.encode(texts, batch_size=EMB_BATCH)
        embeddings = out["dense_vecs"].astype("float32")
    else:
        # generic
        vecs = model.encode(texts, batch_size=EMB_BATCH)
        embeddings = np.array(vecs, dtype="float32").copy()
    
    # 결과 캐싱
    cache_manager.cache_embeddings(texts, embeddings)
    logger.debug(f"임베딩 생성 완료 및 캐싱: {embeddings.shape}")
    
    return embeddings


def preload_embedding_model() -> bool:
    """
    서버 시작 시 임베딩 모델 사전 로딩
    """
    try:
        logger.info("임베딩 모델 사전 로딩 시작...")
        _load_model()
        logger.info("임베딩 모델 사전 로딩 완료")
        return True
    except Exception as e:
        logger.error(f"임베딩 모델 사전 로딩 실패: {e}")
        return False


def get_embedding_model_info() -> Dict[str, Any]:
    """
    현재 로딩된 임베딩 모델 정보 반환
    """
    try:
        kind, model = _load_model()
        return {
            "model_name": EMB_NAME,
            "model_type": kind,
            "batch_size": EMB_BATCH,
            "status": "loaded"
        }
    except Exception as e:
        return {
            "model_name": EMB_NAME,
            "model_type": "unknown",
            "batch_size": EMB_BATCH,
            "status": f"error: {e}"
        }