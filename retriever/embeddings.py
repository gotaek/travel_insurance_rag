import os
from functools import lru_cache
from typing import List
import numpy as np

# FlagEmbedding (BGE 계열)
from FlagEmbedding import BGEM3FlagModel # m3용
from FlagEmbedding import FlagModel       # generic wrapper

EMB_NAME = os.getenv("EMB_MODEL_NAME", "dragonkue/multilingual-e5-small-ko")
EMB_BATCH = int(os.getenv("EMB_BATCH", "32"))

@lru_cache()
def _load_model():
    name = EMB_NAME.strip().lower()
    
    # Hugging Face 토큰 설정
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        print("🔑 Hugging Face 토큰이 설정되었습니다.")
    
    print(f"🔄 {EMB_NAME} 모델 로딩 중...")
    
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
    kind, model = _load_model()
    if kind == "m3":
        # returns {"dense_vecs": np.ndarray, ...}
        out = model.encode(texts, batch_size=EMB_BATCH)
        return out["dense_vecs"].astype("float32")
    # generic
    vecs = model.encode(texts, batch_size=EMB_BATCH)
    return np.array(vecs, dtype="float32").copy()