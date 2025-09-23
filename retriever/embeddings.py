import os
from functools import lru_cache
from typing import List
import numpy as np

# FlagEmbedding (BGE 계열)
from FlagEmbedding import BGEM3FlagModel # m3용
from FlagEmbedding import FlagModel       # generic wrapper (bge-small-ko등 일부)

EMB_NAME = os.getenv("EMB_MODEL_NAME", "BAAI/bge-small-ko-v1")
EMB_BATCH = int(os.getenv("EMB_BATCH", "32"))

@lru_cache()
def _load_model():
    name = EMB_NAME.strip().lower()
    # bge-m3 (멀티링구얼) 우선 지원
    if "m3" in name:
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        return ("m3", model)
    # bge-small-ko 및 기타 모델: generic FlagModel 사용
    return ("generic", FlagModel(EMB_NAME, use_fp16=True))

def embed_texts(texts: List[str]) -> np.ndarray:
    kind, model = _load_model()
    if kind == "m3":
        # returns {"dense_vecs": np.ndarray, ...}
        out = model.encode(texts, batch_size=EMB_BATCH)
        return out["dense_vecs"].astype("float32")
    # generic
    vecs = model.encode(texts, batch_size=EMB_BATCH)
    return np.array(vecs, dtype="float32").copy()