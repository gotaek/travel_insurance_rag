import os
import pickle
import glob
from pathlib import Path
import numpy as np
from tqdm import tqdm

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

from retriever.vector import _embed_stub

# 경로 설정
DOC_DIR = os.getenv("DOC_DIR", "data/documents")
OUT_DIR = os.getenv("VECTOR_DIR", "data/vector_db")
INDEX_PATH = Path(OUT_DIR) / "index.faiss"
META_PATH = Path(OUT_DIR) / "index.pkl"

def _load_documents(doc_dir: str):
    """
    간단 문서 로더 (Stub)
    - .txt 파일만 읽음
    - PDF 파서는 이후 확장
    """
    docs = []
    for fpath in glob.glob(os.path.join(doc_dir, "*.txt")):
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({"doc_id": Path(fpath).stem, "text": text})
    return docs

def _chunk_text(text: str, max_len: int = 200):
    """
    단순 청킹: max_len 글자 단위로 슬라이딩
    """
    chunks = []
    for i in range(0, len(text), max_len):
        chunk = text[i : i + max_len]
        chunks.append(chunk)
    return chunks

def build_index():
    os.makedirs(OUT_DIR, exist_ok=True)
    docs = _load_documents(DOC_DIR)
    meta = []
    vectors = []

    for d in tqdm(docs, desc="Embedding docs"):
        chunks = _chunk_text(d["text"])
        for i, chunk in enumerate(chunks):
            emb = _embed_stub(chunk)  # 임시 스텁 임베딩
            vectors.append(emb[0])
            meta.append(
                {
                    "doc_id": d["doc_id"],
                    "page": i + 1,  # 단순히 chunk 번호를 page로 사용
                    "text": chunk,
                    "insurer": None,
                    "version": None,
                }
            )

    if not vectors:
        print("⚠️ No documents found, empty index created.")
        return

    dim = len(vectors[0])
    xb = np.array(vectors).astype("float32")
    if faiss:
        index = faiss.IndexFlatL2(dim)
        index.add(xb)
        faiss.write_index(index, str(INDEX_PATH))
        with open(META_PATH, "wb") as f:
            pickle.dump(meta, f)
        print(f"✅ Built index with {len(meta)} chunks → {INDEX_PATH}")
    else:
        print("⚠️ faiss not available. Skipped index build.")

if __name__ == "__main__":
    build_index()