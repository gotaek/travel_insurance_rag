import os, re, pickle, json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import numpy as np

import fitz  # PyMuPDF
import pdfplumber

try:
    import faiss
except Exception:
    faiss = None

from sklearn.feature_extraction.text import TfidfVectorizer
from retriever.embeddings import embed_texts

DOC_DIR = os.getenv("DOCUMENT_DIR", "data/documents")
OUT_DIR = os.getenv("VECTOR_DIR", "data/vector_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

INDEX_PATH = Path(OUT_DIR) / "index.faiss"
META_PATH = Path(OUT_DIR) / "index.pkl"

# ex) 2025_DB손보_여행자보험약관.pdf
FNAME_RE = re.compile(r"(?P<year>\d{4})[_-](?P<insurer>[^_-]+)[_-](?P<title>.+)\.pdf$", re.IGNORECASE)

# 헤딩/섹션 규칙
RE_ARTICLE = re.compile(r"^\s*제\s*\d+\s*조")
RE_CHAPTER = re.compile(r"^\s*제\s*\d+\s*관")
RE_SPECIAL = re.compile(r"^\s*특별약관")
RE_APPENDIX = re.compile(r"^\s*(부록|별표)")

KEYWORD_WHITELIST = [
    "사망", "후유장해", "질병", "상해", "지연", "결항", "수하물", "자기부담",
    "대기기간", "면책", "배상책임", "실손의료비", "식중독", "감염병", "여권분실",
    "반려견", "지수형", "특정감염병", "입원", "구조송환"
]

def _parse_filename_meta(fp: Path) -> Dict[str, Any]:
    m = FNAME_RE.search(fp.name)
    if not m:
        return {"version_year": None, "insurer": None, "title": fp.stem}
    d = m.groupdict()
    return {
        "version_year": d["year"],
        "insurer": d["insurer"],
        "title": d["title"].replace("_"," ").replace("-"," ").strip(),
    }

def _section_type_by_heading(text: str) -> str:
    if RE_APPENDIX.search(text): return "appendix"
    if "별표" in text[:20] or "부록" in text[:20]: return "appendix"
    return "body"

def _chunk_text(txt: str, size: int, overlap: int) -> List[str]:
    if len(txt) <= size:
        return [txt]
    chunks = []
    i = 0
    while i < len(txt):
        chunks.append(txt[i:i+size])
        i += max(1, size - overlap)
        if i >= len(txt):
            break
    return chunks

def _keyword_tags(docs: List[str], topk: int = 8) -> List[List[str]]:
    if not docs:
        return [[]]
    # TF-IDF + 화이트리스트 보강
    vec = TfidfVectorizer(max_df=0.9, min_df=2, ngram_range=(1,2))
    try:
        X = vec.fit_transform(docs)
        terms = np.array(vec.get_feature_names_out())
    except ValueError:
        return [[kw for kw in KEYWORD_WHITELIST if kw in d] for d in docs]

    tags: List[List[str]] = []
    for row in range(X.shape[0]):
        data = X[row].toarray().ravel()
        idx = data.argsort()[::-1][:topk]
        tfidf_tags = [t for t in terms[idx] if len(t) > 1]
        wl = [kw for kw in KEYWORD_WHITELIST if kw in docs[row]]
        # 합치되도록 중복 제거
        uniq = []
        for t in wl + tfidf_tags:
            if t not in uniq:
                uniq.append(t)
        tags.append(uniq[:topk])
    return tags

def _extract_tables(pdf_path: Path) -> Dict[int, List[Dict[str, Any]]]:
    """
    페이지별 테이블(행 리스트) 추출.
    - pdfplumber tables: 각 행을 리스트로 반환 → 헤더/셀 정리 후 JSON화
    반환 구조: {page_index(1-base): [ { "rows": [[...],[...]], "bbox": (x0,y0,x1,y1) }, ... ]}
    """
    results: Dict[int, List[Dict[str, Any]]] = {}
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = []
                items = []
                for tb in tables or []:
                    rows = []
                    for r in tb:
                        rows.append([ (c or "").strip() for c in r ])
                    if rows:
                        items.append({"rows": rows})
                if items:
                    results[i] = items
    except Exception:
        pass
    return results

def _blocks_from_pymupdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    PyMuPDF로 page→blocks 텍스트 추출.
    block→line→span 순서로 텍스트를 연결하며, 폰트/크기 등은 필요 시 확장.
    """
    out = []
    meta = _parse_filename_meta(pdf_path)
    doc_id = f"{meta['insurer']}_{meta['version_year']}_{meta['title']}".strip()
    with fitz.open(str(pdf_path)) as doc:
        for pno, page in enumerate(doc, start=1):
            raw = page.get_text("dict")
            for b in raw.get("blocks", []):
                if "lines" not in b:  # 이미지/그림
                    continue
                texts = []
                for ln in b["lines"]:
                    spans = ln.get("spans", [])
                    line_text = "".join([s.get("text","") for s in spans]).strip()
                    if line_text:
                        texts.append(line_text)
                text = "\n".join(texts).strip()
                if not text:
                    continue
                out.append({
                    "doc_id": doc_id,
                    "insurer": meta["insurer"],
                    "version": meta["version_year"],
                    "title": meta["title"],
                    "page": pno,
                    "text": text,
                })
    return out

def _label_sections(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    헤딩 규칙/키워드로 body/appendix 라벨. 표는 pdfplumber 결과에 따로 합류.
    """
    labeled = []
    last_heading = None
    for b in blocks:
        t = b["text"].splitlines()[0] if b["text"] else ""
        if RE_SPECIAL.search(t) or RE_CHAPTER.search(t) or RE_ARTICLE.search(t) or RE_APPENDIX.search(t):
            last_heading = t
        section_type = "body"
        if last_heading and RE_APPENDIX.search(last_heading):
            section_type = "appendix"
        elif RE_APPENDIX.search(t):
            section_type = "appendix"
        labeled.append({**b, "section_type": section_type, "heading": last_heading})
    return labeled

def _merge_tables(labeled_blocks: List[Dict[str, Any]], table_map: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    pdfplumber에서 뽑은 테이블을 section_type='table' 로 삽입.
    - 각 페이지 끝에 추가(간단). 고급 매칭(좌표 근접)은 후속 개선에서.
    """
    merged: List[Dict[str, Any]] = []
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for b in labeled_blocks:
        by_page.setdefault(b["page"], []).append(b)

    for pno in sorted(by_page.keys()):
        merged.extend(by_page[pno])
        for tb in table_map.get(pno, []):
            text_rows = [" | ".join(r) for r in tb["rows"]]
            merged.append({**by_page[pno][0], "section_type": "table", "text": "\n".join(text_rows)})
    return merged

def _build_index(chunks_meta: List[Dict[str, Any]]):
    os.makedirs(OUT_DIR, exist_ok=True)
    texts = [c["text"] for c in chunks_meta]
    if not texts:
        print("⚠️ No text chunks parsed. Abort.")
        return
    vecs = embed_texts(texts)  # (N, D) float32
    dim = vecs.shape[1]
    if faiss is None:
        print("⚠️ faiss not available. Skipping index build.")
        return
    index = faiss.IndexFlatL2(dim)
    index.add(vecs)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(chunks_meta, f)
    print(f"✅ Built FAISS index: {len(chunks_meta)} chunks → {INDEX_PATH}")

def main():
    pdfs = sorted([p for p in Path(DOC_DIR).glob("*.pdf")])
    if not pdfs:
        print(f"⚠️ No PDFs under {DOC_DIR}. Place files like '2025_보험사_문서제목.pdf'")
        return

    all_sections: List[Dict[str, Any]] = []
    for p in tqdm(pdfs, desc="Parsing PDFs (PyMuPDF+pdfplumber)"):
        blocks = _blocks_from_pymupdf(p)
        labeled = _label_sections(blocks)
        tables = _extract_tables(p)
        merged = _merge_tables(labeled, tables)
        all_sections.extend(merged)

    # 섹션 텍스트 기반 키워드 태깅
    section_texts = [s["text"] for s in all_sections]
    section_tags = _keyword_tags(section_texts, topk=8)
    for s, tags in zip(all_sections, section_tags):
        s["tags"] = tags

    # 청킹(+메타 전개)
    chunks_meta: List[Dict[str, Any]] = []
    for s in tqdm(all_sections, desc="Chunking"):
        # 표는 행 단위로 이미 짧은 편 → 바로 저장. (크면 일반 청킹)
        if s.get("section_type") == "table" and len(s["text"]) <= CHUNK_SIZE:
            m = dict(s)
            m["chunk_no"] = 1
            chunks_meta.append(m)
        else:
            for i, ch in enumerate(_chunk_text(s["text"], CHUNK_SIZE, CHUNK_OVERLAP), start=1):
                m = dict(s)
                m["text"] = ch
                m["chunk_no"] = i
                chunks_meta.append(m)

    _build_index(chunks_meta)

if __name__ == "__main__":
    main()