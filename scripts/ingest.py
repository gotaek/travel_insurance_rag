import os, re, json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import numpy as np

import fitz  # PyMuPDF

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None

from sklearn.feature_extraction.text import TfidfVectorizer
from retriever.embeddings import embed_texts

# LangChain text splitters 추가
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOC_DIR = os.getenv("DOCUMENT_DIR", "data/documents")
OUT_DIR = os.getenv("VECTOR_DIR", "data/vector_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

COLLECTION_NAME = "insurance_docs"

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

def _merge_small_blocks(blocks: List[Dict[str, Any]], min_size: int = 700) -> List[Dict[str, Any]]:
    """
    작은 블록들을 병합하여 의미 있는 텍스트 생성 (개선된 버전)
    
    Args:
        blocks: PDF에서 추출된 블록 리스트
        min_size: 최소 블록 크기 (문자 수) - 700자로 증가
        
    Returns:
        병합된 블록 리스트
    """
    if not blocks:
        return []
    
    merged_blocks = []
    current_block = None
    current_text = ""
    
    for block in blocks:
        block_text = block["text"].strip()
        
        # 빈 텍스트는 건너뛰기
        if not block_text:
            continue
            
        # 현재 블록이 없으면 새로 시작
        if current_block is None:
            current_block = dict(block)
            current_text = block_text
        else:
            # 현재 텍스트와 합쳤을 때 최소 크기 미만이면 병합
            if len(current_text) + len(block_text) < min_size * 1.5:  # 1.5배로 제한 완화
                current_text += "\n" + block_text
            else:
                # 현재까지의 텍스트를 블록으로 저장
                current_block["text"] = current_text
                merged_blocks.append(current_block)
                
                # 새 블록 시작
                current_block = dict(block)
                current_text = block_text
    
    # 마지막 블록 처리
    if current_block and current_text:
        current_block["text"] = current_text
        merged_blocks.append(current_block)
    
    # 🔧 개선: 최종 검증 - 너무 작은 블록은 다음 블록과 강제 병합
    final_blocks = []
    i = 0
    while i < len(merged_blocks):
        current_block = merged_blocks[i]
        current_text = current_block["text"]
        
        if len(current_text) < min_size and i + 1 < len(merged_blocks):
            # 다음 블록과 강제 병합
            next_block = merged_blocks[i + 1]
            merged_text = current_text + "\n" + next_block["text"]
            
            # 병합된 블록이 너무 크면 분할
            if len(merged_text) > min_size * 2:
                # 중간점에서 분할
                mid_point = len(merged_text) // 2
                # 문장 경계에서 분할점 찾기
                for offset in range(0, len(merged_text) // 4):
                    if mid_point - offset > 0:
                        if merged_text[mid_point - offset:mid_point - offset + 2] == '. ':
                            mid_point = mid_point - offset + 2
                            break
                    if mid_point + offset < len(merged_text):
                        if merged_text[mid_point + offset:mid_point + offset + 2] == '. ':
                            mid_point = mid_point + offset + 2
                            break
                
                # 분할 실행
                if mid_point > 0 and mid_point < len(merged_text):
                    final_blocks.append({
                        **current_block,
                        "text": merged_text[:mid_point]
                    })
                    final_blocks.append({
                        **next_block,
                        "text": merged_text[mid_point:]
                    })
                else:
                    final_blocks.append({
                        **current_block,
                        "text": merged_text
                    })
            else:
                final_blocks.append({
                    **current_block,
                    "text": merged_text
                })
            i += 2  # 두 블록을 처리했으므로 2 증가
        else:
            final_blocks.append(current_block)
            i += 1
    
    return final_blocks

def _create_recursive_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    한국어에 최적화된 Recursive Text Splitter 생성
    
    Args:
        chunk_size: 청크 크기
        chunk_overlap: 청크 오버랩
        
    Returns:
        RecursiveCharacterTextSplitter 인스턴스
    """
    # 한국어에 최적화된 구분자 설정
    separators = [
        "\n\n",      # 문단 구분
        "\n",        # 줄 구분  
        ". ",        # 문장 구분
        "。",        # 한국어 문장 구분
        " ",         # 단어 구분
        ""           # 문자 구분
    ]
    
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False
    )

def _uniform_chunking_with_recursive_splitter(txt: str, target_size: int, overlap: int, tolerance: float = 0.15) -> List[str]:
    """
    RecursiveCharacterTextSplitter와 후처리를 결합한 균등 청킹 (개선된 버전)
    
    Args:
        txt: 청킹할 텍스트
        target_size: 목표 청크 크기
        overlap: 청크 오버랩
        tolerance: 허용 오차 (0.15 = 15%)
        
    Returns:
        균등한 크기의 청크 리스트
    """
    if len(txt) <= target_size:
        return [txt]
    
    # 🔧 개선: 최소 크기 보장 로직 추가
    if len(txt) < target_size * 0.8:  # 80% 미만이면
        # 작은 텍스트는 그대로 반환하지 말고 강제 분할 시도
        if len(txt) < target_size * 0.5:  # 50% 미만이면
            return [txt]  # 너무 작으면 그대로 반환
        else:
            # 50-80% 범위면 강제로 목표 크기에 맞춰 분할
            return _force_chunk_to_target_size(txt, target_size, overlap)
    
    # 1단계: RecursiveCharacterTextSplitter로 의미 단위 분할
    text_splitter = _create_recursive_text_splitter(target_size, overlap)
    initial_chunks = text_splitter.split_text(txt)
    
    # 2단계: 청크 크기 균등화
    min_size = int(target_size * (1 - tolerance))
    max_size = int(target_size * (1 + tolerance))
    
    balanced_chunks = []
    i = 0
    
    while i < len(initial_chunks):
        current_chunk = initial_chunks[i]
        chunk_len = len(current_chunk)
        
        if min_size <= chunk_len <= max_size:
            # 적절한 크기의 청크는 그대로 사용
            balanced_chunks.append(current_chunk)
            i += 1
        elif chunk_len < min_size:
            # 너무 작은 청크는 다음 청크와 병합
            if i + 1 < len(initial_chunks):
                next_chunk = initial_chunks[i + 1]
                merged = current_chunk + "\n" + next_chunk
                
                if len(merged) <= max_size:
                    # 병합된 청크가 최대 크기 이내면 사용
                    balanced_chunks.append(merged)
                    i += 2
                else:
                    # 병합하면 너무 크면 현재 청크만 사용
                    balanced_chunks.append(current_chunk)
                    i += 1
            else:
                # 마지막 청크는 그대로 사용
                balanced_chunks.append(current_chunk)
                i += 1
        else:
            # 너무 큰 청크는 재분할
            sub_chunks = _split_large_chunk(current_chunk, target_size, overlap)
            balanced_chunks.extend(sub_chunks)
            i += 1
    
    # 3단계: 마지막 최적화 - 너무 작은 마지막 청크 처리
    if len(balanced_chunks) > 1 and len(balanced_chunks[-1]) < min_size:
        # 마지막 청크를 이전 청크와 병합
        last_chunk = balanced_chunks.pop()
        balanced_chunks[-1] += "\n" + last_chunk
    
    # 🔧 개선: 최종 검증 - 목표 크기 달성 여부 확인
    final_chunks = []
    for chunk in balanced_chunks:
        if len(chunk) < target_size * 0.7:  # 70% 미만이면
            # 다음 청크와 강제 병합 시도
            if len(final_chunks) > 0:
                # 이전 청크와 병합
                final_chunks[-1] += "\n" + chunk
            else:
                final_chunks.append(chunk)
        else:
            final_chunks.append(chunk)
    
    return final_chunks

def _force_chunk_to_target_size(txt: str, target_size: int, overlap: int) -> List[str]:
    """
    작은 텍스트를 목표 크기에 맞춰 강제 분할
    
    Args:
        txt: 분할할 텍스트
        target_size: 목표 청크 크기
        overlap: 청크 오버랩
        
    Returns:
        강제 분할된 청크 리스트
    """
    if len(txt) <= target_size:
        return [txt]
    
    chunks = []
    start = 0
    
    while start < len(txt):
        end = min(start + target_size, len(txt))
        
        # 마지막 청크가 아니면 적절한 분할점 찾기
        if end < len(txt):
            # 문장 경계에서 분할 시도
            for sep in ['. ', '.\n', '\n\n', '\n']:
                sep_pos = txt.rfind(sep, start, end)
                if sep_pos > start + target_size * 0.7:  # 70% 이상 위치에서 발견
                    end = sep_pos + len(sep)
                    break
        
        chunk_text = txt[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)
        
        start = max(start + 1, end - overlap)
    
    return chunks

def _split_large_chunk(chunk: str, target_size: int, overlap: int) -> List[str]:
    """
    큰 청크를 목표 크기로 분할
    """
    if len(chunk) <= target_size:
        return [chunk]
    
    chunks = []
    start = 0
    
    while start < len(chunk):
        end = min(start + target_size, len(chunk))
        
        # 마지막 청크가 아니면 적절한 분할점 찾기
        if end < len(chunk):
            # 문장 경계에서 분할 시도
            for sep in ['. ', '.\n', '\n\n', '\n']:
                sep_pos = chunk.rfind(sep, start, end)
                if sep_pos > start + target_size * 0.7:  # 70% 이상 위치에서 발견
                    end = sep_pos + len(sep)
                    break
        
        chunk_text = chunk[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)
        
        start = max(start + 1, end - overlap)
    
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

def _extract_tables_pymupdf(pdf_path: Path) -> Dict[int, List[Dict[str, Any]]]:
    """
    PyMuPDF를 사용한 테이블 추출 (간단한 형태)
    - 테이블 형태의 텍스트를 감지하여 구조화
    """
    results: Dict[int, List[Dict[str, Any]]] = {}
    try:
        with fitz.open(str(pdf_path)) as doc:
            for pno, page in enumerate(doc, start=1):
                # 페이지의 텍스트를 블록 단위로 추출
                blocks = page.get_text("dict").get("blocks", [])
                tables = []
                
                for block in blocks:
                    if "lines" not in block:
                        continue
                    
                    # 블록의 텍스트 추출
                    block_text = ""
                    for line in block["lines"]:
                        line_text = "".join([span.get("text", "") for span in line.get("spans", [])])
                        block_text += line_text + "\n"
                    
                    # 테이블 패턴 감지 (간단한 휴리스틱)
                    if _is_table_pattern(block_text):
                        rows = _parse_table_text(block_text)
                        if rows:
                            tables.append({"rows": rows})
                
                if tables:
                    results[pno] = tables
    except Exception as e:
        print(f"⚠️ 테이블 추출 중 오류: {e}")
    
    return results

def _is_table_pattern(text: str) -> bool:
    """
    텍스트가 테이블 패턴인지 판단
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) < 2:
        return False
    
    # 여러 구분자로 분리된 컬럼이 있는지 확인
    separators = ['|', '\t', '  ', '    ']  # 파이프, 탭, 2개 이상 공백
    
    for sep in separators:
        if sep in text:
            # 구분자로 분리된 행이 2개 이상 있는지 확인
            separated_lines = [line for line in lines if sep in line]
            if len(separated_lines) >= 2:
                return True
    
    return False

def _parse_table_text(text: str) -> List[List[str]]:
    """
    테이블 형태의 텍스트를 행/열 구조로 파싱
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    rows = []
    
    for line in lines:
        # 가장 적절한 구분자 찾기
        if '|' in line:
            cells = [cell.strip() for cell in line.split('|')]
        elif '\t' in line:
            cells = [cell.strip() for cell in line.split('\t')]
        else:
            # 2개 이상의 공백으로 분리
            cells = [cell.strip() for cell in re.split(r'\s{2,}', line)]
        
        # 빈 셀 제거하고 유효한 행만 추가
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(cells)
    
    return rows

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
    PyMuPDF로 추출한 테이블을 section_type='table' 로 삽입.
    - 중복 방지를 위해 테이블이 있는 페이지는 일반 텍스트에서 테이블 패턴 제외
    """
    merged: List[Dict[str, Any]] = []
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    
    # 페이지별로 블록 그룹화
    for b in labeled_blocks:
        by_page.setdefault(b["page"], []).append(b)

    for pno in sorted(by_page.keys()):
        page_blocks = by_page[pno]
        
        # 해당 페이지에 테이블이 있는지 확인
        has_tables = pno in table_map and table_map[pno]
        
        if has_tables:
            # 테이블이 있는 페이지는 테이블 패턴을 제외한 일반 텍스트만 추가
            for block in page_blocks:
                if not _is_table_pattern(block["text"]):
                    merged.append(block)
            
            # 구조화된 테이블 추가
            for tb in table_map[pno]:
                text_rows = [" | ".join(r) for r in tb["rows"]]
                merged.append({
                    **page_blocks[0], 
                    "section_type": "table", 
                    "text": "\n".join(text_rows)
                })
        else:
            # 테이블이 없는 페이지는 그대로 추가
            merged.extend(page_blocks)
    
    return merged

def _build_index(chunks_meta: List[Dict[str, Any]]):
    """Chroma DB에 벡터 인덱스 구축 - multilingual-e5-small-ko 모델 사용"""
    os.makedirs(OUT_DIR, exist_ok=True)
    texts = [c["text"] for c in chunks_meta]
    if not texts:
        print("⚠️ No text chunks parsed. Abort.")
        return
    
    if chromadb is None:
        print("⚠️ chromadb not available. Skipping index build.")
        return
    
    try:
        # Chroma DB 클라이언트 초기화
        client = chromadb.PersistentClient(
            path=OUT_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 기존 컬렉션 확인 및 처리
        collection = None
        try:
            # 기존 컬렉션이 있는지 확인
            collection = client.get_collection(COLLECTION_NAME)
            print(f"📋 Found existing collection: {COLLECTION_NAME}")
            # 기존 컬렉션 삭제
            client.delete_collection(COLLECTION_NAME)
            print(f"🗑️ Deleted existing collection: {COLLECTION_NAME}")
        except Exception:
            print(f"📋 No existing collection found: {COLLECTION_NAME}")
        
        # 새 컬렉션 생성
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "여행자보험 문서 벡터 데이터베이스"}
        )
        print(f"✨ Created new collection: {COLLECTION_NAME}")
        
        # 문서 ID 생성 및 메타데이터 준비
        doc_ids = [f"doc_{i}" for i in range(len(chunks_meta))]
        metadatas = []
        
        for chunk in chunks_meta:
            # Chroma DB 메타데이터는 문자열 값만 허용
            metadata = {}
            for key, value in chunk.items():
                if key != "text" and isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)
            metadatas.append(metadata)
        
        # multilingual-e5-small-ko 모델을 사용하여 임베딩 생성
        print("🔄 Generating embeddings with multilingual-e5-small-ko model...")
        embeddings = embed_texts(texts)
        
        # 배치 단위로 컬렉션에 문서와 임베딩 추가 (Chroma DB 배치 크기 제한 대응)
        BATCH_SIZE = 5000  # Chroma DB 최대 배치 크기보다 작게 설정
        total_chunks = len(chunks_meta)
        
        print(f"🔄 배치 단위로 벡터 DB에 저장 중... (총 {total_chunks}개 청크)")
        
        for i in range(0, total_chunks, BATCH_SIZE):
            end_idx = min(i + BATCH_SIZE, total_chunks)
            batch_texts = texts[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            batch_ids = doc_ids[i:end_idx]
            batch_embeddings = embeddings[i:end_idx].tolist()
            
            print(f"  📦 배치 {i//BATCH_SIZE + 1}: {len(batch_texts)}개 청크 저장 중...")
            
            collection.add(
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids,
                embeddings=batch_embeddings
            )
        
        print(f"✅ Built Chroma DB index: {len(chunks_meta)} chunks → {OUT_DIR}/{COLLECTION_NAME}")
        
    except Exception as e:
        print(f"❌ Failed to build Chroma DB index: {e}")
        return

def _remove_duplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    중복된 청크 제거
    """
    seen_texts = set()
    unique_chunks = []
    duplicate_count = 0
    
    for chunk in chunks:
        # 텍스트 정규화 (공백, 특수문자 정리)
        normalized = re.sub(r'\s+', ' ', chunk["text"].strip())
        
        if normalized and normalized not in seen_texts:
            seen_texts.add(normalized)
            unique_chunks.append(chunk)
        else:
            duplicate_count += 1
            if duplicate_count <= 5:  # 처음 5개만 로그 출력
                print(f"⚠️ 중복 청크 제거: {chunk.get('doc_id', 'unknown')} - {normalized[:50]}...")
    
    if duplicate_count > 5:
        print(f"⚠️ 총 {duplicate_count}개의 중복 청크가 제거되었습니다.")
    
    return unique_chunks

def _filter_empty_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    빈 청크 및 너무 짧은 청크 필터링
    """
    filtered_chunks = []
    empty_count = 0
    
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        
        # 빈 텍스트 또는 너무 짧은 텍스트 제거
        if not text or len(text) < 10:
            empty_count += 1
            continue
        
        # 의미있는 텍스트만 포함 (숫자, 한글, 영문이 포함된 경우)
        if re.search(r'[가-힣a-zA-Z0-9]', text):
            filtered_chunks.append(chunk)
        else:
            empty_count += 1
    
    if empty_count > 0:
        print(f"⚠️ {empty_count}개의 빈/무의미한 청크가 제거되었습니다.")
    
    return filtered_chunks

def _analyze_chunk_sizes(chunks: List[Dict[str, Any]], target_size: int = 800) -> None:
    """청크 크기 분석 및 통계 출력 (개선된 버전)"""
    sizes = [len(chunk["text"]) for chunk in chunks]
    if not sizes:
        return
    
    avg_size = sum(sizes) / len(sizes)
    min_size = min(sizes)
    max_size = max(sizes)
    std_dev = (sum((x - avg_size) ** 2 for x in sizes) / len(sizes)) ** 0.5
    
    # 목표 크기 기준 통계
    target_range_count = sum(1 for s in sizes if target_size * 0.8 <= s <= target_size * 1.2)
    target_ratio = target_range_count / len(sizes) * 100
    
    # 크기 분포 분석
    size_ranges = [
        (0, target_size * 0.5, "매우 작음"),
        (target_size * 0.5, target_size * 0.8, "작음"),
        (target_size * 0.8, target_size * 1.2, "적절함"),
        (target_size * 1.2, target_size * 1.5, "큼"),
        (target_size * 1.5, float('inf'), "매우 큼")
    ]
    
    print(f"\n📊 청크 크기 분석 (개선된 버전):")
    print(f"  - 총 청크 수: {len(chunks)}")
    print(f"  - 평균 크기: {avg_size:.1f}자")
    print(f"  - 최소/최대 크기: {min_size}/{max_size}자")
    print(f"  - 표준편차: {std_dev:.1f}자")
    print(f"  - 목표 크기({target_size}±20%) 범위: {target_range_count}개 ({target_ratio:.1f}%)")
    
    print(f"\n📈 크기 분포:")
    for start, end, label in size_ranges:
        count = sum(1 for s in sizes if start <= s < end)
        if count > 0:
            ratio = count / len(sizes) * 100
            print(f"  - {label}: {count}개 ({ratio:.1f}%)")
    
    # 품질 평가
    quality_score = 0
    if target_ratio >= 80:
        quality_score = 5  # 우수
    elif target_ratio >= 60:
        quality_score = 4  # 양호
    elif target_ratio >= 40:
        quality_score = 3  # 보통
    elif target_ratio >= 20:
        quality_score = 2  # 미흡
    else:
        quality_score = 1  # 불량
    
    quality_labels = ["불량", "미흡", "보통", "양호", "우수"]
    print(f"\n🎯 청킹 품질: {quality_labels[quality_score-1]} ({quality_score}/5)")
    
    # 개선 제안
    if target_ratio < 60:
        print(f"\n💡 개선 제안:")
        if target_ratio < 40:
            print(f"  - 블록 병합 임계값을 더 크게 설정 (현재: 700자)")
            print(f"  - 최소 청크 크기 강제 적용 필요")
        if std_dev > target_size * 0.3:
            print(f"  - 청크 크기 일관성 개선 필요 (표준편차: {std_dev:.1f}자)")
        if min_size < target_size * 0.5:
            print(f"  - 너무 작은 청크들 강제 병합 필요")

def main():
    pdfs = sorted([p for p in Path(DOC_DIR).glob("*.pdf")])
    if not pdfs:
        print(f"⚠️ No PDFs under {DOC_DIR}. Place files like '2025_보험사_문서제목.pdf'")
        return

    all_sections: List[Dict[str, Any]] = []
    for p in tqdm(pdfs, desc="Parsing PDFs (PyMuPDF only)"):
        blocks = _blocks_from_pymupdf(p)
        labeled = _label_sections(blocks)
        tables = _extract_tables_pymupdf(p)
        merged = _merge_tables(labeled, tables)
        
        # 🔧 개선: 작은 블록 병합 적용 (700자 임계값)
        print(f"📄 {p.name}: {len(merged)}개 블록 추출")
        merged_blocks = _merge_small_blocks(merged, min_size=700)
        print(f"📄 {p.name}: {len(merged_blocks)}개 블록으로 병합 (병합률: {len(merged_blocks)/len(merged)*100:.1f}%)")
        
        # 🔧 개선: 병합된 블록 크기 검증
        avg_block_size = sum(len(block["text"]) for block in merged_blocks) / len(merged_blocks) if merged_blocks else 0
        print(f"📄 {p.name}: 평균 블록 크기 {avg_block_size:.1f}자")
        
        all_sections.extend(merged_blocks)

    print(f"📄 총 {len(all_sections)}개의 섹션이 추출되었습니다.")

    # 섹션 텍스트 기반 키워드 태깅
    section_texts = [s["text"] for s in all_sections]
    section_tags = _keyword_tags(section_texts, topk=8)
    for s, tags in zip(all_sections, section_tags):
        s["tags"] = tags

    # 🔧 개선: RecursiveCharacterTextSplitter + 균등화를 사용한 일정한 크기 청크 생성
    chunks_meta: List[Dict[str, Any]] = []
    for s in tqdm(all_sections, desc="Uniform Chunking with Recursive Splitter"):
        # 표는 행 단위로 이미 짧은 편 → 바로 저장. (크면 일반 청킹)
        if s.get("section_type") == "table" and len(s["text"]) <= CHUNK_SIZE:
            m = dict(s)
            m["chunk_no"] = 1
            chunks_meta.append(m)
        else:
            # RecursiveCharacterTextSplitter + 균등화 사용
            chunked_texts = _uniform_chunking_with_recursive_splitter(
                s["text"], CHUNK_SIZE, CHUNK_OVERLAP
            )
            for i, ch in enumerate(chunked_texts, start=1):
                m = dict(s)
                m["text"] = ch
                m["chunk_no"] = i
                chunks_meta.append(m)

    print(f"🔧 총 {len(chunks_meta)}개의 청크가 생성되었습니다.")

    # 중복 및 빈 청크 제거
    chunks_meta = _remove_duplicate_chunks(chunks_meta)
    chunks_meta = _filter_empty_chunks(chunks_meta)
    
    # 청크 크기 분석
    _analyze_chunk_sizes(chunks_meta, CHUNK_SIZE)
    
    print(f"✅ 최종 {len(chunks_meta)}개의 청크가 벡터 DB에 저장됩니다.")

    _build_index(chunks_meta)

if __name__ == "__main__":
    main()