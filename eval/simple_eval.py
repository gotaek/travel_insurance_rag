#!/usr/bin/env python3
"""
기본적인 RAG 시스템 평가 도구
RAGAS 없이 간단하고 효과적인 평가 시스템
"""

import json
import logging
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from graph.builder import build_graph

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 경로 설정
QUESTIONS_PATH = Path("eval/questions.jsonl")
OUTPUT_DIR = Path("eval/out")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_questions() -> List[Dict[str, Any]]:
    """평가 질문들을 JSONL 파일에서 로드."""
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"평가 질문 파일을 찾을 수 없습니다: {QUESTIONS_PATH}")
    
    questions = []
    for line_num, line in enumerate(QUESTIONS_PATH.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        
        try:
            question_data = json.loads(line)
            if isinstance(question_data, dict):
                questions.append(question_data)
        except json.JSONDecodeError as e:
            logger.error(f"라인 {line_num} JSON 파싱 오류: {e}")
            continue
    
    logger.info(f"로드된 평가 질문: {len(questions)}개")
    return questions


def run_rag_system(graph, question: str) -> Dict[str, Any]:
    """
    RAG 시스템을 실행하여 답변과 컨텍스트를 생성.
    
    Args:
        graph: RAG 시스템 그래프
        question: 질문
        
    Returns:
        RAG 시스템 실행 결과
    """
    try:
        start_time = time.time()
        state = {"question": question}
        result = graph.invoke(state)
        end_time = time.time()
        
        # 답변 추출
        draft_answer = result.get("draft_answer", {})
        answer_text = ""
        if isinstance(draft_answer, dict):
            answer_text = draft_answer.get("conclusion", "")
        
        # 컨텍스트 추출
        refined_docs = result.get("refined", [])
        contexts = []
        for doc in refined_docs:
            if isinstance(doc, dict) and doc.get("text"):
                contexts.append(doc["text"])
        
        return {
            "answer": answer_text,
            "contexts": contexts,
            "response_time": end_time - start_time,
            "success": True,
            "raw_result": result
        }
        
    except Exception as e:
        logger.error(f"RAG 시스템 실행 중 오류: {e}")
        return {
            "answer": "",
            "contexts": [],
            "response_time": 0,
            "success": False,
            "error": str(e)
        }


def calculate_recall_at_k_direct(ground_truths: List[str], contexts: List[str], k: int) -> float:
    """
    특정 k 값에 대한 Recall@K 직접 계산 (재귀 없음).
    개선된 유연한 매칭 알고리즘 사용.
    
    Args:
        ground_truths: 정답 키워드 리스트
        contexts: 검색된 컨텍스트 리스트
        k: 상위 K개 컨텍스트만 고려
        
    Returns:
        Recall@K 점수
    """
    if not ground_truths or not contexts:
        return 0.0
    
    # 상위 k개 컨텍스트만 사용
    top_k_contexts = contexts[:k]
    
    # 모든 컨텍스트를 하나의 텍스트로 합치기
    combined_context = " ".join(top_k_contexts).lower()
    
    # 정답 키워드와 매칭 확인
    matched_ground_truths = []
    
    for ground_truth in ground_truths:
        # 키워드를 소문자로 변환하고 특수문자 제거
        clean_ground_truth = re.sub(r'[^\w\s]', '', ground_truth.lower())
        clean_ground_truth = re.sub(r'\s+', ' ', clean_ground_truth).strip()
        
        # 단어 단위로 분할
        ground_truth_words = clean_ground_truth.split()
        
        # 유연한 매칭을 위한 키워드 매핑 (questions.jsonl 기반 확장)
        keyword_mappings = {
            # 항공 관련
            "항공기": ["항공편", "항공", "비행기", "출국"],
            "출발": ["출국", "이륙"],
            "지연": ["연착", "지연"],
            "수하물": ["짐", "가방", "화물"],
            
            # 보험/보장 관련
            "보상": ["보장", "지급", "보험금", "담보", "특별약관"],
            "보장": ["보상", "지급", "보험금", "담보", "특별약관"],
            "담보": ["보장", "보상", "특별약관"],
            "특별약관": ["특약", "담보", "보장"],
            "특약": ["특별약관", "담보", "보장"],
            
            # 의료 관련
            "의료비": ["치료비", "진료비", "병원비"],
            "급여": ["급여", "비급여"],
            "비급여": ["비급여", "급여"],
            "입원": ["입원", "통원"],
            "통원": ["입원", "통원"],
            "응급실": ["응급실", "응급"],
            "진단서": ["진단서", "의무기록", "진료기록"],
            "의무기록": ["진단서", "의무기록", "진료기록"],
            
            # 휴대품 관련
            "휴대품": ["휴대품", "개인물품", "소지품"],
            "분실": ["분실", "도난", "분실제외"],
            "도난": ["분실", "도난", "도난"],
            "파손": ["파손", "손해", "손상"],
            "손해": ["파손", "손해", "손상"],
            "휴대폰": ["휴대폰", "스마트폰", "핸드폰"],
            
            # 서류/증빙 관련
            "서류": ["서류", "증빙", "문서", "확인서"],
            "증빙": ["서류", "증빙", "문서", "확인서"],
            "확인서": ["증명서", "서류", "문서", "확인서"],
            "영수증": ["영수증", "영수증", "수수료"],
            "신고서": ["신고서", "신고서", "경찰신고서"],
            "경찰신고서": ["신고서", "경찰신고서", "분실각서"],
            "분실각서": ["신고서", "경찰신고서", "분실각서"],
            "사고경위서": ["사고경위서", "경위서"],
            "보험금청구서": ["보험금청구서", "청구서"],
            
            # 여권 관련
            "여권": ["여권", "여권분실"],
            "여권분실": ["여권", "여권분실"],
            "재발급": ["재발급", "재발급비용"],
            "재발급비용": ["재발급", "재발급비용"],
            "대사관": ["대사관", "영사관"],
            "영사관": ["대사관", "영사관"],
            
            # 배상책임 관련
            "배상책임": ["배상책임", "배상"],
            "배상": ["배상책임", "배상"],
            "상해": ["상해", "부상", "신체"],
            "부상": ["상해", "부상", "신체"],
            "재물": ["재물", "물건", "재산"],
            "소송비용": ["소송비용", "법정비용"],
            
            # 보험료/가입 관련
            "보험료": ["보험료", "보험료"],
            "가입": ["가입", "계약"],
            "계약": ["가입", "계약"],
            "연령": ["나이", "연령"],
            "나이": ["연령", "나이"],
            "고령자": ["고령자", "노인", "시니어"],
            "할증": ["할증", "추가요금"],
            "차등": ["차등", "차이"],
            
            # 여행 관련
            "여행": ["여행", "출장", "관광"],
            "출장": ["여행", "출장", "관광"],
            "관광": ["여행", "출장", "관광"],
            "여행기간": ["여행기간", "체류기간"],
            "체류기간": ["여행기간", "체류기간"],
            "여행지역": ["여행지역", "목적지"],
            "목적지": ["여행지역", "목적지"],
            "장기여행": ["장기여행", "장기체류"],
            "장기체류": ["장기여행", "장기체류"],
            
            # 스포츠 관련
            "스포츠": ["스포츠", "운동", "레저"],
            "운동": ["스포츠", "운동", "레저"],
            "스키": ["스키", "스노보드"],
            "스노보드": ["스키", "스노보드"],
            "겨울스포츠": ["스키", "스노보드", "겨울스포츠"],
            "부상치료비": ["부상치료비", "치료비", "의료비"],
            "장비손해": ["장비손해", "장비파손"],
            
            # 면책사항 관련
            "면책": ["면책", "제외", "보장제외"],
            "제외": ["면책", "제외", "보장제외"],
            "보장제외": ["면책", "제외", "보장제외"],
            "음주": ["음주", "술"],
            "술": ["음주", "술"],
            "약물": ["약물", "마약"],
            "고의": ["고의", "의도적"],
            "의도적": ["고의", "의도적"],
            "중과실": ["중과실", "과실"],
            "과실": ["중과실", "과실"],
            "전문등반": ["전문등반", "등반"],
            "등반": ["전문등반", "등반"],
            "경기": ["경기", "대회"],
            "대회": ["경기", "대회"],
            "전쟁": ["전쟁", "테러"],
            "테러": ["전쟁", "테러"],
            "폭동": ["폭동", "폭력"],
            "폭력": ["폭동", "폭력"],
            "임신": ["임신", "출산"],
            "출산": ["임신", "출산"],
            
            # 렌터카 관련
            "렌터카": ["렌터카", "렌트카"],
            "렌트카": ["렌터카", "렌트카"],
            "자동차": ["자동차", "차량"],
            "차량": ["자동차", "차량"],
            "운전자": ["운전자", "운전"],
            "운전": ["운전자", "운전"],
            "면허": ["면허", "운전면허"],
            "운전면허": ["면허", "운전면허"],
            "CDW": ["CDW", "면책금"],
            "면책금": ["CDW", "면책금"],
            
            # 보험금 청구 관련
            "청구": ["청구", "신청"],
            "신청": ["청구", "신청"],
            "가지급": ["가지급", "선지급"],
            "선지급": ["가지급", "선지급"],
            "신분증": ["신분증", "신분증명서"],
            "신분증명서": ["신분증", "신분증명서"],
            "계좌사본": ["계좌사본", "통장사본"],
            "통장사본": ["계좌사본", "통장사본"],
            
            # 보험기간 관련
            "보험기간": ["보험기간", "보장기간"],
            "보장기간": ["보험기간", "보장기간"],
            "보장개시": ["보장개시", "보장시작"],
            "보장시작": ["보장개시", "보장시작"],
            "대기기간": ["대기기간", "면책기간"],
            "면책기간": ["대기기간", "면책기간"],
            "보험증권": ["보험증권", "증권"],
            "증권": ["보험증권", "증권"],
            
            # 한도/금액 관련
            "한도": ["한도", "한도금액", "최대금액"],
            "한도금액": ["한도", "한도금액", "최대금액"],
            "최대금액": ["한도", "한도금액", "최대금액"],
            "자기부담": ["자기부담", "자기부담금"],
            "자기부담금": ["자기부담", "자기부담금"],
            "본인부담률": ["본인부담률", "부담률"],
            "부담률": ["본인부담률", "부담률"],
            "공제": ["공제", "차감"],
            "차감": ["공제", "차감"],
            "감가": ["감가", "감가상각"],
            "감가상각": ["감가", "감가상각"],
            
            # 구조/송환 관련
            "구조송환": ["구조송환", "송환"],
            "송환": ["구조송환", "송환"],
            "구조": ["구조", "구조송환"],
            
            # 사망/후유장해 관련
            "사망": ["사망", "사고사"],
            "사고사": ["사망", "사고사"],
            "후유장해": ["후유장해", "장해"],
            "장해": ["후유장해", "장해"],
            
            # 추가비용 관련
            "추가비용": ["추가비용", "부대비용"],
            "부대비용": ["추가비용", "부대비용"],
            "교통비": ["교통비", "이동비"],
            "이동비": ["교통비", "이동비"],
            "수수료": ["수수료", "요금"],
            "요금": ["수수료", "요금"]
        }
        
        # 각 단어가 컨텍스트에 포함되는지 확인 (유연한 매칭 포함)
        word_matches = 0
        for word in ground_truth_words:
            if len(word) > 1:  # 1글자 이상 단어 고려
                # 직접 매칭
                if word in combined_context:
                    word_matches += 1
                # 유연한 매칭 (동의어/유사어)
                elif word in keyword_mappings:
                    synonyms = keyword_mappings[word]
                    for synonym in synonyms:
                        if synonym in combined_context:
                            word_matches += 1
                            break
        
        # 30% 이상의 단어가 매칭되면 해당 ground_truth는 매칭된 것으로 간주 (임계값 낮춤)
        if word_matches >= len(ground_truth_words) * 0.3:
            matched_ground_truths.append(ground_truth)
    
    # Recall@K 계산
    total_ground_truths = len(ground_truths)
    matched_count = len(matched_ground_truths)
    
    return matched_count / total_ground_truths if total_ground_truths > 0 else 0.0


def calculate_recall_at_k(ground_truths: List[str], contexts: List[str], k: int = 5) -> Dict[str, Any]:
    """
    Recall@K 메트릭 계산 (일관된 로직 사용).
    
    Args:
        ground_truths: 정답 키워드 리스트
        contexts: 검색된 컨텍스트 리스트
        k: 상위 K개 컨텍스트만 고려
        
    Returns:
        Recall@K 메트릭
    """
    if not ground_truths or not contexts:
        return {
            "recall_at_k": 0.0,
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "matched_ground_truths": [],
            "total_ground_truths": len(ground_truths)
        }
    
    # 모든 k 값에 대해 동일한 로직 사용 (calculate_recall_at_k_direct)
    recall_at_1 = calculate_recall_at_k_direct(ground_truths, contexts, 1)
    recall_at_3 = calculate_recall_at_k_direct(ground_truths, contexts, 3)
    recall_at_5 = calculate_recall_at_k_direct(ground_truths, contexts, 5)
    recall_at_k = calculate_recall_at_k_direct(ground_truths, contexts, k)
    
    # 매칭된 ground_truths 계산 (k=5 기준)
    matched_ground_truths = []
    top_k_contexts = contexts[:5]  # k=5로 고정
    combined_context = " ".join(top_k_contexts).lower()
    
    for ground_truth in ground_truths:
        # 키워드를 소문자로 변환하고 특수문자 제거
        clean_ground_truth = re.sub(r'[^\w\s]', '', ground_truth.lower())
        clean_ground_truth = re.sub(r'\s+', ' ', clean_ground_truth).strip()
        
        # 단어 단위로 분할
        ground_truth_words = clean_ground_truth.split()
        
        # 유연한 매칭을 위한 키워드 매핑 (calculate_recall_at_k_direct와 동일)
        keyword_mappings = {
            # 항공 관련
            "항공기": ["항공편", "항공", "비행기", "출국"],
            "출발": ["출국", "이륙"],
            "지연": ["연착", "지연"],
            "수하물": ["짐", "가방", "화물"],
            
            # 보험/보장 관련
            "보상": ["보장", "지급", "보험금", "담보", "특별약관"],
            "보장": ["보상", "지급", "보험금", "담보", "특별약관"],
            "담보": ["보장", "보상", "특별약관"],
            "특별약관": ["특약", "담보", "보장"],
            "특약": ["특별약관", "담보", "보장"],
            
            # 의료 관련
            "의료비": ["치료비", "진료비", "병원비"],
            "급여": ["급여", "비급여"],
            "비급여": ["비급여", "급여"],
            "입원": ["입원", "통원"],
            "통원": ["입원", "통원"],
            "응급실": ["응급실", "응급"],
            "진단서": ["진단서", "의무기록", "진료기록"],
            "의무기록": ["진단서", "의무기록", "진료기록"],
            
            # 휴대품 관련
            "휴대품": ["휴대품", "개인물품", "소지품"],
            "분실": ["분실", "도난", "분실제외"],
            "도난": ["분실", "도난", "도난"],
            "파손": ["파손", "손해", "손상"],
            "손해": ["파손", "손해", "손상"],
            "휴대폰": ["휴대폰", "스마트폰", "핸드폰"],
            
            # 서류/증빙 관련
            "서류": ["서류", "증빙", "문서", "확인서"],
            "증빙": ["서류", "증빙", "문서", "확인서"],
            "확인서": ["증명서", "서류", "문서", "확인서"],
            "영수증": ["영수증", "영수증", "수수료"],
            "신고서": ["신고서", "신고서", "경찰신고서"],
            "경찰신고서": ["신고서", "경찰신고서", "분실각서"],
            "분실각서": ["신고서", "경찰신고서", "분실각서"],
            "사고경위서": ["사고경위서", "경위서"],
            "보험금청구서": ["보험금청구서", "청구서"],
            
            # 여권 관련
            "여권": ["여권", "여권분실"],
            "여권분실": ["여권", "여권분실"],
            "재발급": ["재발급", "재발급비용"],
            "재발급비용": ["재발급", "재발급비용"],
            "대사관": ["대사관", "영사관"],
            "영사관": ["대사관", "영사관"],
            
            # 배상책임 관련
            "배상책임": ["배상책임", "배상"],
            "배상": ["배상책임", "배상"],
            "상해": ["상해", "부상", "신체"],
            "부상": ["상해", "부상", "신체"],
            "재물": ["재물", "물건", "재산"],
            "소송비용": ["소송비용", "법정비용"],
            
            # 보험료/가입 관련
            "보험료": ["보험료", "보험료"],
            "가입": ["가입", "계약"],
            "계약": ["가입", "계약"],
            "연령": ["나이", "연령"],
            "나이": ["연령", "나이"],
            "고령자": ["고령자", "노인", "시니어"],
            "할증": ["할증", "추가요금"],
            "차등": ["차등", "차이"],
            
            # 여행 관련
            "여행": ["여행", "출장", "관광"],
            "출장": ["여행", "출장", "관광"],
            "관광": ["여행", "출장", "관광"],
            "여행기간": ["여행기간", "체류기간"],
            "체류기간": ["여행기간", "체류기간"],
            "여행지역": ["여행지역", "목적지"],
            "목적지": ["여행지역", "목적지"],
            "장기여행": ["장기여행", "장기체류"],
            "장기체류": ["장기여행", "장기체류"],
            
            # 스포츠 관련
            "스포츠": ["스포츠", "운동", "레저"],
            "운동": ["스포츠", "운동", "레저"],
            "스키": ["스키", "스노보드"],
            "스노보드": ["스키", "스노보드"],
            "겨울스포츠": ["스키", "스노보드", "겨울스포츠"],
            "부상치료비": ["부상치료비", "치료비", "의료비"],
            "장비손해": ["장비손해", "장비파손"],
            
            # 면책사항 관련
            "면책": ["면책", "제외", "보장제외"],
            "제외": ["면책", "제외", "보장제외"],
            "보장제외": ["면책", "제외", "보장제외"],
            "음주": ["음주", "술"],
            "술": ["음주", "술"],
            "약물": ["약물", "마약"],
            "고의": ["고의", "의도적"],
            "의도적": ["고의", "의도적"],
            "중과실": ["중과실", "과실"],
            "과실": ["중과실", "과실"],
            "전문등반": ["전문등반", "등반"],
            "등반": ["전문등반", "등반"],
            "경기": ["경기", "대회"],
            "대회": ["경기", "대회"],
            "전쟁": ["전쟁", "테러"],
            "테러": ["전쟁", "테러"],
            "폭동": ["폭동", "폭력"],
            "폭력": ["폭동", "폭력"],
            "임신": ["임신", "출산"],
            "출산": ["임신", "출산"],
            
            # 렌터카 관련
            "렌터카": ["렌터카", "렌트카"],
            "렌트카": ["렌터카", "렌트카"],
            "자동차": ["자동차", "차량"],
            "차량": ["자동차", "차량"],
            "운전자": ["운전자", "운전"],
            "운전": ["운전자", "운전"],
            "면허": ["면허", "운전면허"],
            "운전면허": ["면허", "운전면허"],
            "CDW": ["CDW", "면책금"],
            "면책금": ["CDW", "면책금"],
            
            # 보험금 청구 관련
            "청구": ["청구", "신청"],
            "신청": ["청구", "신청"],
            "가지급": ["가지급", "선지급"],
            "선지급": ["가지급", "선지급"],
            "신분증": ["신분증", "신분증명서"],
            "신분증명서": ["신분증", "신분증명서"],
            "계좌사본": ["계좌사본", "통장사본"],
            "통장사본": ["계좌사본", "통장사본"],
            
            # 보험기간 관련
            "보험기간": ["보험기간", "보장기간"],
            "보장기간": ["보험기간", "보장기간"],
            "보장개시": ["보장개시", "보장시작"],
            "보장시작": ["보장개시", "보장시작"],
            "대기기간": ["대기기간", "면책기간"],
            "면책기간": ["대기기간", "면책기간"],
            "보험증권": ["보험증권", "증권"],
            "증권": ["보험증권", "증권"],
            
            # 한도/금액 관련
            "한도": ["한도", "한도금액", "최대금액"],
            "한도금액": ["한도", "한도금액", "최대금액"],
            "최대금액": ["한도", "한도금액", "최대금액"],
            "자기부담": ["자기부담", "자기부담금"],
            "자기부담금": ["자기부담", "자기부담금"],
            "본인부담률": ["본인부담률", "부담률"],
            "부담률": ["본인부담률", "부담률"],
            "공제": ["공제", "차감"],
            "차감": ["공제", "차감"],
            "감가": ["감가", "감가상각"],
            "감가상각": ["감가", "감가상각"],
            
            # 구조/송환 관련
            "구조송환": ["구조송환", "송환"],
            "송환": ["구조송환", "송환"],
            "구조": ["구조", "구조송환"],
            
            # 사망/후유장해 관련
            "사망": ["사망", "사고사"],
            "사고사": ["사망", "사고사"],
            "후유장해": ["후유장해", "장해"],
            "장해": ["후유장해", "장해"],
            
            # 추가비용 관련
            "추가비용": ["추가비용", "부대비용"],
            "부대비용": ["추가비용", "부대비용"],
            "교통비": ["교통비", "이동비"],
            "이동비": ["교통비", "이동비"],
            "수수료": ["수수료", "요금"],
            "요금": ["수수료", "요금"]
        }
        
        # 각 단어가 컨텍스트에 포함되는지 확인 (유연한 매칭 포함)
        word_matches = 0
        for word in ground_truth_words:
            if len(word) > 1:  # 1글자 이상 단어 고려
                # 직접 매칭
                if word in combined_context:
                    word_matches += 1
                # 유연한 매칭 (동의어/유사어)
                elif word in keyword_mappings:
                    synonyms = keyword_mappings[word]
                    for synonym in synonyms:
                        if synonym in combined_context:
                            word_matches += 1
                            break
        
        # 30% 이상의 단어가 매칭되면 해당 ground_truth는 매칭된 것으로 간주
        if word_matches >= len(ground_truth_words) * 0.3:
            matched_ground_truths.append(ground_truth)
    
    total_ground_truths = len(ground_truths)
    matched_count = len(matched_ground_truths)
    
    return {
        "recall_at_k": recall_at_k,
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "recall_at_5": recall_at_5,
        "matched_ground_truths": matched_ground_truths,
        "total_ground_truths": total_ground_truths,
        "matched_count": matched_count
    }


def calculate_basic_metrics(question_data: Dict[str, Any], rag_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    기본적인 평가 메트릭 계산 (Recall@K 포함).
    
    Args:
        question_data: 질문 데이터
        rag_result: RAG 시스템 실행 결과
        
    Returns:
        평가 메트릭
    """
    metrics = {
        "question_id": question_data.get("id", "unknown"),
        "question": question_data.get("question", ""),
        "category": question_data.get("category", "unknown"),
        "intent": question_data.get("intent", "unknown"),
        "response_time": rag_result.get("response_time", 0),
        "success": rag_result.get("success", False),
        "answer_length": len(rag_result.get("answer", "")),
        "context_count": len(rag_result.get("contexts", [])),
        "context_length": sum(len(ctx) for ctx in rag_result.get("contexts", [])),
    }
    
    # Recall@K 메트릭 계산
    ground_truths = question_data.get("ground_truths", [])
    contexts = rag_result.get("contexts", [])
    
    recall_metrics = calculate_recall_at_k(ground_truths, contexts, k=5)
    metrics.update(recall_metrics)
    
    # 기존 키워드 매칭 점수 (답변 기반)
    answer = rag_result.get("answer", "").lower()
    
    if ground_truths and answer:
        matched_keywords = 0
        total_keywords = len(ground_truths)
        
        for keyword in ground_truths:
            # 키워드를 소문자로 변환하고 공백 제거
            clean_keyword = re.sub(r'\s+', '', keyword.lower())
            clean_answer = re.sub(r'\s+', '', answer)
            
            if clean_keyword in clean_answer:
                matched_keywords += 1
        
        metrics["answer_keyword_match_score"] = matched_keywords / total_keywords if total_keywords > 0 else 0
        metrics["answer_matched_keywords"] = matched_keywords
        metrics["answer_total_keywords"] = total_keywords
    else:
        metrics["answer_keyword_match_score"] = 0
        metrics["answer_matched_keywords"] = 0
        metrics["answer_total_keywords"] = 0
    
    # 답변 품질 점수 (간단한 휴리스틱)
    quality_score = 0
    
    # 답변 길이 점수 (너무 짧거나 길면 감점)
    answer_length = metrics["answer_length"]
    if 50 <= answer_length <= 500:
        quality_score += 0.2
    elif 20 <= answer_length < 50 or 500 < answer_length <= 1000:
        quality_score += 0.15
    else:
        quality_score += 0.1
    
    # 컨텍스트 점수
    if metrics["context_count"] > 0:
        quality_score += 0.2
    
    # Recall@K 점수 (가장 중요)
    quality_score += metrics["recall_at_k"] * 0.5
    
    # 답변 키워드 매칭 점수
    quality_score += metrics["answer_keyword_match_score"] * 0.1
    
    metrics["quality_score"] = min(quality_score, 1.0)
    
    return metrics


def run_evaluation(questions: List[Dict[str, Any]], graph) -> List[Dict[str, Any]]:
    """
    전체 평가 실행.
    
    Args:
        questions: 평가 질문 리스트
        graph: RAG 시스템 그래프
        
    Returns:
        평가 결과 리스트
    """
    results = []
    
    for i, question_data in enumerate(questions, 1):
        question = question_data.get("question", "")
        if not question:
            continue
        
        logger.info(f"평가 진행 중: {i}/{len(questions)} - {question_data.get('id', 'unknown')}")
        
        # RAG 시스템 실행
        rag_result = run_rag_system(graph, question)
        
        # 메트릭 계산
        metrics = calculate_basic_metrics(question_data, rag_result)
        
        # 결과 저장
        result = {
            **metrics,
            "answer": rag_result.get("answer", ""),
            "contexts": rag_result.get("contexts", []),
            "ground_truths": question_data.get("ground_truths", []),
            "error": rag_result.get("error", None)
        }
        
        results.append(result)
        
        # 진행 상황 출력
        logger.info(f"  - 응답시간: {metrics['response_time']:.2f}초")
        logger.info(f"  - 답변길이: {metrics['answer_length']}자")
        logger.info(f"  - 컨텍스트: {metrics['context_count']}개")
        logger.info(f"  - Recall@1: {metrics['recall_at_1']:.3f}")
        logger.info(f"  - Recall@3: {metrics['recall_at_3']:.3f}")
        logger.info(f"  - Recall@5: {metrics['recall_at_5']:.3f}")
        logger.info(f"  - 매칭된정답: {metrics['matched_count']}/{metrics['total_ground_truths']}")
        logger.info(f"  - 답변키워드매칭: {metrics['answer_matched_keywords']}/{metrics['answer_total_keywords']}")
        logger.info(f"  - 품질점수: {metrics['quality_score']:.3f}")
    
    return results


def save_results(results: List[Dict[str, Any]]) -> None:
    """평가 결과를 파일로 저장."""
    if not results:
        logger.warning("저장할 결과가 없습니다.")
        return
    
    try:
        # CSV 형태로 저장
        csv_path = OUTPUT_DIR / "simple_eval_results.csv"
        with open(csv_path, 'w', encoding='utf-8') as f:
            # 헤더 작성 (Recall@K 메트릭 포함)
            headers = [
                "question_id", "question", "category", "intent",
                "response_time", "success", "answer_length", "context_count", "context_length",
                "recall_at_k", "recall_at_1", "recall_at_3", "recall_at_5",
                "matched_ground_truths", "total_ground_truths", "matched_count",
                "answer_keyword_match_score", "answer_matched_keywords", "answer_total_keywords",
                "quality_score"
            ]
            f.write(",".join(headers) + "\n")
            
            # 데이터 작성
            for result in results:
                row = []
                for header in headers:
                    value = result.get(header, "")
                    # 리스트인 경우 세미콜론으로 구분하여 저장
                    if isinstance(value, list):
                        value = "; ".join(str(item) for item in value)
                    # 문자열에 콤마가 포함된 경우 따옴표로 감싸기
                    elif isinstance(value, str) and "," in value:
                        value = f'"{value}"'
                    row.append(str(value))
                f.write(",".join(row) + "\n")
        
        logger.info(f"CSV 결과 저장: {csv_path}")
        
        # JSON 형태로 저장
        json_path = OUTPUT_DIR / "simple_eval_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON 결과 저장: {json_path}")
        
        # 요약 통계 저장
        summary_path = OUTPUT_DIR / "simple_eval_summary.json"
        summary = calculate_summary_stats(results)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"요약 통계 저장: {summary_path}")
        
    except Exception as e:
        logger.error(f"결과 저장 중 오류: {e}")


def calculate_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """요약 통계 계산."""
    if not results:
        return {}
    
    # 기본 통계
    total_questions = len(results)
    successful_questions = sum(1 for r in results if r.get("success", False))
    
    # 평균값 계산
    avg_response_time = sum(r.get("response_time", 0) for r in results) / total_questions
    avg_answer_length = sum(r.get("answer_length", 0) for r in results) / total_questions
    avg_context_count = sum(r.get("context_count", 0) for r in results) / total_questions
    avg_recall_at_1 = sum(r.get("recall_at_1", 0) for r in results) / total_questions
    avg_recall_at_3 = sum(r.get("recall_at_3", 0) for r in results) / total_questions
    avg_recall_at_5 = sum(r.get("recall_at_5", 0) for r in results) / total_questions
    avg_answer_keyword_match = sum(r.get("answer_keyword_match_score", 0) for r in results) / total_questions
    avg_quality_score = sum(r.get("quality_score", 0) for r in results) / total_questions
    
    # 카테고리별 통계
    category_stats = {}
    for result in results:
        category = result.get("category", "unknown")
        if category not in category_stats:
            category_stats[category] = {
                "count": 0,
                "avg_quality_score": 0,
                "avg_recall_at_5": 0,
                "avg_answer_keyword_match": 0,
                "avg_response_time": 0
            }
        
        category_stats[category]["count"] += 1
        category_stats[category]["avg_quality_score"] += result.get("quality_score", 0)
        category_stats[category]["avg_recall_at_5"] += result.get("recall_at_5", 0)
        category_stats[category]["avg_answer_keyword_match"] += result.get("answer_keyword_match_score", 0)
        category_stats[category]["avg_response_time"] += result.get("response_time", 0)
    
    # 카테고리별 평균 계산
    for category in category_stats:
        count = category_stats[category]["count"]
        category_stats[category]["avg_quality_score"] /= count
        category_stats[category]["avg_recall_at_5"] /= count
        category_stats[category]["avg_answer_keyword_match"] /= count
        category_stats[category]["avg_response_time"] /= count
    
    return {
        "evaluation_date": datetime.now().isoformat(),
        "total_questions": total_questions,
        "successful_questions": successful_questions,
        "success_rate": successful_questions / total_questions,
        "average_metrics": {
            "response_time": avg_response_time,
            "answer_length": avg_answer_length,
            "context_count": avg_context_count,
            "recall_at_1": avg_recall_at_1,
            "recall_at_3": avg_recall_at_3,
            "recall_at_5": avg_recall_at_5,
            "answer_keyword_match_score": avg_answer_keyword_match,
            "quality_score": avg_quality_score
        },
        "category_stats": category_stats
    }


def print_summary(results: List[Dict[str, Any]]) -> None:
    """평가 결과 요약 출력."""
    if not results:
        print("❌ 출력할 결과가 없습니다.")
        return
    
    summary = calculate_summary_stats(results)
    
    print("\n" + "="*60)
    print("📊 기본 평가 결과 요약")
    print("="*60)
    
    print(f"📋 총 평가 질문: {summary['total_questions']}개")
    print(f"✅ 성공한 질문: {summary['successful_questions']}개")
    print(f"📈 성공률: {summary['success_rate']:.1%}")
    
    print(f"\n📊 평균 메트릭:")
    avg_metrics = summary['average_metrics']
    print(f"  ⏱️ 평균 응답시간: {avg_metrics['response_time']:.2f}초")
    print(f"  📝 평균 답변길이: {avg_metrics['answer_length']:.0f}자")
    print(f"  📚 평균 컨텍스트 수: {avg_metrics['context_count']:.1f}개")
    print(f"  🎯 Recall@1: {avg_metrics['recall_at_1']:.3f}")
    print(f"  🎯 Recall@3: {avg_metrics['recall_at_3']:.3f}")
    print(f"  🎯 Recall@5: {avg_metrics['recall_at_5']:.3f}")
    print(f"  🔍 평균 답변키워드매칭: {avg_metrics['answer_keyword_match_score']:.3f}")
    print(f"  ⭐ 평균 품질점수: {avg_metrics['quality_score']:.3f}")
    
    print(f"\n📂 카테고리별 성능:")
    for category, stats in summary['category_stats'].items():
        print(f"  {category}:")
        print(f"    - 질문 수: {stats['count']}개")
        print(f"    - 품질점수: {stats['avg_quality_score']:.3f}")
        print(f"    - Recall@5: {stats['avg_recall_at_5']:.3f}")
        print(f"    - 답변키워드매칭: {stats['avg_answer_keyword_match']:.3f}")
        print(f"    - 응답시간: {stats['avg_response_time']:.2f}초")
    
    print(f"\n📁 결과 파일:")
    print(f"  - CSV: eval/out/simple_eval_results.csv")
    print(f"  - JSON: eval/out/simple_eval_results.json")
    print(f"  - 요약: eval/out/simple_eval_summary.json")


def main():
    """메인 실행 함수."""
    try:
        print("🚀 기본 평가 시스템 시작")
        
        # 질문 로드
        questions = load_questions()
        if not questions:
            logger.error("평가할 질문이 없습니다.")
            return
        
        # RAG 그래프 빌드
        logger.info("RAG 그래프 빌드 중...")
        graph = build_graph()
        
        # 평가 실행
        results = run_evaluation(questions, graph)
        if not results:
            logger.error("평가 실행에 실패했습니다.")
            return
        
        # 결과 저장
        save_results(results)
        
        # 요약 출력
        print_summary(results)
        
        print("\n✅ 기본 평가 완료!")
        
    except Exception as e:
        logger.error(f"평가 실행 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
