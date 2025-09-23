from typing import Dict, Any
import json
import re
from app.deps import get_llm

INTENTS = ["summary", "compare", "qa", "recommend"]

def _llm_classify_intent(question: str) -> Dict[str, Any]:
    """
    LLM을 사용하여 질문의 intent와 needs_web을 분류
    """
    prompt = f"""
다음 질문을 분석하여 여행자보험 RAG 시스템에서 적절한 처리 방식을 결정해주세요.

질문: "{question}"

다음 중 하나의 intent를 선택하세요:
- "qa": 일반적인 질문-답변 (보장 내용, 가입 조건, 보험료 등)
- "summary": 문서 요약 (약관 요약, 상품 정리 등)
- "compare": 비교 분석 (보험 상품 간 비교, 차이점 분석 등)
- "recommend": 추천 및 권장 (특약 추천, 여행지별 보험 추천 등)

또한 다음 조건을 확인하여 needs_web을 결정하세요:
- 최신 뉴스나 실시간 정보가 필요한가?
- 특정 날짜나 지역의 현재 상황이 필요한가?
- 여행지의 현재 안전 상황이나 규제가 필요한가?

반드시 다음 JSON 형식으로만 답변하세요:
{{
    "intent": "qa|summary|compare|recommend",
    "needs_web": true|false,
    "reasoning": "분류 근거를 간단히 설명"
}}
"""

    try:
        llm = get_llm()
        response = llm.generate_content(prompt, request_options={"timeout": 30})
        
        # JSON 파싱
        response_text = response.text.strip()
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        else:
            json_text = response_text
        
        result = json.loads(json_text)
        
        # 유효성 검증
        if result.get("intent") not in INTENTS:
            result["intent"] = "qa"
        if not isinstance(result.get("needs_web"), bool):
            result["needs_web"] = False
            
        return result
        
    except Exception as e:
        # LLM 호출 실패 시 fallback (디버깅 정보 포함)
        print(f"⚠️ LLM 분류 실패, fallback 사용: {str(e)}")
        return _fallback_classify(question)

def _fallback_classify(question: str) -> Dict[str, Any]:
    """
    LLM 호출 실패 시 사용하는 향상된 키워드 기반 fallback
    """
    ql = question.lower()
    
    # Intent 분류 (더 정교한 패턴 매칭)
    intent = "qa"  # 기본값
    
    # Summary 키워드 (우선순위 높음)
    summary_keywords = ["요약", "정리", "summary", "약관 요약", "상품 요약", "핵심 내용"]
    if any(k in question for k in summary_keywords):
        intent = "summary"
    
    # Compare 키워드
    elif any(k in question for k in ["비교", "차이", "다른 점", "compare", "vs", "대비", "구분"]):
        intent = "compare"
    
    # Recommend 키워드
    elif any(k in question for k in ["추천", "특약", "권장", "recommend", "어떤", "선택", "가장 좋은"]):
        intent = "recommend"
    
    # Web 검색 필요성 (더 정교한 판단)
    needs_web = False
    
    # 날짜 패턴 (2024, 2025, 3월, 12월 등)
    date_patterns = [
        r"\d{4}년", r"\d{4}-\d{2}", r"\d{4}/\d{2}",
        r"\d{1,2}월", r"내년", r"올해", r"다음 달"
    ]
    has_date = any(re.search(pattern, question) for pattern in date_patterns)
    
    # 지역 키워드 (확장)
    city_keywords = [
        # 미국 도시
        "la", "los angeles", "엘에이", "로스앤젤레스",
        "new york", "뉴욕", "nyc", "맨해튼",
        "san francisco", "샌프란시스코", "sf",
        "chicago", "시카고",
        "miami", "마이애미",
        "las vegas", "라스베가스",
        "seattle", "시애틀",
        "boston", "보스턴",
        "washington", "워싱턴", "dc",
        "philadelphia", "필라델피아",
        "houston", "휴스턴",
        "dallas", "댈러스",
        "denver", "덴버",
        "atlanta", "애틀랜타",
        "phoenix", "피닉스",
        "san diego", "샌디에고",
        "portland", "포틀랜드",
        
        # 일본 도시
        "도쿄", "tokyo", "東京",
        "오사카", "osaka", "大阪",
        "교토", "kyoto", "京都",
        "요코하마", "yokohama", "横浜",
        "나고야", "nagoya", "名古屋",
        "삿포로", "sapporo", "札幌",
        "고베", "kobe", "神戸",
        "후쿠오카", "fukuoka", "福岡",
        "히로시마", "hiroshima", "広島",
        "센다이", "sendai", "仙台",
        
        # 유럽 도시
        "파리", "paris",
        "런던", "london",
        "베를린", "berlin",
        "로마", "rome", "roma",
        "마드리드", "madrid",
        "바르셀로나", "barcelona",
        "암스테르담", "amsterdam",
        "빈", "vienna", "wien",
        "프라하", "prague", "praha",
        "부다페스트", "budapest",
        "취리히", "zurich",
        "스톡홀름", "stockholm",
        "코펜하겐", "copenhagen",
        "오슬로", "oslo",
        "헬싱키", "helsinki",
        "밀라노", "milan", "milano",
        "베니스", "venice", "venezia",
        "플로렌스", "florence", "firenze",
        "리스본", "lisbon", "lisboa",
        "아테네", "athens",
        "이스탄불", "istanbul",
        "모스크바", "moscow", "москва",
        "상트페테르부르크", "st petersburg",
        
        # 아시아 도시 (중국, 홍콩, 싱가포르)
        "베이징", "beijing", "北京",
        "상하이", "shanghai", "上海",
        "광저우", "guangzhou", "广州",
        "선전", "shenzhen", "深圳",
        "청두", "chengdu", "成都",
        "시안", "xian", "西安",
        "난징", "nanjing", "南京",
        "항저우", "hangzhou", "杭州",
        "홍콩", "hong kong", "香港",
        "싱가포르", "singapore",
        "마카오", "macau", "澳門",
        "타이페이", "taipei", "台北",
        "가오슝", "kaohsiung", "高雄",
        
        # 동남아시아
        "방콕", "bangkok", "กรุงเทพ",
        "치앙마이", "chiang mai",
        "푸켓", "phuket",
        "파타야", "pattaya",
        "호치민", "ho chi minh", "사이공", "saigon",
        "하노이", "hanoi",
        "다낭", "da nang",
        "호이안", "hoi an",
        "쿠알라룸푸르", "kuala lumpur", "kl",
        "페낭", "penang",
        "조호르바루", "johor bahru",
        "자카르타", "jakarta",
        "발리", "bali",
        "욕야카르타", "yogyakarta",
        "수라바야", "surabaya",
        "마닐라", "manila",
        "세부", "cebu",
        "보라카이", "boracay",
        "팔라완", "palawan",
        "프놈펜", "phnom penh",
        "시엠립", "siem reap",
        "비엔티안", "vientiane",
        "루앙프라방", "luang prabang",
        "양곤", "yangon",
        "만달레이", "mandalay",
        
        # 인도
        "뭄바이", "mumbai",
        "델리", "delhi", "new delhi",
        "방갈로르", "bangalore", "bengaluru",
        "첸나이", "chennai",
        "콜카타", "kolkata", "calcutta",
        "하이데라바드", "hyderabad",
        "푸네", "pune",
        "아그라", "agra",
        "자이푸르", "jaipur",
        "고아", "goa",
        
        # 중동
        "두바이", "dubai",
        "아부다비", "abu dhabi",
        "도하", "doha",
        "리야드", "riyadh",
        "쿠웨이트", "kuwait",
        "암만", "amman",
        "카이로", "cairo",
        "텔아비브", "tel aviv",
        "예루살렘", "jerusalem",
        "이스파한", "isfahan",
        "테헤란", "tehran",
        
        # 오세아니아
        "시드니", "sydney",
        "멜버른", "melbourne",
        "브리즈번", "brisbane",
        "퍼스", "perth",
        "애들레이드", "adelaide",
        "캔버라", "canberra",
        "골드코스트", "gold coast",
        "오클랜드", "auckland",
        "웰링턴", "wellington",
        "크라이스트처치", "christchurch",
        "퀸스타운", "queenstown",
        
        # 캐나다
        "토론토", "toronto",
        "밴쿠버", "vancouver",
        "몬트리올", "montreal",
        "캘거리", "calgary",
        "에드먼턴", "edmonton",
        "오타와", "ottawa",
        "퀘벡", "quebec",
        "위니펙", "winnipeg",
        
        # 남미
        "상파울루", "sao paulo", "são paulo",
        "리우데자네이루", "rio de janeiro",
        "부에노스아이레스", "buenos aires",
        "산티아고", "santiago",
        "리마", "lima",
        "보고타", "bogota",
        "카라카스", "caracas",
        "몬테비데오", "montevideo",
        
        # 아프리카
        "케이프타운", "cape town",
        "요하네스버그", "johannesburg",
        "카이로", "cairo",
        "카사블랑카", "casablanca",
        "마라케시", "marrakech",
        "나이로비", "nairobi",
        "아디스아바바", "addis ababa",
        "라고스", "lagos",
        "다카르", "dakar",
        
        # 러시아/동유럽
        "모스크바", "moscow",
        "상트페테르부르크", "saint petersburg",
        "키예프", "kiev", "kyiv",
        "바르샤바", "warsaw",
        "크라쿠프", "krakow",
        "부다페스트", "budapest",
        "프라하", "prague",
        "부카레스트", "bucharest",
        "소피아", "sofia",
        "베오그라드", "belgrade",
        
        # 국가/지역명
        "미국", "usa", "america", "united states",
        "일본", "japan", "日本",
        "중국", "china", "中国",
        "한국", "korea", "south korea", "대한민국",
        "태국", "thailand",
        "베트남", "vietnam",
        "싱가포르", "singapore",
        "말레이시아", "malaysia",
        "인도네시아", "indonesia",
        "필리핀", "philippines",
        "인도", "india",
        "호주", "australia",
        "뉴질랜드", "new zealand",
        "캐나다", "canada",
        "영국", "uk", "united kingdom", "britain",
        "프랑스", "france",
        "독일", "germany",
        "이탈리아", "italy",
        "스페인", "spain",
        "네덜란드", "netherlands",
        "스위스", "switzerland",
        "오스트리아", "austria",
        "벨기에", "belgium",
        "덴마크", "denmark",
        "스웨덴", "sweden",
        "노르웨이", "norway",
        "핀란드", "finland",
        "러시아", "russia",
        "터키", "turkey",
        "그리스", "greece",
        "포르투갈", "portugal",
        "아일랜드", "ireland",
        "폴란드", "poland",
        "체코", "czech republic",
        "헝가리", "hungary",
        "루마니아", "romania",
        "불가리아", "bulgaria",
        "크로아티아", "croatia",
        "세르비아", "serbia",
        "우크라이나", "ukraine",
        "벨라루스", "belarus",
        "리투아니아", "lithuania",
        "라트비아", "latvia",
        "에스토니아", "estonia",
        
        # 지역명
        "유럽", "europe",
        "아시아", "asia",
        "아메리카", "america",
        "북미", "north america",
        "남미", "south america",
        "오세아니아", "oceania",
        "아프리카", "africa",
        "중동", "middle east",
        "동남아시아", "southeast asia",
        "동아시아", "east asia",
        "서아시아", "west asia",
        "남아시아", "south asia",
        "동유럽", "eastern europe",
        "서유럽", "western europe",
        "북유럽", "northern europe",
        "남유럽", "southern europe",
        "중앙아시아", "central asia"
    ]
    has_city = any(x in ql for x in city_keywords)
    
    # 실시간 정보 키워드
    live_keywords = ["뉴스", "현지", "실시간", "최신", "현재", "지금", "요즘"]
    has_live = any(x in question for x in live_keywords)
    
    # Recommend intent이면서 날짜/지역/실시간 정보가 있는 경우
    if intent == "recommend" and (has_date or has_city or has_live):
        needs_web = True
    
    # 일반적인 실시간 정보 요청
    if has_live and any(x in question for x in ["상황", "정보", "뉴스", "현재"]):
        needs_web = True
    
    return {
        "intent": intent,
        "needs_web": needs_web,
        "reasoning": f"Enhanced fallback: {intent} (date:{has_date}, city:{has_city}, live:{has_live})"
    }

def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM 기반 질문 분석 및 분기 결정
    """
    q = state.get("question", "")
    
    # LLM을 사용한 분류
    classification = _llm_classify_intent(q)
    
    intent = classification["intent"]
    needs_web = classification["needs_web"]
    reasoning = classification.get("reasoning", "")
    
    # 실행 계획 생성
    plan = ["planner", "search", "rank_filter", "verify_refine", f"answer:{intent}"]
    if needs_web:
        plan.insert(1, "websearch")
    
    return {
        **state, 
        "intent": intent, 
        "needs_web": needs_web, 
        "plan": plan,
        "classification_reasoning": reasoning
    }