import json
from typing import Dict, Any

def parse_llm_output(text: str) -> Dict[str, Any]:
    """
    LLM 출력에서 JSON 부분만 파싱.
    - JSON 앞뒤에 텍스트/마크다운 섞여 있어도 안전하게 추출
    - 실패 시 기본 fallback 반환
    """
    if not text:
        return _fallback()
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            snippet = text[start:end+1]
            return json.loads(snippet)
    except Exception:
        pass
    return _fallback()

def _fallback() -> Dict[str, Any]:
    return {
        "conclusion": "(LLM 출력 파싱 실패)",
        "evidence": [],
        "caveats": [],
        "quotes": []
    }