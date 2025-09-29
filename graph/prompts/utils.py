# prompts/utils.py — 프롬프트 캐싱 유틸리티

from functools import lru_cache
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# 프롬프트 캐시
_prompt_cache: Dict[str, str] = {}


@lru_cache(maxsize=10)
def get_cached_prompt(prompt_name: str) -> str:
    """캐시된 프롬프트 반환 (빠른 접근용)"""
    if prompt_name not in _prompt_cache:
        _prompt_cache[prompt_name] = _load_prompt(prompt_name)
    return _prompt_cache[prompt_name]


def _load_prompt(prompt_name: str) -> str:
    """프롬프트 파일 로드"""
    try:
        # 현재 작업 디렉토리 기준으로 경로 설정
        current_dir = Path(__file__).parent.parent
        prompt_path = current_dir / "prompts" / f"{prompt_name}.md"
        return prompt_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"프롬프트 로드 실패: {prompt_name} - {e}")
        return f"프롬프트를 로드할 수 없습니다: {prompt_name}"


def clear_prompt_cache():
    """프롬프트 캐시 초기화"""
    global _prompt_cache
    _prompt_cache.clear()
    get_cached_prompt.cache_clear()


def get_simple_fallback_response(question: str, node_type: str = "QA") -> Dict[str, Any]:
    """간단한 fallback 응답 생성"""
    from graph.models import EvidenceInfo, CaveatInfo
    
    return {
        "conclusion": f"질문을 확인했습니다: '{question[:100]}{'...' if len(question) > 100 else ''}'",
        "evidence": [EvidenceInfo(text=f"{node_type} 시스템 응답", source="시스템")],
        "caveats": [CaveatInfo(text="추가 확인이 필요할 수 있습니다.", source="시스템")],
        "quotes": [],
        "web_quotes": [],
        "web_info": {},
        "recommendations": [] if node_type == "Recommend" else None,
        "comparison_table": None if node_type != "Compare" else {
            "headers": ["항목", "결과"],
            "rows": [["시스템 응답", "추가 확인 필요"]]
        }
    }