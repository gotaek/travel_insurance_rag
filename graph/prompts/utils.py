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
    """최적화된 fallback 응답 생성 - 프롬프트 엔지니어링 원칙 적용"""
    from graph.models import EvidenceInfo, CaveatInfo
    
    # 노드 타입별 맞춤형 fallback 메시지
    node_messages = {
        "QA": "질문을 확인했습니다. 관련 정보를 찾기 위해 추가 검색이 필요할 수 있습니다.",
        "Recommend": "추천을 위해 여행 정보와 최신 데이터를 추가로 확인해야 합니다.",
        "Summarize": "요약을 위해 해당 보험사의 약관 문서를 추가로 확인해야 합니다.",
        "Compare": "비교를 위해 관련 보험사들의 최신 약관 정보를 확인해야 합니다."
    }
    
    conclusion = node_messages.get(node_type, "요청을 확인했습니다. 추가 정보 확인이 필요합니다.")
    
    return {
        "conclusion": conclusion,
        "evidence": [EvidenceInfo(text=f"{node_type} 시스템 응답 - 추가 검색 필요", source="시스템")],
        "caveats": [
            CaveatInfo(text="관련 문서를 찾을 수 없어 정확한 답변을 제공하기 어렵습니다.", source="시스템"),
            CaveatInfo(text="보험 약관을 직접 확인하시기 바랍니다.", source="시스템")
        ],
        "web_quotes": [],
        "web_info": {},
        "recommendations": [] if node_type == "Recommend" else None,
        "comparison_table": None if node_type != "Compare" else {
            "headers": ["항목", "상태"],
            "rows": [["정보 확인", "추가 검색 필요"]]
        }
    }