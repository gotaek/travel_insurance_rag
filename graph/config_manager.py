"""
시스템 전역 설정 관리자
YAML 설정 파일을 로드하고 시스템 전체에서 공유할 수 있는 설정을 제공합니다.
"""
import yaml
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """시스템 전역 설정을 관리하는 싱글톤 클래스"""
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[Dict[str, Any]] = None
    
    def __new__(cls) -> 'ConfigManager':
        """싱글톤 패턴 구현"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """설정 초기화 (한 번만 실행)"""
        if self._config is None:
            self._load_config()
    
    def _load_config(self) -> None:
        """YAML 설정 파일을 로드합니다."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'policies.yaml')
            with open(config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
            logger.info("설정 파일이 성공적으로 로드되었습니다.")
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {str(e)}")
            # 기본 설정으로 fallback
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정을 반환합니다."""
        return {
            "system": {
                "replan": {
                    "max_attempts": 2,
                    "max_structured_failures": 2,
                    "quality_threshold": 0.7
                },
                "performance": {
                    "enable_llm_classification": True,
                    "fallback_priority": True,
                    "complex_case_threshold": 2
                }
            }
        }
    
    def get_replan_config(self) -> Dict[str, Any]:
        """재검색 관련 설정을 반환합니다."""
        return self._config.get("system", {}).get("replan", {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """성능 최적화 관련 설정을 반환합니다."""
        return self._config.get("system", {}).get("performance", {})
    
    def get_max_replan_attempts(self) -> int:
        """최대 재검색 시도 횟수를 반환합니다."""
        return self.get_replan_config().get("max_attempts", 3)
    
    def get_max_structured_failures(self) -> int:
        """최대 연속 구조화 실패 허용 횟수를 반환합니다."""
        return self.get_replan_config().get("max_structured_failures", 2)
    
    def get_quality_threshold(self) -> float:
        """품질 평가 임계값을 반환합니다."""
        return self.get_replan_config().get("quality_threshold", 0.7)
    
    
    def is_llm_classification_enabled(self) -> bool:
        """LLM 분류 사용 여부를 반환합니다."""
        return self.get_performance_config().get("enable_llm_classification", True)
    
    def is_fallback_priority(self) -> bool:
        """fallback 분류 우선 사용 여부를 반환합니다."""
        return self.get_performance_config().get("fallback_priority", True)
    
    def get_complex_case_threshold(self) -> int:
        """복잡한 케이스 판단 임계값을 반환합니다."""
        return self.get_performance_config().get("complex_case_threshold", 2)
    
    def get_all_config(self) -> Dict[str, Any]:
        """전체 설정을 반환합니다."""
        return self._config.copy() if self._config else {}
    
    def reload_config(self) -> None:
        """설정을 다시 로드합니다."""
        self._config = None
        self._load_config()
        logger.info("설정이 다시 로드되었습니다.")

# 전역 설정 관리자 인스턴스
config_manager = ConfigManager()

def get_system_config() -> ConfigManager:
    """시스템 설정 관리자를 반환합니다."""
    return config_manager
