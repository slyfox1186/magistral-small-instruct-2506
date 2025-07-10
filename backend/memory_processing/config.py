"""Configuration Management for Memory Processing System

Provides multiple deployment profiles optimized for different use cases:
- default: Balanced for general use
- production: Optimized for production environments
- development: Detailed logging and debugging
- fast: High-speed processing for performance-critical scenarios
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class MemoryProcessingConfig:
    """Configuration for memory processing system"""

    # Timeout settings (seconds)
    max_processing_time: int = 30
    llm_timeout: int = 20
    analysis_timeout: int = 10

    # Confidence thresholds
    min_confidence_threshold: float = 0.3
    high_confidence_threshold: float = 0.7

    # Importance scoring
    importance_multiplier: float = 1.0
    personal_info_boost: float = 0.3
    emotional_boost: float = 0.2

    # Deduplication settings
    similarity_threshold: float = 0.85
    merge_threshold: float = 0.9

    # Processing limits
    max_memories_per_conversation: int = 5
    max_content_length: int = 2000

    # Logging and monitoring
    log_level: str = "INFO"
    enable_metrics: bool = True
    enable_detailed_logging: bool = False

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    # Feature flags
    enable_deduplication: bool = True
    enable_importance_scoring: bool = True
    enable_content_analysis: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> 'MemoryProcessingConfig':
        """Create config from dictionary"""
        return cls(**config_dict)

# Predefined configuration profiles
CONFIGS = {
    'default': MemoryProcessingConfig(
        max_processing_time=30,
        llm_timeout=20,
        min_confidence_threshold=0.3,
        high_confidence_threshold=0.7,
        similarity_threshold=0.85,
        log_level="INFO",
        enable_detailed_logging=False
    ),

    'production': MemoryProcessingConfig(
        max_processing_time=20,
        llm_timeout=15,
        min_confidence_threshold=0.4,
        high_confidence_threshold=0.8,
        similarity_threshold=0.9,
        log_level="WARNING",
        enable_detailed_logging=False,
        max_retries=2,
        importance_multiplier=1.2
    ),

    'development': MemoryProcessingConfig(
        max_processing_time=60,
        llm_timeout=30,
        min_confidence_threshold=0.2,
        high_confidence_threshold=0.6,
        similarity_threshold=0.8,
        log_level="DEBUG",
        enable_detailed_logging=True,
        max_retries=1,
        enable_metrics=True
    ),

    'fast': MemoryProcessingConfig(
        max_processing_time=10,
        llm_timeout=8,
        min_confidence_threshold=0.5,
        high_confidence_threshold=0.8,
        similarity_threshold=0.9,
        log_level="ERROR",
        enable_detailed_logging=False,
        max_retries=1,
        max_memories_per_conversation=3
    )
}

def get_config(profile: str = 'default') -> MemoryProcessingConfig:
    """Get configuration for specified profile
    
    Args:
        profile: Configuration profile name ('default', 'production', 'development', 'fast')
        
    Returns:
        MemoryProcessingConfig object
        
    Raises:
        ValueError: If profile is not recognized
    """
    if profile not in CONFIGS:
        available_profiles = ', '.join(CONFIGS.keys())
        raise ValueError(f"Unknown profile '{profile}'. Available profiles: {available_profiles}")

    return CONFIGS[profile]

def list_available_profiles() -> dict[str, str]:
    """List all available configuration profiles with descriptions
    
    Returns:
        Dictionary mapping profile names to descriptions
    """
    return {
        'default': 'Balanced configuration for general use',
        'production': 'Optimized for production environments with stricter thresholds',
        'development': 'Detailed logging and debugging enabled',
        'fast': 'High-speed processing with minimal overhead'
    }

def create_custom_config(**kwargs) -> MemoryProcessingConfig:
    """Create a custom configuration by overriding default values
    
    Args:
        **kwargs: Configuration parameters to override
        
    Returns:
        MemoryProcessingConfig with custom values
    """
    base_config = get_config('default')
    config_dict = base_config.to_dict()
    config_dict.update(kwargs)
    return MemoryProcessingConfig.from_dict(config_dict)

# Configuration validation
def validate_config(config: MemoryProcessingConfig) -> bool:
    """Validate configuration parameters
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    if config.max_processing_time <= 0:
        raise ValueError("max_processing_time must be positive")

    if config.llm_timeout <= 0:
        raise ValueError("llm_timeout must be positive")

    if not (0 <= config.min_confidence_threshold <= 1):
        raise ValueError("min_confidence_threshold must be between 0 and 1")

    if not (0 <= config.high_confidence_threshold <= 1):
        raise ValueError("high_confidence_threshold must be between 0 and 1")

    if config.min_confidence_threshold > config.high_confidence_threshold:
        raise ValueError("min_confidence_threshold cannot be greater than high_confidence_threshold")

    if not (0 <= config.similarity_threshold <= 1):
        raise ValueError("similarity_threshold must be between 0 and 1")

    if config.max_memories_per_conversation <= 0:
        raise ValueError("max_memories_per_conversation must be positive")

    if config.max_content_length <= 0:
        raise ValueError("max_content_length must be positive")

    return True
