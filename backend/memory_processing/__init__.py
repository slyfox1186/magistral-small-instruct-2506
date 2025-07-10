"""
Advanced Memory Processing System for AI Assistant

This package provides a world-class, production-ready memory processing system
that transforms conversations into structured, meaningful memories.

Key Components:
- ContentAnalyzer: LLM-powered content analysis and categorization
- MemoryExtractor: Structured memory extraction with confidence scoring
- DeduplicationEngine: Semantic similarity detection and intelligent merging
- AdvancedMemoryProcessor: Main orchestrator with 5-stage pipeline

Features:
- Multi-category memory support (personal facts, preferences, experiences, etc.)
- Sophisticated importance scoring with context awareness
- Intelligent deduplication beyond exact matching
- Production-ready error handling with graceful degradation
- Comprehensive monitoring and metrics collection
- Multiple deployment profiles (default, production, development, fast)

Usage:
    from memory_processing import AdvancedMemoryProcessor
    from memory_processing.config import get_config
    
    config = get_config('production')
    processor = AdvancedMemoryProcessor(memory_system, config)
    result = await processor.process_conversation(user_prompt, response, session_id)
"""

from .advanced_memory_processor import AdvancedMemoryProcessor
from .config import get_config, MemoryProcessingConfig
from .content_analyzer import ContentAnalyzer
from .memory_extractor import MemoryExtractor
from .deduplication_engine import DeduplicationEngine
from .utils import extract_entities, calculate_text_similarity, format_memory_content

__version__ = "1.0.0"
__all__ = [
    "AdvancedMemoryProcessor",
    "get_config",
    "MemoryProcessingConfig",
    "ContentAnalyzer",
    "MemoryExtractor",
    "DeduplicationEngine",
    "extract_entities",
    "calculate_text_similarity",
    "format_memory_content"
]