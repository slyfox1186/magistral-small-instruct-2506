"""AI Memory System.

================

A sophisticated memory system for AI assistants using Redis Stack and GPU-accelerated embeddings.
Implements a tiered memory architecture inspired by human cognitive processes.
"""

from .schemas.memory_schema import LongTermMemory, MemoryBase, MemoryCircle, ShortTermMemory
from .services.consolidation_worker import ConsolidationWorker
from .services.embedding_service import EmbeddingService
from .services.memory_api import MemoryAPI
from .services.retrieval_engine import RetrievalEngine

__all__ = [
    "ConsolidationWorker",
    "EmbeddingService",
    "LongTermMemory",
    "MemoryAPI",
    "MemoryBase",
    "MemoryCircle",
    "RetrievalEngine",
    "ShortTermMemory",
]
