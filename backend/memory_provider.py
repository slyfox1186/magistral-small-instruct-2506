"""Memory provider factory for switching between Redis and SQLite backends.

Uses feature flags for safe migration and rollback.
"""

import logging
import os
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from personal_memory_system import PersonalMemorySystem
    from redis_compat_memory import RedisCompatMemorySystem

logger = logging.getLogger(__name__)


def get_memory_system(config=None) -> Union["PersonalMemorySystem", "RedisCompatMemorySystem"]:
    """Factory function to get the appropriate memory system based on configuration.

    This allows safe switching between the old Redis system and new SQLite system
    via environment variable or config setting.
    """
    memory_backend = _get_backend_choice(config)
    logger.info(f"🧠 Initializing memory system with backend: {memory_backend}")

    if memory_backend == "sqlite":
        return _create_sqlite_memory_system(config)
    elif memory_backend == "redis":
        return _create_redis_memory_system()
    else:
        raise ValueError(f"Unknown memory backend: {memory_backend}")


def _get_backend_choice(config) -> str:
    """Get the memory backend choice from config or environment."""
    memory_backend = os.getenv("MEMORY_BACKEND", "sqlite").lower()

    if config and hasattr(config, "MEMORY_BACKEND"):
        memory_backend = config.MEMORY_BACKEND.lower()

    return memory_backend


def _get_embedding_model():
    """Get embedding model from ResourceManager."""
    try:
        from resource_manager import ResourceManager

        resource_manager = ResourceManager()
        logger.info("✅ Loaded embedding model from ResourceManager")
    except Exception as e:
        logger.warning(f"⚠️ Could not load embedding model: {e}")
        return None
    else:
        return resource_manager


def _get_db_path(config) -> str:
    """Get database path from config or use default."""
    db_path = "memory/personal_ai_memories.db"
    if config and hasattr(config, "SQLITE_DB_PATH"):
        db_path = config.SQLITE_DB_PATH
    return db_path


def _create_async_memory_system(config):
    """Create async personal memory system."""
    from async_personal_memory_system import AsyncPersonalMemorySystem

    db_path = _get_db_path(config)
    embedding_model = _get_embedding_model()

    memory_system = AsyncPersonalMemorySystem(db_path, embedding_model=embedding_model)
    logger.info(f"✅ Initialized AsyncPersonalMemorySystem with database: {db_path}")
    return memory_system


def _create_sync_memory_system(config):
    """Create synchronous personal memory system."""
    from personal_memory_system import PersonalMemorySystem

    db_path = _get_db_path(config)
    embedding_model = _get_embedding_model()

    memory_system = PersonalMemorySystem(db_path, embedding_model=embedding_model)
    logger.info(f"✅ Initialized PersonalMemorySystem with database: {db_path}")
    return memory_system


def _should_use_redis_compat(config) -> bool:
    """Check if Redis compatibility mode should be used."""
    use_compat = os.getenv("USE_REDIS_COMPAT", "false").lower() == "true"
    if config and hasattr(config, "USE_REDIS_COMPAT"):
        use_compat = config.USE_REDIS_COMPAT
    return use_compat


def _wrap_with_redis_compat(memory_system):
    """Wrap memory system with Redis compatibility layer."""
    from redis_compat_memory import RedisCompatMemorySystem

    wrapped_system = RedisCompatMemorySystem(memory_system)
    logger.info("🔄 Wrapped PersonalMemorySystem with Redis compatibility layer")
    return wrapped_system


def _create_sqlite_memory_system(config):
    """Create SQLite-based memory system."""
    use_async = os.getenv("USE_ASYNC_MEMORY", "false").lower() == "true"

    if use_async:
        memory_system = _create_async_memory_system(config)
    else:
        memory_system = _create_sync_memory_system(config)

    if _should_use_redis_compat(config):
        memory_system = _wrap_with_redis_compat(memory_system)

    return memory_system


def _create_redis_memory_system():
    """Create Redis-based memory system with fallback."""
    logger.warning("⚠️ Using legacy Redis memory system. Consider migrating to SQLite.")

    try:
        from memory.redis_utils import RedisMemoryStore

        return RedisMemoryStore()
    except ImportError:
        logger.exception("❌ Redis backend requested but redis_utils not found!")
        logger.info("Falling back to SQLite with compatibility mode...")

        # Fallback to SQLite with compat mode
        from personal_memory_system import PersonalMemorySystem
        from redis_compat_memory import RedisCompatMemorySystem

        personal_memory = PersonalMemorySystem("memory/personal_ai_memories.db")
        return RedisCompatMemorySystem(personal_memory)


def get_memory_stats() -> dict:
    """Get statistics from the current memory system."""
    memory_system = get_memory_system()

    # Check if it's the personal memory system
    if hasattr(memory_system, "get_stats"):
        return memory_system.get_stats()

    # Check if it's wrapped in compatibility layer
    if hasattr(memory_system, "memory") and hasattr(memory_system.memory, "get_stats"):
        return memory_system.memory.get_stats()

    # Fallback for Redis or unknown systems
    return {"backend": "unknown", "stats_available": False}


# Configuration class for settings
class MemoryConfig:
    """Configuration for memory system."""

    def __init__(self):
        """Initialize memory configuration from environment variables."""
        # Read from environment with defaults
        self.MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "sqlite")
        self.USE_REDIS_COMPAT = os.getenv("USE_REDIS_COMPAT", "false").lower() == "true"
        self.SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "memory/personal_ai_memories.db")

        # Migration settings
        self.MIGRATE_ON_STARTUP = os.getenv("MIGRATE_ON_STARTUP", "false").lower() == "true"
        self.REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


# Example .env file content:
"""
# Memory System Configuration
# Options: "redis" (legacy) or "sqlite" (new, recommended)
MEMORY_BACKEND=sqlite

# Use Redis compatibility layer for gradual migration
USE_REDIS_COMPAT=false

# SQLite database path
SQLITE_DB_PATH=memory/personal_ai_memories.db

# Auto-migrate Redis data on startup (one-time operation)
MIGRATE_ON_STARTUP=false

# Redis connection (only needed if using Redis backend or migration)
REDIS_URL=redis://localhost:6379
"""
