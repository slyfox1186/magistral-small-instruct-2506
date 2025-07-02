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
    # Get backend choice from environment or config
    memory_backend = os.getenv("MEMORY_BACKEND", "sqlite").lower()

    if config and hasattr(config, "MEMORY_BACKEND"):
        memory_backend = config.MEMORY_BACKEND.lower()

    logger.info(f"🧠 Initializing memory system with backend: {memory_backend}")

    if memory_backend == "sqlite":
        # Check if async version is requested
        use_async = os.getenv("USE_ASYNC_MEMORY", "false").lower() == "true"

        if use_async:
            # Use the new async personal memory system
            from async_personal_memory_system import AsyncPersonalMemorySystem

            db_path = "personal_ai_memories.db"
            if config and hasattr(config, "SQLITE_DB_PATH"):
                db_path = config.SQLITE_DB_PATH

            # Get embedding model from ResourceManager for generating embeddings
            embedding_model = None
            try:
                from resource_manager import ResourceManager

                resource_manager = ResourceManager()
                embedding_model = resource_manager
                logger.info("✅ Loaded embedding model from ResourceManager")
            except Exception as e:
                logger.warning(f"⚠️ Could not load embedding model: {e}")

            memory_system = AsyncPersonalMemorySystem(db_path, embedding_model=embedding_model)
            logger.info(f"✅ Initialized AsyncPersonalMemorySystem with database: {db_path}")
        else:
            # Use the original synchronous personal memory system
            from personal_memory_system import PersonalMemorySystem

            db_path = "personal_ai_memories.db"
            if config and hasattr(config, "SQLITE_DB_PATH"):
                db_path = config.SQLITE_DB_PATH

            # Get embedding model from ResourceManager for generating embeddings
            embedding_model = None
            try:
                from resource_manager import ResourceManager

                resource_manager = ResourceManager()
                embedding_model = resource_manager
                logger.info("✅ Loaded embedding model from ResourceManager")
            except Exception as e:
                logger.warning(f"⚠️ Could not load embedding model: {e}")

            memory_system = PersonalMemorySystem(db_path, embedding_model=embedding_model)
            logger.info(f"✅ Initialized PersonalMemorySystem with database: {db_path}")

        # Check if we need Redis compatibility mode
        use_compat = os.getenv("USE_REDIS_COMPAT", "false").lower() == "true"
        if config and hasattr(config, "USE_REDIS_COMPAT"):
            use_compat = config.USE_REDIS_COMPAT

        if use_compat:
            from redis_compat_memory import RedisCompatMemorySystem

            memory_system = RedisCompatMemorySystem(memory_system)
            logger.info("🔄 Wrapped PersonalMemorySystem with Redis compatibility layer")

        return memory_system

    elif memory_backend == "redis":
        # Use the old Redis system (for rollback capability)
        logger.warning("⚠️ Using legacy Redis memory system. Consider migrating to SQLite.")

        try:
            # Import the old system - you'll need to keep this around temporarily
            from memory.redis_utils import RedisMemoryStore

            return RedisMemoryStore()
        except ImportError:
            logger.error("❌ Redis backend requested but redis_utils not found!")
            logger.info("Falling back to SQLite with compatibility mode...")

            # Fallback to SQLite with compat mode
            from personal_memory_system import PersonalMemorySystem
            from redis_compat_memory import RedisCompatMemorySystem

            personal_memory = PersonalMemorySystem("personal_ai_memories.db")
            return RedisCompatMemorySystem(personal_memory)

    else:
        raise ValueError(f"Unknown memory backend: {memory_backend}")


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
        # Read from environment with defaults
        self.MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "sqlite")
        self.USE_REDIS_COMPAT = os.getenv("USE_REDIS_COMPAT", "false").lower() == "true"
        self.SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "personal_ai_memories.db")

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
SQLITE_DB_PATH=personal_ai_memories.db

# Auto-migrate Redis data on startup (one-time operation)
MIGRATE_ON_STARTUP=false

# Redis connection (only needed if using Redis backend or migration)
REDIS_URL=redis://localhost:6379
"""
