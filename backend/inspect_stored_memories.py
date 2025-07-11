#!/usr/bin/env python3
"""Inspect what memories are actually stored in the database."""

import asyncio
import logging
import sqlite3
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants for content preview lengths
MEMORY_CONTENT_PREVIEW_LENGTH = 200
CORE_VALUE_PREVIEW_LENGTH = 100
RECENT_MEMORY_PREVIEW_LENGTH = 150
SEARCH_RESULT_PREVIEW_LENGTH = 100


def inspect_memory_database():
    """Directly inspect the SQLite database to see stored memories."""
    try:
        # Find the database file
        db_path = (
            "/home/jman/tmp/models-to-test/Magistral-Small-2506-1.2/backend/personal_ai_memories.db"
        )

        if not Path(db_path).exists():
            logger.error(f"❌ Database not found at {db_path}")
            return

        logger.info(f"📁 Inspecting database: {db_path}")

        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        logger.info(f"📊 Available tables: {[table[0] for table in tables]}")

        # Inspect memories table
        cursor.execute("SELECT COUNT(*) FROM memories")
        count = cursor.fetchone()[0]
        logger.info(f"🧠 Total memories in database: {count}")

        if count > 0:
            # Get recent memories with soul/animal keywords
            logger.info(f"\n{'=' * 60}")
            logger.info("SEARCHING FOR ANIMAL/SOUL RELATED MEMORIES")
            logger.info(f"{'=' * 60}")

            cursor.execute("""
                SELECT id, content, conversation_id, importance, created_at
                FROM memories
                WHERE content LIKE '%animal%' OR content LIKE '%soul%' OR content LIKE '%dog%' OR content LIKE '%cat%'
                ORDER BY created_at DESC
                LIMIT 10
            """)

            animal_memories = cursor.fetchall()
            logger.info(f"🐾 Found {len(animal_memories)} animal/soul related memories:")

            for memory in animal_memories:
                memory_id, content, conv_id, importance, created_at = memory
                content_preview = content[:MEMORY_CONTENT_PREVIEW_LENGTH] + "..." if len(content) > MEMORY_CONTENT_PREVIEW_LENGTH else content
                logger.info(f"  ID: {memory_id}")
                logger.info(f"  Conversation: {conv_id}")
                logger.info(f"  Importance: {importance}")
                logger.info(f"  Created: {created_at}")
                logger.info(f"  Content: {content_preview}")
                logger.info("  ---")

        # Check core memories
        cursor.execute("SELECT COUNT(*) FROM core_memories")
        core_count = cursor.fetchone()[0]
        logger.info(f"\n🎯 Core memories count: {core_count}")

        if core_count > 0:
            cursor.execute("SELECT key, value, category FROM core_memories LIMIT 10")
            core_memories = cursor.fetchall()
            logger.info("🎯 Core memories:")
            for key, value, category in core_memories:
                value_preview = value[:CORE_VALUE_PREVIEW_LENGTH] + "..." if len(str(value)) > CORE_VALUE_PREVIEW_LENGTH else str(value)
                logger.info(f"  {key} ({category}): {value_preview}")

        # Get all recent memories regardless of content
        logger.info(f"\n{'=' * 60}")
        logger.info("RECENT MEMORIES (LAST 10)")
        logger.info(f"{'=' * 60}")

        cursor.execute("""
            SELECT id, content, conversation_id, importance, created_at
            FROM memories
            ORDER BY created_at DESC
            LIMIT 10
        """)

        recent_memories = cursor.fetchall()
        for memory in recent_memories:
            memory_id, content, conv_id, importance, created_at = memory
            content_preview = content[:RECENT_MEMORY_PREVIEW_LENGTH] + "..." if len(content) > RECENT_MEMORY_PREVIEW_LENGTH else content
            logger.info(f"  [{memory_id}] {conv_id} ({importance:.2f}): {content_preview}")

        conn.close()

    except Exception as e:
        logger.error(f"Error inspecting database: {e}", exc_info=True)


async def test_memory_search():
    """Test memory search functionality."""
    try:
        # Initialize app state for memory system
        from memory_provider import MemoryConfig, get_memory_system
        from modules.globals import app_state

        if not app_state.personal_memory:
            logger.info("🔧 Initializing memory system...")
            memory_config = MemoryConfig()
            app_state.personal_memory = get_memory_system(memory_config)
            logger.info("✅ Memory system initialized")

        # Test searches for animal/soul content
        search_queries = [
            "animals souls",
            "animals have souls",
            "dog emotions",
            "philosophical",
            "consciousness",
            "believe",
        ]

        for query in search_queries:
            logger.info(f"\n🔍 Searching for: '{query}'")
            try:
                memories = await app_state.personal_memory.get_relevant_memories(
                    query=query, limit=5
                )
                logger.info(f"   Found {len(memories) if memories else 0} results")

                if memories:
                    for i, memory in enumerate(memories[:3]):
                        content_preview = (
                            memory.content[:SEARCH_RESULT_PREVIEW_LENGTH] + "..."
                            if len(memory.content) > SEARCH_RESULT_PREVIEW_LENGTH
                            else memory.content
                        )
                        logger.info(f"   {i + 1}. {content_preview}")

            except Exception:
                logger.exception("   ❌ Search failed")

    except Exception as e:
        logger.error(f"Error in memory search test: {e}", exc_info=True)


if __name__ == "__main__":
    logger.info("🔍 MEMORY INSPECTION STARTING")

    # First inspect the raw database
    inspect_memory_database()

    # Then test the memory search functionality
    logger.info(f"\n{'=' * 80}")
    logger.info("TESTING MEMORY SEARCH FUNCTIONALITY")
    logger.info(f"{'=' * 80}")

    asyncio.run(test_memory_search())
