"""Optimized async memory system for personal AI assistant.

Uses aiosqlite for non-blocking database operations and connection pooling.
"""

import asyncio
import hashlib
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

# High importance threshold for verbose logging
HIGH_IMPORTANCE_THRESHOLD = 0.7


@dataclass
class Memory:
    """A single memory entry."""

    id: str
    content: str
    summary: str | None
    embedding: list[float] | None
    conversation_id: str
    timestamp: datetime
    importance: float = 0.5
    access_count: int = 0
    last_accessed: datetime | None = None
    metadata: dict[str, Any] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["last_accessed"] = self.last_accessed.isoformat() if self.last_accessed else None
        data["embedding"] = json.dumps(self.embedding) if self.embedding else None
        data["metadata"] = json.dumps(self.metadata) if self.metadata else None
        return data


class AsyncPersonalMemorySystem:
    """Async hierarchical memory system optimized for personal AI use.

    Features:
    - Async SQLite operations for non-blocking I/O
    - Connection pooling for improved performance
    - Automatic summarization of old conversations
    - Smart retrieval based on relevance and recency
    - Memory importance scoring
    """

    def __init__(self, db_path: str = "memories.db", embedding_model=None, pool_size: int = 5):
        """Initialize the async personal memory system."""
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.pool_size = pool_size
        self._pool: list[aiosqlite.Connection] = []
        self._pool_semaphore = asyncio.Semaphore(pool_size)
        self._initialized = False

        # Memory windows
        self.short_term_hours = 24  # Keep full detail for 24 hours
        self.medium_term_days = 7  # Keep summaries for a week
        self.long_term_days = 30  # Keep important memories for a month

    async def initialize(self):
        """Initialize the database and connection pool."""
        if self._initialized:
            return

        await self._init_database()
        self._initialized = True
        logger.info(f"Initialized AsyncPersonalMemorySystem with pool size {self.pool_size}")

    async def _init_database(self):
        """Initialize SQLite database with optimized schema."""
        async with self._get_connection() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    summary TEXT,
                    embedding TEXT,
                    conversation_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    importance REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    last_accessed DATETIME,
                    metadata TEXT,

                    -- Indexes for fast retrieval
                    CHECK (importance >= 0 AND importance <= 1)
                )
            """
            )

            # Create indexes for common queries
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversation ON memories(conversation_id)"
            )
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)")

            # Conversation summaries table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    conversation_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME NOT NULL,
                    message_count INTEGER,
                    importance REAL DEFAULT 0.5,
                    topics TEXT  -- JSON array of topics
                )
            """
            )

            # Handle core memories table creation and migration
            await self._init_core_memories_table(conn)

            await conn.commit()

    async def _init_core_memories_table(self, conn):
        """Initialize or migrate core_memories table with proper schema (async version)."""
        try:
            # Check if table exists
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='core_memories'"
            )
            row = await cursor.fetchone()
            table_exists = row is not None
            
            if table_exists:
                # Check current schema
                cursor = await conn.execute("PRAGMA table_info(core_memories)")
                rows = await cursor.fetchall()
                columns = [row[1] for row in rows]
                
                # If conversation_id column doesn't exist, we need to migrate
                if "conversation_id" not in columns:
                    logger.info("🔄 Migrating core_memories table to add conversation_id column...")
                    
                    # Get existing data
                    try:
                        cursor = await conn.execute("SELECT key, value, created_at, updated_at, category FROM core_memories")
                        existing_data = await cursor.fetchall()
                        logger.info(f"Found {len(existing_data)} existing core memories to migrate")
                    except Exception:
                        # If we can't read existing data, assume table is corrupted
                        existing_data = []
                        logger.warning("Could not read existing core memories, proceeding with empty migration")
                    
                    # Drop the old table
                    await conn.execute("DROP TABLE core_memories")
                    logger.info("Dropped old core_memories table")
                    
                    # Create new table with updated schema
                    await conn.execute(
                        """
                        CREATE TABLE core_memories (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            key TEXT NOT NULL,
                            value TEXT NOT NULL,
                            conversation_id TEXT NOT NULL,
                            created_at DATETIME NOT NULL,
                            updated_at DATETIME NOT NULL,
                            category TEXT DEFAULT 'general',
                            UNIQUE(key, conversation_id)
                        )
                        """
                    )
                    logger.info("Created new core_memories table with conversation_id support")
                    
                    # Migrate existing data to "global" conversation
                    if existing_data:
                        for row in existing_data:
                            key, value, created_at, updated_at, category = row
                            await conn.execute(
                                """
                                INSERT INTO core_memories (key, value, conversation_id, created_at, updated_at, category)
                                VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (key, value, "global", created_at, updated_at, category or "general")
                            )
                        logger.info(f"✅ Migrated {len(existing_data)} core memories to 'global' conversation")
                else:
                    logger.debug("✅ Core memories table already has correct schema")
            else:
                # Create new table
                logger.info("Creating new core_memories table...")
                await conn.execute(
                    """
                    CREATE TABLE core_memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        conversation_id TEXT NOT NULL,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL,
                        category TEXT DEFAULT 'general',
                        UNIQUE(key, conversation_id)
                    )
                    """
                )
                logger.info("✅ Created core_memories table with conversation_id support")
            
            # Create index for core memories conversation queries
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_core_memories_conversation ON core_memories(conversation_id)")
            logger.debug("Created index for core_memories conversation queries")
                
        except Exception as e:
            logger.error(f"❌ Error during core_memories table initialization: {e}")
            # If everything fails, create a basic table
            try:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS core_memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        conversation_id TEXT NOT NULL DEFAULT 'global',
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL,
                        category TEXT DEFAULT 'general',
                        UNIQUE(key, conversation_id)
                    )
                    """
                )
                logger.warning("Created fallback core_memories table")
            except Exception as fallback_error:
                logger.error(f"❌ Even fallback table creation failed: {fallback_error}")
                raise

    @asynccontextmanager
    async def _get_connection(self):
        """Get a connection from the pool or create a new one."""
        async with self._pool_semaphore:
            conn = None
            try:
                # Try to get connection from pool
                if self._pool:
                    conn = self._pool.pop()
                else:
                    # Create new connection
                    conn = await aiosqlite.connect(self.db_path, timeout=10.0)
                    conn.row_factory = aiosqlite.Row
                    # Enable WAL mode for better concurrency
                    await conn.execute("PRAGMA journal_mode=WAL")
                    # Faster writes
                    await conn.execute("PRAGMA synchronous=NORMAL")
                    # Increase cache size for better performance (10MB)
                    await conn.execute("PRAGMA cache_size=-10240")
                    # Set busy timeout to avoid lock errors
                    await conn.execute("PRAGMA busy_timeout=5000")

                yield conn

                # Return connection to pool if pool not full
                if len(self._pool) < self.pool_size:
                    self._pool.append(conn)
                    conn = None  # Don't close it

            finally:
                # Close connection if not returned to pool
                if conn:
                    await conn.close()

    def _generate_id(self, content: str, timestamp: datetime) -> str:
        """Generate a unique ID for a memory."""
        data = f"{content}{timestamp.isoformat()}".encode()
        return hashlib.sha256(data).hexdigest()[:16]

    async def add_memory(
        self,
        content: str,
        conversation_id: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Add a new memory to the system."""
        if not self._initialized:
            await self.initialize()

        timestamp = datetime.now(UTC)
        memory_id = self._generate_id(content, timestamp)

        # VERBOSE LOGGING: Log all memory additions
        logger.info("🧠 MEMORY ADD: Adding new memory to database")
        logger.info(f"🧠 MEMORY ADD:   Memory ID: {memory_id}")
        logger.info(f"🧠 MEMORY ADD:   Conversation ID: {conversation_id}")
        logger.info(f"🧠 MEMORY ADD:   Importance: {importance}")
        logger.info(f"🧠 MEMORY ADD:   Timestamp: {timestamp}")
        logger.info(f"🧠 MEMORY ADD:   Content length: {len(content)} characters")
        logger.info(f"🧠 MEMORY ADD:   Content preview (first 300 chars): '{content[:300]}...'")
        if metadata:
            logger.info(f"🧠 MEMORY ADD:   Metadata: {metadata}")

        # Log entry for high importance memories only
        if importance >= HIGH_IMPORTANCE_THRESHOLD:
            logger.warning(
                f"[MEMORY_HIGH_IMPORTANCE] Adding high importance memory:\n"
                f"  Memory ID: {memory_id}\n"
                f"  Importance Score: {importance}\n"
                f"  Content: {content[:100]}..."
            )

        # Generate embedding if model is available
        embedding = None
        if self.embedding_model:
            try:
                embedding = await self._generate_embedding(content)
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")

        memory = Memory(
            id=memory_id,
            content=content,
            summary=None,  # Will be generated during consolidation
            embedding=embedding,
            conversation_id=conversation_id,
            timestamp=timestamp,
            importance=importance,
            metadata=metadata,
        )

        # Store in database
        async with self._get_connection() as conn:
            data = memory.to_dict()
            placeholders = ", ".join(["?" for _ in data])
            columns = ", ".join(data.keys())

            logger.info("🧠 MEMORY ADD: Executing database INSERT")
            logger.info(f"🧠 MEMORY ADD:   SQL columns: {columns}")
            logger.info(f"🧠 MEMORY ADD:   Data values (first 5): {list(data.values())[:5]}")

            # Safe because data.keys() comes from a controlled dataclass, not user input
            await conn.execute(
                f"INSERT INTO memories ({columns}) VALUES ({placeholders})",  # noqa: S608
                list(data.values()),
            )
            await conn.commit()

            logger.info("🧠 MEMORY ADD: ✅ Successfully committed to database")

        logger.info(f"🧠 MEMORY ADD: ✅ Memory {memory_id[:8]}... stored successfully")
        return memory

    async def get_relevant_memories(
        self, query: str, limit: int = 10, time_window_hours: int | None = None, conversation_id: str | None = None
    ) -> list[Memory]:
        """Retrieve relevant memories for the current context.

        Uses a combination of:
        - Semantic similarity (if embeddings available)
        - Recency
        - Importance scores
        - Access patterns
        """
        if not self._initialized:
            await self.initialize()

        logger.info("🧠 MEMORY RETRIEVAL DB: Starting database query for memories")
        logger.info(f"🧠 MEMORY RETRIEVAL DB:   Query: '{query}'")
        logger.info(f"🧠 MEMORY RETRIEVAL DB:   Limit: {limit}")
        logger.info(f"🧠 MEMORY RETRIEVAL DB:   Time window: {time_window_hours}h")

        logger.debug(
            f"Retrieving memories: query={query[:50]}..., limit={limit}, time_window={time_window_hours}h"
        )

        memories = []
        high_importance_memories = []

        async with self._get_connection() as conn:
            # Build query based on available features
            base_query = """
                SELECT * FROM memories
                WHERE 1=1
            """
            params = []

            # Conversation filter for isolation
            if conversation_id:
                base_query += " AND conversation_id = ?"
                params.append(conversation_id)

            # Time window filter
            if time_window_hours:
                cutoff = datetime.now(UTC) - timedelta(hours=time_window_hours)
                base_query += " AND timestamp > ?"
                params.append(cutoff.isoformat())

            # For now, use simple relevance scoring
            # In production, you'd use vector similarity
            base_query += """
                ORDER BY
                    importance DESC,
                    timestamp DESC
                LIMIT ?
            """
            params.append(limit)

            logger.info("🧠 MEMORY RETRIEVAL DB: Executing SQL query")
            logger.info(f"🧠 MEMORY RETRIEVAL DB:   SQL: {base_query}")
            logger.info(f"🧠 MEMORY RETRIEVAL DB:   Params: {params}")

            cursor = await conn.execute(base_query, params)
            rows = await cursor.fetchall()

            logger.info(f"🧠 MEMORY RETRIEVAL DB: Found {len(rows)} rows in database")

            # Collect memories and their IDs for batch update
            memory_ids_to_update = []

            for i, row in enumerate(rows):
                memory = self._row_to_memory(dict(row))
                memories.append(memory)
                memory_ids_to_update.append(memory.id)

                logger.info(f"🧠 MEMORY RETRIEVAL DB: Row {i + 1}:")
                logger.info(f"🧠 MEMORY RETRIEVAL DB:   ID: {memory.id}")
                logger.info(f"🧠 MEMORY RETRIEVAL DB:   Conversation ID: {memory.conversation_id}")
                logger.info(f"🧠 MEMORY RETRIEVAL DB:   Importance: {memory.importance}")
                logger.info(f"🧠 MEMORY RETRIEVAL DB:   Timestamp: {memory.timestamp}")
                logger.info(
                    f"🧠 MEMORY RETRIEVAL DB:   Content (first 200 chars): '{memory.content[:200]}...'"
                )

                # Track high importance memories
                if memory.importance >= HIGH_IMPORTANCE_THRESHOLD:
                    high_importance_memories.append(memory)
                    logger.info("🧠 MEMORY RETRIEVAL DB:   ⭐ HIGH IMPORTANCE MEMORY")

            # Batch update access counts
            if memory_ids_to_update:
                now = datetime.now(UTC).isoformat()
                for memory_id in memory_ids_to_update:
                    await conn.execute(
                        "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                        (now, memory_id),
                    )
                await conn.commit()

        if high_importance_memories:
            logger.info(
                f"Retrieved {len(high_importance_memories)} high importance memories out of {len(memories)} total"
            )

        # Update access patterns for retrieved memories asynchronously
        if memories:
            # Don't await - let it run in background
            self._access_update_task = asyncio.create_task(
                self._update_memories_on_access(memory_ids_to_update)
            )

        return memories

    async def _update_memories_on_access(self, memory_ids: list[str]):
        """Update importance scores for accessed memories (runs in background)."""
        try:
            async with self._get_connection() as conn:
                now = datetime.now(UTC)
                for memory_id in memory_ids:
                    # Fetch current importance
                    cursor = await conn.execute(
                        "SELECT importance FROM memories WHERE id = ?", (memory_id,)
                    )
                    row = await cursor.fetchone()
                    if row:
                        current_importance = row[0]
                        # Boost importance with diminishing returns
                        boost = (1.0 - current_importance) * 0.05  # 5% of remaining gap to 1.0
                        new_importance = min(1.0, current_importance + boost)

                        # Update importance and access stats
                        await conn.execute(
                            """UPDATE memories
                               SET importance = ?,
                                   access_count = access_count + 1,
                                   last_accessed = ?
                               WHERE id = ?""",
                            (new_importance, now.isoformat(), memory_id),
                        )

                        if (
                            new_importance >= HIGH_IMPORTANCE_THRESHOLD
                            and current_importance < HIGH_IMPORTANCE_THRESHOLD
                        ):
                            logger.info(
                                f"Memory {memory_id} promoted to high importance through access pattern"
                            )

                await conn.commit()
        except Exception:
            logger.exception("Error updating memories on access")

    async def get_conversation_context(
        self, conversation_id: str, max_messages: int = 50
    ) -> list[Memory]:
        """Get recent memories from a specific conversation."""
        if not self._initialized:
            await self.initialize()

        async with self._get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT * FROM memories
                WHERE conversation_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (conversation_id, max_messages),
            )
            rows = await cursor.fetchall()

            memories = [self._row_to_memory(dict(row)) for row in rows]

            # Return in chronological order
            return list(reversed(memories))

    async def consolidate_old_memories(self):
        """Consolidate old memories to save space and improve retrieval.

        - Summarize conversations older than short_term_hours
        - Remove detailed memories older than long_term_days (keeping summaries)
        """
        if not self._initialized:
            await self.initialize()

        now = datetime.now(UTC)

        async with self._get_connection() as conn:
            # Find conversations that need summarization
            cutoff = now - timedelta(hours=self.short_term_hours)
            cursor = await conn.execute(
                """
                SELECT DISTINCT conversation_id
                FROM memories
                WHERE timestamp < ?
                AND conversation_id NOT IN (
                    SELECT conversation_id FROM conversation_summaries
                )
                """,
                (cutoff.isoformat(),),
            )
            rows = await cursor.fetchall()
            conversations_to_summarize = [row[0] for row in rows]

            for conv_id in conversations_to_summarize:
                await self._summarize_conversation(conv_id)

            # Remove old detailed memories (keep summaries)
            old_cutoff = now - timedelta(days=self.long_term_days)
            await conn.execute(
                """
                DELETE FROM memories
                WHERE timestamp < ?
                AND importance < ?
                AND conversation_id IN (
                    SELECT conversation_id FROM conversation_summaries
                )
                """,
                (old_cutoff.isoformat(), HIGH_IMPORTANCE_THRESHOLD),
            )
            await conn.commit()

    async def _summarize_conversation(self, conversation_id: str):
        """Create a summary for a conversation."""
        # This would use the LLM to generate summaries
        # For now, just log
        logger.debug(f"Would summarize conversation {conversation_id}")

    async def set_core_memory(self, key: str, value: str, conversation_id: str, category: str = "general"):
        """Set a core memory (persistent fact about the user)."""
        if not self._initialized:
            await self.initialize()

        now = datetime.now(UTC)
        async with self._get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO core_memories (key, value, conversation_id, created_at, updated_at, category)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(key, conversation_id) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at,
                    category = excluded.category
                """,
                (key, value, conversation_id, now.isoformat(), now.isoformat(), category),
            )
            await conn.commit()
        logger.debug(f"Set core memory for conversation {conversation_id}: {key} = {value[:50]}...")

    async def get_core_memory(self, key: str, conversation_id: str) -> str | None:
        """Get a core memory value for a specific conversation."""
        if not self._initialized:
            await self.initialize()

        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT value FROM core_memories WHERE key = ? AND conversation_id = ?", 
                (key, conversation_id)
            )
            row = await cursor.fetchone()
            return row[0] if row else None

    async def get_all_core_memories(self, conversation_id: str) -> dict[str, str]:
        """Get all core memories for a specific conversation."""
        if not self._initialized:
            await self.initialize()

        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT key, value FROM core_memories WHERE conversation_id = ? ORDER BY key", 
                (conversation_id,)
            )
            rows = await cursor.fetchall()
            return {row[0]: row[1] for row in rows}

    async def _generate_embedding(self, text: str) -> list[float] | None:
        """Generate embedding for text using the embedding model."""
        if not self.embedding_model:
            return None

        try:
            # Import the function from resource_manager
            from resource_manager import get_sentence_transformer_embeddings

            # Run embedding generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, get_sentence_transformer_embeddings, [text]
            )

            if embeddings and len(embeddings) > 0:
                # Convert numpy array to list if needed
                if hasattr(embeddings[0], "tolist"):
                    return embeddings[0].tolist()
                else:
                    return list(embeddings[0])
        except Exception:
            logger.exception("Failed to generate embedding")
            return None
        else:
            return None

    def _row_to_memory(self, row: dict) -> Memory:
        """Convert a database row to a Memory object."""
        return Memory(
            id=row["id"],
            content=row["content"],
            summary=row["summary"],
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            conversation_id=row["conversation_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            importance=row["importance"],
            access_count=row["access_count"],
            last_accessed=datetime.fromisoformat(row["last_accessed"])
            if row["last_accessed"]
            else None,
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get synchronous stats - kept for compatibility."""
        # This is a sync wrapper around async functionality
        # In production, you'd want to make this fully async
        return {
            "backend": "aiosqlite",
            "pool_size": self.pool_size,
            "active_connections": self.pool_size - len(self._pool),
            "stats_available": True,
        }

    async def get_stats_async(self) -> dict[str, Any]:
        """Get statistics about the memory system."""
        if not self._initialized:
            await self.initialize()

        async with self._get_connection() as conn:
            # Total memories
            cursor = await conn.execute("SELECT COUNT(*) FROM memories")
            total_memories = (await cursor.fetchone())[0]

            # High importance memories
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM memories WHERE importance >= ?", (HIGH_IMPORTANCE_THRESHOLD,)
            )
            high_importance_count = (await cursor.fetchone())[0]

            # Recent memories (last 24h)
            cutoff = datetime.now(UTC) - timedelta(hours=24)
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM memories WHERE timestamp > ?", (cutoff.isoformat(),)
            )
            recent_count = (await cursor.fetchone())[0]

            return {
                "backend": "aiosqlite",
                "total_memories": total_memories,
                "high_importance_memories": high_importance_count,
                "recent_memories_24h": recent_count,
                "pool_size": self.pool_size,
                "active_connections": self.pool_size - len(self._pool),
                "stats_available": True,
            }

    def _get_high_importance_count(self) -> int:
        """Sync method for compatibility - returns 0."""
        return 0

    async def close(self):
        """Close all connections in the pool."""
        while self._pool:
            conn = self._pool.pop()
            await conn.close()
        self._initialized = False
        logger.info("Closed all database connections")
