"""Optimized memory system for personal AI assistant.

Uses SQLite for robust local storage and implements hierarchical memory.
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Constants for memory system
HIGH_IMPORTANCE_THRESHOLD = 0.7
MEDIUM_IMPORTANCE_THRESHOLD = 0.5
MIN_PHRASE_LENGTH = 2
MIN_CONTENT_LENGTH = 10
MAX_CONTENT_SNIPPET = 80


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


class PersonalMemorySystem:
    """Hierarchical memory system optimized for personal AI use.

    Features:
    - SQLite for robust local storage (no Redis needed)
    - Automatic summarization of old conversations
    - Smart retrieval based on relevance and recency
    - Memory importance scoring
    """

    def __init__(self, db_path: str = "memories.db", embedding_model=None):
        """Initialize the personal memory system."""
        self.db_path = db_path
        self.embedding_model = embedding_model
        self._init_database()

        # Memory windows
        self.short_term_hours = 24  # Keep full detail for 24 hours
        self.medium_term_days = 7  # Keep summaries for a week
        self.long_term_days = 30  # Keep important memories for a month

    def _init_database(self):
        """Initialize SQLite database with optimized schema."""
        with self._get_connection() as conn:
            conn.execute(
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversation ON memories(conversation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)")

            # Conversation summaries table
            conn.execute(
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
            self._init_core_memories_table(conn)

            conn.commit()

    def _init_core_memories_table(self, conn):
        """Initialize or migrate core_memories table with proper schema."""
        try:
            # Check if table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='core_memories'"
            )
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                # Check current schema
                cursor = conn.execute("PRAGMA table_info(core_memories)")
                columns = [row[1] for row in cursor.fetchall()]
                
                # If conversation_id column doesn't exist, we need to migrate
                if "conversation_id" not in columns:
                    logger.info("🔄 Migrating core_memories table to add conversation_id column...")
                    
                    # Get existing data
                    try:
                        cursor = conn.execute("SELECT key, value, created_at, updated_at, category FROM core_memories")
                        existing_data = cursor.fetchall()
                        logger.info(f"Found {len(existing_data)} existing core memories to migrate")
                    except Exception:
                        # If we can't read existing data, assume table is corrupted
                        existing_data = []
                        logger.warning("Could not read existing core memories, proceeding with empty migration")
                    
                    # Drop the old table
                    conn.execute("DROP TABLE core_memories")
                    logger.info("Dropped old core_memories table")
                    
                    # Create new table with updated schema
                    conn.execute(
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
                            conn.execute(
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
                conn.execute(
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_core_memories_conversation ON core_memories(conversation_id)")
            logger.debug("Created index for core_memories conversation queries")
                
        except Exception as e:
            logger.error(f"❌ Error during core_memories table initialization: {e}")
            # If everything fails, create a basic table
            try:
                conn.execute(
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

    @contextmanager
    def _get_connection(self):
        """Get a database connection with optimized settings."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        # Faster writes
        conn.execute("PRAGMA synchronous=NORMAL")
        # Increase cache size for better performance (10MB)
        conn.execute("PRAGMA cache_size=-10240")
        # Set busy timeout to avoid lock errors
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
        finally:
            conn.close()

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
        timestamp = datetime.now(UTC)
        memory_id = self._generate_id(content, timestamp)

        # Log entry for all memories
        logger.debug(
            f"[MEMORY_CREATE] Memory ID: {memory_id[:8]}..., "
            f"Importance: {importance}, Length: {len(content)} chars"
        )

        # Generate embedding if model is available
        embedding = None
        if self.embedding_model:
            try:
                logger.debug(
                    f"[MEMORY_EMBEDDING] Generating embedding for memory {memory_id[:8]}..."
                )
                embedding = await self._generate_embedding(content)
                if embedding is not None:
                    logger.debug(
                        f"[MEMORY_EMBEDDING] Successfully generated {len(embedding)} dimensional embedding"
                    )
                else:
                    logger.warning("[MEMORY_EMBEDDING] Embedding generation returned None")
            except Exception as e:
                logger.warning(f"[MEMORY_EMBEDDING] Failed to generate embedding: {e}")

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

        # Check if this is a high importance memory
        if importance >= HIGH_IMPORTANCE_THRESHOLD:
            logger.warning(
                f"\n[MEMORY_HIGH_IMPORTANCE] 🔥 HIGH IMPORTANCE MEMORY DETECTED! 🔥\n"
                f"  Memory ID: {memory_id}\n"
                f"  Importance Score: {importance}\n"
                f"  Conversation ID: {conversation_id}\n"
                f"  Timestamp: {timestamp.isoformat()}\n"
                f"  Content: {content}\n"
                f"  Metadata: {json.dumps(metadata, indent=2) if metadata else 'None'}\n"
                f"  Embedding Status: {'Generated' if embedding else 'Not Generated'}\n"
                f"  {'=' * 80}"
            )

        # Store in database
        start_time = datetime.now(UTC)
        with self._get_connection() as conn:
            data = memory.to_dict()
            placeholders = ", ".join(["?" for _ in data])
            columns = ", ".join(data.keys())

            # Safe because data.keys() comes from a controlled dataclass, not user input
            conn.execute(
                f"INSERT INTO memories ({columns}) VALUES ({placeholders})",  # noqa: S608
                list(data.values()),
            )
            conn.commit()

        db_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
        logger.debug(f"[MEMORY_STORED] Memory {memory_id[:8]}... stored in {db_time:.2f}ms")

        # Log memory statistics after addition
        if importance >= HIGH_IMPORTANCE_THRESHOLD:
            stats = self.get_stats()
            high_importance_count = self._get_high_importance_count()
            logger.info(
                f"[MEMORY_STATS] Current system statistics:\n"
                f"  Total memories: {stats['total_memories']}\n"
                f"  High importance memories: {high_importance_count}\n"
                f"  High importance percentage: "
                f"{(high_importance_count / stats['total_memories'] * 100):.1f}%"
                if stats["total_memories"] > 0
                else "N/A"
            )

        return memory

    async def get_relevant_memories(
        self, query: str, limit: int = 10, time_window_hours: int | None = None
    ) -> list[Memory]:
        """Retrieve relevant memories for the current context.

        Uses a combination of:
        - Semantic similarity (if embeddings available)
        - Recency
        - Importance scores
        - Access patterns
        """
        logger.debug(
            f"[MEMORY_RETRIEVAL] Query: {query[:50]}..., Limit: {limit}, "
            f"Time window: {time_window_hours}h"
            if time_window_hours
            else "Time window: None"
        )

        memories = []
        high_importance_memories = []
        retrieval_start = datetime.now(UTC)

        with self._get_connection() as conn:
            # Build query based on available features
            base_query = """
                SELECT * FROM memories
                WHERE 1=1
            """
            params = []

            # Time window filter
            if time_window_hours:
                cutoff = datetime.now(UTC) - timedelta(hours=time_window_hours)
                base_query += " AND timestamp > ?"
                params.append(cutoff.isoformat())

            # Implement proper semantic search using embeddings
            try:
                # Generate embedding for the query using ResourceManager
                from resource_manager import get_sentence_transformer_embeddings
                query_embeddings = get_sentence_transformer_embeddings([query])
                if query_embeddings and len(query_embeddings) > 0:
                    query_embedding = query_embeddings[0]
                    
                    # Get all memories with embeddings for similarity calculation
                    embedding_query = base_query + " AND embedding IS NOT NULL"
                    cursor = conn.execute(embedding_query, params[:-1])  # Remove limit for now
                    
                    memories_with_similarity = []
                    
                    for row in cursor:
                        try:
                            memory = self._row_to_memory(row)
                            if memory.embedding:
                                # Calculate cosine similarity
                                import numpy as np
                                stored_embedding = np.array(memory.embedding)
                                query_emb = np.array(query_embedding)
                                
                                # Cosine similarity calculation
                                dot_product = np.dot(query_emb, stored_embedding)
                                norm_query = np.linalg.norm(query_emb)
                                norm_stored = np.linalg.norm(stored_embedding)
                                
                                if norm_query > 0 and norm_stored > 0:
                                    similarity = dot_product / (norm_query * norm_stored)
                                    memories_with_similarity.append((similarity, memory))
                        except Exception as e:
                            logger.debug(f"Error calculating similarity for memory {row[0]}: {e}")
                            continue
                    
                    # Sort by similarity (desc), then importance (desc), then timestamp (desc)
                    memories_with_similarity.sort(key=lambda x: (x[0], x[1].importance, x[1].timestamp), reverse=True)
                    
                    # Apply similarity threshold (0.2) and limit
                    similarity_threshold = 0.2
                    memories = [memory for similarity, memory in memories_with_similarity 
                              if similarity >= similarity_threshold][:limit]
                    
                    if memories:
                        logger.debug(f"[MEMORY_RETRIEVAL] Found {len(memories)} semantically similar memories")
                    else:
                        # If no semantic matches, fall back to importance/timestamp
                        raise ValueError("No semantic matches found")
                else:
                    raise ValueError("Could not generate query embedding")
                    
            except Exception as e:
                logger.debug(f"[MEMORY_RETRIEVAL] Semantic search failed: {e}, falling back to importance/timestamp")
                # Fallback to original importance/timestamp sorting
                base_query += """
                    ORDER BY
                        importance DESC,
                        timestamp DESC
                    LIMIT ?
                """
                params.append(limit)
                cursor = conn.execute(base_query, params)
                memories = [self._row_to_memory(row) for row in cursor]

            # Collect memory IDs for batch update and track high importance memories
            memory_ids_to_update = []
            
            for memory in memories:
                memory_ids_to_update.append(memory.id)
                
                # Track high importance memories
                if memory.importance >= HIGH_IMPORTANCE_THRESHOLD:
                    high_importance_memories.append(memory)
                    logger.info(
                        f"[MEMORY_HIGH_IMPORTANCE_RETRIEVED] Retrieved high importance memory:\n"
                        f"  Memory ID: {memory.id}\n"
                        f"  Importance: {memory.importance}\n"
                        f"  Created: {memory.timestamp.isoformat()}\n"
                        f"  Access count: {memory.access_count}\n"
                        f"  Content preview: {memory.content[:100]}..."
                    )

            # Batch update access counts
            if memory_ids_to_update:
                now = datetime.now(UTC).isoformat()
                for memory_id in memory_ids_to_update:
                    conn.execute(
                        "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                        (now, memory_id),
                    )
                conn.commit()

        retrieval_time = (datetime.now(UTC) - retrieval_start).total_seconds() * 1000

        logger.debug(
            f"[MEMORY_RETRIEVAL_COMPLETE] Retrieved {len(memories)} memories in {retrieval_time:.2f}ms"
        )

        # Log details of high importance memories if any
        if high_importance_memories:
            logger.warning(
                "[MEMORY_HIGH_IMPORTANCE_SUMMARY] High importance memories in retrieval:\n"
                + "\n".join(
                    [
                        f"  - ID: {m.id[:8]}... | Importance: {m.importance} | "
                        f"Accessed: {m.access_count} times"
                        for m in high_importance_memories
                    ]
                )
            )

        return memories

    async def get_conversation_context(
        self, conversation_id: str, max_messages: int = 50
    ) -> list[Memory]:
        """Get recent memories from a specific conversation."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM memories
                WHERE conversation_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (conversation_id, max_messages),
            )

            memories = [self._row_to_memory(row) for row in cursor]

            # Return in chronological order
            return list(reversed(memories))

    async def consolidate_old_memories(self):
        """Consolidate old memories to save space and improve retrieval.

        - Summarize conversations older than short_term_hours
        - Remove detailed memories older than long_term_days (keeping summaries)
        """
        now = datetime.now(UTC)

        with self._get_connection() as conn:
            # Find conversations that need summarization
            cutoff = now - timedelta(hours=self.short_term_hours)
            cursor = conn.execute(
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

            conversations_to_summarize = [row[0] for row in cursor]

            for conv_id in conversations_to_summarize:
                await self._summarize_conversation(conv_id)

            # Clean up very old detailed memories
            old_cutoff = now - timedelta(days=self.long_term_days)
            deleted = conn.execute(
                """
                DELETE FROM memories
                WHERE timestamp < ?
                AND importance < 0.7
                AND conversation_id IN (
                    SELECT conversation_id FROM conversation_summaries
                )
                """,
                (old_cutoff.isoformat(),),
            ).rowcount

            conn.commit()

            # Count high importance memories before deletion
            high_importance_deleted = conn.execute(
                """
                SELECT COUNT(*) FROM memories
                WHERE timestamp < ?
                AND importance >= ?
                AND importance < 0.7
                AND conversation_id IN (
                    SELECT conversation_id FROM conversation_summaries
                )
                """,
                (old_cutoff.isoformat(), HIGH_IMPORTANCE_THRESHOLD),
            ).fetchone()[0]

            if deleted > 0:
                logger.info(
                    f"[MEMORY_CONSOLIDATION] Cleaned up {deleted} old memories\n"
                    f"  High importance memories protected: {high_importance_deleted}\n"
                    f"  Note: Memories with importance >= 0.7 are always preserved"
                )

    async def _summarize_conversation(self, conversation_id: str):
        """Create a summary of a conversation with proper topic extraction."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT content, timestamp, importance, metadata
                FROM memories
                WHERE conversation_id = ?
                ORDER BY timestamp
                """,
                (conversation_id,),
            )

            messages = list(cursor)
            if not messages:
                return

            message_count = len(messages)
            avg_importance = sum(row[2] for row in messages) / message_count
            start_time = messages[0][1]
            end_time = messages[-1][1]

            # Extract topics from high-importance messages
            topics = set()
            key_points = []

            for content, _timestamp, importance, metadata_json in messages:
                # Parse metadata to extract key phrases
                if metadata_json:
                    try:
                        metadata = json.loads(metadata_json)
                        if "analysis" in metadata and "key_phrases" in metadata["analysis"]:
                            for phrase in metadata["analysis"]["key_phrases"]:
                                if phrase and len(phrase) > MIN_PHRASE_LENGTH:
                                    topics.add(phrase[:50])  # Limit length
                    except Exception as e:
                        logger.debug(f"Failed to parse metadata for phrase extraction: {e}")

                # Add content-based topics for high importance messages
                if importance >= HIGH_IMPORTANCE_THRESHOLD:
                    # Extract first 50 chars as key point
                    clean_content = content.replace("User: ", "").replace("Assistant: ", "")
                    if len(clean_content) > MIN_CONTENT_LENGTH:
                        key_points.append(
                            clean_content[:MAX_CONTENT_SNIPPET]
                            + ("..." if len(clean_content) > MAX_CONTENT_SNIPPET else "")
                        )

            # Build summary
            if key_points:
                summary = f"Discussion covered: {'; '.join(key_points[:3])}"
            else:
                summary = f"Conversation with {message_count} exchanges"

            # Add topic list
            topic_list = list(topics)[:5]  # Limit to 5 topics
            topics_str = ", ".join(topic_list) if topic_list else "general discussion"

            conn.execute(
                """
                INSERT OR REPLACE INTO conversation_summaries
                (conversation_id, summary, start_time, end_time, message_count, importance, topics)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    summary,
                    start_time,
                    end_time,
                    message_count,
                    avg_importance,
                    topics_str,
                ),
            )
            conn.commit()

            logger.info(
                f"📋 Summarized conversation {conversation_id[:8]} - {message_count} messages, "
                f"topics: {topics_str[:50]}"
            )

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        """Convert a database row to a Memory object."""
        data = dict(row)
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["last_accessed"] = (
            datetime.fromisoformat(data["last_accessed"]) if data["last_accessed"] else None
        )
        data["embedding"] = json.loads(data["embedding"]) if data["embedding"] else None
        data["metadata"] = json.loads(data["metadata"]) if data["metadata"] else None
        return Memory(**data)

    async def _generate_embedding(self, text: str) -> list[float] | None:
        """Generate embedding for text using ResourceManager.

        Returns a list of floats on success, None on failure.
        """
        if not self.embedding_model:
            logger.warning("[MEMORY_EMBEDDING] No embedding model available")
            return None

        try:
            # Use ResourceManager to generate embeddings
            import numpy as np

            from resource_manager import get_sentence_transformer_embeddings

            embeddings = get_sentence_transformer_embeddings([text])

            # Consolidated check for a valid, non-empty list/array of embeddings
            if embeddings is None or not hasattr(embeddings, "__len__") or len(embeddings) == 0:
                logger.warning("[MEMORY_EMBEDDING] ResourceManager returned no valid embeddings")
                return None

            embedding = embeddings[0]

            # Robustly check the actual embedding object before converting
            if isinstance(embedding, np.ndarray):
                return embedding.tolist()
            elif isinstance(embedding, list | tuple) and all(
                isinstance(x, int | float) for x in embedding
            ):
                return list(embedding)
            else:
                logger.warning(
                    f"[MEMORY_EMBEDDING] Unexpected embedding format or type: {type(embedding)}"
                )
                return None

        except Exception as e:
            # Include full stack trace for easier debugging
            logger.error(f"[MEMORY_EMBEDDING] Failed to generate embedding: {e}", exc_info=True)
            return None

    async def set_core_memory(self, key: str, value: str, conversation_id: str, category: str = "general"):
        """Set a core memory (persistent fact about the user) for a specific conversation."""
        now = datetime.now(UTC)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO core_memories (key, value, conversation_id, created_at, updated_at, category)
                VALUES (?, ?, ?,
                    COALESCE((SELECT created_at FROM core_memories WHERE key = ? AND conversation_id = ?), ?),
                    ?, ?)
                """,
                (key, value, conversation_id, key, conversation_id, now.isoformat(), now.isoformat(), category),
            )
            conn.commit()

        logger.info(f"Core memory set for conversation {conversation_id}: {key} = {value[:50]}...")

    async def get_core_memory(self, key: str, conversation_id: str) -> str | None:
        """Get a specific core memory for a conversation."""
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT value FROM core_memories WHERE key = ? AND conversation_id = ?", (key, conversation_id)
            ).fetchone()

            return result[0] if result else None

    async def get_all_core_memories(self, conversation_id: str) -> dict[str, str]:
        """Get all core memories for a specific conversation as a dictionary."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT key, value FROM core_memories WHERE conversation_id = ? ORDER BY category, key", 
                (conversation_id,)
            )

            return {row[0]: row[1] for row in cursor}

    def _get_high_importance_count(self) -> int:
        """Get count of high importance memories."""
        with self._get_connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM memories WHERE importance >= ?", (HIGH_IMPORTANCE_THRESHOLD,)
            ).fetchone()[0]

    def get_stats(self) -> dict[str, Any]:
        """Get memory system statistics."""
        with self._get_connection() as conn:
            total_memories = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            total_conversations = conn.execute(
                "SELECT COUNT(DISTINCT conversation_id) FROM memories"
            ).fetchone()[0]
            total_summaries = conn.execute(
                "SELECT COUNT(*) FROM conversation_summaries"
            ).fetchone()[0]
            total_core_memories = conn.execute("SELECT COUNT(*) FROM core_memories").fetchone()[0]

            # High importance statistics
            high_importance_count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE importance >= ?", (HIGH_IMPORTANCE_THRESHOLD,)
            ).fetchone()[0]

            # Importance distribution
            importance_distribution = conn.execute(
                """
                SELECT
                    CASE
                        WHEN importance >= 0.9 THEN 'Critical (0.9-1.0)'
                        WHEN importance >= 0.7 THEN 'High (0.7-0.9)'
                        WHEN importance >= 0.5 THEN 'Medium (0.5-0.7)'
                        ELSE 'Low (0.0-0.5)'
                    END as importance_level,
                    COUNT(*) as count
                FROM memories
                GROUP BY importance_level
                ORDER BY MIN(importance) DESC
                """
            ).fetchall()

            # Average importance
            avg_importance = (
                conn.execute("SELECT AVG(importance) FROM memories").fetchone()[0] or 0.0
            )

            # Get memory distribution by age
            now = datetime.now(UTC)
            day_ago = now - timedelta(days=1)
            week_ago = now - timedelta(days=7)
            month_ago = now - timedelta(days=30)

            recent_24h = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE timestamp > ?", (day_ago.isoformat(),)
            ).fetchone()[0]

            recent_week = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE timestamp > ?", (week_ago.isoformat(),)
            ).fetchone()[0]

            recent_month = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE timestamp > ?", (month_ago.isoformat(),)
            ).fetchone()[0]

            # Database size
            db_size = Path(self.db_path).stat().st_size / (1024 * 1024)  # MB

            stats = {
                "total_memories": total_memories,
                "total_conversations": total_conversations,
                "total_summaries": total_summaries,
                "total_core_memories": total_core_memories,
                "high_importance_memories": high_importance_count,
                "high_importance_percentage": (
                    (high_importance_count / total_memories * 100) if total_memories > 0 else 0
                ),
                "average_importance": round(avg_importance, 3),
                "importance_distribution": {row[0]: row[1] for row in importance_distribution},
                "memories_24h": recent_24h,
                "memories_7d": recent_week,
                "memories_30d": recent_month,
                "database_size_mb": round(db_size, 2),
            }

            # Log statistics if there are high importance memories
            if high_importance_count > 0:
                logger.info(
                    "[MEMORY_STATS] Memory system statistics:\n"
                    + f"  Total memories: {total_memories}\n"
                    + f"  High importance: {high_importance_count} "
                    + f"({stats['high_importance_percentage']:.1f}%)\n"
                    + f"  Average importance: {avg_importance:.3f}\n"
                    + f"  Distribution: {stats['importance_distribution']}"
                )

            return stats

    def clear_conversation_memories(self, conversation_id: str) -> int:
        """Clear all memories for a specific conversation and return count of deleted items."""
        deleted_count = 0

        with self._get_connection() as conn:
            # Count items before deletion
            memories_count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE conversation_id = ?",
                (conversation_id,)
            ).fetchone()[0]
            summaries_count = conn.execute(
                "SELECT COUNT(*) FROM conversation_summaries WHERE conversation_id = ?",
                (conversation_id,)
            ).fetchone()[0]
            core_memories_count = conn.execute(
                "SELECT COUNT(*) FROM core_memories WHERE conversation_id = ?",
                (conversation_id,)
            ).fetchone()[0]

            # Clear conversation-specific data
            conn.execute("DELETE FROM memories WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM conversation_summaries WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM core_memories WHERE conversation_id = ?", (conversation_id,))

            deleted_count = memories_count + summaries_count + core_memories_count
            
            logger.info(f"🗑️ Cleared {deleted_count} memories for conversation {conversation_id}")
            
            conn.commit()

        return deleted_count

    def clear_all_memories(self) -> int:
        """Clear all memories from the database and return count of deleted items."""
        deleted_count = 0

        with self._get_connection() as conn:
            # Count total items before deletion
            total_memories = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            total_summaries = conn.execute(
                "SELECT COUNT(*) FROM conversation_summaries"
            ).fetchone()[0]
            total_core_memories = conn.execute("SELECT COUNT(*) FROM core_memories").fetchone()[0]

            # Clear all tables
            conn.execute("DELETE FROM memories")
            conn.execute("DELETE FROM conversation_summaries")
            conn.execute("DELETE FROM core_memories")

            # Reset auto-increment sequences (only if table exists)
            try:
                conn.execute(
                    "DELETE FROM sqlite_sequence WHERE name IN "
                    "('memories', 'conversation_summaries', 'core_memories')"
                )
            except Exception as e:
                logger.debug(f"sqlite_sequence table doesn't exist or couldn't be cleared: {e}")

            conn.commit()

            deleted_count = total_memories + total_summaries + total_core_memories
            logger.info(
                f"Cleared {total_memories} memories, {total_summaries} summaries, "
                f"{total_core_memories} core memories"
            )

        return deleted_count


# Example usage
async def demo():
    """Demo of the personal memory system."""
    memory_system = PersonalMemorySystem("my_ai_memories.db")

    # Add some memories
    conv_id = "conv_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    await memory_system.add_memory(
        "User asked about quantum computing", conv_id, importance=0.8, metadata={"topic": "physics"}
    )

    await memory_system.add_memory(
        "Explained superposition and entanglement", conv_id, importance=0.7
    )

    # Retrieve relevant memories
    memories = await memory_system.get_relevant_memories("Tell me about quantum mechanics", limit=5)

    for memory in memories:
        print(f"- {memory.content} (importance: {memory.importance})")

    # Get stats
    stats = memory_system.get_stats()
    print(f"\nMemory Stats: {stats}")


if __name__ == "__main__":
    asyncio.run(demo())
