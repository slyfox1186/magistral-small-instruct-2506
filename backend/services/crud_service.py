#!/usr/bin/env python3
"""CRUD service for managing conversations, messages, and user settings."""

import json
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from modules.models import (
    ConversationCreate,
    ConversationListResponse,
    ConversationResponse,
    ConversationUpdate,
    MessageCreate,
    MessageListResponse,
    MessageResponse,
    MessageUpdate,
    UserSettingsCreate,
    UserSettingsResponse,
    UserSettingsUpdate,
)


class CRUDService:
    """Service for handling CRUD operations on conversations, messages, and user settings."""

    def __init__(self, db_path: str = "neural_consciousness.db"):
        """Initialize CRUDService with database path."""
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    archived BOOLEAN DEFAULT FALSE,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0
                )
            """)

            # Create messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
                )
            """)

            # Create user_settings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL UNIQUE,
                    theme TEXT DEFAULT 'celestial-indigo',
                    ai_personality TEXT DEFAULT 'helpful',
                    response_style TEXT DEFAULT 'balanced',
                    memory_retention BOOLEAN DEFAULT TRUE,
                    auto_summarize BOOLEAN DEFAULT TRUE,
                    preferred_language TEXT DEFAULT 'en',
                    custom_prompts TEXT NOT NULL DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for better performance
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_archived ON conversations(archived)"
            )

            conn.commit()

    @asynccontextmanager
    async def get_connection(self):
        """Get async database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ===================== Conversation CRUD =====================

    async def create_conversation(
        self, conversation_data: ConversationCreate
    ) -> ConversationResponse:
        """Create a new conversation."""
        conversation_id = str(uuid.uuid4())
        now = datetime.now(UTC)

        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO conversations (id, title, tags, archived, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    conversation_id,
                    conversation_data.title,
                    json.dumps(conversation_data.tags),
                    conversation_data.archived,
                    json.dumps(conversation_data.metadata),
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            conn.commit()

        return ConversationResponse(
            id=conversation_id,
            title=conversation_data.title,
            tags=conversation_data.tags,
            archived=conversation_data.archived,
            metadata=conversation_data.metadata,
            created_at=now,
            updated_at=now,
            message_count=0,
        )

    async def get_conversation(self, conversation_id: str) -> ConversationResponse | None:
        """Get a conversation by ID."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, title, tags, archived, metadata, created_at, updated_at, message_count
                FROM conversations WHERE id = ?
            """,
                (conversation_id,),
            )
            row = cursor.fetchone()

            if not row:
                return None

            return ConversationResponse(
                id=row["id"],
                title=row["title"],
                tags=json.loads(row["tags"]),
                archived=bool(row["archived"]),
                metadata=json.loads(row["metadata"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                message_count=row["message_count"],
            )

    async def list_conversations(
        self, page: int = 1, page_size: int = 20, archived: bool | None = None
    ) -> ConversationListResponse:
        """List conversations with pagination."""
        offset = (page - 1) * page_size

        async with self.get_connection() as conn:
            cursor = conn.cursor()

            # Build query with optional archived filter
            if archived is not None:
                count_query = "SELECT COUNT(*) FROM conversations WHERE archived = ?"
                list_query = """
                    SELECT id, title, tags, archived, metadata, created_at, updated_at, message_count
                    FROM conversations WHERE archived = ?
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                """
                count_params = [archived]
                list_params = [archived, page_size, offset]
            else:
                count_query = "SELECT COUNT(*) FROM conversations"
                list_query = """
                    SELECT id, title, tags, archived, metadata, created_at, updated_at, message_count
                    FROM conversations
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                """
                count_params = []
                list_params = [page_size, offset]

            # Get total count
            cursor.execute(count_query, count_params)
            total = cursor.fetchone()[0]

            # Get conversations
            cursor.execute(list_query, list_params)

            rows = cursor.fetchall()

            conversations = [
                ConversationResponse(
                    id=row["id"],
                    title=row["title"],
                    tags=json.loads(row["tags"]),
                    archived=bool(row["archived"]),
                    metadata=json.loads(row["metadata"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    message_count=row["message_count"],
                )
                for row in rows
            ]

            return ConversationListResponse(
                conversations=conversations, total=total, page=page, page_size=page_size
            )

    async def update_conversation(
        self, conversation_id: str, update_data: ConversationUpdate
    ) -> ConversationResponse | None:
        """Update a conversation."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()

            # Check if conversation exists
            cursor.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
            if not cursor.fetchone():
                return None

            # Build update query using a whitelist approach
            set_clauses = []
            params = []

            # Whitelist of allowed fields to prevent injection
            allowed_updates = {
                "title": update_data.title,
                "tags": json.dumps(update_data.tags) if update_data.tags is not None else None,
                "archived": update_data.archived,
                "metadata": json.dumps(update_data.metadata)
                if update_data.metadata is not None
                else None,
            }

            for field, value in allowed_updates.items():
                if value is not None:
                    set_clauses.append(f"{field} = ?")
                    params.append(value)

            if set_clauses:
                set_clauses.append("updated_at = ?")
                params.append(datetime.now(UTC).isoformat())
                params.append(conversation_id)

                # Construct safe SQL with whitelisted field names (safe from injection)
                set_clause = ", ".join(set_clauses)  # Only contains whitelisted field names
                sql = f"UPDATE conversations SET {set_clause} WHERE id = ?"  # noqa: S608
                cursor.execute(sql, params)
                conn.commit()

            # Return updated conversation
            return await self.get_conversation(conversation_id)

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages AND associated memories."""
        # CRITICAL: Also clear memories to prevent cross-contamination
        from modules.globals import app_state
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info(f"🗑️ Deleting conversation {conversation_id} and associated memories...")
        
        # First, clear memories from the personal memory system
        memories_deleted = 0
        if app_state.personal_memory:
            try:
                memories_deleted = app_state.personal_memory.clear_conversation_memories(conversation_id)
                logger.info(f"🗑️ Cleared {memories_deleted} memories for conversation {conversation_id}")
            except Exception as e:
                logger.error(f"🗑️ Failed to clear memories for conversation {conversation_id}: {e}")
                # Continue with conversation deletion even if memory clearing fails
        
        # Then delete the conversation from CRUD database
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
            
            if deleted:
                logger.info(f"🗑️ Successfully deleted conversation {conversation_id} and {memories_deleted} associated memories")
            else:
                logger.warning(f"🗑️ Conversation {conversation_id} not found in database")
                
            return deleted

    # ===================== Message CRUD =====================

    async def create_message(self, message_data: MessageCreate) -> MessageResponse:
        """Create a new message."""
        message_id = str(uuid.uuid4())
        now = datetime.now(UTC)

        async with self.get_connection() as conn:
            cursor = conn.cursor()

            # Insert message
            cursor.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    message_id,
                    message_data.conversation_id,
                    message_data.role,
                    message_data.content,
                    json.dumps(message_data.metadata),
                    now.isoformat(),
                    now.isoformat(),
                ),
            )

            # Update conversation message count and updated_at
            cursor.execute(
                """
                UPDATE conversations
                SET message_count = message_count + 1,
                    updated_at = ?
                WHERE id = ?
            """,
                (now.isoformat(), message_data.conversation_id),
            )

            conn.commit()

        return MessageResponse(
            id=message_id,
            conversation_id=message_data.conversation_id,
            role=message_data.role,
            content=message_data.content,
            metadata=message_data.metadata,
            created_at=now,
            updated_at=now,
        )

    async def get_message(self, message_id: str) -> MessageResponse | None:
        """Get a message by ID."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, conversation_id, role, content, metadata, created_at, updated_at
                FROM messages WHERE id = ?
            """,
                (message_id,),
            )
            row = cursor.fetchone()

            if not row:
                return None

            return MessageResponse(
                id=row["id"],
                conversation_id=row["conversation_id"],
                role=row["role"],
                content=row["content"],
                metadata=json.loads(row["metadata"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    async def list_messages(
        self, conversation_id: str, page: int = 1, page_size: int = 50
    ) -> MessageListResponse:
        """List messages for a conversation with pagination."""
        offset = (page - 1) * page_size

        async with self.get_connection() as conn:
            cursor = conn.cursor()

            # Get total count
            cursor.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,)
            )
            total = cursor.fetchone()[0]

            # Get messages
            cursor.execute(
                """
                SELECT id, conversation_id, role, content, metadata, created_at, updated_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                LIMIT ? OFFSET ?
            """,
                (conversation_id, page_size, offset),
            )

            rows = cursor.fetchall()

            messages = [
                MessageResponse(
                    id=row["id"],
                    conversation_id=row["conversation_id"],
                    role=row["role"],
                    content=row["content"],
                    metadata=json.loads(row["metadata"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
                for row in rows
            ]

            return MessageListResponse(
                messages=messages, total=total, page=page, page_size=page_size
            )

    async def update_message(
        self, message_id: str, update_data: MessageUpdate
    ) -> MessageResponse | None:
        """Update a message."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()

            # Check if message exists
            cursor.execute("SELECT id FROM messages WHERE id = ?", (message_id,))
            if not cursor.fetchone():
                return None

            # Build update query using a whitelist approach
            set_clauses = []
            params = []

            # Whitelist of allowed fields to prevent injection
            allowed_updates = {
                "content": update_data.content,
                "metadata": json.dumps(update_data.metadata)
                if update_data.metadata is not None
                else None,
            }

            for field, value in allowed_updates.items():
                if value is not None:
                    set_clauses.append(f"{field} = ?")
                    params.append(value)

            if set_clauses:
                set_clauses.append("updated_at = ?")
                params.append(datetime.now(UTC).isoformat())
                params.append(message_id)

                # Construct safe SQL with whitelisted field names (safe from injection)
                set_clause = ", ".join(set_clauses)  # Only contains whitelisted field names
                sql = f"UPDATE messages SET {set_clause} WHERE id = ?"  # noqa: S608
                cursor.execute(sql, params)
                conn.commit()

            # Return updated message
            return await self.get_message(message_id)

    async def delete_message(self, message_id: str) -> bool:
        """Delete a message and update conversation message count."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()

            # Get conversation_id before deleting
            cursor.execute("SELECT conversation_id FROM messages WHERE id = ?", (message_id,))
            row = cursor.fetchone()
            if not row:
                return False

            conversation_id = row["conversation_id"]

            # Delete message
            cursor.execute("DELETE FROM messages WHERE id = ?", (message_id,))

            # Update conversation message count
            cursor.execute(
                """
                UPDATE conversations
                SET message_count = message_count - 1,
                    updated_at = ?
                WHERE id = ?
            """,
                (datetime.now(UTC).isoformat(), conversation_id),
            )

            conn.commit()
            return cursor.rowcount > 0

    # ===================== User Settings CRUD =====================

    async def create_user_settings(self, settings_data: UserSettingsCreate) -> UserSettingsResponse:
        """Create user settings."""
        settings_id = str(uuid.uuid4())
        now = datetime.now(UTC)

        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO user_settings (
                    id, user_id, theme, ai_personality, response_style,
                    memory_retention, auto_summarize, preferred_language,
                    custom_prompts, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    settings_id,
                    settings_data.user_id,
                    settings_data.theme,
                    settings_data.ai_personality,
                    settings_data.response_style,
                    settings_data.memory_retention,
                    settings_data.auto_summarize,
                    settings_data.preferred_language,
                    json.dumps(settings_data.custom_prompts),
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            conn.commit()

        return UserSettingsResponse(
            id=settings_id,
            user_id=settings_data.user_id,
            theme=settings_data.theme,
            ai_personality=settings_data.ai_personality,
            response_style=settings_data.response_style,
            memory_retention=settings_data.memory_retention,
            auto_summarize=settings_data.auto_summarize,
            preferred_language=settings_data.preferred_language,
            custom_prompts=settings_data.custom_prompts,
            created_at=now,
            updated_at=now,
        )

    async def get_user_settings(self, user_id: str) -> UserSettingsResponse | None:
        """Get user settings by user ID."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, user_id, theme, ai_personality, response_style,
                       memory_retention, auto_summarize, preferred_language,
                       custom_prompts, created_at, updated_at
                FROM user_settings WHERE user_id = ?
            """,
                (user_id,),
            )
            row = cursor.fetchone()

            if not row:
                return None

            return UserSettingsResponse(
                id=row["id"],
                user_id=row["user_id"],
                theme=row["theme"],
                ai_personality=row["ai_personality"],
                response_style=row["response_style"],
                memory_retention=bool(row["memory_retention"]),
                auto_summarize=bool(row["auto_summarize"]),
                preferred_language=row["preferred_language"],
                custom_prompts=json.loads(row["custom_prompts"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    async def update_user_settings(
        self, user_id: str, update_data: UserSettingsUpdate
    ) -> UserSettingsResponse | None:
        """Update user settings."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()

            # Check if settings exist
            cursor.execute("SELECT id FROM user_settings WHERE user_id = ?", (user_id,))
            if not cursor.fetchone():
                return None

            # Build update query using a whitelist approach
            set_clauses = []
            params = []

            # Whitelist of allowed fields to prevent injection
            allowed_updates = {
                "theme": update_data.theme,
                "ai_personality": update_data.ai_personality,
                "response_style": update_data.response_style,
                "memory_retention": update_data.memory_retention,
                "auto_summarize": update_data.auto_summarize,
                "preferred_language": update_data.preferred_language,
                "custom_prompts": json.dumps(update_data.custom_prompts)
                if update_data.custom_prompts is not None
                else None,
            }

            for field, value in allowed_updates.items():
                if value is not None:
                    set_clauses.append(f"{field} = ?")
                    params.append(value)

            if set_clauses:
                set_clauses.append("updated_at = ?")
                params.append(datetime.now(UTC).isoformat())
                params.append(user_id)

                # Construct safe SQL with whitelisted field names (safe from injection)
                set_clause = ", ".join(set_clauses)  # Only contains whitelisted field names
                sql = f"UPDATE user_settings SET {set_clause} WHERE user_id = ?"  # noqa: S608
                cursor.execute(sql, params)
                conn.commit()

            # Return updated settings
            return await self.get_user_settings(user_id)

    async def delete_user_settings(self, user_id: str) -> bool:
        """Delete user settings."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_settings WHERE user_id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount > 0
