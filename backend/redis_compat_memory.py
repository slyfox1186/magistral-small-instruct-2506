"""Redis compatibility layer for PersonalMemorySystem.
Provides Redis-like API while using SQLite backend.
"""

import asyncio
import logging

from personal_memory_system import PersonalMemorySystem

logger = logging.getLogger(__name__)


class RedisCompatMemorySystem:
    """Adapter that makes PersonalMemorySystem look like Redis.
    This is a transitional layer to minimize changes to dependent code.
    """

    def __init__(self, personal_memory: PersonalMemorySystem):
        self.memory = personal_memory
        self._pub_sub_handlers = {}

    async def get(self, key: str) -> str | None:
        """Redis GET command compatibility."""
        # Check if it's a core memory key
        if key.startswith("core_memory:"):
            core_key = key.replace("core_memory:", "")
            value = await self.memory.get_core_memory(core_key)
            return value

        # For regular memories, return the most recent one
        memories = await self.memory.get_relevant_memories(key, limit=1)
        if memories:
            return memories[0].content
        return None

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        """Redis SET command compatibility."""
        # If it's a core memory
        if key.startswith("core_memory:"):
            core_key = key.replace("core_memory:", "")
            await self.memory.set_core_memory(core_key, value)
            return True

        # Otherwise, add as a regular memory
        await self.memory.add_memory(content=value, conversation_id=key, importance=0.5)
        return True

    async def lpush(self, key: str, *values: str) -> int:
        """Redis LPUSH command compatibility - adds to beginning of list."""
        # Store each value as a separate memory with the same conversation_id
        for value in values:
            await self.memory.add_memory(content=value, conversation_id=key, importance=0.5)

        # Return approximate count
        memories = await self.memory.get_conversation_context(key, max_messages=1000)
        return len(memories)

    async def lrange(self, key: str, start: int, stop: int) -> list[str]:
        """Redis LRANGE command compatibility."""
        # Get memories for this conversation
        max_messages = stop - start + 1 if stop >= 0 else 1000
        memories = await self.memory.get_conversation_context(key, max_messages=max_messages)

        # Convert to string list
        result = [m.content for m in memories]

        # Apply range slicing
        if stop < 0:
            return result[start:]
        else:
            return result[start : stop + 1]

    async def hset(self, name: str, key: str, value: str) -> int:
        """Redis HSET command compatibility."""
        # Store as metadata in a memory
        await self.memory.add_memory(
            content=value,
            conversation_id=name,
            metadata={"field": key, "type": "hash"},
            importance=0.3,
        )
        return 1

    async def hget(self, name: str, key: str) -> str | None:
        """Redis HGET command compatibility."""
        # Search for the specific hash field
        memories = await self.memory.get_relevant_memories(query=f"{name}:{key}", limit=10)

        for memory in memories:
            if (
                memory.metadata
                and memory.metadata.get("field") == key
                and memory.conversation_id == name
            ):
                return memory.content
        return None

    async def exists(self, *keys: str) -> int:
        """Redis EXISTS command compatibility."""
        count = 0
        for key in keys:
            if await self.get(key) is not None:
                count += 1
        return count

    async def delete(self, *keys: str) -> int:
        """Redis DELETE command compatibility - returns count of deleted keys."""
        # Note: PersonalMemorySystem doesn't have delete, so we'll track this
        logger.warning(
            f"DELETE operation requested for keys: {keys}. Not implemented in SQLite backend."
        )
        return 0

    def clear_all_memories(self) -> int:
        """Clear all memories - delegates to underlying PersonalMemorySystem."""
        logger.info(
            "🧹 RedisCompatMemorySystem: Delegating clear_all_memories to underlying SQLite system"
        )
        return self.memory.clear_all_memories()

    async def expire(self, key: str, seconds: int) -> bool:
        """Redis EXPIRE command compatibility."""
        # SQLite doesn't have TTL, so we'll ignore this
        logger.debug(f"EXPIRE operation ignored for key: {key}")
        return True

    async def publish(self, channel: str, message: str) -> int:
        """Redis PUBLISH command compatibility."""
        # Since we don't have real pub/sub, we'll call handlers directly
        handlers = self._pub_sub_handlers.get(channel, [])
        for handler in handlers:
            try:
                await handler(channel, message)
            except Exception as e:
                logger.error(f"Error in pub/sub handler: {e}")
        return len(handlers)

    def subscribe_handler(self, channel: str, handler):
        """Register a handler for a channel (replaces Redis pub/sub)."""
        if channel not in self._pub_sub_handlers:
            self._pub_sub_handlers[channel] = []
        self._pub_sub_handlers[channel].append(handler)

    async def zadd(self, key: str, mapping: dict) -> int:
        """Redis ZADD command compatibility - sorted sets."""
        # Store as memories with importance as the score
        count = 0
        for member, score in mapping.items():
            await self.memory.add_memory(
                content=member,
                conversation_id=key,
                importance=float(score) / 100.0,  # Normalize score to 0-1
                metadata={"type": "sorted_set"},
            )
            count += 1
        return count

    async def zrange(
        self, key: str, start: int, stop: int, withscores: bool = False
    ) -> list[str | tuple]:
        """Redis ZRANGE command compatibility."""
        memories = await self.memory.get_relevant_memories(
            query=key, limit=stop - start + 1 if stop >= 0 else 100
        )

        # Filter for sorted set entries
        sorted_memories = [
            m for m in memories if m.metadata and m.metadata.get("type") == "sorted_set"
        ]

        # Sort by importance (score)
        sorted_memories.sort(key=lambda m: m.importance)

        if withscores:
            return [(m.content, m.importance * 100) for m in sorted_memories[start : stop + 1]]
        else:
            return [m.content for m in sorted_memories[start : stop + 1]]

    # Convenience methods for common patterns
    async def get_conversation_history(self, session_id: str) -> list[dict]:
        """Get conversation history in the format expected by the old system."""
        memories = await self.memory.get_conversation_context(session_id, max_messages=100)

        history = []
        for memory in memories:
            # Reconstruct the expected format
            entry = {
                "content": memory.content,
                "timestamp": memory.timestamp.isoformat(),
                "metadata": memory.metadata or {},
            }

            # Determine role from metadata or content pattern
            if memory.metadata and "role" in memory.metadata:
                entry["role"] = memory.metadata["role"]
            elif memory.content.startswith("User:"):
                entry["role"] = "user"
            elif memory.content.startswith("Assistant:"):
                entry["role"] = "assistant"
            else:
                entry["role"] = "system"

            history.append(entry)

        return history


# Example usage for migration
async def demo_compatibility():
    """Demo showing how the compatibility layer works."""
    # Create the underlying personal memory system
    personal_memory = PersonalMemorySystem("test_compat.db")

    # Wrap it with Redis compatibility
    redis_compat = RedisCompatMemorySystem(personal_memory)

    # Now you can use Redis-like commands
    await redis_compat.set("user:1:name", "Alice")
    name = await redis_compat.get("user:1:name")
    print(f"Retrieved: {name}")

    # List operations
    await redis_compat.lpush("conversation:123", "Hello!", "How are you?")
    messages = await redis_compat.lrange("conversation:123", 0, -1)
    print(f"Messages: {messages}")

    # Hash operations
    await redis_compat.hset("user:1", "email", "alice@example.com")
    email = await redis_compat.hget("user:1", "email")
    print(f"Email: {email}")


if __name__ == "__main__":
    asyncio.run(demo_compatibility())
