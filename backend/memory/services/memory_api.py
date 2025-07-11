"""Memory API Service.

==================

Core CRUD operations for the memory system.
Handles memory creation, updates, deletion, and tag extraction.
"""

import json
import logging
from pathlib import Path
from typing import Any

import redis.asyncio as redis
import yaml

from ..schemas import (
    ConsolidationCandidate,
    LongTermMemory,
    MemoryCircle,
    MemorySource,
    ShortTermMemory,
)
from ..utils import get_metrics
from .storage_rules_manager import StorageRulesManager

logger = logging.getLogger(__name__)


class MemoryAPI:
    """Core memory operations API."""

    def __init__(self, redis_client: redis.Redis, config_path: str = "config/redis.yaml"):
        """Initialize Memory API.

        Args:
            redis_client: Async Redis client
            config_path: Path to Redis configuration
        """
        self.redis = redis_client
        self.config = self._load_config(config_path)
        self.circles = self._load_circles()
        self.metrics = get_metrics()
        self.storage_rules = StorageRulesManager()

        # Key prefixes
        self.stm_prefix = "stm:"
        self.ltm_prefix = "ltm:"
        self.consolidation_prefix = "consolidation:"

    def _load_config(self, config_path: str) -> dict:
        """Load Redis configuration."""
        config_file = Path(__file__).parent.parent / config_path
        return yaml.safe_load(config_file.read_text())

    def _load_circles(self) -> dict[str, MemoryCircle]:
        """Load memory circle definitions."""
        circles_file = Path(__file__).parent.parent / "config/circles.yaml"
        circles_config = yaml.safe_load(circles_file.read_text())

        circles = {}
        for name, data in circles_config["circles"].items():
            circles[name] = MemoryCircle(name=name, **data)

        self.selection_rules = circles_config.get("selection_rules", [])
        self.default_circle = circles_config.get("default_circle", "experiences")

        return circles

    async def create_stm(
        self,
        content: str,
        source: MemorySource,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> ShortTermMemory:
        """Create a new short-term memory.

        Args:
            content: Memory content
            source: Source of the memory (user/assistant/system)
            tags: Optional explicit tags
            metadata: Optional metadata

        Returns:
            Created ShortTermMemory object
        """
        async with self.metrics.track_operation("memory_creation"):
            # Extract tags if not provided
            if tags is None:
                tags = self._extract_tags(content)

            # Determine memory circle
            circle = self._determine_circle(content)

            # Create memory object
            memory = ShortTermMemory(
                content=content,
                source=source,
                tags=tags,
                circle=circle,
                metadata=metadata or {},
                ttl_hours=self.config["memory"]["stm"]["ttl_hours"],
            )

            # Enhance importance score using storage rules
            memory.importance_score = self.storage_rules.calculate_enhanced_importance(
                memory.dict()
            )

            # Check capacity and prune if necessary
            await self._check_and_prune_stm_capacity()

            # Store in Redis with TTL
            key = f"{self.stm_prefix}{memory.id}"
            ttl_seconds = memory.ttl_hours * 3600

            await self.redis.hset(key, mapping=memory.to_redis_dict())
            await self.redis.expire(key, ttl_seconds)

            # Update metrics
            self.metrics.increment_counter("stm_count")

            logger.info(f"Created STM {memory.id} in circle '{circle}' with {len(tags)} tags")

            return memory

    async def get_stm(self, memory_id: str) -> ShortTermMemory | None:
        """Retrieve a short-term memory by ID."""
        key = f"{self.stm_prefix}{memory_id}"
        try:
            data = await self.redis.hgetall(key)

            if not data:
                logger.debug(f"🧠 MEMORY DEBUG: STM not found: {memory_id}")
                return None
        except Exception as e:
            logger.error(
                f"🧠 MEMORY DEBUG: ❌ Error retrieving STM {memory_id}: {e}", exc_info=True
            )
            return None

        # Convert byte strings to proper types
        memory_dict = {k.decode(): v.decode() for k, v in data.items()}
        return ShortTermMemory.from_redis_dict(memory_dict)

    async def update_stm(
        self,
        memory_id: str,
        content: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> ShortTermMemory | None:
        """Update an existing STM."""
        memory = await self.get_stm(memory_id)
        if not memory:
            logger.warning(f"🧠 MEMORY DEBUG: Cannot update STM - not found: {memory_id}")
            return None

        if content is not None:
            memory.content = content
        if tags is not None:
            memory.tags = tags
        if metadata is not None:
            memory.metadata.update(metadata)

        # Re-determine circle if content changed
        if content is not None:
            memory.circle = self._determine_circle(memory.content)

        # Update in Redis
        key = f"{self.stm_prefix}{memory_id}"
        await self.redis.hset(key, mapping=memory.to_redis_dict())

        return memory

    async def delete_stm(self, memory_id: str) -> bool:
        """Delete a short-term memory."""
        key = f"{self.stm_prefix}{memory_id}"
        result = await self.redis.delete(key)

        if result:
            self.metrics.increment_counter("stm_count", -1)

        return bool(result)

    async def delete_ltm(self, memory_id: str) -> bool:
        """Delete a long-term memory."""
        key = f"{self.ltm_prefix}{memory_id}"
        result = await self.redis.delete(key)

        if result:
            self.metrics.increment_counter("ltm_count", -1)

        return bool(result)

    async def promote_to_ltm(
        self, stm_ids: list[str], summary: str, user_approved: bool = True
    ) -> LongTermMemory:
        """Promote STMs to a consolidated LTM.

        Args:
            stm_ids: List of STM IDs to consolidate
            summary: Consolidated summary
            user_approved: Whether user approved this consolidation

        Returns:
            Created LongTermMemory object
        """
        async with self.metrics.track_operation("stm_to_ltm_promotion"):
            # Gather all STMs
            stms = []
            all_tags = set()

            for stm_id in stm_ids:
                stm = await self.get_stm(stm_id)
                if stm:
                    stms.append(stm)
                    all_tags.update(stm.tags)

            if not stms:
                raise ValueError("No valid STMs found for promotion")

            # Determine circle based on majority
            circle_counts = {}
            for stm in stms:
                circle_counts[stm.circle] = circle_counts.get(stm.circle, 0) + 1
            circle = max(circle_counts, key=circle_counts.get)

            # Create LTM
            ltm = LongTermMemory(
                content=summary,
                source=MemorySource.SYSTEM,
                tags=list(all_tags),
                circle=circle,
                original_memories=stm_ids,
                user_approved=user_approved,
                consolidation_summary=summary,
            )

            # Enhance importance score using storage rules
            ltm.importance_score = self.storage_rules.calculate_enhanced_importance(ltm.dict())

            # Check capacity and prune if necessary
            await self._check_and_prune_ltm_capacity()

            # Store in Redis (no TTL for LTM)
            key = f"{self.ltm_prefix}{ltm.id}"
            await self.redis.hset(key, mapping=ltm.to_redis_dict())

            # Optionally delete original STMs
            if user_approved:
                for stm_id in stm_ids:
                    await self.delete_stm(stm_id)

            # Update metrics
            self.metrics.increment_counter("ltm_count")

            logger.info(f"Promoted {len(stm_ids)} STMs to LTM {ltm.id}")

            return ltm

    async def get_ltm(self, memory_id: str) -> LongTermMemory | None:
        """Retrieve a long-term memory by ID."""
        key = f"{self.ltm_prefix}{memory_id}"
        data = await self.redis.hgetall(key)

        if not data:
            return None

        # Convert byte strings
        memory_dict = {k.decode(): v.decode() for k, v in data.items()}
        ltm = LongTermMemory.from_redis_dict(memory_dict)

        # Update access tracking
        ltm.update_access()
        await self.redis.hset(key, mapping=ltm.to_redis_dict())

        return ltm

    async def update_ltm_links(
        self, memory_id: str, linked_ids: list[str]
    ) -> LongTermMemory | None:
        """Update links between LTMs."""
        ltm = await self.get_ltm(memory_id)
        if not ltm:
            return None

        ltm.links = linked_ids

        key = f"{self.ltm_prefix}{memory_id}"
        await self.redis.hset(key, mapping=ltm.to_redis_dict())

        return ltm

    async def queue_consolidation_candidate(self, candidate: ConsolidationCandidate):
        """Queue a consolidation candidate for user review."""
        key = f"{self.consolidation_prefix}{candidate.created_at}"

        await self.redis.hset(
            key,
            mapping={
                "memory_ids": json.dumps(candidate.memory_ids),
                "suggested_summary": candidate.suggested_summary,
                "common_tags": json.dumps(candidate.common_tags),
                "circle": candidate.circle,
                "confidence": str(candidate.confidence),
                "created_at": str(candidate.created_at),
            },
        )

        # Set expiry for unreviewed candidates (7 days)
        await self.redis.expire(key, 7 * 24 * 3600)

    async def get_pending_consolidations(self) -> list[ConsolidationCandidate]:
        """Get all pending consolidation candidates."""
        pattern = f"{self.consolidation_prefix}*"
        keys = [key async for key in self.redis.scan_iter(match=pattern)]

        candidates = []
        for key in keys:
            data = await self.redis.hgetall(key)
            if data:
                candidate = ConsolidationCandidate(
                    memory_ids=json.loads(data[b"memory_ids"].decode()),
                    suggested_summary=data[b"suggested_summary"].decode(),
                    common_tags=json.loads(data[b"common_tags"].decode()),
                    circle=data[b"circle"].decode(),
                    confidence=float(data[b"confidence"].decode()),
                    created_at=float(data[b"created_at"].decode()),
                )
                candidates.append(candidate)

        return sorted(candidates, key=lambda x: x.created_at, reverse=True)

    def _extract_tags(self, content: str) -> list[str]:
        """Extract hashtags from content."""
        # Simple hashtag extraction without regex
        hashtags = []
        words = content.split()
        for word in words:
            if word.startswith("#") and len(word) > 1:
                # Take the hashtag part (until first non-word character)
                hashtag = "#"
                for char in word[1:]:
                    if char.isalnum() or char == "_":
                        hashtag += char
                    else:
                        break
                if len(hashtag) > 1:  # Ensure it's not just "#"
                    hashtags.append(hashtag)
        return list(set(hashtags))  # Remove duplicates

    def _determine_circle(self, content: str) -> str:
        """Determine which memory circle this content belongs to."""
        content_lower = content.lower()

        # Check selection rules in order
        for rule in self.selection_rules:
            pattern = rule["pattern"]
            circle = rule["circle"]

            # Simple pattern matching without regex
            if pattern in content_lower:
                return circle

        # Default circle
        return self.default_circle

    async def get_memory_stats(self) -> dict[str, Any]:
        """Get memory system statistics."""
        # Count STMs
        stm_count = 0
        async for _ in self.redis.scan_iter(match=f"{self.stm_prefix}*"):
            stm_count += 1

        # Count LTMs
        ltm_count = 0
        async for _ in self.redis.scan_iter(match=f"{self.ltm_prefix}*"):
            ltm_count += 1

        # Count pending consolidations
        consolidation_count = 0
        async for _ in self.redis.scan_iter(match=f"{self.consolidation_prefix}*"):
            consolidation_count += 1

        # Circle distribution
        circle_dist = dict.fromkeys(self.circles, 0)

        # Sample some memories for circle distribution
        sample_size = min(100, stm_count + ltm_count)
        sampled = 0

        async for key in self.redis.scan_iter(match=f"{self.stm_prefix}*"):
            if sampled >= sample_size:
                break
            data = await self.redis.hget(key, "circle")
            if data:
                circle = data.decode()
                if circle in circle_dist:
                    circle_dist[circle] += 1
            sampled += 1

        return {
            "stm_count": stm_count,
            "ltm_count": ltm_count,
            "total_memories": stm_count + ltm_count,
            "pending_consolidations": consolidation_count,
            "circle_distribution": circle_dist,
            "config": {
                "stm_ttl_hours": self.config["memory"]["stm"]["ttl_hours"],
                "stm_max_size": self.config["memory"]["stm"]["max_size"],
                "ltm_max_size": self.config["memory"]["ltm"]["max_size"],
            },
        }

    async def _check_and_prune_stm_capacity(self):
        """Check STM capacity and prune old memories if necessary."""
        capacity_limits = self.storage_rules.get_capacity_limits()
        stm_max = capacity_limits["stm_max"]

        # Count current STM entries
        stm_count = 0
        async for _ in self.redis.scan_iter(match=f"{self.stm_prefix}*"):
            stm_count += 1

        if stm_count >= stm_max:
            # Get memories to prune
            memories_to_prune = await self._get_prunable_stm_memories(stm_count - stm_max + 1)

            for memory_id in memories_to_prune:
                await self.delete_stm(memory_id)
                logger.info(f"Pruned STM {memory_id} due to capacity limit")

    async def _get_prunable_stm_memories(self, count_to_prune: int) -> list[str]:
        """Get list of STM memory IDs that can be pruned based on storage rules."""
        pruneable_memories = []

        async for key in self.redis.scan_iter(match=f"{self.stm_prefix}*"):
            memory_id = key.decode().replace(self.stm_prefix, "")
            memory = await self.get_stm(memory_id)

            if memory and self.storage_rules.should_prune_memory(memory.dict()):
                # Calculate enhanced importance for ranking
                enhanced_importance = self.storage_rules.calculate_enhanced_importance(
                    memory.dict()
                )
                pruneable_memories.append((memory_id, enhanced_importance))

        # Sort by importance (lowest first) and return the requested count
        pruneable_memories.sort(key=lambda x: x[1])
        return [memory_id for memory_id, _ in pruneable_memories[:count_to_prune]]

    async def _check_and_prune_ltm_capacity(self):
        """Check LTM capacity and prune old memories if necessary."""
        capacity_limits = self.storage_rules.get_capacity_limits()
        ltm_max = capacity_limits["ltm_max"]

        # Count current LTM entries
        ltm_count = 0
        async for _ in self.redis.scan_iter(match=f"{self.ltm_prefix}*"):
            ltm_count += 1

        if ltm_count >= ltm_max:
            # Get memories to prune
            memories_to_prune = await self._get_prunable_ltm_memories(ltm_count - ltm_max + 1)

            for memory_id in memories_to_prune:
                # Check if user approval is required
                memory = await self.get_ltm(memory_id)
                if memory and self.storage_rules.requires_user_approval(memory.importance_score):
                    logger.warning(
                        f"LTM {memory_id} requires user approval for deletion (importance: {memory.importance_score})"
                    )
                    continue

                await self.delete_ltm(memory_id)
                logger.info(f"Pruned LTM {memory_id} due to capacity limit")

    async def _get_prunable_ltm_memories(self, count_to_prune: int) -> list[str]:
        """Get list of LTM memory IDs that can be pruned based on storage rules."""
        pruneable_memories = []

        async for key in self.redis.scan_iter(match=f"{self.ltm_prefix}*"):
            memory_id = key.decode().replace(self.ltm_prefix, "")
            memory = await self.get_ltm(memory_id)

            if memory and self.storage_rules.should_prune_memory(memory.dict()):
                # Calculate enhanced importance for ranking
                enhanced_importance = self.storage_rules.calculate_enhanced_importance(
                    memory.dict(), memory.access_count if hasattr(memory, "access_count") else 0
                )
                pruneable_memories.append((memory_id, enhanced_importance))

        # Sort by importance (lowest first) and return the requested count
        pruneable_memories.sort(key=lambda x: x[1])
        return [memory_id for memory_id, _ in pruneable_memories[:count_to_prune]]


async def create_memory_api(redis_url: str = "redis://localhost:6379") -> MemoryAPI:
    """Factory function to create MemoryAPI with Redis connection."""
    redis_client = await redis.from_url(redis_url)
    return MemoryAPI(redis_client)
