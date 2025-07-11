"""Consolidation Worker.

====================

Handles memory consolidation from STM to LTM.
Runs on event-driven and periodic triggers with user-in-the-loop approval.
"""

import asyncio
import logging
import time
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import numpy as np
import redis.asyncio as redis
from sklearn.cluster import DBSCAN

from ..schemas import ConsolidationCandidate, LongTermMemory, ShortTermMemory
from ..services.embedding_service import EmbeddingService
from ..services.memory_api import MemoryAPI
from ..services.retrieval_engine import RetrievalEngine
from ..utils import get_metrics

logger = logging.getLogger(__name__)

# Constants
MIN_CLUSTER_SIZE_FOR_COHERENCE = 2
MAX_SUMMARY_MEMORIES = 5


class ConsolidationWorker:
    """Worker for consolidating short-term memories into long-term."""

    def __init__(
        self,
        redis_client: redis.Redis,
        memory_api: MemoryAPI,
        embedding_service: EmbeddingService,
        retrieval_engine: RetrievalEngine,
        llm_client: Any | None = None,
    ):
        """Initialize consolidation worker.

        Args:
            redis_client: Async Redis client
            memory_api: Memory API instance
            embedding_service: Embedding service instance
            retrieval_engine: Retrieval engine instance
            llm_client: Optional LLM client for summarization
        """
        self.redis = redis_client
        self.memory_api = memory_api
        self.embedding_service = embedding_service
        self.retrieval_engine = retrieval_engine
        self.llm_client = llm_client
        self.metrics = get_metrics()

        # Consolidation parameters
        self.min_cluster_size = 3
        self.similarity_threshold = 0.85  # For DBSCAN eps
        self.inactivity_threshold = 1800  # 30 minutes
        self.consolidation_interval = 86400  # 24 hours

        # Tracking
        self.last_activity_time = time.time()
        self.last_consolidation_time = time.time()
        self.is_running = False

    async def start(self):
        """Start the consolidation worker."""
        self.is_running = True

        # Start background tasks
        self._activity_task = asyncio.create_task(self._activity_monitor())
        self._consolidation_task = asyncio.create_task(self._periodic_consolidation())

        logger.info("Consolidation worker started")

    async def stop(self):
        """Stop the consolidation worker."""
        self.is_running = False

        # Cancel background tasks
        if hasattr(self, "_activity_task"):
            self._activity_task.cancel()
        if hasattr(self, "_consolidation_task"):
            self._consolidation_task.cancel()

        logger.info("Consolidation worker stopped")

    async def _activity_monitor(self):
        """Monitor for inactivity and trigger consolidation."""
        while self.is_running:
            try:
                current_time = time.time()
                time_since_activity = current_time - self.last_activity_time

                if time_since_activity > self.inactivity_threshold:
                    logger.info("Inactivity detected, triggering consolidation")
                    await self.consolidate_memories("inactivity")
                    self.last_activity_time = current_time

                await asyncio.sleep(60)  # Check every minute

            except Exception:
                logger.exception("Error in activity monitor")
                await asyncio.sleep(300)  # Back off on error

    async def _periodic_consolidation(self):
        """Run consolidation periodically."""
        while self.is_running:
            try:
                current_time = time.time()
                time_since_consolidation = current_time - self.last_consolidation_time

                if time_since_consolidation > self.consolidation_interval:
                    logger.info("Running periodic consolidation")
                    await self.consolidate_memories("periodic")
                    self.last_consolidation_time = current_time

                # Sleep until next check
                await asyncio.sleep(3600)  # Check every hour

            except Exception:
                logger.exception("Error in periodic consolidation")
                await asyncio.sleep(3600)

    def update_activity(self):
        """Update last activity time (called by memory creation)."""
        self.last_activity_time = time.time()

    async def consolidate_memories(self, trigger: str = "manual") -> int:
        """Main consolidation process.

        Args:
            trigger: What triggered consolidation (manual/inactivity/periodic)

        Returns:
            Number of consolidation candidates created
        """
        async with self.metrics.track_operation("memory_consolidation"):
            logger.info(f"Starting memory consolidation (trigger: {trigger})")

            # Get all STMs
            stm_memories = await self._get_all_stms()

            if len(stm_memories) < self.min_cluster_size:
                logger.info(f"Not enough STMs for consolidation: {len(stm_memories)}")
                return 0

            # Group by circles
            circle_groups = self._group_by_circle(stm_memories)

            total_candidates = 0

            # Process each circle separately
            for circle, memories in circle_groups.items():
                if len(memories) >= self.min_cluster_size:
                    candidates = await self._process_circle_memories(circle, memories)
                    total_candidates += len(candidates)

            logger.info(f"Consolidation complete. Created {total_candidates} candidates")
            return total_candidates

    async def _get_all_stms(self) -> list[ShortTermMemory]:
        """Retrieve all short-term memories."""
        memories = []

        async for key in self.redis.scan_iter(match="stm:*"):
            key_str = key.decode() if isinstance(key, bytes) else key
            memory_id = key_str.split(":")[-1]
            memory = await self.memory_api.get_stm(memory_id)
            if memory:
                memories.append(memory)

        return memories

    def _group_by_circle(self, memories: list[ShortTermMemory]) -> dict[str, list[ShortTermMemory]]:
        """Group memories by circle."""
        groups = defaultdict(list)
        for memory in memories:
            groups[memory.circle].append(memory)
        return dict(groups)

    async def _process_circle_memories(
        self, circle: str, memories: list[ShortTermMemory]
    ) -> list[ConsolidationCandidate]:
        """Process memories from a single circle."""
        # Ensure all memories have embeddings
        memories_with_embeddings = []
        for memory in memories:
            if not memory.embedding:
                memory.embedding = await self.embedding_service.generate_embedding(memory.content)
            memories_with_embeddings.append(memory)

        # Cluster similar memories
        clusters = self._cluster_memories(memories_with_embeddings)

        # Create consolidation candidates
        candidates = []
        for cluster_indices in clusters:
            if len(cluster_indices) >= self.min_cluster_size:
                cluster_memories = [memories_with_embeddings[i] for i in cluster_indices]
                candidate = await self._create_consolidation_candidate(circle, cluster_memories)
                if candidate:
                    candidates.append(candidate)
                    await self.memory_api.queue_consolidation_candidate(candidate)

        return candidates

    def _cluster_memories(self, memories: list[ShortTermMemory]) -> list[list[int]]:
        """Cluster memories based on embedding similarity."""
        if not memories:
            return []

        # Extract embeddings
        embeddings = np.array([m.embedding for m in memories])

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms

        # Use DBSCAN for clustering
        # Convert similarity threshold to distance for DBSCAN
        eps = 2 * (1 - self.similarity_threshold)  # Euclidean distance for normalized vectors

        clustering = DBSCAN(eps=eps, min_samples=self.min_cluster_size, metric="euclidean").fit(
            normalized_embeddings
        )

        # Group by cluster label
        clusters = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            if label != -1:  # Skip noise points
                clusters[label].append(idx)

        return list(clusters.values())

    async def _create_consolidation_candidate(
        self, circle: str, memories: list[ShortTermMemory]
    ) -> ConsolidationCandidate | None:
        """Create a consolidation candidate from clustered memories."""
        if not memories:
            return None

        # Extract common tags
        all_tags = []
        for memory in memories:
            all_tags.extend(memory.tags)

        # Count tag frequency
        tag_counts = defaultdict(int)
        for tag in all_tags:
            tag_counts[tag] += 1

        # Common tags appear in at least half the memories
        common_tags = [tag for tag, count in tag_counts.items() if count >= len(memories) / 2]

        # Generate summary
        if self.llm_client:
            summary = await self._generate_llm_summary(memories)
        else:
            summary = self._generate_simple_summary(memories, common_tags)

        # Calculate confidence based on similarity
        avg_similarity = self._calculate_cluster_coherence(memories)

        return ConsolidationCandidate(
            memory_ids=[m.id for m in memories],
            suggested_summary=summary,
            common_tags=common_tags,
            circle=circle,
            confidence=avg_similarity,
        )

    def _calculate_cluster_coherence(self, memories: list[ShortTermMemory]) -> float:
        """Calculate average pairwise similarity within cluster."""
        if len(memories) < MIN_CLUSTER_SIZE_FOR_COHERENCE:
            return 1.0

        embeddings = np.array([m.embedding for m in memories])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                similarity = np.dot(normalized[i], normalized[j])
                similarities.append(similarity)

        return float(np.mean(similarities))

    async def _generate_llm_summary(self, memories: list[ShortTermMemory]) -> str:
        """Generate summary using LLM."""
        # Format memories for LLM
        memory_texts = []
        for i, memory in enumerate(memories):
            timestamp = datetime.fromtimestamp(memory.timestamp, tz=UTC).strftime("%Y-%m-%d %H:%M")
            memory_texts.append(f"{i + 1}. [{timestamp}] {memory.content}")

        # TODO: Implement LLM summary generation

        # For now, fall back to simple summary
        return self._generate_simple_summary(memories, [])

    def _generate_simple_summary(
        self, memories: list[ShortTermMemory], common_tags: list[str]
    ) -> str:
        """Generate simple summary without LLM."""
        # Sort by timestamp
        sorted_memories = sorted(memories, key=lambda m: m.timestamp)

        # Create summary parts
        parts = []

        # Add topic if common tags exist
        if common_tags:
            topics = [tag.lstrip("#") for tag in common_tags[:3]]
            parts.append(f"Memories about {', '.join(topics)}:")

        # Add key points from each memory (first sentence)
        for memory in sorted_memories[:MAX_SUMMARY_MEMORIES]:
            first_sentence = memory.content.split(".")[0].strip()
            if first_sentence:
                parts.append(f"- {first_sentence}")

        # Add count if more memories
        if len(memories) > MAX_SUMMARY_MEMORIES:
            parts.append(f"(and {len(memories) - MAX_SUMMARY_MEMORIES} more related memories)")

        return "\n".join(parts)

    async def apply_user_decision(
        self, candidate_timestamp: float, approved: bool, edited_summary: str | None = None
    ) -> LongTermMemory | None:
        """Apply user's decision on a consolidation candidate.

        Args:
            candidate_timestamp: Timestamp identifying the candidate
            approved: Whether user approved consolidation
            edited_summary: Optional edited summary from user

        Returns:
            Created LongTermMemory if approved, None otherwise
        """
        # Retrieve candidate
        key = f"consolidation:{candidate_timestamp}"
        data = await self.redis.hgetall(key)

        if not data:
            logger.error(f"Consolidation candidate not found: {candidate_timestamp}")
            return None

        # Parse candidate data
        import json

        candidate = ConsolidationCandidate(
            memory_ids=json.loads(data[b"memory_ids"].decode()),
            suggested_summary=data[b"suggested_summary"].decode(),
            common_tags=json.loads(data[b"common_tags"].decode()),
            circle=data[b"circle"].decode(),
            confidence=float(data[b"confidence"].decode()),
            created_at=float(data[b"created_at"].decode()),
        )

        # Delete candidate from queue
        await self.redis.delete(key)

        if not approved:
            logger.info("User rejected consolidation candidate")
            return None

        # Use edited summary if provided
        summary = edited_summary or candidate.suggested_summary

        # Create LTM
        ltm = await self.memory_api.promote_to_ltm(
            candidate.memory_ids, summary, user_approved=True
        )

        logger.info(f"Created LTM {ltm.id} from {len(candidate.memory_ids)} STMs")

        return ltm

    async def get_consolidation_stats(self) -> dict[str, Any]:
        """Get consolidation statistics."""
        # Count pending candidates
        pending_count = 0
        async for _ in self.redis.scan_iter(match="consolidation:*"):
            pending_count += 1

        # Get memory stats
        memory_stats = await self.memory_api.get_memory_stats()

        return {
            "pending_candidates": pending_count,
            "stm_count": memory_stats["stm_count"],
            "ltm_count": memory_stats["ltm_count"],
            "last_consolidation": datetime.fromtimestamp(
                self.last_consolidation_time, tz=UTC
            ).isoformat(),
            "consolidation_params": {
                "min_cluster_size": self.min_cluster_size,
                "similarity_threshold": self.similarity_threshold,
                "inactivity_threshold_minutes": self.inactivity_threshold / 60,
                "periodic_interval_hours": self.consolidation_interval / 3600,
            },
        }


# Factory function
async def create_consolidation_worker(
    redis_url: str = "redis://localhost:6379", llm_client: Any | None = None
) -> ConsolidationWorker:
    """Create consolidation worker with dependencies."""
    redis_client = await redis.from_url(redis_url)

    # Create dependencies
    memory_api = MemoryAPI(redis_client)
    embedding_service = EmbeddingService(redis_client)
    await embedding_service.initialize()

    retrieval_engine = RetrievalEngine(redis_client, memory_api, embedding_service)

    return ConsolidationWorker(
        redis_client, memory_api, embedding_service, retrieval_engine, llm_client
    )
