"""Retrieval Engine.

================

Hybrid search implementation combining vector similarity and metadata filtering.
Implements two-stage retrieval with re-ranking for optimal performance.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import redis.asyncio as redis
import yaml
from redis.commands.search.query import Query

from ..schemas import MemoryCircle, MemorySearchResult, ShortTermMemory
from ..services.embedding_service import EmbeddingService
from ..services.memory_api import MemoryAPI
from ..utils import get_metrics

logger = logging.getLogger(__name__)

# Constants for magic values
HYBRID_THRESHOLD = 0.5  # Threshold for hybrid match type classification


class RetrievalEngine:
    """Advanced memory retrieval with hybrid search."""

    def __init__(
        self,
        redis_client: redis.Redis,
        memory_api: MemoryAPI,
        embedding_service: EmbeddingService,
        config_path: str = "config/redis.yaml",
    ):
        """Initialize retrieval engine.

        Args:
            redis_client: Async Redis client
            memory_api: Memory API instance
            embedding_service: Embedding service instance
            config_path: Path to configuration
        """
        self.redis = redis_client
        self.memory_api = memory_api
        self.embedding_service = embedding_service
        self.config = self._load_config(config_path)
        self.metrics = get_metrics()

        # Load circle definitions for scoring
        self.circles = self._load_circles()

        # Retrieval parameters
        self.vector_weight = 0.7
        self.metadata_weight = 0.3
        self.recency_decay = 0.1

    def _load_config(self, config_path: str) -> dict:
        """Load configuration."""
        config_file = Path(__file__).parent.parent / config_path
        with config_file.open() as f:
            return yaml.safe_load(f)

    def _load_circles(self) -> dict[str, MemoryCircle]:
        """Load memory circle definitions."""
        circles_file = Path(__file__).parent.parent / "config/circles.yaml"
        with circles_file.open() as f:
            circles_config = yaml.safe_load(f)

        circles = {}
        for name, data in circles_config["circles"].items():
            circles[name] = MemoryCircle(name=name, **data)
        return circles

    async def search(
        self,
        query: str,
        limit: int = 10,
        memory_types: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> list[MemorySearchResult]:
        """Perform hybrid search across memory types.

        Args:
            query: Search query
            limit: Maximum results to return
            memory_types: Types to search ["stm", "ltm"]
            filters: Additional filters (tags, circles, etc.)
            include_embeddings: Whether to include embeddings in results

        Returns:
            List of search results ranked by relevance
        """
        if memory_types is None:
            memory_types = ["stm", "ltm"]
        async with self.metrics.track_operation("memory_retrieval"):
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding(query)

            # Perform searches in parallel
            search_tasks = []

            if "stm" in memory_types:
                search_tasks.append(self._search_stm(query_embedding, limit * 2, filters))

            if "ltm" in memory_types:
                search_tasks.append(self._search_ltm(query_embedding, limit * 2, filters))

            # Gather results
            all_results = []
            search_results = await asyncio.gather(*search_tasks)

            for results in search_results:
                all_results.extend(results)

            # Re-rank combined results
            ranked_results = await self._rerank_results(all_results, query, query_embedding, limit)

            # Remove embeddings if not requested
            if not include_embeddings:
                for result in ranked_results:
                    result.memory.embedding = None

            return ranked_results

    async def _search_stm(
        self, query_embedding: list[float], limit: int, filters: dict | None = None
    ) -> list[MemorySearchResult]:
        """Search short-term memories."""
        # Vector search
        vector_results = await self.embedding_service.vector_search(
            query_embedding, self.embedding_service.stm_index, limit=limit, filters=filters
        )

        # Convert to MemorySearchResult
        results = []
        for vr in vector_results:
            # Extract memory ID from Redis key
            memory_id = vr["id"].split(":")[-1]
            memory = await self.memory_api.get_stm(memory_id)

            if memory:
                results.append(
                    MemorySearchResult(
                        memory=memory, score=vr["score"], match_type="vector", highlights=[]
                    )
                )

        return results

    async def _search_ltm(
        self, query_embedding: list[float], limit: int, filters: dict | None = None
    ) -> list[MemorySearchResult]:
        """Search long-term memories."""
        # Add retrieval score filter for LTM
        if filters is None:
            filters = {}
        filters["min_score"] = self.config["memory"]["ltm"]["archive_threshold"]

        # Vector search
        vector_results = await self.embedding_service.vector_search(
            query_embedding, self.embedding_service.ltm_index, limit=limit, filters=filters
        )

        # Convert to MemorySearchResult
        results = []
        for vr in vector_results:
            memory_id = vr["id"].split(":")[-1]
            memory = await self.memory_api.get_ltm(memory_id)

            if memory:
                # Apply decay to retrieval score
                circle = self.circles.get(memory.circle)
                if circle:
                    memory.retrieval_score = memory.calculate_decay(circle.decay_rate)

                results.append(
                    MemorySearchResult(
                        memory=memory, score=vr["score"], match_type="vector", highlights=[]
                    )
                )

        return results

    async def _rerank_results(
        self,
        results: list[MemorySearchResult],
        query: str,
        query_embedding: list[float],
        limit: int,
    ) -> list[MemorySearchResult]:
        """Re-rank results using multiple signals."""
        if not results:
            return []

        # Calculate composite scores
        for result in results:
            memory = result.memory

            # Base vector similarity score
            vector_score = result.score

            # Tag match score
            tag_score = self._calculate_tag_score(query, memory.tags)

            # Circle priority score
            circle = self.circles.get(memory.circle)
            circle_priority = circle.priority if circle else 0.5

            # Recency score
            if isinstance(memory, ShortTermMemory):
                age_hours = (time.time() - memory.timestamp) / 3600
                recency_score = np.exp(-self.recency_decay * age_hours)
            else:  # LongTermMemory
                recency_score = memory.retrieval_score

            # Composite score
            result.score = self.vector_weight * vector_score + self.metadata_weight * (
                0.4 * tag_score + 0.3 * circle_priority + 0.3 * recency_score
            )

            # Update match type
            if tag_score > HYBRID_THRESHOLD and vector_score > HYBRID_THRESHOLD:
                result.match_type = "hybrid"
            elif tag_score > vector_score:
                result.match_type = "tag"

            # Add highlights
            result.highlights = self._extract_highlights(query, memory.content)

        # Sort by composite score
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:limit]

    def _calculate_tag_score(self, query: str, tags: list[str]) -> float:
        """Calculate tag relevance score."""
        if not tags:
            return 0.0

        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Check for exact tag matches in query
        matching_tags = 0
        for tag in tags:
            tag_word = tag.lstrip("#").lower()
            if tag_word in query_words or tag_word in query_lower:
                matching_tags += 1

        return matching_tags / len(tags) if tags else 0.0

    def _extract_highlights(self, query: str, content: str, context_words: int = 10) -> list[str]:
        """Extract relevant snippets from content."""
        highlights = []
        query_words = set(query.lower().split())
        content_words = content.split()

        for i, word in enumerate(content_words):
            if word.lower() in query_words:
                # Extract context around the word
                start = max(0, i - context_words)
                end = min(len(content_words), i + context_words + 1)

                snippet = " ".join(content_words[start:end])
                if start > 0:
                    snippet = "..." + snippet
                if end < len(content_words):
                    snippet = snippet + "..."

                highlights.append(snippet)

        return highlights[:3]  # Limit to 3 highlights

    async def find_related_memories(
        self, memory_id: str, memory_type: str = "ltm", limit: int = 5
    ) -> list[MemorySearchResult]:
        """Find memories related to a given memory."""
        # Get the source memory
        if memory_type == "stm":
            memory = await self.memory_api.get_stm(memory_id)
        else:
            memory = await self.memory_api.get_ltm(memory_id)

        if not memory or not memory.embedding:
            return []

        # Search using memory's embedding
        filters = {
            "circle": memory.circle,
            "tags": memory.tags[:3] if memory.tags else [],  # Use top 3 tags
        }

        # Search both STM and LTM for related memories
        results = await self.embedding_service.vector_search(
            memory.embedding,
            self.embedding_service.ltm_index,
            limit=limit + 1,  # +1 to exclude self
            filters=filters,
        )

        # Convert and filter out self
        related = []
        for vr in results:
            if memory_id not in vr["id"]:
                ltm_id = vr["id"].split(":")[-1]
                ltm = await self.memory_api.get_ltm(ltm_id)

                if ltm:
                    related.append(
                        MemorySearchResult(
                            memory=ltm, score=vr["score"], match_type="related", highlights=[]
                        )
                    )

        return related[:limit]

    async def search_by_time_range(
        self,
        start_time: float,
        end_time: float,
        memory_types: list[str] | None = None,
        limit: int = 50,
    ) -> list[MemorySearchResult]:
        """Search memories within a time range."""
        if memory_types is None:
            memory_types = ["stm", "ltm"]
        results = []

        # Build time range query
        if "stm" in memory_types:
            stm_query = Query(f"@timestamp:[{start_time} {end_time}]").paging(0, limit)
            stm_results = await self.redis.ft(self.embedding_service.stm_index).search(stm_query)

            for doc in stm_results.docs:
                memory_id = doc.id.split(":")[-1]
                memory = await self.memory_api.get_stm(memory_id)
                if memory:
                    results.append(
                        MemorySearchResult(
                            memory=memory, score=1.0, match_type="temporal", highlights=[]
                        )
                    )

        if "ltm" in memory_types:
            ltm_query = Query(f"@last_accessed:[{start_time} {end_time}]").paging(0, limit)
            ltm_results = await self.redis.ft(self.embedding_service.ltm_index).search(ltm_query)

            for doc in ltm_results.docs:
                memory_id = doc.id.split(":")[-1]
                memory = await self.memory_api.get_ltm(memory_id)
                if memory:
                    results.append(
                        MemorySearchResult(
                            memory=memory, score=1.0, match_type="temporal", highlights=[]
                        )
                    )

        # Sort by timestamp
        results.sort(
            key=lambda x: (
                x.memory.timestamp if hasattr(x.memory, "timestamp") else x.memory.last_accessed
            ),
            reverse=True,
        )

        return results[:limit]

    async def get_memory_context(
        self, query: str, max_tokens: int = 2000, include_metadata: bool = True
    ) -> str:
        """Get formatted memory context for LLM consumption.

        Args:
            query: Context query
            max_tokens: Approximate token limit
            include_metadata: Whether to include tags and metadata

        Returns:
            Formatted string of relevant memories
        """
        # Search for relevant memories
        results = await self.search(query, limit=20)

        if not results:
            return ""

        # Format memories
        context_parts = ["## Relevant Memories\n"]
        current_tokens = 50  # Header estimate

        for i, result in enumerate(results):
            memory = result.memory

            # Format memory entry
            memory_type = "Recent" if isinstance(memory, ShortTermMemory) else "Established"

            entry = f"\n### {memory_type} Memory {i + 1} (Relevance: {result.score:.2f})\n"
            entry += f"{memory.content}\n"

            if include_metadata and memory.tags:
                entry += f"Tags: {', '.join(memory.tags)}\n"

            # Rough token estimate (4 chars per token)
            entry_tokens = len(entry) // 4

            if current_tokens + entry_tokens > max_tokens:
                break

            context_parts.append(entry)
            current_tokens += entry_tokens

        return "".join(context_parts)


# Factory function
async def create_retrieval_engine(redis_url: str = "redis://localhost:6379") -> RetrievalEngine:
    """Create retrieval engine with dependencies."""
    redis_client = await redis.from_url(redis_url)

    # Create dependencies
    memory_api = MemoryAPI(redis_client)
    embedding_service = EmbeddingService(redis_client)
    await embedding_service.initialize()

    return RetrievalEngine(redis_client, memory_api, embedding_service)
