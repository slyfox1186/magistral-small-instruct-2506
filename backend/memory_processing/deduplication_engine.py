"""Deduplication Engine for Memory Processing.

Handles semantic similarity detection and intelligent merging of memories
to prevent duplicate storage while preserving important information.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

from .config import MemoryProcessingConfig
from .memory_extractor import ExtractedMemory
from .utils import calculate_text_similarity

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationResult:
    """Results of deduplication process."""

    original_count: int
    deduplicated_count: int
    merged_count: int
    filtered_count: int
    processing_time: float
    similarity_matches: list[tuple[str, str, float]]


class DeduplicationEngine:
    """Advanced deduplication engine with semantic similarity detection."""

    def __init__(self, memory_system, config: MemoryProcessingConfig):
        """Initialize the deduplication engine.

        Args:
            memory_system: Memory system instance
            config: Memory processing configuration
        """
        self.memory_system = memory_system
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Similarity thresholds for different actions
        self.similarity_thresholds = {
            "merge": config.merge_threshold,  # Very high similarity -> merge
            "filter": config.similarity_threshold,  # High similarity -> filter duplicate
            "similar": 0.7,  # Moderate similarity -> note as similar
        }

    async def deduplicate_memories(
        self, new_memories: list[ExtractedMemory], session_id: str
    ) -> tuple[list[ExtractedMemory], DeduplicationResult]:
        """Deduplicate memories against existing stored memories.

        Args:
            new_memories: List of newly extracted memories
            session_id: Session identifier

        Returns:
            Tuple of (deduplicated_memories, deduplication_result)
        """
        if not new_memories:
            return [], DeduplicationResult(0, 0, 0, 0, 0.0, [])

        start_time = time.time()
        original_count = len(new_memories)

        try:
            # Get existing memories for comparison
            existing_memories = await self._get_existing_memories(session_id)

            # Perform deduplication
            deduplicated_memories = []
            merged_count = 0
            filtered_count = 0
            similarity_matches = []

            for new_memory in new_memories:
                should_keep, merge_info = await self._check_memory_uniqueness(
                    new_memory, existing_memories, similarity_matches
                )

                if should_keep:
                    # Check if this memory should be merged with an existing one
                    if merge_info and merge_info["action"] == "merge":
                        await self._merge_memories(new_memory, merge_info["existing_memory"])
                        merged_count += 1
                    else:
                        deduplicated_memories.append(new_memory)
                else:
                    filtered_count += 1

                    # Log the filtering for debugging
                    if self.config.enable_detailed_logging:
                        self.logger.debug(
                            f"Filtered duplicate memory: {new_memory.content[:50]}..."
                        )

            processing_time = time.time() - start_time

            result = DeduplicationResult(
                original_count=original_count,
                deduplicated_count=len(deduplicated_memories),
                merged_count=merged_count,
                filtered_count=filtered_count,
                processing_time=processing_time,
                similarity_matches=similarity_matches,
            )

            if self.config.enable_detailed_logging:
                self.logger.debug(
                    f"Deduplication complete for session {session_id}: "
                    f"original={original_count}, kept={len(deduplicated_memories)}, "
                    f"merged={merged_count}, filtered={filtered_count}"
                )

        except Exception:
            self.logger.exception("Error in deduplication")
            processing_time = time.time() - start_time
            return new_memories, DeduplicationResult(
                original_count, len(new_memories), 0, 0, processing_time, []
            )
        else:
            return deduplicated_memories, result

    async def _get_existing_memories(self, session_id: str) -> list[dict[str, Any]]:
        """Get existing memories for comparison.

        Args:
            session_id: Session identifier

        Returns:
            List of existing memory dictionaries
        """
        try:
            # Get recent memories for comparison (last 50 memories)
            recent_memories = await self.memory_system.get_relevant_memories("", limit=50)

            # Get core memories (they persist longer) for this conversation
            core_memories = await self.memory_system.get_all_core_memories(session_id)

            # Convert core memories dict to list format for consistency
            core_memory_list = []
            if isinstance(core_memories, dict):
                core_memory_list.extend(
                    [
                        {"content": value, "category": "core", "memory_type": "core"}
                        for value in core_memories.values()
                    ]
                )
            else:
                core_memory_list = core_memories

            # Combine and deduplicate
            all_memories = recent_memories + core_memory_list

            # Remove duplicates based on content
            unique_memories = []
            seen_content = set()

            for memory in all_memories:
                # Handle both Memory objects and dictionaries
                if hasattr(memory, "content"):
                    content = memory.content
                else:
                    content = memory.get("content", "")

                if content and content not in seen_content:
                    # Convert Memory objects to dictionaries for consistency
                    if hasattr(memory, "content"):
                        memory_dict = {
                            "content": memory.content,
                            "importance": memory.importance,
                            "conversation_id": memory.conversation_id,
                            "id": memory.id,
                            "timestamp": memory.timestamp.isoformat()
                            if hasattr(memory.timestamp, "isoformat")
                            else str(memory.timestamp),
                            "metadata": memory.metadata,
                        }
                        unique_memories.append(memory_dict)
                    else:
                        unique_memories.append(memory)
                    seen_content.add(content)

        except Exception:
            self.logger.exception("Error getting existing memories")
            return []
        else:
            return unique_memories

    async def _check_memory_uniqueness(
        self,
        new_memory: ExtractedMemory,
        existing_memories: list[dict[str, Any]],
        similarity_matches: list[tuple[str, str, float]],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Check if a memory is unique or should be merged/filtered.

        Args:
            new_memory: New memory to check
            existing_memories: List of existing memories
            similarity_matches: List to record similarity matches

        Returns:
            Tuple of (should_keep, merge_info)
        """
        if not existing_memories:
            return True, None

        best_match = None
        best_similarity = 0.0

        for existing_memory in existing_memories:
            similarity = self._calculate_memory_similarity(new_memory, existing_memory)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = existing_memory

        # Record similarity match if significant
        if best_similarity > self.similarity_thresholds["similar"]:
            similarity_matches.append(
                (new_memory.content[:50], best_match.get("content", "")[:50], best_similarity)
            )

        # Determine action based on similarity
        if best_similarity >= self.similarity_thresholds["merge"]:
            # Very high similarity -> merge
            return True, {
                "action": "merge",
                "existing_memory": best_match,
                "similarity": best_similarity,
            }
        elif best_similarity >= self.similarity_thresholds["filter"]:
            # High similarity -> filter out as duplicate
            return False, {
                "action": "filter",
                "existing_memory": best_match,
                "similarity": best_similarity,
            }
        else:
            # Unique enough -> keep
            return True, None

    def _calculate_memory_similarity(
        self, new_memory: ExtractedMemory, existing_memory: dict[str, Any]
    ) -> float:
        """Calculate similarity between new and existing memory.

        Args:
            new_memory: New memory to compare
            existing_memory: Existing memory dictionary

        Returns:
            Similarity score between 0 and 1
        """
        # Get content for comparison
        new_content = new_memory.content
        # Handle both Memory objects and dictionaries
        if hasattr(existing_memory, "content"):
            existing_content = existing_memory.content
        else:
            existing_content = existing_memory.get("content", "")

        if not new_content or not existing_content:
            return 0.0

        # Calculate base text similarity
        base_similarity = calculate_text_similarity(new_content, existing_content)

        # Apply category-specific adjustments
        new_category = new_memory.category
        # Handle both Memory objects and dictionaries
        if hasattr(existing_memory, "category"):
            existing_category = existing_memory.category
        else:
            existing_category = existing_memory.get("category", "")

        # Boost similarity if categories match
        if new_category == existing_category:
            base_similarity *= 1.1

        # Boost similarity for core memories (more strict deduplication)
        existing_memory_type = (
            getattr(existing_memory, "memory_type", existing_memory.get("memory_type", ""))
            if hasattr(existing_memory, "memory_type")
            else existing_memory.get("memory_type", "")
        )
        if new_memory.memory_type == "core" and existing_memory_type == "core":
            base_similarity *= 1.2

        # Consider entity overlap
        entity_similarity = self._calculate_entity_similarity(new_memory, existing_memory)

        # Weighted combination
        final_similarity = (base_similarity * 0.7) + (entity_similarity * 0.3)

        return min(1.0, final_similarity)

    def _calculate_entity_similarity(
        self, new_memory: ExtractedMemory, existing_memory: dict[str, Any]
    ) -> float:
        """Calculate similarity based on entity overlap.

        Args:
            new_memory: New memory to compare
            existing_memory: Existing memory dictionary

        Returns:
            Entity similarity score between 0 and 1
        """
        try:
            new_entities = new_memory.entities
            # Handle both Memory objects and dictionaries
            if hasattr(existing_memory, "entities"):
                existing_entities = existing_memory.entities
            else:
                existing_entities = existing_memory.get("entities", {})

            if not new_entities or not existing_entities:
                return 0.0

            # Calculate overlap for each entity type
            total_overlap = 0
            total_entities = 0

            for entity_type in set(new_entities.keys()) | set(existing_entities.keys()):
                new_set = set(new_entities.get(entity_type, []))
                existing_set = set(existing_entities.get(entity_type, []))

                if new_set or existing_set:
                    overlap = len(new_set & existing_set)
                    total = len(new_set | existing_set)
                    if total > 0:
                        total_overlap += overlap
                        total_entities += total

            if total_entities > 0:
                return total_overlap / total_entities
            else:
                return 0.0

        except Exception:
            self.logger.exception("Error calculating entity similarity")
            return 0.0

    async def _merge_memories(
        self, new_memory: ExtractedMemory, existing_memory: dict[str, Any]
    ) -> None:
        """Merge new memory with existing memory.

        Args:
            new_memory: New memory to merge
            existing_memory: Existing memory to merge with
        """
        try:
            # Create merged content
            merged_content = self._create_merged_content(new_memory, existing_memory)

            # Handle both Memory objects and dictionaries
            existing_importance = (
                getattr(existing_memory, "importance", existing_memory.get("importance", 0))
                if hasattr(existing_memory, "importance")
                else existing_memory.get("importance", 0)
            )
            existing_confidence = (
                getattr(existing_memory, "confidence", existing_memory.get("confidence", 0))
                if hasattr(existing_memory, "confidence")
                else existing_memory.get("confidence", 0)
            )

            # Update existing memory with merged content
            {
                "content": merged_content,
                "importance": max(new_memory.importance_score, existing_importance),
                "confidence": max(new_memory.confidence, existing_confidence),
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Note: update_memory method doesn't exist in PersonalMemorySystem
            # For now, we'll just log the merge attempt
            # In a full implementation, we'd need to add an update method
            memory_id = existing_memory.get("id", "unknown")

            if self.config.enable_detailed_logging:
                self.logger.debug(f"Would merge memory {memory_id} with new content")
                self.logger.debug(f"Merged content: {merged_content[:100]}...")

        except Exception:
            self.logger.exception("Error merging memories")

    def _create_merged_content(
        self, new_memory: ExtractedMemory, existing_memory: dict[str, Any]
    ) -> str:
        """Create merged content from two memories.

        Args:
            new_memory: New memory
            existing_memory: Existing memory

        Returns:
            Merged content string
        """
        new_content = new_memory.content
        # Handle both Memory objects and dictionaries
        if hasattr(existing_memory, "content"):
            existing_content = existing_memory.content
        else:
            existing_content = existing_memory.get("content", "")

        # If one is much shorter, use the longer one
        if len(new_content) > len(existing_content) * 1.5:
            return new_content
        elif len(existing_content) > len(new_content) * 1.5:
            return existing_content

        # If similar length, combine them
        merged = f"{existing_content}\n\nAdditional info: {new_content}"

        # Ensure merged content doesn't exceed limits
        if len(merged) > self.config.max_content_length:
            merged = merged[: self.config.max_content_length - 3] + "..."

        return merged

    async def find_similar_memories(self, content: str, limit: int = 5) -> list[dict[str, Any]]:
        """Find memories similar to given content.

        Args:
            content: Content to find similar memories for
            limit: Maximum number of similar memories to return

        Returns:
            List of similar memories with similarity scores
        """
        try:
            # Get recent memories for comparison since get_all_memories doesn't exist
            recent_memories = await self.memory_system.get_relevant_memories("", limit=100)
            all_memories = recent_memories

            # Calculate similarities
            similarities = []

            for memory in all_memories:
                # Handle both Memory objects and dictionaries
                if hasattr(memory, "content"):
                    memory_content = memory.content
                else:
                    memory_content = memory.get("content", "")

                if memory_content:
                    similarity = calculate_text_similarity(content, memory_content)
                    # Define similarity threshold
                    similarity_threshold = 0.3
                    if (
                        similarity > similarity_threshold
                    ):  # Only include reasonably similar memories
                        similarities.append({"memory": memory, "similarity": similarity})

            # Sort by similarity and return top N
            similarities.sort(key=lambda x: x["similarity"], reverse=True)

            return similarities[:limit]

        except Exception:
            self.logger.exception("Error finding similar memories")
            return []

    def get_deduplication_stats(self, results: list[DeduplicationResult]) -> dict[str, Any]:
        """Get statistics about deduplication performance.

        Args:
            results: List of deduplication results

        Returns:
            Statistics dictionary
        """
        if not results:
            return {
                "total_sessions": 0,
                "avg_processing_time": 0.0,
                "total_original": 0,
                "total_deduplicated": 0,
                "total_merged": 0,
                "total_filtered": 0,
                "effectiveness": 0.0,
            }

        stats = {
            "total_sessions": len(results),
            "avg_processing_time": sum(r.processing_time for r in results) / len(results),
            "total_original": sum(r.original_count for r in results),
            "total_deduplicated": sum(r.deduplicated_count for r in results),
            "total_merged": sum(r.merged_count for r in results),
            "total_filtered": sum(r.filtered_count for r in results),
        }

        # Calculate effectiveness (percentage of duplicates removed)
        if stats["total_original"] > 0:
            duplicates_removed = stats["total_merged"] + stats["total_filtered"]
            stats["effectiveness"] = (duplicates_removed / stats["total_original"]) * 100
        else:
            stats["effectiveness"] = 0.0

        return stats
