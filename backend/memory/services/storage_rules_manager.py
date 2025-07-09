"""Storage Rules Manager - Enhanced memory management with user-configurable rules.

This module implements the valuable suggestions from the LLM:
1. Dynamic Storage Limits - with configurable capacity management
2. Customizable Storage Rules - externalized from magic numbers

Designed to integrate with the existing sophisticated Redis-based memory system.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class StorageRulesManager:
    """Manages configurable storage rules for memory management."""

    def __init__(self, config_path: Path | None = None):
        """Initialize with storage rules configuration."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "storage_rules.yaml"

        self.config_path = config_path
        self.rules = self._load_rules()

    def _load_rules(self) -> dict[str, Any]:
        """Load storage rules from YAML configuration."""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)["storage_rules"]
        except Exception as e:
            logger.error(f"Failed to load storage rules: {e}")
            return self._get_default_rules()

    def _get_default_rules(self) -> dict[str, Any]:
        """Fallback default rules if config loading fails."""
        return {
            "capacity_management": {
                "stm_max_entries": 10000,
                "ltm_max_entries": 100000,
                "pruning_strategy": "importance_based",
                "stm_min_importance": 0.1,
                "ltm_min_importance": 0.3,
                "never_prune": [
                    {"importance_score": 0.9},
                    {"circles": ["identity"]},
                    {"tags": ["core", "permanent", "critical"]},
                ],
            },
            "importance_scoring": {
                "factors": {
                    "recency_weight": 0.3,
                    "frequency_weight": 0.2,
                    "circle_weight": 0.3,
                    "tag_weight": 0.2,
                }
            },
        }

    def should_prune_memory(self, memory: dict[str, Any]) -> bool:
        """Determine if a memory should be pruned based on storage rules.

        Args:
            memory: Memory object with fields like importance_score, circle, tags, timestamp

        Returns:
            bool: True if memory should be pruned, False to keep
        """
        # Never prune rules
        never_prune_rules = self.rules["capacity_management"].get("never_prune", [])

        # Check each never prune rule
        for rule in never_prune_rules:
            # Check importance threshold
            if "importance_score" in rule:
                min_importance = rule["importance_score"]
                if memory.get("importance_score", 0) >= min_importance:
                    return False

            # Check protected circles
            if "circles" in rule:
                protected_circles = rule["circles"]
                if memory.get("circle") in protected_circles:
                    return False

            # Check protected tags
            if "tags" in rule:
                protected_tags = rule["tags"]
                memory_tags = memory.get("tags", [])
                if any(tag in protected_tags for tag in memory_tags):
                    return False

        # Check circle-specific policies
        circle_policies = self.rules.get("circle_policies", {})
        memory_circle = memory.get("circle")

        if memory_circle in circle_policies:
            policy = circle_policies[memory_circle]
            min_retention_days = policy.get("min_retention_days", 0)

            # Check if memory is within minimum retention period
            if memory.get("timestamp"):
                # Handle both datetime objects and timestamps
                timestamp = memory["timestamp"]
                if isinstance(timestamp, (int, float)):
                    memory_datetime = datetime.fromtimestamp(timestamp)
                else:
                    memory_datetime = timestamp

                memory_age = datetime.now() - memory_datetime
                if memory_age.days < min_retention_days:
                    return False

        # If none of the protection rules apply, memory can be pruned
        return True

    def calculate_enhanced_importance(
        self, memory: dict[str, Any], access_frequency: int = 0
    ) -> float:
        """Calculate enhanced importance score using configurable factors.

        Args:
            memory: Memory object
            access_frequency: How often this memory has been accessed

        Returns:
            float: Enhanced importance score (0.0 to 1.0)
        """
        factors = self.rules["importance_scoring"]["factors"]
        base_importance = memory.get("importance_score", 0.5)

        # Recency factor
        recency_score = 0.5  # Default
        if memory.get("timestamp"):
            # Handle both datetime objects and timestamps
            timestamp = memory["timestamp"]
            if isinstance(timestamp, (int, float)):
                memory_datetime = datetime.fromtimestamp(timestamp)
            else:
                memory_datetime = timestamp

            age_hours = (datetime.now() - memory_datetime).total_seconds() / 3600
            recency_score = max(0.1, 1.0 - (age_hours / (24 * 30)))  # Decay over 30 days

        # Frequency factor
        frequency_score = min(1.0, access_frequency / 10.0)  # Normalize to 0-1

        # Circle factor (from existing circles.yaml)
        circle_score = 0.5  # Default
        memory_circle = memory.get("circle")
        if memory_circle:
            # This would integrate with existing circle priorities
            circle_priorities = {
                "identity": 0.9,
                "relationships": 0.8,
                "experiences": 0.7,
                "knowledge": 0.6,
                "temporal": 0.5,
                "context": 0.9,
                "communication": 0.7,
            }
            circle_score = circle_priorities.get(memory_circle, 0.5)

        # Tag factor
        tag_score = 0.5  # Default
        memory_tags = memory.get("tags", [])
        tag_importance = self.rules["importance_scoring"].get("tag_importance", {})

        if memory_tags:
            tag_scores = [tag_importance.get(tag, 0.5) for tag in memory_tags]
            tag_score = max(tag_scores) if tag_scores else 0.5

        # Weighted combination
        enhanced_importance = (
            base_importance * 0.4  # Base importance still matters
            + recency_score * factors["recency_weight"]
            + frequency_score * factors["frequency_weight"]
            + circle_score * factors["circle_weight"]
            + tag_score * factors["tag_weight"]
        )

        return min(1.0, enhanced_importance)

    def get_capacity_limits(self) -> dict[str, int]:
        """Get current capacity limits for STM and LTM."""
        capacity = self.rules["capacity_management"]
        return {
            "stm_max": capacity.get("stm_max_entries", 10000),
            "ltm_max": capacity.get("ltm_max_entries", 100000),
        }

    def get_pruning_strategy(self) -> str:
        """Get current pruning strategy."""
        return self.rules["capacity_management"].get("pruning_strategy", "importance_based")

    def should_enable_consolidation(self) -> bool:
        """Check if memory consolidation is enabled."""
        return self.rules.get("advanced_pruning", {}).get("enable_consolidation", True)

    def get_similarity_threshold(self) -> float:
        """Get similarity threshold for memory consolidation."""
        return self.rules.get("advanced_pruning", {}).get("similarity_threshold", 0.85)

    def requires_user_approval(self, importance_score: float) -> bool:
        """Check if user approval is required for deletion."""
        threshold = self.rules.get("advanced_pruning", {}).get(
            "require_approval_above_importance", 0.8
        )
        return importance_score >= threshold


# Utility function for easy integration
def create_storage_rules_manager() -> StorageRulesManager:
    """Factory function to create a storage rules manager."""
    return StorageRulesManager()
