"""Memory Schema Definitions
========================

Pydantic models for the memory system data structures.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class MemorySource(str, Enum):
    """Source of memory creation"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MemoryCircle(BaseModel):
    """Memory circle/category definition"""

    name: str
    description: str
    priority: float = Field(ge=0.0, le=1.0)
    decay_rate: float = Field(ge=0.0, le=1.0)
    color: str = "#808080"
    sub_circles: list[str] = []


class MemoryBase(BaseModel):
    """Base memory structure shared by STM and LTM"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    source: MemorySource
    embedding: list[float] | None = None
    tags: list[str] = []
    circle: str = "experiences"
    metadata: dict[str, Any] = {}

    @model_validator(mode="after")
    def extract_hashtags(self):
        """Extract hashtags from content if not provided"""
        if not self.tags and self.content:
            # Simple hashtag extraction without regex
            hashtags = []
            words = self.content.split()
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
            self.tags = hashtags
        return self


class ShortTermMemory(MemoryBase):
    """Short-term memory structure with TTL"""

    ttl_hours: int = 48
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)

    def to_redis_dict(self) -> dict[str, Any]:
        """Convert to Redis-compatible dictionary"""
        data = self.dict()
        # Convert embedding to JSON string for Redis
        if data.get("embedding"):
            import json

            data["embedding"] = json.dumps(data["embedding"])
        return data

    @classmethod
    def from_redis_dict(cls, data: dict[str, Any]) -> "ShortTermMemory":
        """Create from Redis dictionary"""
        # Parse embedding from JSON string
        if data.get("embedding") and isinstance(data["embedding"], str):
            import json

            data["embedding"] = json.loads(data["embedding"])
        return cls(**data)


class LongTermMemory(MemoryBase):
    """Long-term consolidated memory"""

    original_memories: list[str] = []  # STM IDs that were consolidated
    retrieval_score: float = Field(default=1.0, ge=0.0, le=1.0)
    last_accessed: float = Field(default_factory=lambda: datetime.now().timestamp())
    access_count: int = 0
    links: list[str] = []  # Related LTM IDs
    user_approved: bool = True
    consolidation_summary: str | None = None

    def update_access(self):
        """Update access tracking"""
        self.last_accessed = datetime.now().timestamp()
        self.access_count += 1

    def calculate_decay(self, decay_rate: float) -> float:
        """Calculate retrieval score decay based on time"""
        time_since_access = datetime.now().timestamp() - self.last_accessed
        days_elapsed = time_since_access / 86400  # Convert to days

        # Exponential decay
        import math

        decayed_score = self.retrieval_score * math.exp(-decay_rate * days_elapsed)
        return max(0.1, decayed_score)  # Minimum score of 0.1

    def to_redis_dict(self) -> dict[str, Any]:
        """Convert to Redis-compatible dictionary"""
        data = self.dict()
        # Convert lists to JSON strings
        import json

        if data.get("embedding"):
            data["embedding"] = json.dumps(data["embedding"])
        if data.get("original_memories"):
            data["original_memories"] = json.dumps(data["original_memories"])
        if data.get("links"):
            data["links"] = json.dumps(data["links"])
        return data

    @classmethod
    def from_redis_dict(cls, data: dict[str, Any]) -> "LongTermMemory":
        """Create from Redis dictionary"""
        import json

        # Parse JSON strings back to lists
        if data.get("embedding") and isinstance(data["embedding"], str):
            data["embedding"] = json.loads(data["embedding"])
        if data.get("original_memories") and isinstance(data["original_memories"], str):
            data["original_memories"] = json.loads(data["original_memories"])
        if data.get("links") and isinstance(data["links"], str):
            data["links"] = json.loads(data["links"])
        return cls(**data)


class MemorySearchResult(BaseModel):
    """Search result with relevance scoring"""

    memory: ShortTermMemory | LongTermMemory
    score: float = Field(ge=0.0, le=1.0)
    match_type: str  # "vector", "tag", "metadata", "hybrid"
    highlights: list[str] = []  # Matched portions

    class Config:
        arbitrary_types_allowed = True


class ConsolidationCandidate(BaseModel):
    """Candidate memories for consolidation"""

    memory_ids: list[str]
    suggested_summary: str
    common_tags: list[str]
    circle: str
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: float = Field(default_factory=lambda: datetime.now().timestamp())

    def to_prompt(self) -> str:
        """Generate user prompt for approval"""
        return f"""
Memory Consolidation Suggestion:

Common themes: {", ".join(self.common_tags)}
Category: {self.circle}

Suggested summary:
{self.suggested_summary}

Based on {len(self.memory_ids)} related memories.
"""


__all__ = [
    "ConsolidationCandidate",
    "LongTermMemory",
    "MemoryBase",
    "MemoryCircle",
    "MemorySearchResult",
    "MemorySource",
    "ShortTermMemory",
]
