"""LLM Call Optimizer
==================

Optimizes LLM usage through caching, batching, and smart call reduction.
Addresses the performance bottleneck from chained LLM calls.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from cachetools import TTLCache

logger = logging.getLogger(__name__)


@dataclass
class LLMCacheEntry:
    """Cached LLM response with metadata."""

    response: str
    timestamp: float
    tokens_used: int
    execution_time_ms: float


class LLMOptimizer:
    """Optimizes LLM calls through caching and smart batching."""

    def __init__(self, redis_client=None, cache_ttl: int = 3600):
        self.redis_client = redis_client
        self.cache_ttl = cache_ttl

        # In-memory cache for fast access
        self.memory_cache = TTLCache(maxsize=1000, ttl=cache_ttl)

        # Call statistics
        self.stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens": 0,
            "total_time_ms": 0,
        }

        logger.info(f"LLM Optimizer initialized with cache TTL: {cache_ttl}s")

    def _generate_cache_key(self, prompt: str, function_type: str, **kwargs) -> str:
        """Generate a cache key for the LLM call."""
        # Include function type and key parameters in hash
        key_data = {"prompt": prompt, "function": function_type, **kwargs}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    async def cached_llm_call(
        self,
        llm_instance,
        prompt: str,
        function_type: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMCacheEntry:
        """Make an LLM call with caching.

        Args:
            llm_instance: The LLM instance to use
            prompt: The prompt to send
            function_type: Type of function (for cache grouping)
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            **kwargs: Additional parameters

        Returns:
            LLMCacheEntry: The cached or fresh response
        """
        cache_key = self._generate_cache_key(
            prompt, function_type, max_tokens=max_tokens, temperature=temperature
        )

        # Check memory cache first
        if cache_key in self.memory_cache:
            self.stats["cache_hits"] += 1
            logger.debug(f"Cache HIT for {function_type}")
            return self.memory_cache[cache_key]

        # Check Redis cache if available
        if self.redis_client:
            try:
                redis_key = f"llm_cache:{cache_key}"
                cached_data = self.redis_client.get(redis_key)
                if cached_data:
                    entry = LLMCacheEntry(**json.loads(cached_data))
                    self.memory_cache[cache_key] = entry
                    self.stats["cache_hits"] += 1
                    logger.debug(f"Redis cache HIT for {function_type}")
                    return entry
            except Exception as e:
                logger.error(f"Redis cache error: {e}")

        # Cache miss - make the actual LLM call
        self.stats["cache_misses"] += 1
        self.stats["total_calls"] += 1

        start_time = time.time()

        try:
            # Make the LLM call - ensure stop tokens are always included
            kwargs_with_stop = kwargs.copy()
            kwargs_with_stop.setdefault("stop", ["[/INST]", "[/SYSTEM_PROMPT]"])

            response = llm_instance.create_completion(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature, **kwargs_with_stop
            )

            response_text = response["choices"][0]["text"].strip()
            tokens_used = response.get("usage", {}).get("total_tokens", 0)
            execution_time = (time.time() - start_time) * 1000

            # Create cache entry
            entry = LLMCacheEntry(
                response=response_text,
                timestamp=time.time(),
                tokens_used=tokens_used,
                execution_time_ms=execution_time,
            )

            # Update statistics
            self.stats["total_tokens"] += tokens_used
            self.stats["total_time_ms"] += execution_time

            # Cache the result
            self.memory_cache[cache_key] = entry

            if self.redis_client:
                try:
                    redis_key = f"llm_cache:{cache_key}"
                    self.redis_client.setex(redis_key, self.cache_ttl, json.dumps(entry.__dict__))
                except Exception as e:
                    logger.error(f"Failed to cache in Redis: {e}")

            logger.debug(
                f"LLM call completed for {function_type}: "
                f"{tokens_used} tokens, {execution_time:.1f}ms"
            )

            return entry

        except Exception as e:
            logger.error(f"LLM call failed for {function_type}: {e}")
            raise

    async def classify_query_cached(self, llm_instance, user_prompt: str) -> str:
        """Cached query classification."""
        prompt = f"""Classify this user query into one of these categories:
- conversation: General chat/questions
- instruction: Commands or requests to do something
- preference: Statements about likes/dislikes
- fact: Factual statements or information

Query: {user_prompt}

Category:"""

        result = await self.cached_llm_call(
            llm_instance, prompt, "query_classification", max_tokens=10, temperature=0.1
        )

        return result.response.lower().strip()

    async def analyze_importance_cached(self, llm_instance, content: str) -> float:
        """Cached importance analysis."""
        prompt = f"""Rate the importance of this content for future conversations on a scale of 0.0 to 1.0:

Content: {content}

Importance (0.0-1.0):"""

        result = await self.cached_llm_call(
            llm_instance, prompt, "importance_analysis", max_tokens=10, temperature=0.1
        )

        try:
            # Extract numerical value
            importance = float(result.response.strip())
            return max(0.0, min(1.0, importance))  # Clamp to valid range
        except ValueError:
            logger.warning(f"Failed to parse importance: {result.response}")
            return 0.5  # Default importance

    async def batch_importance_analysis(self, llm_instance, contents: list[str]) -> list[float]:
        """Batch importance analysis for multiple contents."""
        if not contents:
            return []

        # For small batches, use individual cached calls
        if len(contents) <= 3:
            tasks = [self.analyze_importance_cached(llm_instance, content) for content in contents]
            return await asyncio.gather(*tasks)

        # For larger batches, use a single batched call
        prompt = f"""Rate the importance of each content item for future conversations on a scale of 0.0 to 1.0.
Respond with only the numbers, one per line.

Contents:
{chr(10).join(f"{i + 1}. {content}" for i, content in enumerate(contents))}

Importance scores (0.0-1.0):"""

        result = await self.cached_llm_call(
            llm_instance,
            prompt,
            "batch_importance_analysis",
            max_tokens=len(contents) * 10,
            temperature=0.1,
        )

        try:
            scores = []
            lines = result.response.strip().split("\n")
            for _i, line in enumerate(lines[: len(contents)]):
                try:
                    score = float(line.strip())
                    scores.append(max(0.0, min(1.0, score)))
                except ValueError:
                    scores.append(0.5)  # Default for parse errors

            # Fill any missing scores
            while len(scores) < len(contents):
                scores.append(0.5)

            return scores

        except Exception as e:
            logger.error(f"Batch importance analysis failed: {e}")
            return [0.5] * len(contents)  # Default scores

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        hit_rate = 0.0
        if self.stats["total_calls"] > 0:
            hit_rate = self.stats["cache_hits"] / (
                self.stats["cache_hits"] + self.stats["cache_misses"]
            )

        avg_time = 0.0
        if self.stats["total_calls"] > 0:
            avg_time = self.stats["total_time_ms"] / self.stats["total_calls"]

        return {
            "cache_hit_rate": hit_rate,
            "total_calls": self.stats["total_calls"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "total_tokens": self.stats["total_tokens"],
            "avg_response_time_ms": avg_time,
            "cache_size": len(self.memory_cache),
        }

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()

        if self.redis_client:
            try:
                # Clear Redis cache entries
                keys = self.redis_client.keys("llm_cache:*")
                if keys:
                    self.redis_client.delete(*keys)
                    logger.info(f"Cleared {len(keys)} entries from Redis cache")
            except Exception as e:
                logger.error(f"Failed to clear Redis cache: {e}")

        logger.info("LLM cache cleared")


# Global optimizer instance
_llm_optimizer: LLMOptimizer | None = None


def get_llm_optimizer() -> LLMOptimizer | None:
    """Get the global LLM optimizer instance."""
    return _llm_optimizer


def initialize_llm_optimizer(redis_client=None, cache_ttl: int = 3600) -> LLMOptimizer:
    """Initialize the global LLM optimizer."""
    global _llm_optimizer
    _llm_optimizer = LLMOptimizer(redis_client=redis_client, cache_ttl=cache_ttl)
    logger.info("LLM Optimizer initialized")
    return _llm_optimizer
