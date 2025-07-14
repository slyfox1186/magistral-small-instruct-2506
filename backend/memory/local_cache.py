#!/usr/bin/env python3
"""Local cache and resilient Redis client for memory system."""

import logging
import redis
from typing import Optional, Any

logger = logging.getLogger(__name__)


class ResilientRedisClient:
    """A resilient Redis client with fallback capabilities."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.is_available = True
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis with fallback."""
        try:
            return self.redis_client.get(key)
        except Exception as e:
            logger.warning(f"Redis get failed for key {key}: {e}")
            self.is_available = False
            return None
            
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set value in Redis with fallback."""
        try:
            return self.redis_client.set(key, value, ex=ex)
        except Exception as e:
            logger.warning(f"Redis set failed for key {key}: {e}")
            self.is_available = False
            return False
            
    def delete(self, key: str) -> bool:
        """Delete key from Redis with fallback."""
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.warning(f"Redis delete failed for key {key}: {e}")
            self.is_available = False
            return False