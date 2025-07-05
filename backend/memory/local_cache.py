#!/usr/bin/env python3
"""Resilient Redis client with local fallback and connection pooling.

This module implements Redis resilience features based on best practices:
- Connection pooling for performance
- Automatic reconnection with exponential backoff
- Local in-memory fallback when Redis is unavailable
- Health monitoring with circuit breaker pattern
- Memory optimization and eviction policies
"""

import asyncio
import logging
import time
import threading
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict
from functools import wraps

try:
    import redis
    import redis.asyncio as aioredis
    from redis.exceptions import ConnectionError, TimeoutError, RedisError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass

class CircuitBreaker:
    """Circuit breaker for Redis connections."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = 'HALF_OPEN'
                    logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    raise CircuitBreakerError("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                    self.failure_count = 0
                    logger.info("Circuit breaker reset to CLOSED state")
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'
                    logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
                
                raise e

class LocalCache:
    """Thread-safe in-memory cache for Redis fallback."""
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._access_times: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            if time.time() - entry['timestamp'] > self.ttl:
                del self._cache[key]
                del self._access_times[key]
                return None
            
            self._access_times[key] = time.time()
            return entry['value']
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        with self._lock:
            # Evict old entries if cache is full
            if len(self._cache) >= self.max_size:
                self._evict_lru()
            
            self._cache[key] = {
                'value': value,
                'timestamp': time.time()
            }
            self._access_times[key] = time.time()
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._access_times[key]
                return True
            return False
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.keys(), key=self._access_times.get)
        del self._cache[lru_key]
        del self._access_times[lru_key]
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

class ResilientRedisClient:
    """Resilient Redis client with automatic fallback and reconnection."""
    
    def __init__(self, redis_client, fallback_enabled: bool = True, max_retries: int = 3):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package is required for ResilientRedisClient")
        
        self.redis_client = redis_client
        self.fallback_enabled = fallback_enabled
        self.max_retries = max_retries
        self.circuit_breaker = CircuitBreaker()
        self.local_cache = LocalCache() if fallback_enabled else None
        self.connection_healthy = True
        self._lock = threading.Lock()
        
        logger.info(f"ResilientRedisClient initialized with fallback: {fallback_enabled}")
    
    def _execute_with_fallback(self, operation: str, func, *args, **kwargs):
        """Execute Redis operation with automatic fallback."""
        for attempt in range(self.max_retries):
            try:
                return self.circuit_breaker.call(func, *args, **kwargs)
            except (ConnectionError, TimeoutError, CircuitBreakerError) as e:
                logger.warning(f"Redis {operation} failed (attempt {attempt + 1}): {e}")
                
                if attempt == self.max_retries - 1:
                    if self.fallback_enabled and operation in ['get', 'set', 'delete']:
                        return self._fallback_operation(operation, *args, **kwargs)
                    else:
                        logger.error(f"Redis {operation} failed after {self.max_retries} attempts")
                        raise e
                
                # Exponential backoff
                time.sleep(2 ** attempt)
        
        return None
    
    def _fallback_operation(self, operation: str, *args, **kwargs):
        """Execute operation using local cache fallback."""
        if not self.local_cache:
            return None
        
        logger.debug(f"Using local cache fallback for {operation}")
        
        if operation == 'get':
            return self.local_cache.get(args[0])
        elif operation == 'set':
            self.local_cache.set(args[0], args[1])
            return True
        elif operation == 'delete':
            return self.local_cache.delete(args[0])
        
        return None
    
    def get(self, key: str):
        """Get value from Redis with fallback."""
        return self._execute_with_fallback('get', self.redis_client.get, key)
    
    def set(self, key: str, value: Any, ex: Optional[int] = None):
        """Set value in Redis with fallback."""
        return self._execute_with_fallback('set', self.redis_client.set, key, value, ex=ex)
    
    def delete(self, *keys):
        """Delete keys from Redis with fallback."""
        return self._execute_with_fallback('delete', self.redis_client.delete, *keys)
    
    def exists(self, key: str):
        """Check if key exists in Redis."""
        return self._execute_with_fallback('exists', self.redis_client.exists, key)
    
    def ping(self):
        """Ping Redis server."""
        return self._execute_with_fallback('ping', self.redis_client.ping)
    
    def flushdb(self):
        """Flush current database."""
        return self._execute_with_fallback('flushdb', self.redis_client.flushdb)
    
    def keys(self, pattern: str = '*'):
        """Get keys matching pattern."""
        return self._execute_with_fallback('keys', self.redis_client.keys, pattern)
    
    def scan(self, cursor: int = 0, match: Optional[str] = None, count: Optional[int] = None):
        """Scan keys with pagination."""
        return self._execute_with_fallback('scan', self.redis_client.scan, cursor, match=match, count=count)
    
    def lpush(self, key: str, *values):
        """Push values to head of list."""
        return self._execute_with_fallback('lpush', self.redis_client.lpush, key, *values)
    
    def lrange(self, key: str, start: int, end: int):
        """Get list slice."""
        return self._execute_with_fallback('lrange', self.redis_client.lrange, key, start, end)
    
    def ltrim(self, key: str, start: int, end: int):
        """Trim list to specified range."""
        return self._execute_with_fallback('ltrim', self.redis_client.ltrim, key, start, end)
    
    def json(self):
        """Access JSON operations (if RedisJSON is available)."""
        return JsonOperations(self)
    
    def ft(self, index_name: str):
        """Access search operations (if RediSearch is available)."""
        return SearchOperations(self, index_name)
    
    def config_get(self, parameter: str):
        """Get configuration parameter."""
        return self._execute_with_fallback('config_get', self.redis_client.config_get, parameter)
    
    def config_set(self, parameter: str, value: str):
        """Set configuration parameter."""
        return self._execute_with_fallback('config_set', self.redis_client.config_set, parameter, value)
    
    def save(self):
        """Save Redis database."""
        return self._execute_with_fallback('save', self.redis_client.save)
    
    def bgsave(self):
        """Background save Redis database."""
        return self._execute_with_fallback('bgsave', self.redis_client.bgsave)
    
    def execute_command(self, *args):
        """Execute raw Redis command."""
        return self._execute_with_fallback('execute_command', self.redis_client.execute_command, *args)
    
    def module_list(self):
        """List loaded Redis modules."""
        return self._execute_with_fallback('module_list', self.redis_client.module_list)
    
    def health_check(self) -> bool:
        """Check Redis connection health."""
        try:
            self.ping()
            self.connection_healthy = True
            return True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            self.connection_healthy = False
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        stats = {
            'connection_healthy': self.connection_healthy,
            'circuit_breaker_state': self.circuit_breaker.state,
            'failure_count': self.circuit_breaker.failure_count,
            'fallback_enabled': self.fallback_enabled
        }
        
        if self.local_cache:
            stats['local_cache_size'] = self.local_cache.size()
        
        return stats

class JsonOperations:
    """JSON operations wrapper for resilient client."""
    
    def __init__(self, resilient_client: ResilientRedisClient):
        self.client = resilient_client
    
    def set(self, key: str, path: str, value: Any):
        """Set JSON value."""
        return self.client._execute_with_fallback(
            'json_set', 
            self.client.redis_client.json().set, 
            key, path, value
        )
    
    def get(self, key: str, path: str = '.'):
        """Get JSON value."""
        return self.client._execute_with_fallback(
            'json_get',
            self.client.redis_client.json().get,
            key, path
        )

class SearchOperations:
    """Search operations wrapper for resilient client."""
    
    def __init__(self, resilient_client: ResilientRedisClient, index_name: str):
        self.client = resilient_client
        self.index_name = index_name
    
    def create_index(self, fields, definition=None):
        """Create search index."""
        return self.client._execute_with_fallback(
            'ft_create_index',
            self.client.redis_client.ft(self.index_name).create_index,
            fields=fields, definition=definition
        )
    
    def search(self, query, query_params=None):
        """Search index."""
        return self.client._execute_with_fallback(
            'ft_search',
            self.client.redis_client.ft(self.index_name).search,
            query, query_params
        )
    
    def info(self):
        """Get index info."""
        return self.client._execute_with_fallback(
            'ft_info',
            self.client.redis_client.ft(self.index_name).info
        )

def create_resilient_redis_client(
    host: str = 'localhost',
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
    fallback_enabled: bool = True,
    max_connections: int = 20,
    socket_timeout: int = 5,
    socket_keepalive: bool = True,
    retry_on_timeout: bool = True,
    health_check_interval: int = 30
) -> ResilientRedisClient:
    """Create a resilient Redis client with connection pooling."""
    
    if not REDIS_AVAILABLE:
        raise ImportError("redis package is required")
    
    # Create connection pool with optimization settings
    connection_pool = redis.ConnectionPool(
        host=host,
        port=port,
        db=db,
        password=password,
        decode_responses=True,
        socket_timeout=socket_timeout,
        socket_keepalive=socket_keepalive,
        socket_keepalive_options={},
        retry_on_timeout=retry_on_timeout,
        max_connections=max_connections,
        socket_connect_timeout=socket_timeout,
        health_check_interval=health_check_interval,
    )
    
    # Create base Redis client
    base_client = redis.Redis(connection_pool=connection_pool)
    
    # Test connection
    try:
        base_client.ping()
        logger.info(f"Redis connection established: {host}:{port}")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        if not fallback_enabled:
            raise
    
    # Return resilient wrapper
    return ResilientRedisClient(base_client, fallback_enabled=fallback_enabled)